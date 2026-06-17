#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/classifier/trainer.h"
#include "core/classifier/learning_config.h"

#include "core/lib/cache.h"
#include "core/lib/matrix_cache.h"
#include "core/lib/neural_network_loader.h"
#include "core/lib/neural_network_applier.h"
#include "core/lib/directory_dataset.h"
#include "core/lib/neural_network.h"

#include "png/pngreader.h"
#include "matrix/matrix.h"

#include <CLI11/CLI11.hpp>
#include <nlohmann/json.hpp>

namespace Neural {
namespace Classifier {

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(EpochLog,
    epochNumber,
    learningRate,
    trainAccuracy,
    bestTrainAccuracy,
    stagnateEpochsCount
);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TestStatistics, passedTests, totalTests);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TestResult, stats);

}
}


namespace {

bool validateDataset(const std::filesystem::path& datasetPath, const Neural::NeuralNetwork& network) {
    const auto classCount = network.outputSize();
    for (int i = 0; i < classCount; ++i) {
        const auto subdirectory = datasetPath / std::to_string(i);
        if (!std::filesystem::exists(subdirectory) || !std::filesystem::is_directory(subdirectory)) {
            return false;
        }
    }
    return true;
}

const char* activationName(Neural::ActivationType a) {
    switch (a) {
        case Neural::ActivationType::Sigmoid:
            return "sigmoid";
        case Neural::ActivationType::ReLU:
            return "relu";
        case Neural::ActivationType::Tanh:
            return "tanh";
        case Neural::ActivationType::Softmax:
            return "softmax";
    }
    return "?";
}

// Loads `config` from a JSON file (as written by the "train" subcommand's config dump).
// Leaves `config` untouched if `path` is empty. Returns false (after printing an error) on
// a missing file or malformed JSON.
template <typename Config>
bool loadJsonConfig(const std::string& path, Config& config, const char* label) {
    if (path.empty()) {
        return true;
    }
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open " << label << " config file: " << path << "\n";
        return false;
    }
    try {
        nlohmann::json j;
        in >> j;
        config = j.get<Config>();
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse " << label << " config (" << path << "): " << e.what() << "\n";
        return false;
    }
    return true;
}

std::size_t maxProbabilityClassIndex(const Matrix& output) {
    if (output.getRows() != 1 || output.getCols() == 0) {
        throw std::runtime_error("Network output must be a single row (1xN)");
    }
    std::size_t best = 0;
    double bestValue = output(0, 0);
    for (std::size_t i = 1; i < output.getCols(); ++i) {
        const double v = output(0, i);
        if (v > bestValue) {
            bestValue = v;
            best = i;
        }
    }
    return best;
}

int runPredictImage(
    const std::string& networkPath,
    const std::string& imagePath,
    const std::size_t imageHeight,
    const std::size_t imageWidth
) {
    auto loaded = Neural::LoadNetwork(networkPath);
    if (!loaded) {
        std::cerr << "Failed to load network: " << networkPath << "\n";
        return 5;
    }

    Matrix input = PngUtils::fromImage(imagePath, imageHeight, imageWidth)
        .transform(1, imageHeight * imageWidth);
    Neural::NeuralNetworkApplier applier(std::move(loaded.value()));
    const Matrix out = applier.predict(input);
    const std::size_t digit = maxProbabilityClassIndex(out);
    std::cout << digit << '\n';
    return 0;
}

} // namespace


int main(int argc, char** argv) {
    CLI::App app{"MatrixGui headless classifier CLI"};
    app.require_subcommand(1);

    // ---- predict ----
    CLI::App* predictCmd = app.add_subcommand("predict", "Classify a single PNG image using a trained network");

    std::string predictNetworkPath;
    std::string imagePath;
    std::size_t imageWidth = 28;
    std::size_t imageHeight = 28;

    predictCmd->add_option("--network", predictNetworkPath, "Trained network file (.wgt) to load")
        ->required()
        ->check(CLI::ExistingFile);
    predictCmd->add_option("--image", imagePath, "PNG image to classify")
        ->required()
        ->check(CLI::ExistingFile);
    predictCmd->add_option("--dataset-img-width", imageWidth, "Width of test images in pixels.")
        ->capture_default_str();
    predictCmd->add_option("--dataset-img-height", imageHeight, "Height of test images in pixels.")
        ->group("Dataset")
        ->capture_default_str();


    // ---- train ----
    CLI::App* trainCmd = app.add_subcommand("train", "Train (or continue training) a classifier network on a directory dataset");

    std::string trainNetworkPath;
    std::string workingDirectoryPath = "training-data";
    std::string datasetPath;
    std::string trainDatasetPath;
    std::string testDatasetPath;
    Neural::Classifier::LearningConfig learningConfig{};
    Neural::NeuralNetworkConfiguration netConfig{};
    netConfig.layersSizes = {28 * 28, 50, 20, 10};
    std::size_t testFileLimit = 0;

    // --learning-config/--network-config load their respective structs from JSON files (as
    // written by this command's own config dump) before the per-field options below are
    // registered, so the loaded values become the new defaults: any per-field flag passed on
    // the command line still overrides just that field.
    const auto findOptionValue = [argc, argv](const std::string& flag) -> std::string {
        const std::string eqPrefix = flag + "=";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == flag && i + 1 < argc) {
                return argv[i + 1];
            }
            if (arg.rfind(eqPrefix, 0) == 0) {
                return arg.substr(eqPrefix.size());
            }
        }
        return {};
    };

    std::string learningConfigPath = findOptionValue("--learning-config");
    std::string networkConfigPath = findOptionValue("--network-config");
    if (!loadJsonConfig(learningConfigPath, learningConfig, "learning")) {
        return 4;
    }
    if (!loadJsonConfig(networkConfigPath, netConfig, "network")) {
        return 4;
    }

    trainCmd->add_option("--network", trainNetworkPath, "Network save path (.wgt). Created if missing.")
        ->required();

    trainCmd->add_option("--working-directory", workingDirectoryPath, "Working directory for training data.")
        ->capture_default_str();

    trainCmd->add_option("--learning-config", learningConfigPath,
            "Load training hyperparameters from a JSON file (as written to learning-config.json); "
            "flags below override individual fields from the loaded config.")
        ->group("Config")
        ->check(CLI::ExistingFile);
    trainCmd->add_option("--network-config", networkConfigPath,
            "Load network topology from a JSON file (as written to network-config.json), used only "
            "when creating a new network; flags below override individual fields from the loaded config.")
        ->group("Config")
        ->check(CLI::ExistingFile);

    trainCmd->add_option("--dataset", datasetPath,
            "Dataset root used for both training and testing (subdirs 0..9). "
            "Overridden per-side by --train-dataset/--test-dataset.")
        ->group("Dataset");
    trainCmd->add_option("--train-dataset", trainDatasetPath,
            "Training data root (subdirs 0..9); falls back to --dataset if not set.")
        ->group("Dataset");
    trainCmd->add_option("--test-dataset", testDatasetPath,
            "Testing data root (subdirs 0..9); falls back to --dataset if not set.")
        ->group("Dataset");
    trainCmd->add_option("--test-file-limit", testFileLimit,
            "Max test files per class (0 = all). Runs evaluation after training.")
        ->group("Dataset")
        ->capture_default_str();
    trainCmd->add_option("--dataset-img-width", imageWidth, "Width of test images in pixels.")
        ->group("Dataset")
        ->capture_default_str();
    trainCmd->add_option("--dataset-img-height", imageHeight, "Height of test images in pixels.")
        ->group("Dataset")
        ->capture_default_str();

    trainCmd->add_option("--layers", netConfig.layersSizes,
            "Layer sizes, comma-separated (used only when creating a new network)")
        ->delimiter(',')
        ->group("Network")
        ->capture_default_str();

    const std::map<std::string, Neural::ActivationType> activationMap{
        {"sigmoid", Neural::ActivationType::Sigmoid}, {"relu", Neural::ActivationType::ReLU},
        {"tanh", Neural::ActivationType::Tanh}, {"softmax", Neural::ActivationType::Softmax}};

    trainCmd->add_option("--hidden-activation", netConfig.hiddenActivation,
            "Hidden layer activation (used only when creating a new network)")
        ->transform(CLI::CheckedTransformer(activationMap, CLI::ignore_case))
        ->group("Network")
        ->capture_default_str();
    trainCmd->add_option("--output-activation", netConfig.outputActivation,
            "Output layer activation (used only when creating a new network)")
        ->transform(CLI::CheckedTransformer(activationMap, CLI::ignore_case))
        ->group("Network")
        ->capture_default_str();

    trainCmd->add_option("--initial-lr", learningConfig.initialLearningRate, "Initial learning rate")
        ->group("Training")->capture_default_str();
    trainCmd->add_option("--min-lr", learningConfig.minLearningRate, "Minimum learning rate")
        ->group("Training")->capture_default_str();
    trainCmd->add_option("--lr-decay", learningConfig.learningRateDecay, "Learning rate decay factor")
        ->group("Training")->capture_default_str();
    trainCmd->add_option("--max-epochs", learningConfig.maxEpochs, "Maximum training epochs")
        ->group("Training")->capture_default_str();
    trainCmd->add_option("--patience", learningConfig.patience, "Early-stopping patience (epochs)")
        ->group("Training")->capture_default_str();
    trainCmd->add_option("--inner-epochs", learningConfig.innerEpochs, "Backprop passes per sample per epoch")
        ->group("Training")->capture_default_str();
    trainCmd->add_option("--dropout", learningConfig.dropoutRate, "Dropout rate")
        ->group("Training")->capture_default_str();
    trainCmd->add_option("--dataset-limit-per-label,--dataset-file-limit", learningConfig.datasetLimitPerLabel,
            "Max training files per class (0 = all)")
        ->group("Training")->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    if (*predictCmd) {
        try {
            return runPredictImage(predictNetworkPath, imagePath, imageHeight, imageWidth);
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            return 4;
        }
    }

    // *trainCmd
    std::string trainingPath = !trainDatasetPath.empty() ? trainDatasetPath : datasetPath;
    std::string testingPath  = !testDatasetPath.empty()  ? testDatasetPath  : datasetPath;

    if (trainingPath.empty() || testingPath.empty()) {
        std::cerr << "Specify data directories: use --dataset <dir> for both train and test, or set\n"
                     "  --train-dataset and/or --test-dataset (unspecified side falls back to --dataset).\n";
        return 2;
    }

    if (netConfig.layersSizes.size() < 2) {
        std::cerr << "--layers must list at least input and output sizes (two or more integers).\n";
        return 4;
    }

    std::cout << "Starting with parameters:\n"
              << "\tNetwork path: " << trainNetworkPath << "\n"
              << "\tTraining dataset: " << trainingPath << "\n"
              << "\tTesting dataset: " << testingPath << "\n"
              << "\tLayers (for new network): ";
    for (std::size_t i = 0; i < netConfig.layersSizes.size(); ++i) {
        std::cout << netConfig.layersSizes[i] << (i + 1 < netConfig.layersSizes.size() ? ", " : "");
    }
    std::cout << "\n"
              << "\tHidden activation: " << activationName(netConfig.hiddenActivation) << "\n"
              << "\tOutput activation: " << activationName(netConfig.outputActivation) << "\n"
              << "\tInitial LR: " << learningConfig.initialLearningRate << "\n"
              << "\tMin LR: " << learningConfig.minLearningRate << "\n"
              << "\tLR decay: " << learningConfig.learningRateDecay << "\n"
              << "\tMax epochs: " << learningConfig.maxEpochs << "\n"
              << "\tPatience: " << learningConfig.patience << "\n"
              << "\tInner epochs: " << learningConfig.innerEpochs << "\n"
              << "\tDropout: " << learningConfig.dropoutRate << "\n"
              << "\tDataset limit per label: " << learningConfig.datasetLimitPerLabel << "\n"
              << "\tTest file limit per label: " << testFileLimit << "\n"
              << std::endl;

    Neural::Classifier::Trainer recognizer;

    Neural::NeuralNetwork network;
    if (auto loadedMaybe = Neural::LoadNetwork(trainNetworkPath); loadedMaybe.has_value()) {
        network = *loadedMaybe;
    } else {
        network = Neural::CreateNetwork(netConfig);
    }

    if (network.inputSize() != imageWidth * imageHeight) {
        std::cerr << "Invalid image sizes: " << imageWidth << "x" << imageHeight
            << " = " << imageWidth * imageHeight << ", while network input layer is " << network.inputSize() << "\n";
        return 5;
    }

    if (!validateDataset(trainingPath, network)) {
        std::cerr << "Invalid training dataset (expected subdirectories 0..9): " << trainingPath << "\n";
        return 3;
    }
    if (!validateDataset(testingPath, network)) {
        std::cerr << "Invalid testing dataset (expected subdirectories 0..9): " << testingPath << "\n";
        return 3;
    }

    cache::LRUCache<std::filesystem::path, Matrix> pngCache;
    const auto reader = [&pngCache, imageWidth, imageHeight](const std::filesystem::path& path) {
        if (const auto cached = pngCache.get(path); cached.has_value()) {
            return *cached;
        }

        const auto result = PngUtils::fromImage(path.string(), imageHeight, imageWidth)
            .transform(1, imageHeight * imageWidth);
        pngCache.put(path, result);
        return result;
    };
    auto trainingDataset = std::make_unique<Neural::DirectoryDataset>(trainingPath);
    auto testingDataset = std::make_unique<Neural::DirectoryDataset>(testingPath);
    trainingDataset->setFileReader(reader);
    testingDataset->setFileReader(reader);


    const std::string currentRunDirectoryName
        = std::filesystem::path(trainNetworkPath).filename().string() + "_"
        + std::to_string(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count()
        );
    const auto currentWorkingPath = std::filesystem::path(workingDirectoryPath) / currentRunDirectoryName;
    std::filesystem::create_directories(currentWorkingPath);

    // Save configs
    {
        std::ofstream learningConfigDump(currentWorkingPath / "learning-config.json");
        std::ofstream neuralNetworkConfigDump(currentWorkingPath / "network-config.json");
        nlohmann::json learningConfigSerialized = learningConfig;
        nlohmann::json networkConfigSerialized = network.config;
        learningConfigDump << learningConfigSerialized.dump(2);
        neuralNetworkConfigDump << networkConfigSerialized.dump(2);
    }

    recognizer.setNetwork(network);
    recognizer.setTrainingDataset(std::move(trainingDataset));
    recognizer.setTestingDataset(std::move(testingDataset));
    std::ofstream learningLog(currentWorkingPath / "log.jsonl", std::ios_base::app);
    std::ofstream testingLog(currentWorkingPath / "testing-log.jsonl", std::ios_base::app);
    recognizer.setEpochCallback([&] (const Neural::Classifier::EpochLog& log) {
        Neural::SaveNetwork(recognizer.getNetwork(), trainNetworkPath);
        Neural::SaveNetwork(recognizer.getNetwork(), currentWorkingPath / (std::string("backup-") + std::to_string(log.epochNumber)));
        learningLog << nlohmann::json(log).dump() << std::endl;

        if ((log.epochNumber + 1) % 20 == 0) {
            nlohmann::json testResult = recognizer.test(testFileLimit);
            testResult["epoch"] = log.epochNumber;
            testingLog << testResult.dump() << std::endl;
        }
    });

    recognizer.train(learningConfig);

    const auto testResult = recognizer.test(testFileLimit);
    const auto total = testResult.getTotal();
    if (total.totalTests > 0) {
        std::cout << "Final test accuracy: " << 100.0 * total.passedTests / total.totalTests << "% ("
                  << total.passedTests << "/" << total.totalTests << ")\n";
    }

    nlohmann::json json = testResult;
    json["epoch"] = -1;
    testingLog << json.dump() << std::endl;

    return 0;
}
