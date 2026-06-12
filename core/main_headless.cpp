#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/classifier/trainer.h"
#include "core/classifier/learning_config.h"

#include "core/lib/neural_network_loader.h"
#include "core/lib/neural_network_applier.h"
#include "core/lib/directory_dataset.h"
#include "core/lib/neural_network.h"

#include "png/pngreader.h"
#include "matrix/matrix.h"

#include <CLI11/CLI11.hpp>


namespace {

bool validateDataset(const std::filesystem::path& datasetPath) {
    for (int i = 0; i < 10; ++i) {
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

int runPredictImage(const std::string& networkPath, const std::string& imagePath) {
    auto loaded = Neural::LoadNetwork(networkPath);
    if (!loaded) {
        std::cerr << "Failed to load network: " << networkPath << "\n";
        return 5;
    }

    PngUtils::Cache cache;
    Matrix input = PngUtils::fromImage(imagePath, 28, 28, cache).transform(1, 28 * 28);
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

    predictCmd->add_option("--network", predictNetworkPath, "Trained network file (.wgt) to load")
        ->required()
        ->check(CLI::ExistingFile);
    predictCmd->add_option("--image", imagePath, "PNG image to classify")
        ->required()
        ->check(CLI::ExistingFile);

    // ---- train ----
    CLI::App* trainCmd = app.add_subcommand("train", "Train (or continue training) a classifier network on a directory dataset");

    std::string trainNetworkPath;
    std::string datasetPath;
    std::string trainDatasetPath;
    std::string testDatasetPath;
    Neural::Classifier::LearningConfig learningConfig{};
    Neural::NeuralNetworkConfiguration netConfig{};
    netConfig.layersSizes = {28 * 28, 50, 20, 10};
    std::size_t testFileLimit = 0;

    trainCmd->add_option("--network", trainNetworkPath, "Network save path (.wgt). Created if missing.")
        ->required();

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
            return runPredictImage(predictNetworkPath, imagePath);
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

    if (!validateDataset(trainingPath)) {
        std::cerr << "Invalid training dataset (expected subdirectories 0..9): " << trainingPath << "\n";
        return 3;
    }
    if (!validateDataset(testingPath)) {
        std::cerr << "Invalid testing dataset (expected subdirectories 0..9): " << testingPath << "\n";
        return 3;
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
    PngUtils::Cache pngCache;
    const auto reader = [&pngCache](const std::filesystem::path& path) {
        return PngUtils::fromImage(path.string(), 28, 28, pngCache).transform(1, 28 * 28);
    };
    auto trainingDataset = std::make_unique<Neural::DirectoryDataset>(trainingPath);
    auto testingDataset = std::make_unique<Neural::DirectoryDataset>(testingPath);
    trainingDataset->setFileReader(reader);
    testingDataset->setFileReader(reader);

    std::optional<Neural::NeuralNetwork> loadedMaybe = Neural::LoadNetwork(trainNetworkPath);
    if (!loadedMaybe) {
        loadedMaybe = Neural::CreateNetwork(netConfig);
    }

    recognizer.setNetwork(loadedMaybe.value());
    recognizer.setTrainingDataset(std::move(trainingDataset));
    recognizer.setTestingDataset(std::move(testingDataset));
    recognizer.setEpochCallback([&recognizer, &trainNetworkPath] {
        Neural::SaveNetwork(recognizer.getNetwork(), trainNetworkPath);
    });

    recognizer.train(learningConfig);

    const auto testResult = recognizer.test(testFileLimit);
    const auto total = testResult.getTotal();
    if (total.totalTests > 0) {
        std::cout << "Final test accuracy: " << 100.0 * total.passedTests / total.totalTests << "% ("
                  << total.passedTests << "/" << total.totalTests << ")\n";
    }

    return 0;
}
