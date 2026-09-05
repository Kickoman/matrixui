#include "cli/classifier/commands.h"

#include "cli/lib/png_reader.h"

#include "core/classifier/trainer.h"
#include "core/classifier/trainer_json.h"

#include "core/lib/file_stream.h"
#include "core/nn/directory_dataset.h"
#include "core/nn/neural_network.h"
#include "core/nn/neural_network_applier.h"
#include "core/nn/neural_network_loader.h"

#include "core/matrix/matrix.h"
#include "core/png/pngreader.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>

namespace ClassifierCli {

namespace {

bool ValidateDataset(const std::filesystem::path& datasetPath, const Neural::NeuralNetwork& network) {
    return Neural::DirectoryDataset::IsDirectoryValid(datasetPath, network.outputSize());
}

// Deliberately narrower than the NLOHMANN_JSON_SERIALIZE_ENUM in core/nn/layers.h:
// a --network-config may name an activation the --hidden-activation flag rejects.
const char* ActivationName(Neural::ActivationType activation) {
    switch (activation) {
        case Neural::ActivationType::Sigmoid:
            return "sigmoid";
        case Neural::ActivationType::ReLU:
            return "relu";
        case Neural::ActivationType::Tanh:
            return "tanh";
        case Neural::ActivationType::Softmax:
            return "softmax";
        case Neural::ActivationType::LeakyReLU:
            return "leakyrelu";
    }
    return "?";
}

std::size_t MaxProbabilityClassIndex(const Matrix& output) {
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

void PrintTrainingParameters(
    std::ostream& out,
    const TrainOptions& options,
    const std::string& trainingPath,
    const std::string& testingPath
) {
    out << "Starting with parameters:\n"
        << "\tNetwork path: " << options.networkPath << "\n"
        << "\tTraining dataset: " << trainingPath << "\n"
        << "\tTesting dataset: " << testingPath << "\n"
        << "\tLayers (for new network): ";
    for (std::size_t i = 0; i < options.network.layersSizes.size(); ++i) {
        out << options.network.layersSizes[i] << (i + 1 < options.network.layersSizes.size() ? ", " : "");
    }
    out << "\n"
        << "\tHidden activation: " << ActivationName(options.network.hiddenActivation) << "\n"
        << "\tOutput activation: " << ActivationName(options.network.outputActivation) << "\n"
        << "\tInitial LR: " << options.learning.initialLearningRate << "\n"
        << "\tMin LR: " << options.learning.minLearningRate << "\n"
        << "\tLR decay: " << options.learning.learningRateDecay << "\n"
        << "\tMax epochs: " << options.learning.maxEpochs << "\n"
        << "\tPatience: " << options.learning.patience << "\n"
        << "\tInner epochs: " << options.learning.innerEpochs << "\n"
        << "\tDropout: " << options.learning.dropoutRate << "\n"
        << "\tDataset limit per label: " << options.learning.datasetLimitPerLabel << "\n"
        << "\tTest file limit per label: " << options.testFileLimit << "\n"
        << std::endl;
}

Neural::NeuralNetwork LoadOrCreateNetwork(const TrainOptions& options) {
    if (auto loadedMaybe = Io::TryReadFile(options.networkPath, [](std::istream& in) { return Neural::LoadNetwork(in); }, std::ios::binary);
        loadedMaybe.has_value()) {
        return *loadedMaybe;
    }
    return Neural::CreateNetwork(options.network);
}

// One timestamped directory per run, holding the config dumps, the logs and the
// per-epoch network backups.
std::filesystem::path MakeRunDirectory(const TrainOptions& options) {
    const std::string runDirectoryName
        = std::filesystem::path(options.networkPath).filename().string() + "_"
        + std::to_string(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count()
        );
    const auto path = std::filesystem::path(options.workingDirectory) / runDirectoryName;
    std::filesystem::create_directories(path);
    return path;
}

int RunPredict(std::ostream& out, std::ostream& err, const PredictOptions& options) {
    auto loaded = Io::TryReadFile(options.networkPath, [](std::istream& in) { return Neural::LoadNetwork(in); }, std::ios::binary);
    if (!loaded) {
        err << "Failed to load network: " << options.networkPath << "\n";
        return kSizeMismatch;
    }

    Matrix input = PngUtils::fromImage(options.imagePath, options.imageHeight, options.imageWidth)
        .transform(1, options.imageHeight * options.imageWidth);
    Neural::NeuralNetworkApplier applier(std::move(loaded.value()));
    const Matrix output = applier.predict(input);
    out << MaxProbabilityClassIndex(output) << '\n';
    return kSuccess;
}

}  // namespace

int Predict(std::ostream& out, std::ostream& err, const PredictOptions& options) {
    try {
        return RunPredict(out, err, options);
    } catch (const std::exception& e) {
        err << e.what() << '\n';
        return kBadConfig;
    }
}

int Train(std::ostream& out, std::ostream& err, const TrainOptions& options) {
    const std::string trainingPath = !options.trainDatasetPath.empty() ? options.trainDatasetPath : options.datasetPath;
    const std::string testingPath  = !options.testDatasetPath.empty()  ? options.testDatasetPath  : options.datasetPath;

    if (trainingPath.empty() || testingPath.empty()) {
        err << "Specify data directories: use --dataset <dir> for both train and test, or set\n"
               "  --train-dataset and/or --test-dataset (unspecified side falls back to --dataset).\n";
        return kNoDatasetResolved;
    }

    if (options.network.layersSizes.size() < 2) {
        err << "--layers must list at least input and output sizes (two or more integers).\n";
        return kBadConfig;
    }

    PrintTrainingParameters(out, options, trainingPath, testingPath);

    Neural::Classifier::Trainer recognizer;
    Neural::NeuralNetwork network = LoadOrCreateNetwork(options);

    if (network.inputSize() != options.imageWidth * options.imageHeight) {
        err << "Invalid image sizes: " << options.imageWidth << "x" << options.imageHeight
            << " = " << options.imageWidth * options.imageHeight
            << ", while network input layer is " << network.inputSize() << "\n";
        return kSizeMismatch;
    }

    if (!ValidateDataset(trainingPath, network)) {
        err << "Invalid training dataset (expected subdirectories 0.." << network.outputSize() - 1 << "): " << trainingPath << "\n";
        return kInvalidDataset;
    }
    if (!ValidateDataset(testingPath, network)) {
        err << "Invalid testing dataset (expected subdirectories 0.." << network.outputSize() - 1 << "): " << testingPath << "\n";
        return kInvalidDataset;
    }

    const auto reader = CliLib::MakeCachedPngReader(options.imageWidth, options.imageHeight);
    auto trainingDataset = std::make_unique<Neural::DirectoryDataset>(trainingPath);
    auto testingDataset = std::make_unique<Neural::DirectoryDataset>(testingPath);
    trainingDataset->setFileReader(reader);
    testingDataset->setFileReader(reader);

    const auto currentWorkingPath = MakeRunDirectory(options);

    {
        std::ofstream learningConfigDump(currentWorkingPath / "learning-config.json");
        std::ofstream neuralNetworkConfigDump(currentWorkingPath / "network-config.json");
        nlohmann::json learningConfigSerialized = options.learning;
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
        const auto saveNetwork = [&](const std::filesystem::path& target) {
            Io::WriteFile(target, [&](std::ostream& file) {
                Neural::SaveNetwork(file, recognizer.getNetwork());
            }, std::ios::binary);
        };
        saveNetwork(options.networkPath);
        saveNetwork(currentWorkingPath / (std::string("backup-") + std::to_string(log.epochNumber)));
        learningLog << nlohmann::json(log).dump() << std::endl;

        if ((log.epochNumber + 1) % 20 == 0) {
            nlohmann::json testResult = recognizer.test(options.testFileLimit);
            testResult["epoch"] = log.epochNumber;
            testingLog << testResult.dump() << std::endl;
        }
    });

    recognizer.train(options.learning);

    const auto testResult = recognizer.test(options.testFileLimit);
    const auto total = testResult.getTotal();
    if (total.totalTests > 0) {
        out << "Final test accuracy: " << 100.0 * total.passedTests / total.totalTests << "% ("
            << total.passedTests << "/" << total.totalTests << ")\n";
    }

    nlohmann::json json = testResult;
    json["epoch"] = -1;
    testingLog << json.dump() << std::endl;

    return kSuccess;
}

}  // namespace ClassifierCli
