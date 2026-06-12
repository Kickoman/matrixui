#include <cctype>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

#include "core/classifier/trainer.h"
#include "core/classifier/learning_config.h"

#include "core/lib/neural_network_loader.h"
#include "core/lib/neural_network_applier.h"
#include "core/lib/directory_dataset.h"
#include "core/lib/neural_network.h"

#include "png/pngreader.h"
#include "matrix/matrix.h"


class InputParser {
public:
    InputParser() = default;
    InputParser(const InputParser& other) : tokens(other.tokens) {}
    explicit InputParser(int& argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            this->tokens.push_back(std::string(argv[i]));
        }
    }

    const std::string& getCmdOption(const std::string& option, const std::string& defaultValue = {}) const {
        const auto itr = std::find(this->tokens.cbegin(), this->tokens.cend(), option);
        if (itr != this->tokens.cend() && std::next(itr) != this->tokens.cend()) {
            return *std::next(itr);
        }
        return defaultValue;
    }

    bool cmdOptionExists(const std::string& option) const {
        return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
    }

private:
    std::vector<std::string> tokens;
};


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

std::vector<std::size_t> parseLayers(const std::string& layersParameter) {
    try {
        std::vector<std::size_t> layers;
        if (layersParameter.empty()) {
            return layers;
        }
        std::istringstream stream(layersParameter);
        std::string token;
        while (std::getline(stream, token, ',')) {
            while (!token.empty() && std::isspace(static_cast<unsigned char>(token.front()))) {
                token.erase(0, 1);
            }
            while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back()))) {
                token.pop_back();
            }
            if (token.empty()) {
                continue;
            }
            layers.push_back(std::stoul(token));
        }
        return layers;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse layers: " << e.what() << std::endl;
        throw std::runtime_error("Failed to parse layers");
    }
}

Neural::ActivationType parseActivation(const std::string& s) {
    std::string lower;
    lower.reserve(s.size());
    for (char c : s) {
        lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (lower == "sigmoid") {
        return Neural::ActivationType::Sigmoid;
    }
    if (lower == "relu") {
        return Neural::ActivationType::ReLU;
    }
    if (lower == "tanh") {
        return Neural::ActivationType::Tanh;
    }
    if (lower == "softmax") {
        return Neural::ActivationType::Softmax;
    }
    throw std::runtime_error("Unknown activation \"" + s + "\" (expected sigmoid, relu, tanh, or softmax)");
}

double parseDoubleOpt(const InputParser& cmd, const std::string& option, double defaultValue) {
    if (!cmd.cmdOptionExists(option)) {
        return defaultValue;
    }
    return std::stod(cmd.getCmdOption(option));
}

std::size_t parseSizeOpt(const InputParser& cmd, const std::string& option, std::size_t defaultValue) {
    if (!cmd.cmdOptionExists(option)) {
        return defaultValue;
    }
    const auto v = std::stoull(cmd.getCmdOption(option));
    return static_cast<std::size_t>(v);
}

Neural::Classifier::LearningConfig parseLearningConfig(const InputParser& cmd) {
    Neural::Classifier::LearningConfig d{};
    std::size_t datasetLimitPerLabel = d.datasetLimitPerLabel;
    if (cmd.cmdOptionExists("--dataset-file-limit")) {
        datasetLimitPerLabel = parseSizeOpt(cmd, "--dataset-file-limit", datasetLimitPerLabel);
    }
    if (cmd.cmdOptionExists("--dataset-limit-per-label")) {
        datasetLimitPerLabel = parseSizeOpt(cmd, "--dataset-limit-per-label", datasetLimitPerLabel);
    }
    return {
        .initialLearningRate = parseDoubleOpt(cmd, "--initial-lr", d.initialLearningRate),
        .minLearningRate = parseDoubleOpt(cmd, "--min-lr", d.minLearningRate),
        .learningRateDecay = parseDoubleOpt(cmd, "--lr-decay", d.learningRateDecay),
        .maxEpochs = parseSizeOpt(cmd, "--max-epochs", d.maxEpochs),
        .patience = parseSizeOpt(cmd, "--patience", d.patience),
        .innerEpochs = parseSizeOpt(cmd, "--inner-epochs", d.innerEpochs),
        .datasetLimitPerLabel = datasetLimitPerLabel,
        .dropoutRate = parseDoubleOpt(cmd, "--dropout", d.dropoutRate),
    };
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

Neural::NeuralNetworkConfiguration parseNetworkConfiguration(const InputParser& cmd) {
    static const std::vector<std::size_t> kDefaultLayers = {28 * 28, 50, 20, 10};
    Neural::NeuralNetworkConfiguration config;
    if (cmd.cmdOptionExists("--layers")) {
        config.layersSizes = parseLayers(cmd.getCmdOption("--layers"));
    } else {
        config.layersSizes = kDefaultLayers;
    }
    if (cmd.cmdOptionExists("--hidden-activation")) {
        config.hiddenActivation = parseActivation(cmd.getCmdOption("--hidden-activation"));
    }
    if (cmd.cmdOptionExists("--output-activation")) {
        config.outputActivation = parseActivation(cmd.getCmdOption("--output-activation"));
    }
    return config;
}

void printUsage(const char* argv0) {
    const Neural::Classifier::LearningConfig d{};
    std::cerr
        << "Usage:\n  " << argv0 << " --network <path.wgt> (--dataset <dir> | --train-dataset ... --test-dataset ...)\n"
        << "  " << argv0 << " --network <path.wgt> --predict-image <image.png>\n\n"
        << "Single-image classification (no dataset required):\n"
        << "  --predict-image <path>     Load the network and print the predicted digit (0-9) to stdout.\n\n"
        << "Training mode — required:\n"
        << "  --network <path>           Network save path (.wgt). Created if missing.\n"
        << "  --dataset <dir>            Use the same root for training and testing (subdirs 0..9).\n"
        << "  --train-dataset <dir>     Training data root; if omitted, uses --dataset.\n"
        << "  --test-dataset <dir>      Testing data root; if omitted, uses --dataset.\n"
        << "                             You must end up with both paths: e.g. only --dataset (same\n"
        << "                             tree for train and test), or --train-dataset + --test-dataset,\n"
        << "                             or --dataset plus one override flag.\n\n"
        << "Network (used when creating a new network; ignored when loading existing):\n"
        << "  --layers <n,n,...>         Layer sizes, comma-separated (default 784,50,20,10).\n"
        << "  --hidden-activation <name>  sigmoid | relu | tanh | softmax (default relu).\n"
        << "  --output-activation <name>  sigmoid | relu | tanh | softmax (default softmax).\n\n"
        << "Training (Neural::Classifier::LearningConfig):\n"
        << "  --initial-lr <x>           (default " << d.initialLearningRate << ").\n"
        << "  --min-lr <x>               (default " << d.minLearningRate << ").\n"
        << "  --lr-decay <x>             (default " << d.learningRateDecay << ").\n"
        << "  --max-epochs <n>           (default " << d.maxEpochs << ").\n"
        << "  --patience <n>             (default " << d.patience << ").\n"
        << "  --inner-epochs <n>         Backprop passes per sample per epoch (default " << d.innerEpochs << ").\n"
        << "  --dropout <x>              (default " << d.dropoutRate << ").\n"
        << "  --dataset-limit-per-label <n>   Max training files per class (0 = all; default " << d.datasetLimitPerLabel << ").\n"
        << "  --dataset-file-limit <n>   Same as --dataset-limit-per-label (deprecated alias).\n\n"
        << "Testing:\n"
        << "  --test-file-limit <n>      Max test files per class (0 = all). Runs evaluation after training.\n\n"
        << "  -h, --help                 Show this text.\n";
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
    if (!std::filesystem::exists(imagePath)) {
        std::cerr << "Image not found: " << imagePath << "\n";
        return 6;
    }
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
    InputParser cmd(argc, argv);

    if (cmd.cmdOptionExists("--help") || cmd.cmdOptionExists("-h")) {
        printUsage(argc > 0 ? argv[0] : "matrixgui_headless");
        return 0;
    }

    if (cmd.cmdOptionExists("--predict-image")) {
        if (!cmd.cmdOptionExists("--network")) {
            std::cerr << "--predict-image requires --network\n";
            printUsage(argc > 0 ? argv[0] : "matrixgui_headless");
            return 1;
        }
        const std::string networkName = cmd.getCmdOption("--network");
        const std::string imagePath = cmd.getCmdOption("--predict-image");
        if (networkName.empty() || imagePath.empty()) {
            std::cerr << "--network and --predict-image require values\n";
            return 1;
        }
        try {
            return runPredictImage(networkName, imagePath);
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            return 4;
        }
    }

    if (!cmd.cmdOptionExists("--network")) {
        std::cerr << "Specify network path with --network\n";
        printUsage(argc > 0 ? argv[0] : "matrixgui_headless");
        return 1;
    }

    const std::string networkName = cmd.getCmdOption("--network");
    if (networkName.empty()) {
        std::cerr << "--network requires a value\n";
        return 1;
    }

    std::string trainingPath;
    std::string testingPath;
    if (cmd.cmdOptionExists("--train-dataset")) {
        trainingPath = cmd.getCmdOption("--train-dataset");
    } else if (cmd.cmdOptionExists("--dataset")) {
        trainingPath = cmd.getCmdOption("--dataset");
    }
    if (cmd.cmdOptionExists("--test-dataset")) {
        testingPath = cmd.getCmdOption("--test-dataset");
    } else if (cmd.cmdOptionExists("--dataset")) {
        testingPath = cmd.getCmdOption("--dataset");
    }

    if (trainingPath.empty() || testingPath.empty()) {
        std::cerr << "Specify data directories: use --dataset <dir> for both train and test, or set\n"
                     "  --train-dataset and/or --test-dataset (unspecified side falls back to --dataset).\n";
        printUsage(argc > 0 ? argv[0] : "matrixgui_headless");
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

    Neural::Classifier::LearningConfig learningConfig;
    Neural::NeuralNetworkConfiguration netConfig;
    std::size_t testFileLimit = 0;

    try {
        learningConfig = parseLearningConfig(cmd);
        netConfig = parseNetworkConfiguration(cmd);
        if (cmd.cmdOptionExists("--test-file-limit")) {
            testFileLimit = parseSizeOpt(cmd, "--test-file-limit", 0);
        }
    } catch (const std::exception& e) {
        std::cerr << "Invalid arguments: " << e.what() << std::endl;
        return 4;
    }

    if (netConfig.layersSizes.size() < 2) {
        std::cerr << "--layers must list at least input and output sizes (two or more integers).\n";
        return 4;
    }

    std::cout << "Starting with parameters:\n"
              << "\tNetwork path: " << networkName << "\n"
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

    std::optional<Neural::NeuralNetwork> loadedMaybe = Neural::LoadNetwork(networkName);
    if (!loadedMaybe) {
        loadedMaybe = Neural::CreateNetwork(netConfig);
    }

    recognizer.setNetwork(loadedMaybe.value());
    recognizer.setTrainingDataset(std::move(trainingDataset));
    recognizer.setTestingDataset(std::move(testingDataset));
    recognizer.setEpochCallback([&recognizer, &networkName] {
        Neural::SaveNetwork(recognizer.getNetwork(), networkName);
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
