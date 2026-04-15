#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

#include "neural_network_loader.h"
#include "neural_network_applier.h"
#include "directory_dataset.h"
#include "dataset.h"
#include "gan_config.h"
#include "gan_trainer.h"
#include "generator.h"
#include "discriminator.h"
#include "pngreader.h"
#include "matrix.h"


// ---------------------------------------------------------------------------
// Minimal CLI argument parser (same interface as main_headless.cpp)
// ---------------------------------------------------------------------------
class InputParser {
public:
    InputParser() = default;
    explicit InputParser(int& argc, char** argv) {
        for (int i = 1; i < argc; ++i)
            tokens.emplace_back(argv[i]);
    }

    const std::string& getCmdOption(const std::string& option, const std::string& defaultValue = {}) const {
        const auto it = std::find(tokens.cbegin(), tokens.cend(), option);
        if (it != tokens.cend() && std::next(it) != tokens.cend())
            return *std::next(it);
        return defaultValue;
    }

    bool cmdOptionExists(const std::string& option) const {
        return std::find(tokens.cbegin(), tokens.cend(), option) != tokens.cend();
    }

private:
    std::vector<std::string> tokens;
};


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

double parseDoubleOpt(const InputParser& cmd, const std::string& option, double def) {
    return cmd.cmdOptionExists(option) ? std::stod(cmd.getCmdOption(option)) : def;
}

std::size_t parseSizeOpt(const InputParser& cmd, const std::string& option, std::size_t def) {
    return cmd.cmdOptionExists(option)
        ? static_cast<std::size_t>(std::stoull(cmd.getCmdOption(option)))
        : def;
}

std::vector<std::size_t> parseLayers(const std::string& s) {
    std::vector<std::size_t> result;
    std::istringstream stream(s);
    std::string token;
    while (std::getline(stream, token, ','))
        if (!token.empty())
            result.push_back(std::stoull(token));
    return result;
}

Neural::NeuralNetwork loadOrCreate(
    const std::string& path,
    const Neural::NeuralNetworkConfiguration& config
) {
    if (!path.empty()) {
        auto loaded = Neural::LoadNetwork(path);
        if (loaded) return std::move(*loaded);
    }
    return Neural::CreateNetwork(config);
}

void printUsage(const char* argv0) {
    const Neural::GanConfig d{};
    std::cerr
        << "Usage:\n  " << argv0 << " --classifier <path.wgt> --dataset <dir> [options]\n\n"
        << "Required:\n"
        << "  --classifier <path.wgt>    Trained classifier (frozen signal source).\n"
        << "  --dataset <dir>            Training images root (subdirs 0..9 of PNGs).\n\n"
        << "Network paths (created fresh if the file does not exist):\n"
        << "  --generator <path.wgt>     Generator save path (default: generator.wgt).\n"
        << "  --discriminator <path.wgt> Discriminator save path (default: discriminator.wgt).\n\n"
        << "Network topology (used only when creating new networks):\n"
        << "  --latent-dim <n>           Noise vector size fed to generator (default " << d.latentDim << ").\n"
        << "  --gen-layers <n,n,...>     Generator hidden layer sizes (default 256,512).\n"
        << "  --disc-layers <n,n,...>    Discriminator hidden layer sizes (default 512,256).\n\n"
        << "GAN training:\n"
        << "  --gen-lr <x>               Generator learning rate (default " << d.generatorLr << ").\n"
        << "  --disc-lr <x>              Discriminator learning rate (default " << d.discriminatorLr << ").\n"
        << "  --epochs <n>               Total epochs (default " << d.epochs << ").\n"
        << "  --batch-size <n>           Samples per batch (default " << d.batchSize << ").\n"
        << "  --disc-steps <n>           Discriminator updates per generator update (default "
                                         << d.discriminatorStepsPerGenStep << ").\n"
        << "  --dropout <x>              Discriminator dropout rate (default " << d.dropoutRate << ").\n"
        << "  --classifier-weight <x>    Weight of classifier loss in generator update (default "
                                         << d.classifierLossWeight << ").\n"
        << "  --num-classes <n>          Number of digit classes (default " << d.numClasses << ").\n"
        << "  --dataset-limit <n>        Max images per label to load (0 = all; default 0).\n\n"
        << "Generation mode (no training, no dataset required):\n"
        << "  --generate                 Load the generator and produce sample images.\n"
        << "  --label <n>                Digit to generate (required with --generate).\n"
        << "  --generator <path.wgt>     Generator to load.\n"
        << "  --num-samples <n>          How many images to generate (default 1).\n"
        << "  --output <path.png>        Output file. When generating multiple samples,\n"
        << "                             a counter is inserted before the extension\n"
        << "                             (e.g. digit_0.png, digit_1.png, ...).\n"
        << "  --classifier <path.wgt>    Optional: print the classifier's predicted label.\n\n"
        << "  -h, --help                 Show this text.\n";
}

// For "digit.png" + index 3 → "digit_3.png"
std::string indexedPath(const std::string& base, std::size_t index) {
    const auto dot = base.rfind('.');
    if (dot == std::string::npos)
        return base + "_" + std::to_string(index);
    return base.substr(0, dot) + "_" + std::to_string(index) + base.substr(dot);
}

int runGenerate(const InputParser& cmd) {
    if (!cmd.cmdOptionExists("--label")) {
        std::cerr << "--label <n> is required in generation mode\n";
        return 1;
    }
    const std::size_t label      = parseSizeOpt(cmd, "--label", 0);
    const std::size_t numClasses = parseSizeOpt(cmd, "--num-classes", 10);

    const std::string generatorPath = cmd.getCmdOption("--generator", "generator.wgt");
    const std::string outputPath    = cmd.getCmdOption("--output", "generated.png");
    const std::size_t numSamples    = parseSizeOpt(cmd, "--num-samples", 1);

    auto generatorNet = Neural::LoadNetwork(generatorPath);
    if (!generatorNet) {
        std::cerr << "Failed to load generator: " << generatorPath << "\n";
        return 2;
    }

    // Input size = latentDim + numClasses; infer latentDim from the saved network.
    const std::size_t inputSize = generatorNet->config.layersSizes.front();
    const std::size_t latentDim = inputSize > numClasses ? inputSize - numClasses : inputSize;
    Neural::Generator generator(std::move(*generatorNet), latentDim, numClasses);

    // Optional classifier for printing the predicted label alongside the output.
    std::optional<Neural::NeuralNetworkApplier> classifier;
    if (cmd.cmdOptionExists("--classifier")) {
        auto net = Neural::LoadNetwork(cmd.getCmdOption("--classifier"));
        if (!net) {
            std::cerr << "Failed to load classifier: " << cmd.getCmdOption("--classifier") << "\n";
            return 2;
        }
        classifier.emplace(std::move(*net));
    }

    for (std::size_t i = 0; i < numSamples; ++i) {
        const Matrix flat = generator.generate(label);   // 1 x 784

        // Reshape flat 1x784 row into a 28x28 matrix for saving.
        const Matrix image(28, 28, [&](std::size_t row, std::size_t col) {
            return flat(0, row * 28 + col);
        });

        const std::string path = (numSamples == 1) ? outputPath : indexedPath(outputPath, i);
        PngUtils::toImage(image, path);

        std::cout << "Saved " << path;
        if (classifier) {
            const Matrix out = classifier->predict(flat);
            std::size_t label = 0;
            for (std::size_t c = 1; c < out.getCols(); ++c)
                if (out(0, c) > out(0, label)) label = c;
            std::cout << "  (classifier: " << label << ")";
        }
        std::cout << "\n";
    }
    return 0;
}

} // namespace


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    InputParser cmd(argc, argv);

    if (cmd.cmdOptionExists("--help") || cmd.cmdOptionExists("-h")) {
        printUsage(argc > 0 ? argv[0] : "matrixgui_gan");
        return 0;
    }

    if (cmd.cmdOptionExists("--generate")) {
        try {
            return runGenerate(cmd);
        } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 4;
        }
    }

    if (!cmd.cmdOptionExists("--classifier")) {
        std::cerr << "--classifier <path.wgt> is required\n";
        printUsage(argc > 0 ? argv[0] : "matrixgui_gan");
        return 1;
    }
    if (!cmd.cmdOptionExists("--dataset")) {
        std::cerr << "--dataset <dir> is required\n";
        printUsage(argc > 0 ? argv[0] : "matrixgui_gan");
        return 1;
    }

    const std::string classifierPath   = cmd.getCmdOption("--classifier");
    const std::string datasetPath      = cmd.getCmdOption("--dataset");
    const std::string generatorPath    = cmd.getCmdOption("--generator",    "generator.wgt");
    const std::string discriminatorPath= cmd.getCmdOption("--discriminator","discriminator.wgt");

    // GAN config
    Neural::GanConfig ganConfig{};
    ganConfig.latentDim                 = parseSizeOpt  (cmd, "--latent-dim",  ganConfig.latentDim);
    ganConfig.generatorLr               = parseDoubleOpt(cmd, "--gen-lr",      ganConfig.generatorLr);
    ganConfig.discriminatorLr           = parseDoubleOpt(cmd, "--disc-lr",     ganConfig.discriminatorLr);
    ganConfig.epochs                    = parseSizeOpt  (cmd, "--epochs",      ganConfig.epochs);
    ganConfig.batchSize                 = parseSizeOpt  (cmd, "--batch-size",  ganConfig.batchSize);
    ganConfig.discriminatorStepsPerGenStep = parseSizeOpt(cmd, "--disc-steps",          ganConfig.discriminatorStepsPerGenStep);
    ganConfig.dropoutRate               = parseDoubleOpt(cmd, "--dropout",             ganConfig.dropoutRate);
    ganConfig.classifierLossWeight      = parseDoubleOpt(cmd, "--classifier-weight",   ganConfig.classifierLossWeight);
    ganConfig.numClasses                = parseSizeOpt  (cmd, "--num-classes",         ganConfig.numClasses);
    ganConfig.datasetLimitPerLabel      = parseSizeOpt  (cmd, "--dataset-limit",       ganConfig.datasetLimitPerLabel);

    // Generator topology: latentDim -> hidden... -> 784, Sigmoid output
    const std::vector<std::size_t> kDefaultGenHidden  = {256, 512};
    const std::vector<std::size_t> kDefaultDiscHidden = {512, 256};
    constexpr std::size_t kImageSize = 28 * 28;

    auto buildGenLayers = [&]() {
        // Input = noise (latentDim) concatenated with one-hot label (numClasses).
        std::vector<std::size_t> layers = {ganConfig.latentDim + ganConfig.numClasses};
        const auto hidden = cmd.cmdOptionExists("--gen-layers")
            ? parseLayers(cmd.getCmdOption("--gen-layers"))
            : kDefaultGenHidden;
        layers.insert(layers.end(), hidden.begin(), hidden.end());
        layers.push_back(kImageSize);
        return layers;
    };

    auto buildDiscLayers = [&]() {
        std::vector<std::size_t> layers = {kImageSize};
        const auto hidden = cmd.cmdOptionExists("--disc-layers")
            ? parseLayers(cmd.getCmdOption("--disc-layers"))
            : kDefaultDiscHidden;
        layers.insert(layers.end(), hidden.begin(), hidden.end());
        layers.push_back(1);
        return layers;
    };

    // Load classifier (frozen)
    auto classifierNet = Neural::LoadNetwork(classifierPath);
    if (!classifierNet) {
        std::cerr << "Failed to load classifier: " << classifierPath << "\n";
        return 2;
    }
    Neural::NeuralNetworkApplier classifier(std::move(*classifierNet));

    // Load or create generator
    Neural::NeuralNetworkConfiguration genConfig;
    genConfig.layersSizes      = buildGenLayers();
    genConfig.hiddenActivation = Neural::ActivationType::ReLU;
    genConfig.outputActivation = Neural::ActivationType::Sigmoid;
    Neural::Generator generator(loadOrCreate(generatorPath, genConfig), ganConfig.latentDim, ganConfig.numClasses);

    // Load or create discriminator
    Neural::NeuralNetworkConfiguration discConfig;
    discConfig.layersSizes      = buildDiscLayers();
    discConfig.hiddenActivation = Neural::ActivationType::LeakyReLU;
    discConfig.outputActivation = Neural::ActivationType::Sigmoid;
    Neural::Discriminator discriminator(loadOrCreate(discriminatorPath, discConfig));

    // Load real training samples
    PngUtils::Cache pngCache;
    const auto reader = [&pngCache](const std::filesystem::path& path) {
        return PngUtils::fromImage(path.string(), 28, 28, pngCache).transform(1, kImageSize);
    };
    Neural::DirectoryDataset dataset(datasetPath);
    dataset.setFileReader(reader);

    const auto allSamples = dataset.getAllSamples(ganConfig.datasetLimitPerLabel);
    std::vector<Matrix> realImages;
    realImages.reserve(allSamples.size());
    for (const auto& s : allSamples)
        realImages.push_back(s.input);

    std::cout << "Loaded " << realImages.size() << " real images.\n";
    std::cout << "Generator:     ";
    for (std::size_t sz : genConfig.layersSizes)  std::cout << sz << " ";
    std::cout << "\nDiscriminator: ";
    for (std::size_t sz : discConfig.layersSizes) std::cout << sz << " ";
    std::cout << "\nLatent dim: " << ganConfig.latentDim
              << "  epochs: " << ganConfig.epochs
              << "  batch: " << ganConfig.batchSize
              << "  disc-steps: " << ganConfig.discriminatorStepsPerGenStep
              << "  gen-lr: " << ganConfig.generatorLr
              << "  disc-lr: " << ganConfig.discriminatorLr
              << "\n\n";

    // Train
    Neural::GanTrainer trainer(std::move(generator), std::move(discriminator), std::move(classifier));
    trainer.train(realImages, ganConfig, &std::cout);

    // Save
    Neural::SaveNetwork(trainer.getGenerator().getNetwork(), generatorPath);
    Neural::SaveNetwork(trainer.getDiscriminator().getNetwork(), discriminatorPath);
    std::cout << "Saved generator to " << generatorPath << "\n";
    std::cout << "Saved discriminator to " << discriminatorPath << "\n";

    return 0;
}
