#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "core/lib/neural_network_loader.h"
#include "core/lib/neural_network_applier.h"
#include "core/lib/directory_dataset.h"
#include "core/lib/dataset.h"
#include "core/lib/cache.h"

#include "core/generator/learning_config.h"
#include "core/generator/trainer.h"

#include "png/pngreader.h"
#include "matrix/matrix.h"

#include <CLI11/CLI11.hpp>


namespace {

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

// For "digit.png" + index 3 → "digit_3.png"
std::string indexedPath(const std::string& base, std::size_t index) {
    const auto dot = base.rfind('.');
    if (dot == std::string::npos)
        return base + "_" + std::to_string(index);
    return base.substr(0, dot) + "_" + std::to_string(index) + base.substr(dot);
}

int runGenerate(
    std::size_t label,
    std::size_t numClasses,
    const std::string& generatorPath,
    const std::string& outputPath,
    std::size_t numSamples,
    const std::string& classifierPath,
    std::size_t imageWidth,
    std::size_t imageHeight
) {
    auto generatorNet = Neural::LoadNetwork(generatorPath);
    if (!generatorNet) {
        std::cerr << "Failed to load generator: " << generatorPath << "\n";
        return 2;
    }

    // Optional classifier: if provided, derive numClasses from it (authoritative).
    std::optional<Neural::NeuralNetworkApplier> classifier;
    if (!classifierPath.empty()) {
        auto net = Neural::LoadNetwork(classifierPath);
        if (!net) {
            std::cerr << "Failed to load classifier: " << classifierPath << "\n";
            return 2;
        }
        numClasses = net->outputSize();
        classifier.emplace(std::move(*net));
    }

    const std::size_t latentDim = Neural::GAN::inferLatentDim(generatorNet->config.layersSizes, numClasses);
    std::mt19937 rng{std::random_device{}()};
    Neural::NeuralNetworkApplier generator(std::move(*generatorNet));

    for (std::size_t i = 0; i < numSamples; ++i) {
        const auto flat = Neural::GAN::Generate(generator, numClasses, label, latentDim, rng);

        // Reshape the flat generator output into a height x width image.
        const Matrix image(imageHeight, imageWidth, [&](std::size_t row, std::size_t col) {
            return flat(0, row * imageWidth + col);
        });

        const std::string path = (numSamples == 1) ? outputPath : indexedPath(outputPath, i);
        PngUtils::toImage(image, path);

        std::cout << "Saved " << path;
        if (classifier) {
            const Matrix out = classifier->predict(flat);
            std::size_t predicted = 0;
            for (std::size_t c = 1; c < out.getCols(); ++c)
                if (out(0, c) > out(0, predicted)) predicted = c;
            std::cout << "  (classifier: " << predicted << ")";
        }
        std::cout << "\n";
    }
    return 0;
}

}  // namespace


int main(int argc, char** argv) {
    CLI::App app{"MatrixGui GAN CLI (generator/discriminator training & sampling)"};
    app.require_subcommand(1);

    CLI::App* generateCmd = app.add_subcommand("generate", "Generate sample images from a trained generator");

    std::size_t genLabel = 0;
    std::size_t genNumClasses = 10;
    std::size_t numSamples = 1;
    std::string generatorPathArg = "generator.wgt";
    std::string outputPath = "generated.png";
    std::string classifierPathArg;
    std::size_t imageWidth = 28;
    std::size_t imageHeight = 28;

    generateCmd->add_option("--label", genLabel, "Digit to generate")->required();
    generateCmd->add_option("--num-classes", genNumClasses, "Number of digit classes")->capture_default_str();
    generateCmd->add_option("--generator", generatorPathArg, "Generator network file to load")
        ->check(CLI::ExistingFile)
        ->capture_default_str();
    generateCmd->add_option("--num-samples", numSamples, "How many images to generate")->capture_default_str();
    generateCmd->add_option("--output", outputPath,
            "Output PNG path. With --num-samples > 1, a counter is inserted before the extension.")
        ->capture_default_str();
    generateCmd->add_option("--classifier", classifierPathArg,
            "Optional: print this classifier's predicted label alongside each image")
        ->check(CLI::ExistingFile);
    generateCmd->add_option("--dataset-img-width", imageWidth, "Width of test images in pixels.")
        ->capture_default_str();
    generateCmd->add_option("--dataset-img-height", imageHeight, "Height of test images in pixels.")
        ->capture_default_str();

    CLI::App* trainCmd = app.add_subcommand("train", "Train generator+discriminator against a frozen classifier");

    std::string classifierPath;
    std::string datasetPath;
    std::string generatorPath = "generator.wgt";
    std::string discriminatorPath = "discriminator.wgt";
    Neural::GAN::LearningConfig ganConfig{};
    std::vector<std::size_t> genHidden = {256, 512};
    std::vector<std::size_t> discHidden = {512, 256};

    trainCmd->add_option("--classifier", classifierPath, "Trained classifier (frozen signal source)")->required();
    trainCmd->add_option("--dataset", datasetPath, "Training images root (subdirs 0..9 of PNGs)")
        ->required()
        ->check(CLI::ExistingDirectory);

    trainCmd->add_option("--generator", generatorPath, "Generator save path (created if missing)")
        ->group("Network paths")->capture_default_str();
    trainCmd->add_option("--discriminator", discriminatorPath, "Discriminator save path (created if missing)")
        ->group("Network paths")->capture_default_str();

    trainCmd->add_option("--latent-dim", ganConfig.latentDim, "Noise vector size fed to generator")
        ->group("Topology")->capture_default_str();
    trainCmd->add_option("--gen-layers", genHidden, "Generator hidden layer sizes (new network only)")
        ->delimiter(',')->group("Topology")->capture_default_str();
    trainCmd->add_option("--disc-layers", discHidden, "Discriminator hidden layer sizes (new network only)")
        ->delimiter(',')->group("Topology")->capture_default_str();
    trainCmd->add_option("--dataset-img-width", imageWidth, "Width of test images in pixels.")
        ->group("Topology")
        ->capture_default_str();
    trainCmd->add_option("--dataset-img-height", imageHeight, "Height of test images in pixels.")
        ->group("Topology")
        ->capture_default_str();

    trainCmd->add_option("--gen-lr", ganConfig.generatorLearningRate, "Generator learning rate")
        ->group("GAN training")->capture_default_str();
    trainCmd->add_option("--disc-lr", ganConfig.discriminatorLearningRate, "Discriminator learning rate")
        ->group("GAN training")->capture_default_str();
    trainCmd->add_option("--epochs", ganConfig.epochs, "Total epochs")
        ->group("GAN training")->capture_default_str();
    trainCmd->add_option("--batch-size", ganConfig.batchSize, "Samples per batch")
        ->group("GAN training")->capture_default_str();
    trainCmd->add_option("--disc-steps", ganConfig.discriminatorStepsPerGenStep,
            "Discriminator updates per generator update")
        ->group("GAN training")->capture_default_str();
    trainCmd->add_option("--dropout", ganConfig.dropoutRate, "Discriminator dropout rate")
        ->group("GAN training")->capture_default_str();
    trainCmd->add_option("--classifier-weight", ganConfig.classifierLossWeight,
            "Weight of classifier loss in generator update")
        ->group("GAN training")->capture_default_str();
    trainCmd->add_option("--dataset-limit", ganConfig.datasetLimitPerLabel, "Max images per label to load (0 = all)")
        ->group("GAN training")->capture_default_str();

    trainCmd->add_flag("--adaptive-lr", ganConfig.adaptiveLr.enabled, "Enable adaptive lr adjustment")
        ->group("Adaptive learning rate");
    trainCmd->add_option("--lr-ema-alpha", ganConfig.adaptiveLr.lrEmaAlpha, "EMA smoothing factor; higher = slower reaction")
        ->group("Adaptive learning rate")->capture_default_str();
    trainCmd->add_option("--d-real-target-low", ganConfig.adaptiveLr.dRealTargetLow, "D(real) below this => D collapsed")
        ->group("Adaptive learning rate")->capture_default_str();
    trainCmd->add_option("--d-real-target-high", ganConfig.adaptiveLr.dRealTargetHigh, "D(real) above this => D dominating")
        ->group("Adaptive learning rate")->capture_default_str();
    trainCmd->add_option("--gen-fool-target-low", ganConfig.adaptiveLr.dFakeTargetLow,
            "D(G(z)) below this => D dominating/collapsed")
        ->group("Adaptive learning rate")->capture_default_str();
    trainCmd->add_option("--gen-fool-target-high", ganConfig.adaptiveLr.dFakeTargetHigh, "D(G(z)) above this => G dominating")
        ->group("Adaptive learning rate")->capture_default_str();
    trainCmd->add_option("--lr-adjust-factor", ganConfig.adaptiveLr.lrAdjustFactor, "Multiplicative lr step per epoch")
        ->group("Adaptive learning rate")->capture_default_str();
    trainCmd->add_option("--lr-min", ganConfig.adaptiveLr.lrMin, "Lower lr clamp")
        ->group("Adaptive learning rate")->capture_default_str();
    trainCmd->add_option("--lr-max", ganConfig.adaptiveLr.lrMax, "Upper lr clamp")
        ->group("Adaptive learning rate")->capture_default_str();
    trainCmd->add_option("--lr-warmup-epochs", ganConfig.adaptiveLr.lrWarmupEpochs, "Epochs before adaptive adjustments begin")
        ->group("Adaptive learning rate")->capture_default_str();

    trainCmd->add_flag("--flatness-detection", ganConfig.flatnessDetection.enabled, "Enable plateau detection")
        ->group("Flatness detection");
    trainCmd->add_option("--flatness-threshold", ganConfig.flatnessDetection.threshold,
            "Max EMA change per epoch to count as flat")
        ->group("Flatness detection")->capture_default_str();
    trainCmd->add_option("--flatness-window", ganConfig.flatnessDetection.window, "Consecutive flat epochs before kick fires")
        ->group("Flatness detection")->capture_default_str();
    trainCmd->add_option("--flatness-kick-duration", ganConfig.flatnessDetection.kickDuration, "Epochs to hold the kick")
        ->group("Flatness detection")->capture_default_str();
    trainCmd->add_option("--flatness-dropout-boost", ganConfig.flatnessDetection.discriminatorDropoutBoost,
            "Multiply D dropout by this during kick")
        ->group("Flatness detection")->capture_default_str();
    trainCmd->add_option("--flatness-gen-lr-boost", ganConfig.flatnessDetection.generatorLrBoost, "Multiply G lr by this during kick")
        ->group("Flatness detection")->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    if (*generateCmd) {
        try {
            return runGenerate(
                genLabel, genNumClasses, generatorPathArg, outputPath, numSamples,
                classifierPathArg, imageWidth, imageHeight);
        } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 4;
        }
    }

    auto classifierNet = Neural::LoadNetwork(classifierPath);
    if (!classifierNet) {
        std::cerr << "Failed to load classifier: " << classifierPath << "\n";
        return 2;
    }
    const auto classesCount = classifierNet->outputSize();
    Neural::NeuralNetworkApplier classifier(std::move(*classifierNet));

    auto buildGenLayers = [&]() {
        // Input = noise (latentDim) concatenated with one-hot label (classesCount).
        std::vector<std::size_t> layers = {ganConfig.latentDim + classesCount};
        layers.insert(layers.end(), genHidden.begin(), genHidden.end());
        layers.push_back(imageHeight * imageWidth);
        return layers;
    };

    auto buildDiscLayers = [&]() {
        std::vector<std::size_t> layers = {imageHeight * imageWidth};
        layers.insert(layers.end(), discHidden.begin(), discHidden.end());
        layers.push_back(1);
        return layers;
    };

    Neural::NeuralNetworkConfiguration genConfig;
    genConfig.layersSizes      = buildGenLayers();
    genConfig.hiddenActivation = Neural::ActivationType::ReLU;
    genConfig.outputActivation = Neural::ActivationType::Sigmoid;
    Neural::NeuralNetworkApplier generator(loadOrCreate(generatorPath, genConfig));

    Neural::NeuralNetworkConfiguration discConfig;
    discConfig.layersSizes      = buildDiscLayers();
    discConfig.hiddenActivation = Neural::ActivationType::LeakyReLU;
    discConfig.outputActivation = Neural::ActivationType::Sigmoid;
    Neural::NeuralNetworkApplier discriminator(loadOrCreate(discriminatorPath, discConfig));

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
    Neural::DirectoryDataset dataset(datasetPath);
    dataset.setFileReader(reader);

    auto allSamples = dataset.getAllSamples(ganConfig.datasetLimitPerLabel);
    Neural::Dataset::FilterSamples(allSamples, classesCount);
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
              << "  gen-lr: " << ganConfig.generatorLearningRate
              << "  disc-lr: " << ganConfig.discriminatorLearningRate
              << "\n\n";

    Neural::GAN::GanTrainer trainer(std::move(generator), std::move(discriminator), std::move(classifier));
    trainer.train(realImages, ganConfig, &std::cout);

    Neural::SaveNetwork(trainer.getGenerator().getNeuralNetworkConfig(), generatorPath);
    Neural::SaveNetwork(trainer.getDiscriminator().getNeuralNetworkConfig(), discriminatorPath);
    std::cout << "Saved generator to " << generatorPath << "\n";
    std::cout << "Saved discriminator to " << discriminatorPath << "\n";

    return 0;
}
