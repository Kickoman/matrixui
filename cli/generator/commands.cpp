#include "cli/generator/commands.h"

#include "cli/lib/png_reader.h"

#include "core/lib/file_stream.h"
#include "core/nn/dataset.h"
#include "core/nn/directory_dataset.h"
#include "core/nn/neural_network_applier.h"
#include "core/nn/neural_network_loader.h"

#include "core/generator/trainer.h"

#include "core/matrix/matrix.h"
#include "core/png/pngreader.h"

#include <exception>
#include <optional>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace GeneratorCli {

namespace {

Neural::NeuralNetwork LoadOrCreate(
    const std::string& path,
    const Neural::NeuralNetworkConfiguration& config
) {
    if (!path.empty()) {
        auto loaded = Io::TryReadFile(path, [](std::istream& in) { return Neural::LoadNetwork(in); }, std::ios::binary);
        if (loaded) return std::move(*loaded);
    }
    return Neural::CreateNetwork(config);
}

// For "digit.png" + index 3 -> "digit_3.png"
std::string IndexedPath(const std::string& base, std::size_t index) {
    const auto dot = base.rfind('.');
    if (dot == std::string::npos)
        return base + "_" + std::to_string(index);
    return base.substr(0, dot) + "_" + std::to_string(index) + base.substr(dot);
}

// Input = noise (latentDim) concatenated with one-hot label (classesCount).
std::vector<std::size_t> BuildGeneratorLayers(const TrainOptions& options, const std::size_t classesCount) {
    std::vector<std::size_t> layers = {options.config.latentDim + classesCount};
    layers.insert(layers.end(), options.generatorHidden.begin(), options.generatorHidden.end());
    layers.push_back(options.imageHeight * options.imageWidth);
    return layers;
}

std::vector<std::size_t> BuildDiscriminatorLayers(const TrainOptions& options) {
    std::vector<std::size_t> layers = {options.imageHeight * options.imageWidth};
    layers.insert(layers.end(), options.discriminatorHidden.begin(), options.discriminatorHidden.end());
    layers.push_back(1);
    return layers;
}

int RunGenerate(std::ostream& out, std::ostream& err, const GenerateOptions& options) {
    auto generatorNet = Io::TryReadFile(options.generatorPath, [](std::istream& in) { return Neural::LoadNetwork(in); }, std::ios::binary);
    if (!generatorNet) {
        err << "Failed to load generator: " << options.generatorPath << "\n";
        return kLoadFailed;
    }

    // Optional classifier: if provided, derive numClasses from it (authoritative).
    std::size_t numClasses = options.numClasses;
    std::optional<Neural::NeuralNetworkApplier> classifier;
    if (!options.classifierPath.empty()) {
        auto net = Io::TryReadFile(options.classifierPath, [](std::istream& in) { return Neural::LoadNetwork(in); }, std::ios::binary);
        if (!net) {
            err << "Failed to load classifier: " << options.classifierPath << "\n";
            return kLoadFailed;
        }
        numClasses = net->outputSize();
        classifier.emplace(std::move(*net));
    }

    const std::size_t latentDim = Neural::GAN::inferLatentDim(generatorNet->config.layersSizes, numClasses);
    std::mt19937 rng{std::random_device{}()};
    Neural::NeuralNetworkApplier generator(std::move(*generatorNet));

    for (std::size_t i = 0; i < options.numSamples; ++i) {
        const auto flat = Neural::GAN::Generate(generator, numClasses, options.label, latentDim, rng);

        // Reshape the flat generator output into a height x width image.
        const Matrix image(options.imageHeight, options.imageWidth, [&](std::size_t row, std::size_t col) {
            return flat(0, row * options.imageWidth + col);
        });

        const std::string path = (options.numSamples == 1) ? options.outputPath : IndexedPath(options.outputPath, i);
        PngUtils::toImage(image, path);

        out << "Saved " << path;
        if (classifier) {
            const Matrix predicted = classifier->predict(flat);
            std::size_t best = 0;
            for (std::size_t c = 1; c < predicted.getCols(); ++c)
                if (predicted(0, c) > predicted(0, best)) best = c;
            out << "  (classifier: " << best << ")";
        }
        out << "\n";
    }
    return kSuccess;
}

}  // namespace

int Generate(std::ostream& out, std::ostream& err, const GenerateOptions& options) {
    try {
        return RunGenerate(out, err, options);
    } catch (const std::exception& e) {
        err << e.what() << "\n";
        return kGenerateFailed;
    }
}

int Train(std::ostream& out, std::ostream& err, const TrainOptions& options) {
    auto classifierNet = Io::TryReadFile(options.classifierPath, [](std::istream& in) { return Neural::LoadNetwork(in); }, std::ios::binary);
    if (!classifierNet) {
        err << "Failed to load classifier: " << options.classifierPath << "\n";
        return kLoadFailed;
    }
    const auto classesCount = classifierNet->outputSize();
    Neural::NeuralNetworkApplier classifier(std::move(*classifierNet));

    Neural::NeuralNetworkConfiguration genConfig;
    genConfig.layersSizes      = BuildGeneratorLayers(options, classesCount);
    genConfig.hiddenActivation = Neural::ActivationType::ReLU;
    genConfig.outputActivation = Neural::ActivationType::Sigmoid;
    Neural::NeuralNetworkApplier generator(LoadOrCreate(options.generatorPath, genConfig));

    Neural::NeuralNetworkConfiguration discConfig;
    discConfig.layersSizes      = BuildDiscriminatorLayers(options);
    discConfig.hiddenActivation = Neural::ActivationType::LeakyReLU;
    discConfig.outputActivation = Neural::ActivationType::Sigmoid;
    Neural::NeuralNetworkApplier discriminator(LoadOrCreate(options.discriminatorPath, discConfig));

    Neural::DirectoryDataset dataset(options.datasetPath);
    dataset.setFileReader(CliLib::MakeCachedPngReader(options.imageWidth, options.imageHeight));

    auto allSamples = dataset.getAllSamples(options.config.datasetLimitPerLabel);
    Neural::Dataset::FilterSamples(allSamples, classesCount);
    std::vector<Matrix> realImages;
    realImages.reserve(allSamples.size());
    for (const auto& s : allSamples)
        realImages.push_back(s.input);

    out << "Loaded " << realImages.size() << " real images.\n";
    out << "Generator:     ";
    for (std::size_t sz : genConfig.layersSizes)  out << sz << " ";
    out << "\nDiscriminator: ";
    for (std::size_t sz : discConfig.layersSizes) out << sz << " ";
    out << "\nLatent dim: " << options.config.latentDim
        << "  epochs: " << options.config.epochs
        << "  batch: " << options.config.batchSize
        << "  disc-steps: " << options.config.discriminatorStepsPerGenStep
        << "  gen-lr: " << options.config.generatorLearningRate
        << "  disc-lr: " << options.config.discriminatorLearningRate
        << "\n\n";

    Neural::GAN::GanTrainer trainer(std::move(generator), std::move(discriminator), std::move(classifier));
    trainer.train(realImages, options.config, &out);

    Io::WriteFile(options.generatorPath, [&](std::ostream& file) {
        Neural::SaveNetwork(file, trainer.getGenerator().getNeuralNetworkConfig());
    }, std::ios::binary);
    Io::WriteFile(options.discriminatorPath, [&](std::ostream& file) {
        Neural::SaveNetwork(file, trainer.getDiscriminator().getNeuralNetworkConfig());
    }, std::ios::binary);
    out << "Saved generator to " << options.generatorPath << "\n";
    out << "Saved discriminator to " << options.discriminatorPath << "\n";

    return kSuccess;
}

}  // namespace GeneratorCli
