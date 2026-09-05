#include <doctest/doctest.h>

#include "cli/generator/commands.h"
#include "cli/generator/options.h"

#include "core/lib/file_stream.h"
#include "core/matrix/matrix.h"
#include "core/nn/neural_network_loader.h"
#include "core/png/pngreader.h"

#include "tests/support/temp_dir.h"

#include <cstddef>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kClasses = 3;
constexpr std::size_t kSide = 4;

std::filesystem::path WriteNetwork(const Tests::TempDir& dir, const std::string& name,
                                   const std::vector<std::size_t>& layers) {
    Neural::NeuralNetworkConfiguration configuration{};
    configuration.layersSizes = layers;
    const auto path = dir.file(name);
    Io::WriteFile(path, [&](std::ostream& out) {
        Neural::SaveNetwork(out, Neural::CreateNetwork(configuration));
    }, std::ios::binary);
    return path;
}

// Generator input = latent + one-hot(kClasses); output = kSide * kSide pixels.
std::filesystem::path WriteGenerator(const Tests::TempDir& dir, const std::size_t latentDim) {
    return WriteNetwork(dir, "generator.wgt", {latentDim + kClasses, 8, kSide * kSide});
}

std::filesystem::path MakeDataset(const Tests::TempDir& dir, const std::size_t classes,
                                  const std::size_t perClass) {
    const auto root = dir.file("dataset");
    for (std::size_t label = 0; label < classes; ++label) {
        std::filesystem::create_directories(root / std::to_string(label));
        for (std::size_t i = 0; i < perClass; ++i) {
            const Matrix image(kSide, kSide, [&](std::size_t row, std::size_t col) {
                return static_cast<double>((row * kSide + col + label) % kSide) / kSide;
            });
            PngUtils::toImage(image, (root / std::to_string(label) / (std::to_string(i) + ".png")).string());
        }
    }
    return root;
}

}  // namespace

TEST_CASE("Generate reports a missing generator network") {
    Tests::TempDir dir;
    GeneratorCli::GenerateOptions options;
    options.generatorPath = dir.file("absent.wgt").string();

    std::ostringstream out, err;
    CHECK(GeneratorCli::Generate(out, err, options) == GeneratorCli::kLoadFailed);
    CHECK(err.str() == "Failed to load generator: " + options.generatorPath + "\n");
    CHECK(out.str().empty());
}

TEST_CASE("Generate reports a missing classifier network") {
    Tests::TempDir dir;
    GeneratorCli::GenerateOptions options;
    options.generatorPath = WriteGenerator(dir, 4).string();
    options.classifierPath = dir.file("absent.wgt").string();
    options.numClasses = kClasses;

    std::ostringstream out, err;
    CHECK(GeneratorCli::Generate(out, err, options) == GeneratorCli::kLoadFailed);
    CHECK(err.str() == "Failed to load classifier: " + options.classifierPath + "\n");
}

TEST_CASE("Generate maps a corrupt generator file to the generate-failed code") {
    Tests::TempDir dir;
    GeneratorCli::GenerateOptions options;
    options.generatorPath = dir.write("bad.wgt", "garbage").string();

    std::ostringstream out, err;
    CHECK(GeneratorCli::Generate(out, err, options) == GeneratorCli::kGenerateFailed);
    CHECK_FALSE(err.str().empty());
}

TEST_CASE("Generate writes the requested number of samples") {
    Tests::TempDir dir;
    GeneratorCli::GenerateOptions options;
    options.generatorPath = WriteGenerator(dir, 4).string();
    options.numClasses = kClasses;
    options.label = 1;
    options.outputPath = dir.file("digit.png").string();
    options.imageWidth = kSide;
    options.imageHeight = kSide;

    std::ostringstream out, err;

    SUBCASE("a single sample keeps the plain name") {
        options.numSamples = 1;
        CHECK(GeneratorCli::Generate(out, err, options) == GeneratorCli::kSuccess);
        CHECK(std::filesystem::exists(dir.file("digit.png")));
        CHECK(out.str().find("Saved " + options.outputPath) != std::string::npos);
    }

    SUBCASE("multiple samples get a counter before the extension") {
        options.numSamples = 2;
        CHECK(GeneratorCli::Generate(out, err, options) == GeneratorCli::kSuccess);
        CHECK(std::filesystem::exists(dir.file("digit_0.png")));
        CHECK(std::filesystem::exists(dir.file("digit_1.png")));
        CHECK_FALSE(std::filesystem::exists(dir.file("digit.png")));
    }
}

TEST_CASE("Generate appends the classifier opinion when given one") {
    Tests::TempDir dir;
    GeneratorCli::GenerateOptions options;
    options.generatorPath = WriteGenerator(dir, 4).string();
    options.classifierPath = WriteNetwork(dir, "classifier.wgt", {kSide * kSide, 8, kClasses}).string();
    options.outputPath = dir.file("digit.png").string();
    options.imageWidth = kSide;
    options.imageHeight = kSide;

    std::ostringstream out, err;
    CHECK(GeneratorCli::Generate(out, err, options) == GeneratorCli::kSuccess);
    CHECK(out.str().find("  (classifier: ") != std::string::npos);
}

TEST_CASE("Train reports a missing classifier") {
    Tests::TempDir dir;
    GeneratorCli::TrainOptions options;
    options.classifierPath = dir.file("absent.wgt").string();
    options.datasetPath = dir.file("dataset").string();

    std::ostringstream out, err;
    CHECK(GeneratorCli::Train(out, err, options) == GeneratorCli::kLoadFailed);
    CHECK(err.str() == "Failed to load classifier: " + options.classifierPath + "\n");
}

TEST_CASE("Train runs one epoch end to end on a tiny dataset") {
    Tests::TempDir dir;
    GeneratorCli::TrainOptions options;
    options.classifierPath = WriteNetwork(dir, "classifier.wgt", {kSide * kSide, 4, 2}).string();
    options.datasetPath = MakeDataset(dir, 2, 3).string();
    options.generatorPath = dir.file("gen.wgt").string();
    options.discriminatorPath = dir.file("disc.wgt").string();
    options.generatorHidden = {8};
    options.discriminatorHidden = {8};
    options.imageWidth = kSide;
    options.imageHeight = kSide;
    options.config.latentDim = 4;
    options.config.epochs = 1;
    options.config.batchSize = 1;

    std::ostringstream out, err;
    CHECK(GeneratorCli::Train(out, err, options) == GeneratorCli::kSuccess);
    CHECK(out.str().find("Loaded 6 real images.") != std::string::npos);
    CHECK(out.str().find("Saved generator to " + options.generatorPath) != std::string::npos);
    CHECK(out.str().find("Saved discriminator to " + options.discriminatorPath) != std::string::npos);
    CHECK(std::filesystem::exists(options.generatorPath));
    CHECK(std::filesystem::exists(options.discriminatorPath));
}
