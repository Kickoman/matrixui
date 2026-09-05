#include <doctest/doctest.h>

#include "cli/classifier/commands.h"
#include "cli/classifier/options.h"

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

// Subdirectories 0..classes-1 with deterministic side x side PNGs in each.
std::filesystem::path MakeDataset(const Tests::TempDir& dir, const std::size_t classes,
                                  const std::size_t perClass, const std::size_t side) {
    const auto root = dir.file("dataset");
    for (std::size_t label = 0; label < classes; ++label) {
        std::filesystem::create_directories(root / std::to_string(label));
        for (std::size_t i = 0; i < perClass; ++i) {
            const Matrix image(side, side, [&](std::size_t row, std::size_t col) {
                return static_cast<double>((row + col * classes + label) % side) / side;
            });
            PngUtils::toImage(image, (root / std::to_string(label) / (std::to_string(i) + ".png")).string());
        }
    }
    return root;
}

}  // namespace

TEST_CASE("Predict classifies a PNG with a saved network") {
    Tests::TempDir dir;
    const auto network = WriteNetwork(dir, "net.wgt", {64, 16, 3});
    const auto image = dir.file("digit.png");
    PngUtils::toImage(Matrix(8, 8, 0.5), image.string());

    ClassifierCli::PredictOptions options;
    options.networkPath = network.string();
    options.imagePath = image.string();
    options.imageWidth = 8;
    options.imageHeight = 8;

    std::ostringstream out, err;
    CHECK(ClassifierCli::Predict(out, err, options) == ClassifierCli::kSuccess);
    CHECK(err.str().empty());
    const int digit = std::stoi(out.str());
    CHECK(digit >= 0);
    CHECK(digit < 3);
}

TEST_CASE("Predict reports a missing network file with the size-mismatch code") {
    Tests::TempDir dir;
    ClassifierCli::PredictOptions options;
    options.networkPath = dir.file("absent.wgt").string();
    options.imagePath = dir.file("any.png").string();

    std::ostringstream out, err;
    CHECK(ClassifierCli::Predict(out, err, options) == ClassifierCli::kSizeMismatch);
    CHECK(err.str() == "Failed to load network: " + options.networkPath + "\n");
    CHECK(out.str().empty());
}

TEST_CASE("Predict maps a corrupt network file to the bad-config code") {
    Tests::TempDir dir;
    ClassifierCli::PredictOptions options;
    options.networkPath = dir.write("bad.wgt", "garbage").string();
    options.imagePath = dir.file("any.png").string();

    std::ostringstream out, err;
    CHECK(ClassifierCli::Predict(out, err, options) == ClassifierCli::kBadConfig);
    CHECK_FALSE(err.str().empty());
}

TEST_CASE("Predict maps an unreadable image to the bad-config code") {
    Tests::TempDir dir;
    ClassifierCli::PredictOptions options;
    options.networkPath = WriteNetwork(dir, "net.wgt", {64, 16, 3}).string();
    options.imagePath = dir.write("bad.png", "not a png").string();
    options.imageWidth = 8;
    options.imageHeight = 8;

    std::ostringstream out, err;
    CHECK(ClassifierCli::Predict(out, err, options) == ClassifierCli::kBadConfig);
    CHECK(err.str().find("Failed to load image") != std::string::npos);
}

TEST_CASE("Train demands a dataset") {
    ClassifierCli::TrainOptions options;
    options.networkPath = "unused.wgt";

    std::ostringstream out, err;
    CHECK(ClassifierCli::Train(out, err, options) == ClassifierCli::kNoDatasetResolved);
    CHECK(err.str().find("Specify data directories:") == 0);
    CHECK(out.str().empty());
}

TEST_CASE("Train rejects a single-entry layer list") {
    Tests::TempDir dir;
    ClassifierCli::TrainOptions options;
    options.networkPath = dir.file("net.wgt").string();
    options.datasetPath = dir.file("dataset").string();
    options.network.layersSizes = {64};

    std::ostringstream out, err;
    CHECK(ClassifierCli::Train(out, err, options) == ClassifierCli::kBadConfig);
    CHECK(err.str().find("--layers must list at least") != std::string::npos);
}

TEST_CASE("Train rejects an image size that does not match the network input") {
    Tests::TempDir dir;
    ClassifierCli::TrainOptions options;
    options.networkPath = dir.file("fresh.wgt").string();
    options.datasetPath = dir.file("dataset").string();
    options.network.layersSizes = {16, 8, 2};  // input 16, default image 28x28 = 784

    std::ostringstream out, err;
    CHECK(ClassifierCli::Train(out, err, options) == ClassifierCli::kSizeMismatch);
    CHECK(err.str().find("Invalid image sizes:") != std::string::npos);
}

TEST_CASE("Train rejects a dataset without the class subdirectories") {
    Tests::TempDir dir;
    std::filesystem::create_directories(dir.file("empty"));

    ClassifierCli::TrainOptions options;
    options.networkPath = dir.file("net.wgt").string();
    options.datasetPath = dir.file("empty").string();
    options.network.layersSizes = {16, 8, 2};
    options.imageWidth = 4;
    options.imageHeight = 4;

    std::ostringstream out, err;
    CHECK(ClassifierCli::Train(out, err, options) == ClassifierCli::kInvalidDataset);
    CHECK(err.str().find("Invalid training dataset") != std::string::npos);
}

TEST_CASE("Train completes a one-epoch run on a tiny generated dataset") {
    Tests::TempDir dir;
    const auto dataset = MakeDataset(dir, 2, 3, 4);

    ClassifierCli::TrainOptions options;
    options.networkPath = dir.file("net.wgt").string();
    options.datasetPath = dataset.string();
    options.workingDirectory = dir.file("wd").string();
    options.network.layersSizes = {16, 8, 2};
    options.imageWidth = 4;
    options.imageHeight = 4;
    options.learning.maxEpochs = 1;
    options.learning.patience = 1;

    std::ostringstream out, err;
    CHECK(ClassifierCli::Train(out, err, options) == ClassifierCli::kSuccess);
    CHECK(out.str().find("Starting with parameters:") != std::string::npos);
    CHECK(out.str().find("Final test accuracy:") != std::string::npos);
    CHECK(std::filesystem::exists(options.networkPath));

    // Exactly one timestamped run directory with the dumps and logs.
    std::size_t runDirs = 0;
    for (const auto& entry : std::filesystem::directory_iterator(options.workingDirectory)) {
        ++runDirs;
        CHECK(std::filesystem::exists(entry.path() / "learning-config.json"));
        CHECK(std::filesystem::exists(entry.path() / "network-config.json"));
        CHECK(std::filesystem::exists(entry.path() / "log.jsonl"));
    }
    CHECK(runDirs == 1);
}
