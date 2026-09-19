#include "core/serving/model_loader.h"

#include "core/lib/file_stream.h"
#include "core/lib/sha256.h"
#include "core/nn/neural_network_loader.h"
#include "core/serving/error.h"
#include "core/serving/manifest.h"

#include <fstream>
#include <ios>
#include <istream>
#include <string>
#include <system_error>
#include <utility>

namespace Serving {

namespace {

constexpr const char* kManifestName = "manifest.json";

nlohmann::json ReadManifestDocument(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw ManifestError("Can't open " + path.string());
    }
    try {
        return nlohmann::json::parse(in);
    } catch (const nlohmann::json::exception& error) {
        throw ManifestError(std::string("manifest.json: ") + error.what());
    }
}

void VerifyDigest(const WeightsReference& weights) {
    const auto actual = Hash::ToHex(Io::ReadFile(
        weights.path,
        [](std::istream& in) { return Hash::Sha256OfStream(in); },
        std::ios::binary));
    if (actual != weights.sha256) {
        throw IntegrityError(
            "weights sha256 mismatch for " + weights.path.string()
            + " (manifest " + weights.sha256 + ", file " + actual + ")");
    }
}

void CrossValidate(const ModelManifest& manifest, const Neural::NeuralNetwork& network) {
    if (manifest.input.size != network.inputSize()) {
        throw ContractError(
            "manifest declares input.size " + std::to_string(manifest.input.size)
            + ", but the network takes " + std::to_string(network.inputSize()));
    }
    if (manifest.output.size != network.outputSize()) {
        throw ContractError(
            "manifest declares output.size " + std::to_string(manifest.output.size)
            + ", but the network produces " + std::to_string(network.outputSize()));
    }
}

}  // namespace

LoadedModel LoadModel(const std::filesystem::path& directory) {
    std::error_code ignored;
    if (!std::filesystem::is_directory(directory, ignored)) {
        throw ManifestError("Not a model directory: " + directory.string());
    }

    const auto manifestPath = directory / kManifestName;
    if (!std::filesystem::is_regular_file(manifestPath, ignored)) {
        throw ManifestError("No " + std::string(kManifestName) + " in " + directory.string());
    }

    ModelManifest manifest = ParseManifest(ReadManifestDocument(manifestPath), directory);

    if (!std::filesystem::is_regular_file(manifest.weights.path, ignored)) {
        throw ManifestError("weights.path does not name a file: " + manifest.weights.path.string());
    }

    IntegrityCheck integrity = IntegrityCheck::NotDeclared;
    if (!manifest.weights.sha256.empty()) {
        VerifyDigest(manifest.weights);
        integrity = IntegrityCheck::Verified;
    }

    Neural::NeuralNetwork loaded;
    try {
        loaded = Io::ReadFile(
            manifest.weights.path,
            [](std::istream& in) { return Neural::LoadNetwork(in); },
            std::ios::binary);
    } catch (const std::exception& error) {
        // Io::Error has already appended the path.
        throw IntegrityError(error.what());
    }

    CrossValidate(manifest, loaded);

    return LoadedModel(
        std::move(manifest),
        std::make_shared<const Neural::NeuralNetwork>(std::move(loaded)),
        directory,
        integrity);
}

}  // namespace Serving
