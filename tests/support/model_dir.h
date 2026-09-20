#pragma once

#include "core/lib/file_stream.h"
#include "core/nn/neural_network.h"
#include "core/nn/neural_network_loader.h"
#include "tests/support/temp_dir.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <ios>
#include <ostream>
#include <string>
#include <vector>

namespace Tests {

inline std::filesystem::path MakeModelDirectory(const TempDir& dir, const std::string& name = "model") {
    const auto path = dir.file(name);
    std::filesystem::create_directories(path);
    return path;
}

// Only the required fields, so a test can bend exactly one of them. Name and
// version are parameters because a registry tree needs several of each.
inline nlohmann::json ManifestFor(
    const std::size_t inputSize,
    const std::size_t outputSize,
    const std::string& name = "mnist",
    const std::string& version = "v3"
) {
    return nlohmann::json{
        {"manifestVersion", 1},
        {"name", name},
        {"version", version},
        {"weights", {{"path", "weights.wgt"}}},
        {"input", {{"size", inputSize}}},
        {"output", {{"size", outputSize}}},
    };
}

inline void WriteManifest(const std::filesystem::path& modelDirectory, const nlohmann::json& manifest) {
    Io::WriteFile(modelDirectory / "manifest.json", [&](std::ostream& out) {
        out << manifest.dump(2);
    });
}

inline void WriteNetwork(const std::filesystem::path& target, const std::vector<std::size_t>& layers) {
    Neural::NeuralNetworkConfiguration configuration{};
    configuration.layersSizes = layers;
    Io::WriteFile(target, [&](std::ostream& out) {
        Neural::SaveNetwork(out, Neural::CreateNetwork(configuration));
    }, std::ios::binary);
}

// A directory that loads: manifest.json plus weights.wgt, contract taken from
// the ends of `layers`.
inline std::filesystem::path WriteModelDirectory(
    const TempDir& dir,
    const std::vector<std::size_t>& layers,
    const std::string& name = "model"
) {
    const auto modelDirectory = MakeModelDirectory(dir, name);
    WriteManifest(modelDirectory, ManifestFor(layers.front(), layers.back()));
    WriteNetwork(modelDirectory / "weights.wgt", layers);
    return modelDirectory;
}

inline void Truncate(const std::filesystem::path& path, const std::uintmax_t bytes) {
    std::filesystem::resize_file(path, bytes);
}

// One model directory inside a registry root, named after the pair it declares
// so the directory name and the manifest stay easy to tell apart in a failure.
struct ModelSpec {
    std::string directory;
    std::string name;
    std::string version;
    std::vector<std::size_t> layers{64, 16, 3};
};

inline std::filesystem::path WriteModelTree(const TempDir& dir, const std::vector<ModelSpec>& models,
                                            const std::string& rootName = "models") {
    const auto root = dir.file(rootName);
    std::filesystem::create_directories(root);
    for (const auto& model : models) {
        const auto modelDirectory = root / model.directory;
        std::filesystem::create_directories(modelDirectory);
        WriteManifest(modelDirectory,
                      ManifestFor(model.layers.front(), model.layers.back(), model.name, model.version));
        WriteNetwork(modelDirectory / "weights.wgt", model.layers);
    }
    return root;
}

}  // namespace Tests
