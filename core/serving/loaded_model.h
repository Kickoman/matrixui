#pragma once

#include "core/nn/neural_network.h"
#include "core/serving/manifest.h"

#include <filesystem>
#include <memory>
#include <utility>

namespace Serving {

enum class IntegrityCheck {
    NotDeclared,
    Verified,
};

class LoadedModel {
public:
    LoadedModel(
        ModelManifest manifest,
        std::shared_ptr<const Neural::NeuralNetwork> network,
        std::filesystem::path directory,
        IntegrityCheck integrity
    );

    const ModelManifest& manifest() const { return manifestData; }
    std::shared_ptr<const Neural::NeuralNetwork> network() const { return networkData; }
    const std::filesystem::path& directory() const { return directoryPath; }
    IntegrityCheck integrity() const { return integrityState; }

private:
    ModelManifest manifestData;
    std::shared_ptr<const Neural::NeuralNetwork> networkData;
    std::filesystem::path directoryPath;
    IntegrityCheck integrityState = IntegrityCheck::NotDeclared;
};

}  // namespace Serving
