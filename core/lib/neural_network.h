#pragma once

#include "core/lib/layers.h"

#include <vector>
#include <nlohmann/json.hpp>

namespace Neural {

struct NeuralNetworkConfiguration {
    ActivationType hiddenActivation = ActivationType::ReLU;
    ActivationType outputActivation = ActivationType::Softmax;
    std::vector<std::size_t> layersSizes;
};

struct NeuralNetwork {
    NeuralNetworkConfiguration config;
    std::vector<LayerData> layerStack;

    bool empty() const { return layerStack.empty(); }
    std::size_t inputSize() const { return config.layersSizes.front(); }
    std::size_t outputSize() const { return config.layersSizes.back(); }
    ActivationType hiddenActivation() const { return config.hiddenActivation; }
    ActivationType outputActivation() const { return config.outputActivation; }
};

}

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    Neural::NeuralNetworkConfiguration,
    hiddenActivation,
    outputActivation,
    layersSizes
);
