#pragma once

#include "matrix.h"

#include <cstdint>
#include <vector>

namespace Neural {

enum class ActivationType : std::uint8_t {
    Sigmoid,
    ReLU,
    Tanh,
    Softmax,
};

struct NeuralNetworkConfiguration {
    ActivationType hiddenActivation = ActivationType::ReLU;
    ActivationType outputActivation = ActivationType::Softmax;
    std::vector<std::size_t> layersSizes;
};

struct NeuralNetwork {
    NeuralNetworkConfiguration config;
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;

    void initializeWeights();
    void initializeBiases();


    bool empty() const { return config.layersSizes.empty(); }
    std::size_t layers() const { return config.layersSizes.size(); }
    std::size_t layerSize(const std::size_t index) const { return config.layersSizes[index]; }
    std::size_t inputSize() const { return config.layersSizes.front(); }
    std::size_t outputSize() const { return config.layersSizes.back(); }
    ActivationType hiddenActivation() const { return config.hiddenActivation; }
    ActivationType outputActivation() const { return config.outputActivation; }
};

}
