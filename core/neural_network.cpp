#include "neural_network.h"

#include <random>


namespace {

double StdDevByActivation(
    const Neural::ActivationType activation,
    const std::size_t inputSize,
    const std::size_t outputSize
) {
    using namespace Neural;
    switch (activation) {
        case ActivationType::ReLU:      return std::sqrt(2. / inputSize);
        case ActivationType::Sigmoid:   return std::sqrt(1. / inputSize);
        default:
            return std::sqrt(1. / inputSize);
    }
}

}

namespace Neural {

void NeuralNetwork::initializeWeights() {
    weights.clear();
    if (empty()) {
        return;
    }
    weights.reserve(layers() - 1);
    std::mt19937 generator;
    for (std::size_t i = 0; i < layers() - 1; ++i) {
        const std::size_t inputSize = layerSize(i);
        const std::size_t outputSize = layerSize(i + 1);
        const bool isOutputLayer = i + 2 == layers();

        const double standardDeviation = StdDevByActivation(
            isOutputLayer ? outputActivation() : hiddenActivation(),
            inputSize,
            outputSize
        );
        std::normal_distribution<double> distribution(0., standardDeviation);

        Matrix layerWeights(inputSize, outputSize);
        for (std::size_t row = 0; row < inputSize; ++row) {
            for (std::size_t col = 0; col < outputSize; ++col) {
                layerWeights(row, col) = distribution(generator);
            }
        }
        weights.push_back(layerWeights);
    }
}


void NeuralNetwork::initializeBiases() {
    biases.clear();
    if (empty()) {
        return;
    }
    biases.reserve(layers() - 1);
    for (std::size_t i = 1; i < layers(); ++i) {
        const std::size_t outputSize = layerSize(i);
        biases.push_back(Matrix(1, outputSize, 0.01));
    }
}

}
