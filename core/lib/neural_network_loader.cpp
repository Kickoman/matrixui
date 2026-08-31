#include "core/lib/neural_network_loader.h"
#include "core/lib/neural_network.h"
#include "core/lib/layers.h"
#include "core/lib/write.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <random>
#include <cmath>

namespace Neural {

namespace {


double StdDevByActivation(const ActivationType activation, const std::size_t inputSize) {
    switch (activation) {
        case ActivationType::ReLU:
        case ActivationType::LeakyReLU: return std::sqrt(2. / inputSize);  // He initialization
        case ActivationType::Sigmoid:   return std::sqrt(1. / inputSize);
        default:                        return std::sqrt(1. / inputSize);
    }
}

void PushActivationLayers(
    std::vector<LayerData>& stack,
    const NeuralNetworkConfiguration& config,
    const bool isLastLayer
) {
    if (!isLastLayer) {
        stack.push_back(ActivationLayer{config.hiddenActivation});
        stack.push_back(DropoutLayer{});
    } else if (config.outputActivation == ActivationType::Softmax) {
        stack.push_back(SoftmaxLayer{});
    } else {
        stack.push_back(ActivationLayer{config.outputActivation});
    }
}

} // namespace


void SaveNetwork(const NeuralNetwork& network, const std::filesystem::path& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing");
    }

    constexpr std::uint32_t version = 1;
    WriteBinaryLE(file, version);
    WriteBinaryLE(file, static_cast<std::uint8_t>(network.config.hiddenActivation));
    WriteBinaryLE(file, static_cast<std::uint8_t>(network.config.outputActivation));

    const auto numLayers = static_cast<std::uint32_t>(network.config.layersSizes.size());
    WriteBinaryLE(file, numLayers);
    for (auto size : network.config.layersSizes) {
        WriteBinaryLE(file, static_cast<std::uint64_t>(size));
    }

    for (const auto& layerData : network.layerStack) {
        if (const auto* dense = std::get_if<DenseLayer>(&layerData)) {
            const std::size_t rows = dense->weights.getRows();
            const std::size_t cols = dense->weights.getCols();

            std::vector<double> weightBuf(rows * cols);
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t col = 0; col < cols; ++col) {
                    weightBuf[row * cols + col] = dense->weights(row, col);
                }
            }
            WriteBulkLE(file, weightBuf);

            std::vector<double> biasBuf(cols);
            for (std::size_t col = 0; col < cols; ++col) {
                biasBuf[col] = dense->biases(0, col);
            }
            WriteBulkLE(file, biasBuf);
        }
    }
}

std::optional<NeuralNetworkConfiguration> LoadConfig(const std::filesystem::path& filename) {
    if (!std::filesystem::exists(filename)) {
        return std::nullopt;
    }
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename.string());
    }

    file.seekg(sizeof(std::uint32_t)); // skip version

    NeuralNetworkConfiguration config;
    std::uint8_t hiddenActivation, outputActivation;
    ReadBinaryLE(file, hiddenActivation);
    ReadBinaryLE(file, outputActivation);
    config.hiddenActivation = static_cast<ActivationType>(hiddenActivation);
    config.outputActivation = static_cast<ActivationType>(outputActivation);

    try {
        std::uint32_t numLayers;
        ReadBinaryLE(file, numLayers);
        config.layersSizes.resize(numLayers);
        for (auto& layer : config.layersSizes) {
            std::uint64_t size;
            ReadBinaryLE(file, size);
            layer = static_cast<decltype(layer)>(size);
        }
        return config;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<NeuralNetwork> LoadNetwork(const std::filesystem::path& filename) {
    if (!std::filesystem::exists(filename)) {
        return std::nullopt;
    }
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading");
    }

    std::uint32_t version;
    ReadBinaryLE(file, version);
    if (version != 1) {
        throw std::runtime_error("Unsupported network version: " + std::to_string(version));
    }

    NeuralNetwork network;
    std::uint8_t hiddenActivation, outputActivation;
    ReadBinaryLE(file, hiddenActivation);
    ReadBinaryLE(file, outputActivation);
    network.config.hiddenActivation = static_cast<ActivationType>(hiddenActivation);
    network.config.outputActivation = static_cast<ActivationType>(outputActivation);

    std::uint32_t numLayers;
    ReadBinaryLE(file, numLayers);
    network.config.layersSizes.resize(numLayers);
    for (std::uint32_t i = 0; i < numLayers; ++i) {
        std::uint64_t s;
        ReadBinaryLE(file, s);
        network.config.layersSizes[i] = static_cast<std::size_t>(s);
    }

    const std::size_t numTransitions = numLayers - 1;
    for (std::size_t i = 0; i < numTransitions; ++i) {
        const std::size_t rows = network.config.layersSizes[i];
        const std::size_t cols = network.config.layersSizes[i + 1];
        const bool isLastLayer = (i == numTransitions - 1);

        DenseLayer dense;
        dense.weights = Matrix(rows, cols);
        dense.biases = Matrix(1, cols);
        dense.gradientWeights = Matrix::zeros(rows, cols);
        dense.gradientBiases = Matrix::zeros(1, cols);

        std::vector<double> weightBuf(rows * cols);
        ReadBulkLE(file, weightBuf);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                dense.weights(row, col) = weightBuf[row * cols + col];
            }
        }

        std::vector<double> biasBuf(cols);
        ReadBulkLE(file, biasBuf);
        for (std::size_t col = 0; col < cols; ++col) {
            dense.biases(0, col) = biasBuf[col];
        }

        network.layerStack.push_back(std::move(dense));
        PushActivationLayers(network.layerStack, network.config, isLastLayer);
    }

    return network;
}


NeuralNetwork CreateNetwork(const NeuralNetworkConfiguration& config) {
    NeuralNetwork network;
    network.config = config;

    std::mt19937 generator{std::random_device{}()};
    const std::size_t numTransitions = config.layersSizes.size() - 1;

    for (std::size_t i = 0; i < numTransitions; ++i) {
        const bool isLastLayer = (i == numTransitions - 1);
        const std::size_t inputSize = config.layersSizes[i];
        const std::size_t outputSize = config.layersSizes[i + 1];

        const ActivationType activation = isLastLayer ? config.outputActivation : config.hiddenActivation;
        const double stddev = StdDevByActivation(activation, inputSize);
        std::normal_distribution<double> dist(0., stddev);

        DenseLayer dense;
        dense.weights = Matrix(inputSize, outputSize);
        for (std::size_t row = 0; row < inputSize; ++row) {
            for (std::size_t col = 0; col < outputSize; ++col) {
                dense.weights(row, col) = dist(generator);
            }
        }
        dense.biases = Matrix(1, outputSize, 0.01);
        dense.gradientWeights = Matrix::zeros(inputSize, outputSize);
        dense.gradientBiases = Matrix::zeros(1, outputSize);

        network.layerStack.push_back(std::move(dense));
        PushActivationLayers(network.layerStack, config, isLastLayer);
    }

    return network;
}

}
