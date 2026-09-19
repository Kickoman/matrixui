#include "core/nn/neural_network_loader.h"
#include "core/nn/neural_network.h"
#include "core/nn/layers.h"
#include "core/lib/write.h"

#include <istream>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <random>
#include <cmath>

namespace Neural {

namespace {


constexpr std::uint32_t kMaxLayers = 1u << 12;
constexpr std::uint64_t kMaxLayerSize = std::uint64_t{1} << 24;
constexpr std::uint64_t kMaxWeightBytes = std::uint64_t{4} << 30;

std::int64_t RemainingBytes(std::istream& in) {
    const auto position = in.tellg();
    if (position < 0) {
        in.clear();
        return -1;
    }
    in.seekg(0, std::ios::end);
    const std::int64_t end = in.tellg();
    in.seekg(position);
    return end < 0 ? -1 : end - static_cast<std::int64_t>(position);
}


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

}  // namespace


void SaveNetwork(std::ostream& out, const NeuralNetwork& network) {
    constexpr std::uint32_t version = 1;
    WriteBinaryLE(out, version);
    WriteBinaryLE(out, static_cast<std::uint8_t>(network.config.hiddenActivation));
    WriteBinaryLE(out, static_cast<std::uint8_t>(network.config.outputActivation));

    const auto numLayers = static_cast<std::uint32_t>(network.config.layersSizes.size());
    WriteBinaryLE(out, numLayers);
    for (auto size : network.config.layersSizes) {
        WriteBinaryLE(out, static_cast<std::uint64_t>(size));
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
            WriteBulkLE(out, weightBuf);

            std::vector<double> biasBuf(cols);
            for (std::size_t col = 0; col < cols; ++col) {
                biasBuf[col] = dense->biases(0, col);
            }
            WriteBulkLE(out, biasBuf);
        }
    }
}

std::optional<NeuralNetworkConfiguration> LoadConfig(std::istream& in) {
    std::uint32_t version = 0;
    ReadBinaryLE(in, version);
    if (!in || version != 1) {
        return std::nullopt;
    }

    NeuralNetworkConfiguration config;
    std::uint8_t hiddenActivation = 0;
    std::uint8_t outputActivation = 0;
    ReadBinaryLE(in, hiddenActivation);
    ReadBinaryLE(in, outputActivation);
    config.hiddenActivation = static_cast<ActivationType>(hiddenActivation);
    config.outputActivation = static_cast<ActivationType>(outputActivation);

    try {
        std::uint32_t numLayers = 0;
        ReadBinaryLE(in, numLayers);
        if (!in || numLayers < 2 || numLayers > kMaxLayers) {
            return std::nullopt;
        }

        config.layersSizes.resize(numLayers);
        for (auto& layer : config.layersSizes) {
            std::uint64_t size = 0;
            ReadBinaryLE(in, size);
            layer = static_cast<std::size_t>(size);
        }
        if (!in) {
            return std::nullopt;
        }

        for (const auto layer : config.layersSizes) {
            if (layer == 0 || layer > kMaxLayerSize) {
                return std::nullopt;
            }
        }
        return config;
    } catch (...) {
        return std::nullopt;
    }
}

NeuralNetwork LoadNetwork(std::istream& in) {
    std::uint32_t version = 0;
    ReadBinaryLE(in, version);
    if (!in) {
        throw std::runtime_error("Network file is truncated: no version header");
    }
    if (version != 1) {
        throw std::runtime_error("Unsupported network version: " + std::to_string(version));
    }

    NeuralNetwork network;
    std::uint8_t hiddenActivation = 0;
    std::uint8_t outputActivation = 0;
    ReadBinaryLE(in, hiddenActivation);
    ReadBinaryLE(in, outputActivation);
    if (!in) {
        throw std::runtime_error("Network file is truncated: no activation header");
    }
    network.config.hiddenActivation = static_cast<ActivationType>(hiddenActivation);
    network.config.outputActivation = static_cast<ActivationType>(outputActivation);

    std::uint32_t numLayers = 0;
    ReadBinaryLE(in, numLayers);
    if (!in) {
        throw std::runtime_error("Network file is truncated: no layer count");
    }
    if (numLayers < 2) {
        throw std::runtime_error(
            "Network file declares " + std::to_string(numLayers)
            + " layers, at least 2 are required");
    }
    if (numLayers > kMaxLayers) {
        throw std::runtime_error(
            "Network file declares " + std::to_string(numLayers) + " layers, more than the "
            + std::to_string(kMaxLayers) + " allowed");
    }

    if (const std::int64_t remaining = RemainingBytes(in);
        remaining >= 0
        && static_cast<std::uint64_t>(remaining) < std::uint64_t{numLayers} * sizeof(std::uint64_t)) {
        throw std::runtime_error(
            "Network file declares " + std::to_string(numLayers) + " layers but holds only "
            + std::to_string(remaining) + " bytes after the header");
    }

    network.config.layersSizes.resize(numLayers);
    for (std::uint32_t i = 0; i < numLayers; ++i) {
        std::uint64_t size = 0;
        ReadBinaryLE(in, size);
        network.config.layersSizes[i] = static_cast<std::size_t>(size);
    }
    if (!in) {
        throw std::runtime_error("Network file is truncated: layer sizes are incomplete");
    }
    for (std::uint32_t i = 0; i < numLayers; ++i) {
        const std::uint64_t size = network.config.layersSizes[i];
        if (size == 0 || size > kMaxLayerSize) {
            throw std::runtime_error(
                "Network layer " + std::to_string(i) + " declares an unusable size: "
                + std::to_string(size) + " (allowed 1.." + std::to_string(kMaxLayerSize) + ")");
        }
    }

    std::uint64_t weightBytes = 0;
    const std::size_t numTransitions = numLayers - 1;
    for (std::size_t i = 0; i < numTransitions; ++i) {
        const std::uint64_t rows = network.config.layersSizes[i];
        const std::uint64_t cols = network.config.layersSizes[i + 1];
        weightBytes += (rows * cols + cols) * sizeof(double);
        if (weightBytes > kMaxWeightBytes) {
            throw std::runtime_error(
                "Network file declares more than " + std::to_string(kMaxWeightBytes)
                + " bytes of weights");
        }
    }
    if (const std::int64_t remaining = RemainingBytes(in);
        remaining >= 0 && static_cast<std::uint64_t>(remaining) < weightBytes) {
        throw std::runtime_error(
            "Network file is shorter than its header claims (expected "
            + std::to_string(weightBytes) + " bytes of weights, found "
            + std::to_string(remaining) + ")");
    }

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
        ReadBulkLE(in, weightBuf);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                dense.weights(row, col) = weightBuf[row * cols + col];
            }
        }

        std::vector<double> biasBuf(cols);
        ReadBulkLE(in, biasBuf);
        for (std::size_t col = 0; col < cols; ++col) {
            dense.biases(0, col) = biasBuf[col];
        }

        network.layerStack.push_back(std::move(dense));
        PushActivationLayers(network.layerStack, network.config, isLastLayer);
    }

    if (!in) {
        throw std::runtime_error("Network file is truncated: weights are incomplete");
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
