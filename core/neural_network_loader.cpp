#include "neural_network_loader.h"
#include "neural_network.h"
#include "layers.h"

#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <bit>
#include <random>
#include <cmath>

namespace Neural {

namespace {

constexpr bool IsLittleEndian() {
    return std::endian::native == std::endian::little;
}

inline std::uint16_t SwapBytes(std::uint16_t val) {
    return (val >> 8) | (val << 8);
}

inline std::uint32_t SwapBytes(std::uint32_t val) {
    return ((val >> 24) & 0xFF) |
           ((val >> 8) & 0xFF00) |
           ((val << 8) & 0xFF0000) |
           ((val << 24) & 0xFF000000);
}

inline std::uint64_t SwapBytes(std::uint64_t val) {
    return ((val & 0x00000000000000FFULL) << 56) |
           ((val & 0x000000000000FF00ULL) << 40) |
           ((val & 0x0000000000FF0000ULL) << 24) |
           ((val & 0x00000000FF000000ULL) << 8)  |
           ((val & 0x000000FF00000000ULL) >> 8)  |
           ((val & 0x0000FF0000000000ULL) >> 24) |
           ((val & 0x00FF000000000000ULL) >> 40) |
           ((val & 0xFF00000000000000ULL) >> 56);
}

template<typename T>
void WriteBinaryLE(std::ofstream& file, T value) {
    if constexpr (!IsLittleEndian()) {
        if constexpr (sizeof(T) == 2) {
            value = SwapBytes(static_cast<std::uint16_t>(value));
        } else if constexpr (sizeof(T) == 4) {
            value = SwapBytes(static_cast<std::uint32_t>(value));
        } else if constexpr (sizeof(T) == 8) {
            value = SwapBytes(static_cast<std::uint64_t>(value));
        }
    }
    file.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template<typename T>
void ReadBinaryLE(std::ifstream& file, T& value) {
    file.read(reinterpret_cast<char*>(&value), sizeof(T));

    if constexpr (!IsLittleEndian()) {
        if constexpr (sizeof(T) == 2) {
            value = SwapBytes(static_cast<std::uint16_t>(value));
        } else if constexpr (sizeof(T) == 4) {
            value = SwapBytes(static_cast<std::uint32_t>(value));
        } else if constexpr (sizeof(T) == 8) {
            value = SwapBytes(static_cast<std::uint64_t>(value));
        }
    }
}

template<>
inline void WriteBinaryLE(std::ofstream& file, double value) {
    std::uint64_t temp;
    std::memcpy(&temp, &value, sizeof(double));
    WriteBinaryLE(file, temp);
}

template<>
inline void ReadBinaryLE(std::ifstream& file, double& value) {
    std::uint64_t temp;
    ReadBinaryLE(file, temp);
    std::memcpy(&value, &temp, sizeof(double));
}

void WriteBulkLE(std::ofstream& file, const std::vector<double>& values) {
    if constexpr (IsLittleEndian()) {
        file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
    } else {
        for (double val : values) WriteBinaryLE(file, val);
    }
}

void ReadBulkLE(std::ifstream& file, std::vector<double>& values) {
    if constexpr (IsLittleEndian()) {
        file.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(double));
    } else {
        for (double& val : values) ReadBinaryLE(file, val);
    }
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

} // namespace


void SaveNetwork(const NeuralNetwork& network, const std::string& filename) {
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

std::optional<NeuralNetworkConfiguration> LoadConfig(const std::string& filename) {
    if (!std::filesystem::exists(filename)) {
        return std::nullopt;
    }
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
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

std::optional<NeuralNetwork> LoadNetwork(const std::string& filename) {
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
