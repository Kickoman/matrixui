#include "neural_network_loader.h"
#include "neural_network.h"

#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstring>

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

}


void SaveNetwork(const NeuralNetwork& network, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing");
    }

    constexpr std::uint32_t version = 1;
    WriteBinaryLE(file, version);
    WriteBinaryLE(file, static_cast<std::uint8_t>(network.hiddenActivation));
    WriteBinaryLE(file, static_cast<std::uint8_t>(network.outputActivation));

    const auto numLayers = static_cast<std::uint32_t>(network.layersSizes.size());
    WriteBinaryLE(file, numLayers);
    for (auto size : network.layersSizes) {
        WriteBinaryLE(file, static_cast<std::uint64_t>(size));
    }

    for (std::size_t i = 0; i < network.weights.size(); ++i) {
        const auto& weights = network.weights[i];
        const auto& biases = network.biases[i];

        for (std::size_t row = 0; row < weights.getRows(); ++row) {
            for (std::size_t col = 0; col < weights.getCols(); ++col) {
                WriteBinaryLE(file, weights(row, col));
            }
        }

        for (std::size_t col = 0; col < biases.getCols(); ++col) {
            WriteBinaryLE(file, biases(0, col));
        }
    }
}

std::optional<std::vector<std::size_t>> LoadLayerSizes(const std::string &filename) {
    if (!std::filesystem::exists(filename)) {
        return std::nullopt;
    }
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    // Skip 32bit version, 16bit activations
    file.seekg(sizeof(std::uint32_t) + 2 * sizeof(std::uint8_t));

    try {
        std::uint32_t numLayers;
        std::vector<std::size_t> layers;
        ReadBinaryLE(file, numLayers);
        layers.resize(numLayers);
        for (auto& layer : layers) {
            std::uint64_t size;
            ReadBinaryLE(file, size);
            layer = static_cast<decltype(layer)>(size);
        }
        return layers;
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
    network.hiddenActivation = static_cast<ActivationType>(hiddenActivation);
    network.outputActivation = static_cast<ActivationType>(outputActivation);

    std::uint32_t numLayers;
    ReadBinaryLE(file, numLayers);
    network.layersSizes.resize(numLayers);
    for (std::uint32_t i = 0; i < numLayers; ++i) {
        std::uint64_t s;
        ReadBinaryLE(file, s);
        network.layersSizes[i] = static_cast<std::size_t>(s);
    }

    network.weights.resize(numLayers - 1);
    network.biases.resize(numLayers - 1);

    for (std::size_t i = 0; i < numLayers - 1; ++i) {
        std::size_t rows = network.layersSizes[i];
        std::size_t cols = network.layersSizes[i+1];
        network.weights[i] = Matrix(rows, cols);
        network.biases[i] = Matrix(1, cols);

        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                ReadBinaryLE(file, network.weights[i](row, col));
            }
        }

        for (std::size_t col = 0; col < cols; ++col) {
            ReadBinaryLE(file, network.biases[i](0, col));
        }
    }

    return network;
}


NeuralNetwork LoadNetwork(const std::string& filename, const std::vector<std::size_t>& layers) {
    if (std::filesystem::exists(filename)) {
        auto networkMaybe = LoadNetwork(filename);
        if (networkMaybe.has_value()) return *networkMaybe;
    }

    NeuralNetwork network;
    network.layersSizes = layers;
    network.initializeWeights();
    network.initializeBiases();
    return network;
}

}
