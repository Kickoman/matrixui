#include "neural_network_loader.h"
#include "neural_network.h"

#include <filesystem>
#include <fstream>
#include <random>


namespace Neural {


std::optional<NeuralNetwork> LoadNetwork(const std::string& filename) {
    if (!std::filesystem::exists(filename)) {
        return std::nullopt;
    }
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    std::size_t numLayers;
    file >> numLayers;
    std::vector<std::size_t> layerSizes(numLayers);
    for (std::size_t i = 0; i < numLayers; ++i) {
        file >> layerSizes[i];
    }

    std::vector<Matrix> weights;
    std::vector<Matrix> biases;

    for (std::size_t i = 0; i < numLayers - 1; ++i) {
        std::size_t rows, cols;
        file >> rows >> cols;
        weights.push_back(Matrix(rows, cols));
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                file >> weights.back()(row, col);
            }
        }

        file >> rows >> cols;
        biases.push_back(Matrix(rows, cols));
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                file >> biases.back()(row, col);
            }
        }
    }
    NeuralNetwork network;
    network.initializeNetwork(layerSizes);
    network.setWeights(weights);
    network.setBiases(biases);
    return network;
}


NeuralNetwork LoadNetwork(const std::string& filename, const std::vector<std::size_t>& layers) {
    NeuralNetwork network;
    if (std::filesystem::exists(filename)) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for reading: " + filename);
        }

        std::size_t numLayers;
        file >> numLayers;
        std::vector<std::size_t> layerSizes(numLayers);
        for (std::size_t i = 0; i < numLayers; ++i) {
            file >> layerSizes[i];
        }

        std::vector<Matrix> weights;
        std::vector<Matrix> biases;

        for (std::size_t i = 0; i < numLayers - 1; ++i) {
            std::size_t rows, cols;
            file >> rows >> cols;
            weights.push_back(Matrix(rows, cols));
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t col = 0; col < cols; ++col) {
                    file >> weights.back()(row, col);
                }
            }

            file >> rows >> cols;
            biases.push_back(Matrix(rows, cols));
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t col = 0; col < cols; ++col) {
                    file >> biases.back()(row, col);
                }
            }
        }

        network.initializeNetwork(layerSizes);
        network.setWeights(weights);
        network.setBiases(biases);
        return network;
    }
    std::mt19937 gen;
    network.initializeNetwork(layers);
    network.initializeWeights(gen);
    return network;
}


void SaveNetwork(const NeuralNetwork& network, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    const auto& layerSizes = network.getLayerSizes();
    file << layerSizes.size() << std::endl;
    for (const auto& size : layerSizes) {
        file << size << " ";
    }
    file << std::endl;

    const auto& weights = network.getWeights();
    const auto& biases = network.getBiases();
    for (std::size_t i = 0; i < weights.size(); ++i) {
        file << weights[i].getRows() << " " << weights[i].getCols() << std::endl;
        for (std::size_t row = 0; row < weights[i].getRows(); ++row) {
            for (std::size_t col = 0; col < weights[i].getCols(); ++col) {
                file << weights[i](row, col) << " ";
            }
            file << std::endl;
        }
        file << biases[i].getRows() << " " << biases[i].getCols() << std::endl;
        for (std::size_t row = 0; row < biases[i].getRows(); ++row) {
            for (std::size_t col = 0; col < biases[i].getCols(); ++col) {
                file << biases[i](row, col) << " ";
            }
            file << std::endl;
        }
        file << std::endl;
    }
}

}
