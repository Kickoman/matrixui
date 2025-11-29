#include "neural_network.h"
#include "matrix.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <random>
#include <fstream>

double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

bool NeuralNetwork::isInitialized() const {
    return initialized;
}

const std::vector<std::size_t>& NeuralNetwork::getLayerSizes() const {
    return layerSizes;
}

void NeuralNetwork::initializeNetwork(const std::vector<size_t>& layerSizes) {
    this->layerSizes = layerSizes;
    this->initialized = true;
}

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& sizes)
    : layerSizes(sizes) {
    if (sizes.size() < 2) {
        throw std::invalid_argument("Network must have at least input and output layers");
    }
    initialized = true;
}

void NeuralNetwork::initializeWeights(std::mt19937& gen) {
    if (!isInitialized()) {
        throw std::runtime_error("Network must be initialized before weights initialization");
    }

    weights.clear();
    biases.clear();

    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
        size_t inputSize = layerSizes[i];
        size_t outputSize = layerSizes[i + 1];

        Matrix weightMatrix(inputSize, outputSize);
        for (size_t row = 0; row < inputSize; ++row) {
            for (size_t col = 0; col < outputSize; ++col) {
                weightMatrix(row, col) = dist(gen);
            }
        }
        weights.push_back(weightMatrix);

        Matrix biasMatrix(1, outputSize, 0.0);
        biases.push_back(biasMatrix);
    }
}

std::vector<Matrix> NeuralNetwork::forwardPass(const Matrix& input) const {
    std::vector<Matrix> activations;
    activations.push_back(input);

    Matrix currentActivation = input;

    for (size_t i = 0; i < weights.size(); ++i) {
        Matrix weightedSum = currentActivation * weights[i];

        for (size_t row = 0; row < weightedSum.getRows(); ++row) {
            for (size_t col = 0; col < weightedSum.getCols(); ++col) {
                weightedSum(row, col) += biases[i](0, col);
            }
        }

        Matrix activation = weightedSum;
        for (size_t row = 0; row < activation.getRows(); ++row) {
            for (size_t col = 0; col < activation.getCols(); ++col) {
                activation(row, col) = sigmoid(weightedSum(row, col));
            }
        }

        activations.push_back(activation);
        currentActivation = activation;
    }

    return activations;
}

void NeuralNetwork::backwardPass(
    const Matrix& target,
    const std::vector<Matrix>& activations,
    const double learningRate
) {
    if (activations.size() != weights.size() + 1) {
        throw std::invalid_argument("Invalid number of activations");
    }

    Matrix outputError = target - activations.back();

    std::vector<Matrix> deltas(weights.size());

    Matrix outputDelta = outputError;
    for (size_t row = 0; row < outputDelta.getRows(); ++row) {
        for (size_t col = 0; col < outputDelta.getCols(); ++col) {
            outputDelta(row, col) *= sigmoidDerivative(activations.back()(row, col));
        }
    }
    deltas.back() = outputDelta;

    for (int i = weights.size() - 2; i >= 0; --i) {
        Matrix hiddenError = deltas[i + 1] * weights[i + 1].transpose();
        Matrix hiddenDelta = hiddenError;

        for (size_t row = 0; row < hiddenDelta.getRows(); ++row) {
            for (size_t col = 0; col < hiddenDelta.getCols(); ++col) {
                hiddenDelta(row, col) *= sigmoidDerivative(activations[i + 1](row, col));
            }
        }
        deltas[i] = hiddenDelta;
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        Matrix weightGradient = activations[i].transpose() * deltas[i] * learningRate;
        weights[i] = weights[i] + weightGradient;

        Matrix biasGradient(1, deltas[i].getCols());
        for (size_t col = 0; col < deltas[i].getCols(); ++col) {
            double sum = 0.0;
            for (size_t row = 0; row < deltas[i].getRows(); ++row) {
                sum += deltas[i](row, col);
            }
            biasGradient(0, col) = sum * learningRate;
        }
        biases[i] = biases[i] + biasGradient;
    }
}

void NeuralNetwork::train(
    const Matrix& inputs,
    const Matrix& targets,
    const int epochs,
    const double learningRate,
    std::ostream& logger
) {
    if (!isInitialized()) {
        throw std::runtime_error("Network must be initalized before training");
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto activations = forwardPass(inputs);
        backwardPass(targets, activations, learningRate);

        if (epoch % 10 == 0) {
            Matrix predictions = activations.back();
            Matrix error = targets - predictions;
            double meanError = 0.0;
            for (size_t i = 0; i < error.getCols(); ++i) {
                meanError += std::abs(error(0, i));
            }
            meanError /= error.getCols();
            logger << "\rEpoch " << epoch << ", Error: " << meanError << "                             " << std::flush;
        }
    }
    logger << std::endl;
}

Matrix NeuralNetwork::predict(const Matrix& input) const {
    if (!isInitialized()) {
        throw std::runtime_error("Network must be initalized before prediction");
    }
    auto activations = forwardPass(input);
    return activations.back();
}

void NeuralNetwork::saveWeights(const std::string& filename) const {
    if (!isInitialized()) {
        throw std::runtime_error("Network must be initialized before saving weights");
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << layerSizes.size() << std::endl;
    for (size_t size : layerSizes) {
        file << size << " ";
    }
    file << std::endl;

    for (size_t i = 0; i < weights.size(); ++i) {
        file << weights[i].getRows() << " " << weights[i].getCols() << std::endl;
        for (size_t row = 0; row < weights[i].getRows(); ++row) {
            for (size_t col = 0; col < weights[i].getCols(); ++col) {
                file << weights[i](row, col) << " ";
            }
            file << std::endl;
        }

        file << biases[i].getRows() << " " << biases[i].getCols() << std::endl;
        for (size_t row = 0; row < biases[i].getRows(); ++row) {
            for (size_t col = 0; col < biases[i].getCols(); ++col) {
                file << biases[i](row, col) << " ";
            }
            file << std::endl;
        }
    }

    file.close();
}

void NeuralNetwork::loadWeights(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    size_t numLayers;
    file >> numLayers;
    std::vector<size_t> loadedLayerSizes(numLayers);
    for (size_t i = 0; i < numLayers; ++i) {
        file >> loadedLayerSizes[i];
    }

    layerSizes = loadedLayerSizes;

    weights.clear();
    biases.clear();
    for (size_t i = 0; i < numLayers - 1; ++i) {
        size_t rows, cols;
        file >> rows >> cols;
        Matrix weightMatrix(rows, cols);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                file >> weightMatrix(row, col);
            }
        }
        weights.push_back(weightMatrix);

        file >> rows >> cols;
        Matrix biasMatrix(rows, cols);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                file >> biasMatrix(row, col);
            }
        }
        biases.push_back(biasMatrix);
    }

    file.close();

    initialized = true;
}
