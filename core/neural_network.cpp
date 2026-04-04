#include "neural_network.h"
#include "matrix.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <random>


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

    for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
        size_t inputSize = layerSizes[i];
        size_t outputSize = layerSizes[i + 1];

        double stddev = std::sqrt(1.0 / inputSize);
        std::normal_distribution<double> dist(0.0, stddev);

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
    activations.reserve(layerSizes.size());
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
    const std::vector<Matrix>& activations,
    const Matrix& error,
    const double learningRate
) {
    if (activations.size() != weights.size() + 1) {
        throw std::invalid_argument("Invalid number of activations");
    }

    std::vector<Matrix> deltas(weights.size());
    deltas.back() = error;
    for (size_t row = 0; row < error.getRows(); ++row) {
        for (size_t col = 0; col < error.getCols(); ++col) {
            deltas.back()(row, col) *= sigmoidDerivative(activations.back()(row, col));
        }
    }

    for (int i = weights.size() - 2; i >= 0; --i) {
        Matrix hiddenError = deltas[i + 1] * weights[i + 1].transpose();
        for (size_t row = 0; row < hiddenError.getRows(); ++row) {
            for (size_t col = 0; col < hiddenError.getCols(); ++col) {
                hiddenError(row, col) *= sigmoidDerivative(activations[i + 1](row, col));
            }
        }
        deltas[i] = hiddenError;
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        Matrix weightGradient = activations[i].transpose() * deltas[i] * learningRate;
        weights[i] = weights[i] - weightGradient;

        Matrix biasGradient(1, deltas[i].getCols());
        for (size_t col = 0; col < deltas[i].getCols(); ++col) {
            double sum = 0;
            for (size_t row = 0; row < deltas[i].getRows(); ++row) {
                sum += deltas[i](row, col);
            }
            biasGradient(0, col) = sum * learningRate;
        }
        biases[i] = biases[i] - biasGradient;
    }
}

void NeuralNetwork::train(const Matrix& input, const Matrix& target, const double learningRate) {
    const auto activations = forwardPass(input);
    backwardPass(activations, activations.back() - target, learningRate);
}

Matrix NeuralNetwork::predict(const Matrix& input) const {
    if (!isInitialized()) {
        throw std::runtime_error("Network must be initalized before prediction");
    }
    auto activations = forwardPass(input);
    return activations.back();
}

void NeuralNetwork::setWeights(const std::vector<Matrix>& newWeights) {
    weights = newWeights;
}

void NeuralNetwork::setBiases(const std::vector<Matrix>& newBiases) {
    biases = newBiases;
}

const std::vector<Matrix>& NeuralNetwork::getWeights() const {
    return weights;
}

const std::vector<Matrix>& NeuralNetwork::getBiases() const {
    return biases;
}
