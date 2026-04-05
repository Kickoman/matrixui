#include "neural_network.h"
#include "matrix.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <random>


namespace Neural {

void NeuralNetwork::initializeWeights() {
    weights.clear();
    if (layersSizes.empty()) {
        return;
    }
    weights.reserve(layersSizes.size() - 1);
    std::mt19937 generator;
    for (std::size_t i = 0; i < layersSizes.size() - 1; ++i) {
        const std::size_t inputSize = layersSizes[i];
        const std::size_t outputSize = layersSizes[i + 1];
        const double standardDeviation = std::sqrt(1. / inputSize);
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
    if (layersSizes.empty()) {
        return;
    }
    biases.reserve(layersSizes.size() - 1);
    for (std::size_t i = 1; i < layersSizes.size(); ++i) {
        const std::size_t outputSize = layersSizes[i];
        biases.push_back(Matrix(1, outputSize, 0));
    }
}

double NeuralNetworkApplier::applyActivation(const double x, const ActivationType type) {
    switch(type) {
        case ActivationType::Sigmoid: return 1.0 / (1.0 + std::exp(-x));
        case ActivationType::ReLU:    return std::max(0.0, x);
        case ActivationType::Tanh:    return std::tanh(x);
        default: return x;
    }
}

double NeuralNetworkApplier::applyActivationDerivative(const double x, const ActivationType type) {
    switch(type) {
        case ActivationType::Sigmoid: return x * (1.0 - x);
        case ActivationType::ReLU:    return x > 0.0 ? 1.0 : 0.0;
        case ActivationType::Tanh:    return 1.0 - x*x;
        default: return 1.0;
    }
}

bool NeuralNetworkApplier::isInitialized() const {
    return initialized;
}

const NeuralNetwork& NeuralNetworkApplier::getNeuralNetworkConfig() const {
    return config;
}

void NeuralNetworkApplier::initializeNetwork(const NeuralNetwork& config) {
    this->config = config;
    this->initialized = true;
}

void NeuralNetworkApplier::initializeNetwork(NeuralNetwork&& config) {
    this->config = std::move(config);
    this->initialized = true;
}

NeuralNetworkApplier::NeuralNetworkApplier(const NeuralNetwork& config) {
    initializeNetwork(config);
}

NeuralNetworkApplier::NeuralNetworkApplier(NeuralNetwork&& config) {
    initializeNetwork(std::move(config));
}

std::vector<Matrix> NeuralNetworkApplier::forwardPass(const Matrix& input) const {
    std::vector<Matrix> activations;
    activations.reserve(config.layersSizes.size());
    activations.push_back(input);

    Matrix currentActivation = input;

    for (size_t i = 0; i < config.weights.size(); ++i) {
        Matrix weightedSum = currentActivation * config.weights[i];

        for (size_t row = 0; row < weightedSum.getRows(); ++row) {
            for (size_t col = 0; col < weightedSum.getCols(); ++col) {
                weightedSum(row, col) += config.biases[i](0, col);
            }
        }

        Matrix activation = weightedSum;
        for (size_t row = 0; row < activation.getRows(); ++row) {
            for (size_t col = 0; col < activation.getCols(); ++col) {
                activation(row, col) = applyActivation(
                    weightedSum(row, col),
                    i < config.weights.size() - 1 ? config.hiddenActivation : config.outputActivation
                );
            }
        }

        activations.push_back(activation);
        currentActivation = activation;

    }

    return activations;
}

void NeuralNetworkApplier::backwardPass(
    const std::vector<Matrix>& activations,
    const Matrix& error,
    const double learningRate
) {
    if (activations.size() != config.weights.size() + 1) {
        throw std::invalid_argument("Invalid number of activations");
    }

    std::vector<Matrix> deltas(config.weights.size());
    deltas.back() = error;
    for (size_t row = 0; row < error.getRows(); ++row) {
        for (size_t col = 0; col < error.getCols(); ++col) {
            deltas.back()(row, col) *= applyActivationDerivative(
                activations.back()(row, col),
                config.outputActivation
            );
        }
    }

    for (int i = config.weights.size() - 2; i >= 0; --i) {
        Matrix hiddenError = deltas[i + 1] * config.weights[i + 1].transpose();
        for (size_t row = 0; row < hiddenError.getRows(); ++row) {
            for (size_t col = 0; col < hiddenError.getCols(); ++col) {
                hiddenError(row, col) *= applyActivationDerivative(
                    activations[i + 1](row, col),
                    config.hiddenActivation
                );
            }
        }
        deltas[i] = hiddenError;
    }

    for (size_t i = 0; i < config.weights.size(); ++i) {
        Matrix weightGradient = activations[i].transpose() * deltas[i] * learningRate;
        Matrix biasGradient(1, deltas[i].getCols());
        for (size_t col = 0; col < deltas[i].getCols(); ++col) {
            double sum = 0;
            for (size_t row = 0; row < deltas[i].getRows(); ++row) {
                sum += deltas[i](row, col);
            }
            biasGradient(0, col) = sum * learningRate;
        }

        config.weights[i] = config.weights[i] - weightGradient;
        config.biases[i] = config.biases[i] - biasGradient;
    }
}

void NeuralNetworkApplier::train(const Matrix& input, const Matrix& target, const double learningRate) {
    const auto activations = forwardPass(input);
    backwardPass(activations, activations.back() - target, learningRate);
}

Matrix NeuralNetworkApplier::predict(const Matrix& input) const {
    if (!isInitialized()) {
        throw std::runtime_error("Network must be initalized before prediction");
    }
    auto activations = forwardPass(input);
    return activations.back();
}

}
