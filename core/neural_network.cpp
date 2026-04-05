#include "neural_network.h"
#include "matrix.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <random>


namespace Neural {

double StdDevByActivation(const ActivationType activation, const std::size_t inputSize, const std::size_t outputSize) {
    switch (activation) {
        case ActivationType::ReLU:      return std::sqrt(2. / inputSize);
        case ActivationType::Sigmoid:   return std::sqrt(1. / inputSize);
        default:
            return std::sqrt(1. / inputSize);
    }
}

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
        const bool isOutputLayer = i + 2 == layersSizes.size();

        const double standardDeviation = StdDevByActivation(
            isOutputLayer ? outputActivation : hiddenActivation,
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
    if (layersSizes.empty()) {
        return;
    }
    biases.reserve(layersSizes.size() - 1);
    for (std::size_t i = 1; i < layersSizes.size(); ++i) {
        const std::size_t outputSize = layersSizes[i];
        biases.push_back(Matrix(1, outputSize, 0.01));
    }
}

double NeuralNetworkApplier::applyActivation(const double x, const ActivationType type) {
    assert(type != ActivationType::Softmax);
    switch(type) {
        case ActivationType::Sigmoid: return 1.0 / (1.0 + std::exp(-x));
        case ActivationType::ReLU:    return std::max(0.0, x);
        case ActivationType::Tanh:    return std::tanh(x);
        default: return x;
    }
}

double NeuralNetworkApplier::applyActivationDerivative(const double x, const ActivationType type) {
    assert(type != ActivationType::Softmax);
    switch(type) {
        case ActivationType::Sigmoid: return x * (1.0 - x);
        case ActivationType::ReLU:    return x > 0.0 ? 1.0 : 0.0;
        case ActivationType::Tanh:    return 1.0 - x*x;
        default: return 1.0;
    }
}

void ApplySoftMax(Matrix& m) {
    for (std::size_t row = 0; row < m.getRows(); ++row) {
        double maxValue = m(row, 0);
        for (std::size_t col = 1; col < m.getCols(); ++col) {
            maxValue = std::max(maxValue, m(row, col));
        }

        double sum = 0;
        for (std::size_t col = 0; col < m.getCols(); ++col) {
            m(row, col) = std::exp(m(row, col) - maxValue);
            sum += m(row, col);
        }

        for (std::size_t col = 0; col < m.getCols(); ++col) {
            m(row, col) /= sum;
        }
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

NeuralNetworkApplier::ForwardPassResult NeuralNetworkApplier::forwardPass(const Matrix& input, const double dropoutRate) const {
    static std::mt19937 generator{std::random_device{}()};
    std::vector<Matrix> activations;
    std::vector<Matrix> dropoutMasks;
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

        Matrix dropoutMask = Matrix::ones(activation.getRows(), activation.getCols());
        if (dropoutRate > 0 && i + 1 < config.weights.size()) {
            std::bernoulli_distribution dropDistribution(1. - dropoutRate);
            for (std::size_t row = 0; row < activation.getRows(); ++row) {
                for (std::size_t col = 0; col < activation.getCols(); ++col) {
                    if (!dropDistribution(generator)) {
                        activation(row, col) = 0;
                        dropoutMask(row, col) = 0;
                    } else {
                        activation(row, col) /= (1. - dropoutRate);
                        dropoutMask(row, col) = 1;
                    }
                }
            }
        }

        // softmax hack
        if (i == config.weights.size() - 1 && config.outputActivation == ActivationType::Softmax) {
            ApplySoftMax(activation);
        }

        activations.push_back(activation);
        dropoutMasks.push_back(dropoutMask);
        currentActivation = activation;
    }

    return {
        activations,
        dropoutMasks,
    };
}

void NeuralNetworkApplier::backwardPass(
    const ForwardPassResult& forwardPassResult,
    const Matrix& error,
    const double learningRate,
    const double dropoutRate
) {
    const auto& activations = forwardPassResult.activations;
    const auto& dropoutMasks = forwardPassResult.dropoutMasks;

    if (activations.size() != config.weights.size() + 1) {
        throw std::invalid_argument("Invalid number of activations");
    }

    std::vector<Matrix> deltas(config.weights.size());
    deltas.back() = error;

    if (config.outputActivation != ActivationType::Softmax) {
        for (size_t row = 0; row < error.getRows(); ++row) {
            for (size_t col = 0; col < error.getCols(); ++col) {
                deltas.back()(row, col) *= applyActivationDerivative(
                    activations.back()(row, col),
                    config.outputActivation
                );
            }
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
        deltas[i] = hiddenError.hadamard(dropoutMasks[i]);
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

void NeuralNetworkApplier::train(const Matrix& input, const Matrix& target, const double learningRate, const double dropoutRate) {
    const auto result = forwardPass(input, dropoutRate);
    const auto error = result.activations.back() - target;
    backwardPass(result, error, learningRate, dropoutRate);
}

Matrix NeuralNetworkApplier::predict(const Matrix& input) const {
    if (!isInitialized()) {
        throw std::runtime_error("Network must be initalized before prediction");
    }
    auto result = forwardPass(input);
    return result.activations.back();
}

}
