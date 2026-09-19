#pragma once

#include "core/nn/neural_network.h"

namespace Neural {

Matrix Predict(const NeuralNetwork& network, const Matrix& input);

class NeuralNetworkApplier {
public:
    NeuralNetworkApplier() = default;
    explicit NeuralNetworkApplier(const NeuralNetwork& network);
    explicit NeuralNetworkApplier(NeuralNetwork&& network);

    void initializeNetwork(const NeuralNetwork& network);
    void initializeNetwork(NeuralNetwork&& network);

    bool isInitialized() const;
    const NeuralNetwork& getNeuralNetworkConfig() const;

    Matrix train(const Matrix& input, const Matrix& target, double learningRate, double dropoutRate = 0.);
    Matrix predict(const Matrix& input) const;

    Matrix forward(const Matrix& input, double dropoutRate = 0.0);
    Matrix backward(const Matrix& lossGradient);
    void applyGradients(double learningRate);
    void zeroGradients();

private:
    bool initialized = false;
    NeuralNetwork config;

    std::vector<Matrix> activations;
};

}
