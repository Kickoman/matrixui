#pragma once

#include "neural_network.h"

namespace Neural {

class NeuralNetworkApplier {
public:
    NeuralNetworkApplier() = default;
    explicit NeuralNetworkApplier(const NeuralNetwork& network);
    explicit NeuralNetworkApplier(NeuralNetwork&& network);

    void initializeNetwork(const NeuralNetwork& network);
    void initializeNetwork(NeuralNetwork&& network);

    bool isInitialized() const;
    const NeuralNetwork& getNeuralNetworkConfig() const;

    void train(const Matrix& input, const Matrix& target, double learningRate, double dropoutRate = 0.);
    Matrix predict(const Matrix& input) const;

private:
    bool initialized = false;
    NeuralNetwork config;
};

}
