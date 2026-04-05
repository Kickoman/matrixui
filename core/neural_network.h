#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "matrix.h"
#include <cstdint>
#include <vector>

namespace Neural {

enum class ActivationType : std::uint8_t {
    Sigmoid,
    ReLU,
    Tanh,
    Softmax,
};

struct NeuralNetwork {
    ActivationType hiddenActivation = ActivationType::ReLU;
    ActivationType outputActivation = ActivationType::Sigmoid;
    std::vector<std::size_t> layersSizes;
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;

    void initializeWeights();
    void initializeBiases();
};

class NeuralNetworkApplier {
public:
    NeuralNetworkApplier() = default;
    explicit NeuralNetworkApplier(const NeuralNetwork& config);
    explicit NeuralNetworkApplier(NeuralNetwork&& config);

    void initializeNetwork(const NeuralNetwork& config);
    void initializeNetwork(NeuralNetwork&& config);

    bool isInitialized() const;
    const NeuralNetwork& getNeuralNetworkConfig() const;

    void train(const Matrix& input, const Matrix& target, const double learningRate);
    Matrix predict(const Matrix& input) const;

private:

    static double applyActivation(const double x, const ActivationType type);
    static double applyActivationDerivative(const double x, const ActivationType type);

    std::vector<Matrix> forwardPass(const Matrix& input) const;
    void backwardPass(const std::vector<Matrix>& activations, const Matrix& error, const double learningRate);

    bool initialized = false;
    NeuralNetwork config;
};

}

#endif // NEURAL_NETWORK_H
