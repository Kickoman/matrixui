#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "neural_network.h"
#include <vector>

namespace Neural {


class NeuralNetworkApplier {
public:
    NeuralNetworkApplier() = default;
    explicit NeuralNetworkApplier(const NeuralNetwork& config);
    explicit NeuralNetworkApplier(NeuralNetwork&& config);

    void initializeNetwork(const NeuralNetwork& config);
    void initializeNetwork(NeuralNetwork&& config);

    bool isInitialized() const;
    const NeuralNetwork& getNeuralNetworkConfig() const;

    void train(const Matrix& input, const Matrix& target, const double learningRate, const double dropoutRate = 0.);
    Matrix predict(const Matrix& input) const;

private:
    struct ForwardPassResult {
        std::vector<Matrix> activations;
        std::vector<Matrix> dropoutMasks;
    };

    static double applyActivation(const double x, const ActivationType type);
    static double applyActivationDerivative(const double x, const ActivationType type);

    ForwardPassResult forwardPass(const Matrix& input, const double dropoutRate = 0.0) const;
    void backwardPass(const ForwardPassResult& forwardPassResult, const Matrix& error, const double learningRate, const double dropoutRate);

    bool initialized = false;
    NeuralNetwork config;
};

}

#endif // NEURAL_NETWORK_H
