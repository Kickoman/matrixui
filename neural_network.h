#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "matrix.h"
#include <vector>
#include <random>


class NeuralNetwork {
public:
    NeuralNetwork() = default;
    NeuralNetwork(const std::vector<size_t>& sizes);

    void initializeNetwork(const std::vector<size_t>& layerSizes);
    bool isInitialized() const;
    void initializeWeights(std::mt19937& gen);

    void setWeights(const std::vector<Matrix>& weights);
    void setBiases(const std::vector<Matrix>& biases);

    const std::vector<std::size_t>& getLayerSizes() const;
    const std::vector<Matrix>& getWeights() const;
    const std::vector<Matrix>& getBiases() const;

    void train(const Matrix& input, const Matrix& target, const double learningRate);
    Matrix predict(const Matrix& input) const;

private:
    static double sigmoid(double x);
    static double sigmoidDerivative(double x);

    std::vector<Matrix> forwardPass(const Matrix& input) const;
    void backwardPass(const std::vector<Matrix>& activations, const Matrix& error, const double learningRate);

    bool initialized = false;
    std::vector<size_t> layerSizes;
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;
};

#endif // NEURAL_NETWORK_H
