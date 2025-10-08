#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "matrix.h"
#include <vector>
#include <random>

class NeuralNetwork {
private:
    bool initialized = false;
    std::vector<size_t> layerSizes;
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;

    static double sigmoid(double x);
    static double sigmoidDerivative(double x);

    std::vector<Matrix> forwardPass(const Matrix& input) const;
    void backwardPass(
        const Matrix& input,
        const Matrix& target,
        const std::vector<Matrix>& activations,
        const double learningRate
    );
public:
    NeuralNetwork() = default;
    NeuralNetwork(const std::vector<size_t>& sizes);

    void initializeNetwork(const std::vector<size_t>& layerSizes);
    bool isInitialized() const;

    void initializeWeights(std::mt19937& gen);
    void train(const Matrix& inputs, const Matrix& targets, const int epochs, const double learningRate, std::ostream& logger = std::cout);
    Matrix predict(const Matrix& input) const;

    void saveWeights(const std::string& filename) const;
    void loadWeights(const std::string& filename);
};

#endif // NEURAL_NETWORK_H
