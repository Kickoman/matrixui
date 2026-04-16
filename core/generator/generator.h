#pragma once

#include "core/lib/neural_network_applier.h"
#include <random>

namespace Neural {
namespace GAN {

class Generator {
public:
    Generator(NeuralNetwork network, std::size_t latentDim, std::size_t numClasses);

    // Generate a 1 x latentDim matrix of standard-normal noise.
    Matrix sampleNoise() const;

    // Build the full conditioned input: concatenate noise with the one-hot
    // encoding of label, giving a 1 x (latentDim + numClasses) matrix.
    Matrix conditionedInput(const Matrix& noise, std::size_t label) const;

    // Forward pass conditioned on a digit label.
    Matrix generate(const Matrix& noise, std::size_t label);

    // Convenience: sample noise then generate for the given label.
    Matrix generate(std::size_t label);

    // Backward pass. lossGradient is the gradient w.r.t. the generator's output
    // (the combined gradient from discriminator + classifier).
    // Returns the gradient w.r.t. the generator's input, which is unused.
    Matrix backward(const Matrix& lossGradient);

    void applyGradients(double lr);
    void zeroGradients();

    std::size_t getLatentDim() const;
    std::size_t getNumClasses() const;
    const NeuralNetwork& getNetwork() const;

private:
    NeuralNetworkApplier applier;
    std::size_t latentDim;
    std::size_t numClasses;
    mutable std::mt19937 rng{std::random_device{}()};
};

} // namespace GAN
} // namespace Neural
