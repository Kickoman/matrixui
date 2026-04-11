#pragma once

#include "neural_network_applier.h"

namespace Neural {

// Discriminator: maps an image (784-dim) to a single real/fake score in [0, 1].
// The network must be configured with a Sigmoid output activation.
class Discriminator {
public:
    explicit Discriminator(NeuralNetwork network);

    // Inference-only forward pass (no caching, no dropout). Returns a 1x1 matrix.
    Matrix score(const Matrix& input) const;

    // Training forward pass (caches activations for backward). Returns a 1x1 matrix.
    Matrix forward(const Matrix& input, double dropoutRate = 0.0);

    // Backward pass. Returns gradient w.r.t. input (used to propagate into Generator).
    // Accumulates weight gradients but does NOT apply them — call applyGradients separately.
    Matrix backward(const Matrix& lossGradient);

    void applyGradients(double lr);
    void zeroGradients();

    bool isInitialized() const;
    const NeuralNetwork& getNetwork() const;

private:
    NeuralNetworkApplier applier;
};

}
