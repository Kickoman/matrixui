#pragma once

#include "generator.h"
#include "discriminator.h"
#include "neural_network_applier.h"
#include "gan_config.h"
#include "../matrix/matrix.h"
#include <vector>
#include <ostream>

namespace Neural {

class GanTrainer {
public:
    // classifier is copied — its weights are never updated (applyGradients is
    // never called on it), but forward/backward are called to get the gradient
    // of the classification loss w.r.t. the generated image.
    GanTrainer(
        Generator generator,
        Discriminator discriminator,
        NeuralNetworkApplier classifier
    );

    void train(
        const std::vector<Matrix>& realSamples,
        const GanConfig& config,
        std::ostream* log = nullptr
    );

    const Generator& getGenerator() const;
    const Discriminator& getDiscriminator() const;

private:
    // Train discriminator to score realSample as 1 and fakeSample as 0.
    // Accumulates both real and fake gradients before one applyGradients call.
    void trainDiscriminatorStep(
        const Matrix& realSample,
        const Matrix& fakeSample,
        double lr,
        double dropoutRate
    );

    // Train generator conditioned on label.
    // Loss = discriminator loss (fool D) + classifierLossWeight * classifier loss (produce label).
    // Discriminator and classifier weights are NOT updated in this step.
    // Returns the discriminator score on the generated sample (for logging).
    double trainGeneratorStep(std::size_t label, double classifierLossWeight, double lr);

    Generator generator;
    Discriminator discriminator;
    NeuralNetworkApplier classifier;
};

}
