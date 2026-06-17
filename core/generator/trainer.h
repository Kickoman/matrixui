#pragma once

#include "core/generator/learning_config.h"

#include "core/lib/neural_network_applier.h"
#include "matrix/matrix.h"

#include <random>
#include <vector>
#include <ostream>
#include <functional>
#include <atomic>

namespace Neural {

namespace GAN {

inline std::size_t inferLatentDim(const std::vector<std::size_t>& layerSizes, std::size_t numClasses) {
    const std::size_t inputSize = layerSizes.front();
    return inputSize > numClasses ? inputSize - numClasses : inputSize;
}

Matrix Generate(
    Neural::NeuralNetworkApplier& generator,
    const std::size_t numClasses,
    const std::size_t label,
    const std::size_t latentDim,
    std::mt19937& rng
);

class GanTrainer {
public:
    // classifier is copied — its weights are never updated (applyGradients is
    // never called on it), but forward/backward are called to get the gradient
    // of the classification loss w.r.t. the generated image.
    GanTrainer(
        NeuralNetworkApplier generator,
        NeuralNetworkApplier discriminator,
        NeuralNetworkApplier classifier
    );

    void train(
        const std::vector<Matrix>& realSamples,
        const LearningConfig& config,
        std::ostream* log = nullptr
    );

    // Called at the end of each epoch with:
    //   (epoch, avgDiscReal, avgGenFool, emaReal, emaGen)
    // emaReal/emaGen are EMA-smoothed versions of avgDiscReal/avgGenFool.
    // Always emitted; used by adaptive lr internally and exposed for plotting.
    void setEpochCallback(std::function<void(std::size_t, double, double, double, double)> callback);

    // Signal training to stop after the current epoch completes.
    void requestStop();

    const NeuralNetworkApplier& getGenerator() const;
    const NeuralNetworkApplier& getDiscriminator() const;

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
    double trainGeneratorStep(std::size_t label, const LearningConfig& config, double currentGeneratorLr);

    NeuralNetworkApplier generator;
    NeuralNetworkApplier discriminator;
    NeuralNetworkApplier classifier;

    std::function<void(std::size_t, double, double, double, double)> epochCallback;
    std::atomic<bool> stopFlag{false};

    std::mt19937 rng{std::random_device{}()};
};

} // GAN
} // Neural
