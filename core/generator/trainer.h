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

    void setEpochCallback(std::function<void(std::size_t, double, double, double, double)> callback);
    void requestStop();

    const NeuralNetworkApplier& getGenerator() const;
    const NeuralNetworkApplier& getDiscriminator() const;

private:
    void trainDiscriminatorStep(
        const Matrix& realSample,
        const Matrix& fakeSample,
        double lr,
        double dropoutRate
    );

    double trainGeneratorStep(std::size_t label, const LearningConfig& config, double currentGeneratorLr);

    NeuralNetworkApplier generator;
    NeuralNetworkApplier discriminator;
    NeuralNetworkApplier classifier;

    std::function<void(std::size_t, double, double, double, double)> epochCallback;
    std::atomic<bool> stopFlag{false};

    std::mt19937 rng{std::random_device{}()};
};

}  // namespace GAN
}  // namespace Neural
