#include "core/generator/trainer.h"
#include "core/generator/learning_config.h"
#include "core/lib/loss_functions.h"
#include "core/lib/neural_network.h"
#include "core/lib/neural_network_applier.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>


namespace Neural {

namespace GAN {

Matrix SampleNoise(const std::size_t latentDim, std::mt19937& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    return Matrix(1, latentDim, [&](std::size_t, std::size_t) {
        return dist(rng);
    });
}

Matrix ConditionedInput(
    const Matrix& noise,
    const std::size_t label,
    const std::size_t latentDim,
    const std::size_t numClasses
) {
    return Matrix(1, latentDim + numClasses, [&](std::size_t, std::size_t col) -> double {
        if (col < latentDim)
            return noise(0, col);
        const std::size_t classIdx = col - latentDim;
        return (classIdx == label) ? 1.0 : 0.0;
    });
}

Matrix Generate(
    Neural::NeuralNetworkApplier& generator,
    const std::size_t numClasses,
    const std::size_t label,
    const std::size_t latentDim,
    std::mt19937& rng
) {
    return generator.forward(
        ConditionedInput(
            SampleNoise(latentDim, rng),
            label,
            latentDim,
            numClasses
        )
    );
}


GanTrainer::GanTrainer(
    NeuralNetworkApplier generator,
    NeuralNetworkApplier discriminator,
    NeuralNetworkApplier classifier
)
    : generator(std::move(generator))
    , discriminator(std::move(discriminator))
    , classifier(std::move(classifier))
{}

void GanTrainer::trainDiscriminatorStep(
    const Matrix& realSample,
    const Matrix& fakeSample,
    const double lr,
    const double dropoutRate
) {
    discriminator.zeroGradients();

    // Target is 0.9 to prevent discriminator from being overconfident
    const auto realScore = discriminator.forward(realSample, dropoutRate);
    discriminator.backward(bce_gradient(realScore, Matrix(1, 1, 0.9)));

    const auto fakeScore = discriminator.forward(fakeSample, dropoutRate);
    discriminator.backward(bce_gradient(fakeScore, Matrix(1, 1, 0.0)));

    discriminator.applyGradients(lr);
}

double GanTrainer::trainGeneratorStep(const std::size_t label, const LearningConfig& config, double currentGeneratorLr) {
    generator.zeroGradients();
    discriminator.zeroGradients();
    classifier.zeroGradients();

    const auto fake = Generate(generator, classifier.getNeuralNetworkConfig().outputSize(), label, config.latentDim, rng);
    const auto fakeScore = discriminator.forward(fake, 0.0);
    const auto inputGradFromD = discriminator.backward(bce_gradient(fakeScore, Matrix(1, 1, 1.0)));
    const auto classifierOut = classifier.forward(fake, 0.0);

    Matrix oneHot(1, classifierOut.getCols(), 0.0);
    oneHot(0, label) = 1.0;

    // assuming classifier's output layers is softmax, therefore use (classifier - oneHot) as gradient
    const auto inputGradFromC = classifier.backward(classifierOut - oneHot);
    const auto combinedGrad = inputGradFromD + inputGradFromC * config.classifierLossWeight;

    generator.backward(combinedGrad);
    generator.applyGradients(currentGeneratorLr);

    return fakeScore(0, 0);
}

void GanTrainer::train(
    const std::vector<Matrix>& realSamples,
    const LearningConfig& config,
    std::ostream* log
) {
    if (realSamples.empty()) return;
    stopFlag.store(false, std::memory_order_relaxed);

    const auto numberOfClasses = classifier.getNeuralNetworkConfig().outputSize();
    std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<std::size_t> labelDist(0, numberOfClasses - 1);
    std::vector<std::size_t> indices(realSamples.size());
    std::iota(indices.begin(), indices.end(), 0);

    double currentDLr = config.discriminatorLearningRate;
    double currentGLr = config.generatorLearningRate;
    double emaReal = -1.0;
    double emaGen = -1.0;
    double prevEmaReal = -1.0;
    double prevEmaGen = -1.0;

    std::size_t flatEpochCount = 0;
    std::size_t kickEpochsRemaining = 0;
    double kickSavedGLr = 0.0;

    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        double totalDiscScore = 0.0;
        double totalGenScore = 0.0;
        std::size_t steps = 0;

        const double effectiveDropout = (kickEpochsRemaining > 0)
            ? std::min(config.dropoutRate * config.flatnessDetection.discriminatorDropoutBoost, 0.95)
            : config.dropoutRate;

        for (std::size_t i = 0; i + config.batchSize <= indices.size(); i += config.batchSize) {
            const auto label = labelDist(rng);
            for (std::size_t d = 0; d < config.discriminatorStepsPerGenStep; ++d) {
                const auto idx = indices[i + (d % config.batchSize)];
                const auto fake = Generate(generator, numberOfClasses, label, config.latentDim, rng);
                trainDiscriminatorStep(
                    realSamples[idx], fake,
                    currentDLr, effectiveDropout
                );
                totalDiscScore += discriminator.predict(realSamples[idx])(0, 0);
            }

            totalGenScore += trainGeneratorStep(label, config, currentGLr);
            ++steps;

            if (stopFlag.load(std::memory_order_relaxed)) {
                break;
            }
        }

        const double avgDiscReal = steps > 0 ? totalDiscScore / (steps * config.discriminatorStepsPerGenStep) : 0.0;
        const double avgGenFool  = steps > 0 ? totalGenScore / steps : 0.0;

        prevEmaReal = emaReal;
        prevEmaGen  = emaGen;
        if (emaReal < 0.0) {
            emaReal = avgDiscReal;
            emaGen  = avgGenFool;
        } else {
            emaReal = config.adaptiveLr.lrEmaAlpha * emaReal + (1.0 - config.adaptiveLr.lrEmaAlpha) * avgDiscReal;
            emaGen  = config.adaptiveLr.lrEmaAlpha * emaGen  + (1.0 - config.adaptiveLr.lrEmaAlpha) * avgGenFool;
        }

        if (config.flatnessDetection.enabled && epoch >= config.adaptiveLr.lrWarmupEpochs
                && prevEmaReal >= 0.0 && kickEpochsRemaining == 0) {
            const bool emaRealFlat = std::abs(emaReal - prevEmaReal) < config.flatnessDetection.threshold;
            const bool emaGenFlat  = std::abs(emaGen  - prevEmaGen)  < config.flatnessDetection.threshold;
            if (emaRealFlat && emaGenFlat) {
                ++flatEpochCount;
                if (flatEpochCount >= config.flatnessDetection.window) {
                    kickSavedGLr = currentGLr;
                    currentGLr = std::clamp(currentGLr * config.flatnessDetection.generatorLrBoost, config.adaptiveLr.lrMin, config.adaptiveLr.lrMax);
                    kickEpochsRemaining = config.flatnessDetection.kickDuration;
                    flatEpochCount = 0;
                }
            } else {
                flatEpochCount = 0;
            }
        }

        if (config.adaptiveLr.enabled && epoch >= config.adaptiveLr.lrWarmupEpochs && kickEpochsRemaining == 0) {
            if (emaReal < config.adaptiveLr.dRealTargetLow && emaGen < config.adaptiveLr.dFakeTargetLow) {
                // D collapsed
                currentDLr = std::clamp(currentDLr * config.adaptiveLr.lrAdjustFactor, config.adaptiveLr.lrMin, config.adaptiveLr.lrMax);
                currentGLr = std::clamp(currentGLr / config.adaptiveLr.lrAdjustFactor, config.adaptiveLr.lrMin, config.adaptiveLr.lrMax);
            } else if (emaReal > config.adaptiveLr.dRealTargetHigh && emaGen < config.adaptiveLr.dFakeTargetLow) {
                // D dominating
                currentDLr = std::clamp(currentDLr / config.adaptiveLr.lrAdjustFactor, config.adaptiveLr.lrMin, config.adaptiveLr.lrMax);
                currentGLr = std::clamp(currentGLr * config.adaptiveLr.lrAdjustFactor, config.adaptiveLr.lrMin, config.adaptiveLr.lrMax);
            } else if (emaGen > config.adaptiveLr.dFakeTargetHigh) {
                // G dominating
                currentGLr = std::clamp(currentGLr / config.adaptiveLr.lrAdjustFactor, config.adaptiveLr.lrMin, config.adaptiveLr.lrMax);
                currentDLr = std::clamp(currentDLr * config.adaptiveLr.lrAdjustFactor, config.adaptiveLr.lrMin, config.adaptiveLr.lrMax);
            }
        }

        if (kickEpochsRemaining > 0) {
            --kickEpochsRemaining;
            if (kickEpochsRemaining == 0) {
                currentGLr = kickSavedGLr;
            }
        }

        if (log && steps > 0) {
            const std::size_t logLabel = epoch % numberOfClasses;
            const auto sample = Generate(generator, numberOfClasses, logLabel, config.latentDim, rng);
            const auto classifierOut = classifier.predict(sample);
            std::size_t predictedLabel = 0;
            for (std::size_t c = 1; c < classifierOut.getCols(); ++c) {
                if (classifierOut(0, c) > classifierOut(0, predictedLabel)) {
                    predictedLabel = c;
                }
            }

            *log << "Epoch " << epoch + 1 << "/" << config.epochs
                 << "  D(real)=" << avgDiscReal
                 << "  D(G(z))=" << avgGenFool;
            if (config.adaptiveLr.enabled || config.flatnessDetection.enabled) {
                *log << "  ema_D(real)=" << emaReal
                     << "  ema_D(G(z))=" << emaGen
                     << "  dLr=" << currentDLr
                     << "  gLr=" << currentGLr;
            }
            if (config.flatnessDetection.enabled) {
                *log << "  flat=" << flatEpochCount << "/" << config.flatnessDetection.window
                     << (kickEpochsRemaining > 0 ? "  [KICK]" : "");
            }
            *log << "  sample(label=" << logLabel << " -> classifier=" << predictedLabel << ")"
                 << std::endl;
        }

        if (epochCallback) {
            epochCallback(epoch, avgDiscReal, avgGenFool, emaReal, emaGen);
        }
        if (stopFlag.load(std::memory_order_relaxed)) {
            break;
        }
    }
}

void GanTrainer::setEpochCallback(std::function<void(std::size_t, double, double, double, double)> callback) {
    epochCallback = std::move(callback);
}

void GanTrainer::requestStop() {
    stopFlag.store(true, std::memory_order_relaxed);
}

const NeuralNetworkApplier& GanTrainer::getGenerator() const { return generator; }
const NeuralNetworkApplier& GanTrainer::getDiscriminator() const { return discriminator; }

} // namespace GAN
} // namespace Neural
