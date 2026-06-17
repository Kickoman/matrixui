#include "core/generator/gan_trainer.h"
#include "core/generator/gan_config.h"
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

    // Real sample: target is 0.9 (one-sided label smoothing — prevents D from being overconfident).
    const Matrix realScore = discriminator.forward(realSample, dropoutRate);
    discriminator.backward(bce_gradient(realScore, Matrix(1, 1, 0.9)));

    // Fake sample: target is 0 (fake).
    // Gradients accumulate on top of the real-sample pass above.
    const Matrix fakeScore = discriminator.forward(fakeSample, dropoutRate);
    discriminator.backward(bce_gradient(fakeScore, Matrix(1, 1, 0.0)));

    discriminator.applyGradients(lr);
}

double GanTrainer::trainGeneratorStep(const std::size_t label, const LearningConfig& config, double currentGeneratorLr) {
    generator.zeroGradients();
    // Zero D and C gradients so their backward passes give clean input-space
    // gradients. Their accumulated weight gradients are discarded: D's are
    // overwritten at the start of the next discriminator step; C's are never
    // applied at all.
    discriminator.zeroGradients();
    classifier.zeroGradients();

    const auto fake = Generate(generator, classifier.getNeuralNetworkConfig().outputSize(), label, config.latentDim, rng);

    // --- Discriminator loss: fool D into scoring fake as real ---
    const Matrix fakeScore = discriminator.forward(fake, 0.0);
    const Matrix inputGradFromD = discriminator.backward(bce_gradient(fakeScore, Matrix(1, 1, 1.0)));

    // --- Classifier loss: fake should be classified as 'label' ---
    const Matrix classifierOut = classifier.forward(fake, 0.0);
    // Build one-hot target for the requested label.
    Matrix oneHot(1, classifierOut.getCols(), 0.0);
    oneHot(0, label) = 1.0;
    // SoftmaxLayer::backward is identity (assumes CE loss upstream), so
    // (classifierOut - oneHot) is the correct gradient at the classifier input.
    const Matrix inputGradFromC = classifier.backward(classifierOut - oneHot);

    // Combine: both gradients point toward improving the generator.
    const Matrix combinedGrad = inputGradFromD + inputGradFromC * config.classifierLossWeight;

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

    double currentDLr = config.discriminatorLr;
    double currentGLr = config.generatorLr;
    // Uninitialized sentinel: first epoch sets EMA to the raw value directly,
    // avoiding the bias that comes from starting at an arbitrary 0.5.
    double emaReal = -1.0, emaGen = -1.0;
    double prevEmaReal = -1.0, prevEmaGen = -1.0;
    std::size_t flatEpochCount = 0;
    std::size_t kickEpochsRemaining = 0;
    double kickSavedGLr = 0.0;

    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        double totalDiscScore = 0.0;
        double totalGenScore = 0.0;
        std::size_t steps = 0;

        // During a kick, D dropout is raised to weaken it and give G room.
        const double effectiveDropout = (kickEpochsRemaining > 0)
            ? std::min(config.dropoutRate * config.flatnessDropoutBoost, 0.95)
            : config.dropoutRate;

        for (std::size_t i = 0; i + config.batchSize <= indices.size(); i += config.batchSize) {
            // Pick the label for this step — used for both fake generation in D
            // training and for the conditioned G update.
            const std::size_t label = labelDist(rng);

            // Train discriminator for discriminatorStepsPerGenStep steps.
            for (std::size_t d = 0; d < config.discriminatorStepsPerGenStep; ++d) {
                const std::size_t idx = indices[i + (d % config.batchSize)];
                const Matrix fake = Generate(generator, numberOfClasses, label, config.latentDim, rng);
                trainDiscriminatorStep(
                    realSamples[idx], fake,
                    currentDLr, effectiveDropout
                );
                totalDiscScore += discriminator.predict(realSamples[idx])(0, 0);
            }

            // Train generator one step conditioned on the chosen label.
            totalGenScore += trainGeneratorStep(label, config, currentGLr);
            ++steps;

            if (stopFlag.load(std::memory_order_relaxed)) break;
        }

        const double avgDiscReal = steps > 0 ? totalDiscScore / (steps * config.discriminatorStepsPerGenStep) : 0.0;
        const double avgGenFool  = steps > 0 ? totalGenScore / steps : 0.0;

        // Update EMA — always, so GUI always gets smoothed curves.
        // First epoch: seed from actual values to avoid bias toward the 0.5 init.
        prevEmaReal = emaReal;
        prevEmaGen  = emaGen;
        if (emaReal < 0.0) {
            emaReal = avgDiscReal;
            emaGen  = avgGenFool;
        } else {
            emaReal = config.lrEmaAlpha * emaReal + (1.0 - config.lrEmaAlpha) * avgDiscReal;
            emaGen  = config.lrEmaAlpha * emaGen  + (1.0 - config.lrEmaAlpha) * avgGenFool;
        }

        // Flatness detection — fires only when no kick is already active.
        if (config.flatnessDetection && epoch >= config.lrWarmupEpochs
                && prevEmaReal >= 0.0 && kickEpochsRemaining == 0) {
            const bool emaRealFlat = std::abs(emaReal - prevEmaReal) < config.flatnessThreshold;
            const bool emaGenFlat  = std::abs(emaGen  - prevEmaGen)  < config.flatnessThreshold;
            if (emaRealFlat && emaGenFlat) {
                ++flatEpochCount;
                if (flatEpochCount >= config.flatnessWindow) {
                    // Fire kick: raise D dropout (via effectiveDropout next epoch)
                    // and spike G lr. Adaptive lr is suspended for the duration.
                    kickSavedGLr = currentGLr;
                    currentGLr = std::clamp(
                        currentGLr * config.flatnessGenLrBoost, config.lrMin, config.lrMax);
                    kickEpochsRemaining = config.flatnessKickDuration;
                    flatEpochCount = 0;
                }
            } else {
                flatEpochCount = 0;
            }
        }

        // Adaptive lr adjustment — suspended during a kick to avoid fighting it.
        if (config.adaptiveLr && epoch >= config.lrWarmupEpochs && kickEpochsRemaining == 0) {
            if (emaReal < config.dRealTargetLow && emaGen < config.genFoolTargetLow) {
                // D collapsed: scoring everything near 0. Boost D, slow G.
                currentDLr = std::clamp(currentDLr * config.lrAdjustFactor, config.lrMin, config.lrMax);
                currentGLr = std::clamp(currentGLr / config.lrAdjustFactor, config.lrMin, config.lrMax);
            } else if (emaReal > config.dRealTargetHigh && emaGen < config.genFoolTargetLow) {
                // D dominating: slow D, speed G.
                currentDLr = std::clamp(currentDLr / config.lrAdjustFactor, config.lrMin, config.lrMax);
                currentGLr = std::clamp(currentGLr * config.lrAdjustFactor, config.lrMin, config.lrMax);
            } else if (emaGen > config.genFoolTargetHigh) {
                // G dominating: slow G, speed D.
                currentGLr = std::clamp(currentGLr / config.lrAdjustFactor, config.lrMin, config.lrMax);
                currentDLr = std::clamp(currentDLr * config.lrAdjustFactor, config.lrMin, config.lrMax);
            }
        }

        // Decrement kick counter; restore G lr when it expires.
        if (kickEpochsRemaining > 0) {
            --kickEpochsRemaining;
            if (kickEpochsRemaining == 0)
                currentGLr = kickSavedGLr;
        }

        if (log && steps > 0) {
            // Sample a generated image for each class and log what the classifier thinks.
            const std::size_t logLabel = epoch % numberOfClasses;
            const Matrix sample = Generate(generator, numberOfClasses, logLabel, config.latentDim, rng);
            const Matrix classifierOut = classifier.predict(sample);
            std::size_t predictedLabel = 0;
            for (std::size_t c = 1; c < classifierOut.getCols(); ++c)
                if (classifierOut(0, c) > classifierOut(0, predictedLabel))
                    predictedLabel = c;

            *log << "Epoch " << epoch + 1 << "/" << config.epochs
                 << "  D(real)=" << avgDiscReal
                 << "  D(G(z))=" << avgGenFool;
            if (config.adaptiveLr || config.flatnessDetection)
                *log << "  ema_D(real)=" << emaReal
                     << "  ema_D(G(z))=" << emaGen
                     << "  dLr=" << currentDLr
                     << "  gLr=" << currentGLr;
            if (config.flatnessDetection)
                *log << "  flat=" << flatEpochCount << "/" << config.flatnessWindow
                     << (kickEpochsRemaining > 0 ? "  [KICK]" : "");
            *log << "  sample(label=" << logLabel << " -> classifier=" << predictedLabel << ")"
                 << std::endl;
        }

        if (epochCallback) epochCallback(epoch, avgDiscReal, avgGenFool, emaReal, emaGen);
        if (stopFlag.load(std::memory_order_relaxed)) break;
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
