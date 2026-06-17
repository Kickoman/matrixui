#pragma once

#include <cstddef>
#include <nlohmann/json.hpp>

namespace Neural {
namespace GAN {

/*
    TIPS

    Keep **discriminator** learning rate lower than **generator** learning rate,
    because discriminator converges faster

*/

// Adaptive learning rate: dynamically adjusts dLr and gLr each epoch based
// on EMA-smoothed D(real) and D(G(z)) to counteract discriminator or
// generator domination. Off by default — existing behaviour unchanged.
struct AdaptiveLrConfig {
    bool enabled = false;
    double lrEmaAlpha = 0.9;
    double dRealTargetLow = 0.30;
    double dRealTargetHigh = 0.80;
    double dFakeTargetLow = 0.30;
    double dFakeTargetHigh = 0.60;
    double lrAdjustFactor = 1.05;
    double lrMin = 1e-6;
    double lrMax = 1e-2;
    std::size_t lrWarmupEpochs = 5;
};

// Flatness detection: when both EMA signals are barely moving for
// flatnessWindow consecutive epochs, apply a kick to escape the plateau.
// Kick = raise D dropout + spike G lr for flatnessKickDuration epochs,
// then restore. Adaptive lr is suspended during the kick. Off by default.
struct FlatnessDetectionConfig {
    bool enabled = false;
    double threshold = 0.005; // Max |ema_delta| per epoch to count as flat
    std::size_t window = 10;    // Consecutive flat epochs before kick fires
    std::size_t kickDuration = 5;     // Epochs to hold the kick
    double discriminatorDropoutBoost = 2.0;   // Multiply D dropoutRate by this during kick
    double generatorLrBoost = 3.0;   // Multiply G lr by this during kick
};

struct LearningConfig {
    std::size_t latentDim = 100;
    double generatorLearningRate = 0.0002;
    double discriminatorLearningRate = 0.0001;
    std::size_t epochs = 300;
    std::size_t batchSize = 64;
    std::size_t discriminatorStepsPerGenStep = 3;
    double dropoutRate = 0.3;
    double classifierLossWeight = 1.0;
    std::size_t datasetLimitPerLabel = 0;

    AdaptiveLrConfig adaptiveLr;
    FlatnessDetectionConfig flatnessDetection;
};


NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    AdaptiveLrConfig,
    enabled,
    lrEmaAlpha,
    dRealTargetLow,
    dRealTargetHigh,
    dFakeTargetLow,
    dFakeTargetHigh,
    lrAdjustFactor,
    lrMin,
    lrMax,
    lrWarmupEpochs
);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    FlatnessDetectionConfig,
    enabled,
    threshold,
    window,
    kickDuration,
    discriminatorDropoutBoost,
    generatorLrBoost
);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    LearningConfig,
    latentDim,
    generatorLearningRate,
    discriminatorLearningRate,
    epochs,
    batchSize,
    discriminatorStepsPerGenStep,
    dropoutRate,
    classifierLossWeight,
    datasetLimitPerLabel,
    adaptiveLr,
    flatnessDetection
);

} // namespace GAN
} // namespace Neural
