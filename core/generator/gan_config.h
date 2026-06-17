#pragma once

#include <cstddef>
#include <nlohmann/json.hpp>

namespace Neural {
namespace GAN {

struct GanConfig {
    std::size_t latentDim = 100;
    std::size_t numClasses = 10;
    double generatorLr = 0.0002;
    double discriminatorLr = 0.0001;  // Lower than G: D converges faster
    std::size_t epochs = 300;
    std::size_t batchSize = 64;
    // How many discriminator update steps to run per generator update step.
    std::size_t discriminatorStepsPerGenStep = 3;
    double dropoutRate = 0.3;  // Regularizes D; prevents memorising real samples
    // Weight of the classifier loss relative to the discriminator loss in the
    // generator update. Higher values push the generator harder toward the
    // requested digit class.
    double classifierLossWeight = 1.0;
    // Maximum number of images to load per label (0 = no limit).
    std::size_t datasetLimitPerLabel = 0;

    // Adaptive learning rate: dynamically adjusts dLr and gLr each epoch based
    // on EMA-smoothed D(real) and D(G(z)) to counteract discriminator or
    // generator domination. Off by default — existing behaviour unchanged.
    bool        adaptiveLr         = false;
    double      lrEmaAlpha         = 0.9;   // EMA decay: higher = slower reaction
    double      dRealTargetLow     = 0.30;  // D(real) below this (with genFoolTargetLow) → D collapsed
    double      dRealTargetHigh    = 0.80;  // D(real) above this → D dominating
    double      genFoolTargetLow   = 0.30;  // D(G(z)) below this (combined) → D dominating or D collapsed
    double      genFoolTargetHigh  = 0.60;  // D(G(z)) above this → G dominating
    double      lrAdjustFactor     = 1.05;  // Multiplicative step per epoch (~5%)
    double      lrMin              = 1e-6;
    double      lrMax              = 1e-2;
    std::size_t lrWarmupEpochs     = 5;     // Skip adjustments for first N epochs

    // Flatness detection: when both EMA signals are barely moving for
    // flatnessWindow consecutive epochs, apply a kick to escape the plateau.
    // Kick = raise D dropout + spike G lr for flatnessKickDuration epochs,
    // then restore. Adaptive lr is suspended during the kick. Off by default.
    bool        flatnessDetection    = false;
    double      flatnessThreshold    = 0.005; // Max |ema_delta| per epoch to count as flat
    std::size_t flatnessWindow       = 10;    // Consecutive flat epochs before kick fires
    std::size_t flatnessKickDuration = 5;     // Epochs to hold the kick
    double      flatnessDropoutBoost = 2.0;   // Multiply D dropoutRate by this during kick
    double      flatnessGenLrBoost   = 3.0;   // Multiply G lr by this during kick
};

} // namespace GAN
} // namespace Neural


NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    Neural::GAN::GanConfig,
    latentDim,
    numClasses,
    generatorLr,
    discriminatorLr,
    epochs,
    batchSize,
    discriminatorStepsPerGenStep,
    dropoutRate,
    classifierLossWeight,
    datasetLimitPerLabel,
    adaptiveLr,
    lrEmaAlpha,
    dRealTargetLow,
    dRealTargetHigh,
    genFoolTargetLow,
    genFoolTargetHigh,
    lrAdjustFactor,
    lrMin,
    lrMax,
    lrWarmupEpochs,
    flatnessDetection,
    flatnessThreshold,
    flatnessWindow,
    flatnessKickDuration,
    flatnessDropoutBoost,
    flatnessGenLrBoost
);
