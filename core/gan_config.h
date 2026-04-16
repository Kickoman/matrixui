#pragma once

#include <cstddef>

namespace Neural {

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
};

}
