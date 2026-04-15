#pragma once

#include <cstddef>

namespace Neural {

struct GanConfig {
    std::size_t latentDim = 100;
    std::size_t numClasses = 10;
    double generatorLr = 0.0002;
    double discriminatorLr = 0.0002;
    std::size_t epochs = 100;
    std::size_t batchSize = 32;
    // How many discriminator update steps to run per generator update step.
    std::size_t discriminatorStepsPerGenStep = 1;
    double dropoutRate = 0.0;
    // Weight of the classifier loss relative to the discriminator loss in the
    // generator update. Higher values push the generator harder toward the
    // requested digit class.
    double classifierLossWeight = 1.0;
    // Maximum number of images to load per label (0 = no limit).
    std::size_t datasetLimitPerLabel = 0;
};

}
