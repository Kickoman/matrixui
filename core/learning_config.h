#pragma once

#include <cstddef>


namespace Neural {

struct LearningConfig {
    double initialLearningRate = 0.5;
    double minLearningRate = 0.001;
    double learningRateDecay = 0.5;
    std::size_t sampleBatchSize = 32;
    std::size_t maxEpochs = 200;
    std::size_t patience = 10;
    std::size_t innerEpochs = 5;

    std::size_t datasetLimitPerLabel = 100;
};

}
