#pragma once

#include <cstddef>


namespace Neural {

struct LearningConfig {
    double initialLearningRate = 0.2;
    double minLearningRate = 0.0001;
    double learningRateDecay = 0.7;
    std::size_t sampleBatchSize = 32;
    std::size_t maxEpochs = 500;
    std::size_t patience = 10;
    std::size_t innerEpochs = 1;

    std::size_t datasetLimitPerLabel = 100;
};

}
