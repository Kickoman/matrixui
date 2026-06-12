#pragma once

#include <cstddef>
#include <nlohmann/json.hpp>


namespace Neural {
namespace Classifier {

struct LearningConfig {
    double initialLearningRate = 0.2;
    double minLearningRate = 0.0001;
    double learningRateDecay = 0.7;
    std::size_t maxEpochs = 500;
    std::size_t patience = 10;
    std::size_t innerEpochs = 1;

    std::size_t datasetLimitPerLabel = 100;
    double dropoutRate = 0.0;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    LearningConfig,
    initialLearningRate,
    minLearningRate,
    learningRateDecay,
    maxEpochs,
    patience,
    innerEpochs,
    datasetLimitPerLabel,
    dropoutRate
);


} // namespace Classifier
} // namespace Neural
