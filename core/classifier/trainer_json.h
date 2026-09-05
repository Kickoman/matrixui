#pragma once

#include "core/classifier/trainer.h"

#include <nlohmann/json.hpp>

namespace Neural {
namespace Classifier {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(EpochLog,
    epochNumber,
    learningRate,
    trainAccuracy,
    bestTrainAccuracy,
    stagnateEpochsCount
);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TestStatistics, passedTests, totalTests);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TestResult, stats);

}  // namespace Classifier
}  // namespace Neural
