#pragma once

#include "core/words/config.h"

#include <nlohmann/json.hpp>

namespace Words {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(
    ModelConfig, dim, negatives, initialLearningRate, minLearningRateFactor, minN, maxN, buckets);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(
    SamplingConfig, window, sample, negativeTableSize, negativePower);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(
    TrainConfig, epochs, threads, chunkSize, syncEvery, reportEveryMs, probePairs, seed);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(WordsConfig, model, sampling, train);

}  // namespace Words
