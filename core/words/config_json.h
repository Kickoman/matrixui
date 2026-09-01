#pragma once

#include "core/words/config.h"

#include <nlohmann/json.hpp>

namespace Words {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    ModelConfig, dim, negatives, initialLearningRate, minLearningRateFactor);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    SamplingConfig, window, sample, negativeTableSize, negativePower);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    TrainConfig, epochs, threads, chunkSize, syncEvery, reportEveryMs, probePairs, seed);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WordsConfig, model, sampling, train);

}  // namespace Words
