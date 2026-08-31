#pragma once

// JSON bindings for the words configuration structs.
//
// Kept apart from config.h on purpose: nlohmann/json.hpp is ~25k lines, and
// config.h is included (via model.h and trainer.h) by most of the module.
// Only the layers that actually serialise -- a CLI --config flag, or the GUI's
// QSettings persistence -- need to include this.

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
