#pragma once

#include "core/functions/config.h"

#include <nlohmann/json.hpp>

namespace genetyka {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    GenetizerConfig, maxPopulation, tournamentSize, populationDecreaseFactor);

}  // namespace genetyka

namespace Genetizer {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MutationOptions, operators, scalarRange);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    FitnessConfig, accuracyWeight, complexityWeight, lengthWeight);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    FunctionsConfig, genetizer, mutation, fitness, initialExpressions,
    randomCount, randomDepth, epochs, patience, printTop, printEvery, seed);

}  // namespace Genetizer
