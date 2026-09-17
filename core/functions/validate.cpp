#include "core/functions/validate.h"

#include <stdexcept>
#include <unordered_set>

namespace Genetizer {

void Validate(const FunctionsConfig& config) {
    if (config.mutation.operators.empty()) {
        throw std::runtime_error("config: operators must not be empty");
    }
    std::unordered_set<std::string> seenFunctions;
    for (const auto& name : config.mutation.functions) {
        if (Matematyka::rpn::FindFunction<double>(name) == nullptr) {
            throw std::runtime_error("config: unknown function '" + name + "'");
        }
        if (!seenFunctions.insert(name).second) {
            throw std::runtime_error("config: duplicate function '" + name + "'");
        }
    }
    if (config.genetizer.maxPopulation == 0) {
        throw std::runtime_error("config: maxPopulation must be positive");
    }
    if (config.genetizer.tournamentSize == 0) {
        throw std::runtime_error("config: tournamentSize must be positive");
    }
    if (config.epochs == 0) {
        throw std::runtime_error("config: epochs must be positive");
    }
    if (config.randomCount > 0 && config.randomDepth == 0) {
        throw std::runtime_error("config: randomDepth must be positive when randomCount is");
    }
    if (config.randomCount == 0 && config.initialExpressions.empty()) {
        throw std::runtime_error(
            "config: nothing to seed -- need randomCount > 0 or initialExpressions");
    }
    const auto& fitness = config.fitness;
    if (fitness.accuracyWeight < 0 || fitness.complexityWeight < 0 || fitness.lengthWeight < 0) {
        throw std::runtime_error("config: fitness weights must not be negative");
    }
    if (fitness.accuracyWeight + fitness.complexityWeight + fitness.lengthWeight <= 0) {
        throw std::runtime_error("config: at least one fitness weight must be positive");
    }
}

}  // namespace Genetizer
