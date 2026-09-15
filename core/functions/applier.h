#pragma once

#include "core/lib/cache.h"
#include "core/lib/rpn.h"
#include "core/functions/genetizer.h"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_set>

namespace Genetizer {

using TScalar = double;
using Expression = Matematyka::Expression<TScalar>;
using VariableHolder = Matematyka::VariableHolder<TScalar>;


struct Variable {
    std::string name;
    TScalar value;
};

struct Entry {
    std::vector<Variable> variables;
    TScalar expectedResult;
};

struct OrganismInfo {
    std::size_t epochOfBirth;
    Expression expression;

    const std::string& getPresentation() const {
        if (!parsedPresentation.has_value()) [[ unlikely ]] {
            parsedPresentation = expression.toString();
        }
        return *parsedPresentation;
    }
    mutable std::optional<std::string> parsedPresentation = std::nullopt;
};

struct MutationConfig {
    std::vector<std::string> variables;
    std::vector<char> operators{'+', '-', '*', '/', '^'};
    TScalar scalarRange = 5.;
};

struct FitnessConfig {
    double accuracyWeight = 0.98;
    double complexityWeight = 0.2;
    double lengthWeight = 0.0001;
};

using FunctionGenetizer = genetyka::Genetizer<OrganismInfo, double>;

class FunctionGenetizerApplier {
public:
    static void SeedThreadRng(std::uint64_t seed);

    void resetExpected();
    void addExpected(std::vector<Variable>&& variables, double result);

    void setMutationOptions(std::vector<char> operators, TScalar scalarRange);
    void setFitnessOptions(const FitnessConfig& fitness);

    OrganismInfo makeRandomOrganism(std::size_t maxDepth, bool full) const;
    void seedRandom(FunctionGenetizer& genetizer, std::size_t count, std::size_t maxDepth = 4) const;

    FunctionGenetizer::TRankFunction getRankFunction() {
        return [this](const OrganismInfo& org) { return rankOrganism(org); };
    }
    FunctionGenetizer::TCrossoverFunction getCrossoverFunction() {
        return [this](const OrganismInfo& m, const OrganismInfo& f) { return crossoverFunction(m, f); };
    }
    FunctionGenetizer::TMutateFunction getMutateFunction() {
        return [this](OrganismInfo& o) { return mutateFunction(o); };
    }

    static std::string PrintWorld(const FunctionGenetizer::TWorld& world, std::size_t top = 0);

private:
    double rankOrganism(const OrganismInfo& organism);
    OrganismInfo crossoverFunction(const OrganismInfo& mother, const OrganismInfo& father);
    void mutateFunction(OrganismInfo& organism);

    MutationConfig mutationConfig;
    FitnessConfig fitnessConfig;
    std::unordered_set<std::string> knownVariables;
    VariableHolder vars;
    std::vector<Entry> expectedEntries;

    TScalar expectedMagnitudeSum = 0;
    TScalar errorScale = 1.;

    cache::LRUCache<std::string, double> rankCache{};
};

}
