#pragma once

#include "core/lib/cache.h"
#include "core/lib/rpn.h"
#include "core/lib/rpn_compile.h"
#include "core/functions/genetizer.h"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
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
            presentationHash = std::hash<std::string>{}(*parsedPresentation);
        }
        return *parsedPresentation;
    }

    std::size_t getPresentationHash() const {
        getPresentation();
        return presentationHash;
    }

    mutable std::optional<std::string> parsedPresentation = std::nullopt;
    mutable std::size_t presentationHash = 0;
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

struct DistinctRow {
    const FunctionGenetizer::OrganismInfo* representative;
    std::size_t birth;
    std::size_t copies;
};

struct DistinctWorld {
    std::vector<DistinctRow> rows;
    std::size_t uniqueCount = 0;
    std::size_t totalCount = 0;
};

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

    static DistinctWorld CollectDistinct(const FunctionGenetizer::TWorld& world, std::size_t top = 0);
    static std::string PrintWorld(const FunctionGenetizer::TWorld& world, std::size_t top = 0);

private:
    double rankOrganism(const OrganismInfo& organism);
    OrganismInfo crossoverFunction(const OrganismInfo& mother, const OrganismInfo& father);
    void mutateFunction(OrganismInfo& organism);

    // Slot 0 holds zero and is never written. It is where every variable the
    // data never mentioned resolves to, which is what VariableHolder did by
    // returning T{} for a name it had not been told.
    static constexpr std::uint32_t kZeroSlot = 0;

    struct FlatEntry {
        std::uint32_t firstAssignment;
        std::uint32_t assignmentCount;
        TScalar expectedResult;
    };
    struct SlotAssignment {
        std::uint32_t slot;
        TScalar value;
    };

    std::uint32_t slotOf(const std::string& name) const;

    MutationConfig mutationConfig;
    FitnessConfig fitnessConfig;
    std::unordered_set<std::string> knownVariables;
    std::vector<Entry> expectedEntries;

    std::unordered_map<std::string, std::uint32_t> variableSlots;
    std::vector<SlotAssignment> slotAssignments;
    std::vector<FlatEntry> flatEntries;

    std::vector<TScalar> slotValues{TScalar{}};

    Matematyka::CompiledExpression<TScalar> program;
    std::vector<TScalar> evaluationStack;

    TScalar expectedMagnitudeSum = 0;
    TScalar errorScale = 1.;

    cache::LRUCache<Expression::TRpn, double, Matematyka::RpnHash<TScalar>> rankCache{};
};

}
