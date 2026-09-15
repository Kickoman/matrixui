#pragma once

#include <cstddef>
#include <random>
#include <vector>
#include <algorithm>
#include <functional>

// #include <iostream>

namespace genetyka {

// Function for ranking organisms
template<class TRank, class TOrganism>
using RankFunction = std::function<TRank(const TOrganism&)>;

// Function to mutate an organism inplace
template<class TOrganism>
using MutateInplace = std::function<void(TOrganism&)>;

// Function for organisms crossover
template<class TOrganism>
using Crossover = std::function<TOrganism(const TOrganism&, const TOrganism&)>;


struct GenetizerConfig {
    std::size_t maxPopulation = 100000;
    std::size_t tournamentSize = 3;
    double populationDecreaseFactor = 0.9;
};


template<
    class TOrganism,
    class TRank
>
class Genetizer {
public:
    struct OrganismInfo {
        TOrganism organism;
        TRank rank;
    };
    using TWorld = std::vector<OrganismInfo>;
    using TRankFunction = RankFunction<TRank, TOrganism>;
    using TMutateFunction = MutateInplace<TOrganism>;
    using TCrossoverFunction = Crossover<TOrganism>;

    Genetizer(
        RankFunction<TRank, TOrganism>&& rankFunction,
        MutateInplace<TOrganism>&& mutateFunction,
        Crossover<TOrganism>&& crossoverFunction
    )
        : rankFunction(rankFunction)
        , mutateFunction(mutateFunction)
        , crossoverFunction(crossoverFunction)
        , generator(std::random_device{}())
    {
        setConfig({});
    }

    void reset() {
        world.resize(0);
    }

    void setConfig(const GenetizerConfig& config) {
        this->config = config;
        world.reserve(config.maxPopulation);
        tournamentCompetitors.resize(config.tournamentSize);
        rankedPopulationCount = 0;
    }

    void addOrganism(const TOrganism& organism) {
        world.push_back({
            .organism = organism,
            .rank = rankFunction(organism),
        });
    }

    void addOrganism(TOrganism&& organism) {
        const auto rank = rankFunction(organism);
        world.emplace_back(OrganismInfo{
            .organism = std::move(organism),
            .rank = rank,
        });
    }

    void rankPopulation(const std::size_t populationCount) {
        std::sort(
            std::begin(world),
            std::begin(world) + populationCount,
            [this](const auto& l, const auto& r) -> bool {
                return l.rank > r.rank;
            }
        );
        rankedPopulationCount = populationCount;
    }

    void rankPopulation() {
        rankPopulation(world.size());
    }

    const OrganismInfo& tournamentSelect(const std::size_t populationCount) {
        std::uniform_int_distribution<std::size_t> distribution(0, populationCount - 1);
        for (auto& competitor : tournamentCompetitors) {
            competitor = distribution(generator);
        }

        std::size_t winnerIdx = tournamentCompetitors[0];
        TRank highestRank = rankFunction(world[tournamentCompetitors[0]].organism);
        for (std::size_t i = 1; i < tournamentCompetitors.size(); ++i) {
            const auto organismIdx = tournamentCompetitors[i];
            const auto organismRank = rankFunction(world[organismIdx].organism);
            if (organismRank > highestRank) {
                winnerIdx = organismIdx;
                highestRank = organismRank;
            }
        }
        return world[winnerIdx];
    }

    void runEpoch() {
        auto populationAlive = std::min(world.size(), static_cast<std::size_t>(std::floor(config.maxPopulation * config.populationDecreaseFactor)));
        if (world.size() < config.maxPopulation) [[ unlikely ]] {
            world.resize(config.maxPopulation);
        }
        const auto parentPool = populationAlive;
        while (populationAlive < config.maxPopulation) {
            const auto& mother = tournamentSelect(parentPool);
            const auto& father = tournamentSelect(parentPool);

            auto child = crossoverFunction(mother.organism, father.organism);
            mutateFunction(child);

            const auto rank = rankFunction(child);
            world[populationAlive].organism = std::move(child);
            world[populationAlive].rank = rank;
            ++populationAlive;
        }
        rankPopulation(populationAlive);
    }

    const TWorld& getWorld() const { return world; }

private:
    const RankFunction<TRank, TOrganism> rankFunction;
    const MutateInplace<TOrganism> mutateFunction;
    const Crossover<TOrganism> crossoverFunction;

    GenetizerConfig config;
    TWorld world;
    std::size_t epochNumber;
    std::size_t rankedPopulationCount = 0;

    std::mt19937 generator;
    std::vector<std::size_t> tournamentCompetitors;
};

}
