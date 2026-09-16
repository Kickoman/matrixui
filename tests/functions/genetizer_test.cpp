#include <doctest/doctest.h>

#include "core/functions/genetizer.h"

#include <cstddef>
#include <random>
#include <vector>

namespace {

// A transparent organism: the rank IS the value, so what the engine did is
// readable straight off the world, without expressions in the way.
struct Organism {
    int value = 0;
};

using Engine = genetyka::Genetizer<Organism, double>;

struct Counters {
    std::size_t rankCalls = 0;
    std::size_t crossoverCalls = 0;
    std::vector<std::pair<int, int>> parents;
};

Engine MakeEngine(Counters& counters) {
    return Engine(
        [&counters](const Organism& organism) -> double {
            ++counters.rankCalls;
            return organism.value;
        },
        [](Organism& organism) { organism.value += 1; },
        [&counters](const Organism& mother, const Organism& father) -> Organism {
            ++counters.crossoverCalls;
            counters.parents.emplace_back(mother.value, father.value);
            return Organism{mother.value + father.value};
        });
}

genetyka::GenetizerConfig Config(const std::size_t population, const std::size_t tournament = 3) {
    return genetyka::GenetizerConfig{
        .maxPopulation = population,
        .tournamentSize = tournament,
        .populationDecreaseFactor = 0.9,
    };
}

void Seed(Engine& engine, const std::vector<int>& values) {
    for (const auto value : values) {
        engine.addOrganism(Organism{value});
    }
}

}  // namespace


TEST_CASE("addOrganism ranks the organism exactly once, as it is inserted") {
    Counters counters;
    auto engine = MakeEngine(counters);
    engine.setConfig(Config(10));

    Seed(engine, {4, 7});
    CHECK(counters.rankCalls == 2);
    REQUIRE(engine.getWorld().size() == 2);
    CHECK(engine.getWorld()[0].rank == doctest::Approx(4));
    CHECK(engine.getWorld()[1].rank == doctest::Approx(7));
}

TEST_CASE("Every ranked organism carries the rank its rank function would give") {
    // This is the precondition that lets the tournament read world[i].rank
    // instead of recomputing it. If it ever stops holding, that optimisation is
    // unsound -- so it is pinned here rather than assumed.
    Counters counters;
    auto engine = MakeEngine(counters);
    engine.setConfig(Config(20));
    engine.setSeed(11);
    Seed(engine, {1, 2, 3, 4, 5, 6, 7, 8});

    const auto check = [&engine] {
        for (const auto& info : engine.getWorld()) {
            CHECK(info.rank == doctest::Approx(info.organism.value));
        }
    };

    engine.rankPopulation();
    check();
    for (int epoch = 0; epoch < 3; ++epoch) {
        engine.runEpoch();
        check();
    }
}

TEST_CASE("rankPopulation orders the world by descending rank") {
    Counters counters;
    auto engine = MakeEngine(counters);
    engine.setConfig(Config(10));
    Seed(engine, {3, 9, 1, 7});

    engine.rankPopulation();
    const auto& world = engine.getWorld();
    REQUIRE(world.size() == 4);
    CHECK(world[0].organism.value == 9);
    CHECK(world[1].organism.value == 7);
    CHECK(world[2].organism.value == 3);
    CHECK(world[3].organism.value == 1);
}

TEST_CASE("rankPopulation(k) sorts only the first k and leaves the tail alone") {
    // Selection draws parents by index out of the sorted prefix, which is why a
    // partial sort is not interchangeable with a full one: the prefix would hold
    // the same organisms in a different order, and the draws would land elsewhere.
    Counters counters;
    auto engine = MakeEngine(counters);
    engine.setConfig(Config(10));
    Seed(engine, {3, 9, 1, 7});

    engine.rankPopulation(2);
    const auto& world = engine.getWorld();
    CHECK(world[0].organism.value == 9);
    CHECK(world[1].organism.value == 3);
    CHECK(world[2].organism.value == 1);  // untouched
    CHECK(world[3].organism.value == 7);  // untouched
}

TEST_CASE("The tournament draws tournamentSize indices and returns the best of them") {
    // The draw count, the distribution and the winner rule are all pinned by
    // replaying the engine's own RNG. Deliberately says nothing about how many
    // times the rank function is called to decide the winner.
    constexpr std::uint64_t kSeed = 4242;
    constexpr std::size_t kPool = 8;
    constexpr std::size_t kTournament = 3;

    Counters counters;
    auto engine = MakeEngine(counters);
    engine.setConfig(Config(40, kTournament));
    engine.setSeed(kSeed);
    Seed(engine, {5, 1, 9, 3, 7, 2, 8, 4});
    engine.rankPopulation();

    std::mt19937 replica(kSeed);
    std::uniform_int_distribution<std::size_t> draw(0, kPool - 1);
    std::vector<std::size_t> drawn;
    for (std::size_t i = 0; i < kTournament; ++i) {
        drawn.push_back(draw(replica));
    }

    const auto& world = engine.getWorld();
    std::size_t expected = drawn[0];
    for (std::size_t i = 1; i < drawn.size(); ++i) {
        if (world[drawn[i]].rank > world[expected].rank) {  // strict: first wins a tie
            expected = drawn[i];
        }
    }

    const auto& winner = engine.tournamentSelect(kPool);
    CHECK(&winner == &world[expected]);
}

TEST_CASE("An epoch ranks each child once and re-ranks nobody") {
    // The tournament used to call the fitness function for every competitor,
    // costing 2 * tournamentSize re-rankings of already-ranked parents on top of
    // the one legitimate ranking of the child -- seven evaluations per child at
    // the default tournament size, six of them buying nothing. Reading the
    // stored rank instead is what this pins.
    Counters counters;
    auto engine = MakeEngine(counters);
    engine.setConfig(Config(10, 3));
    engine.setSeed(7);
    Seed(engine, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    const auto afterSeeding = counters.rankCalls;
    engine.runEpoch();
    const auto children = counters.crossoverCalls;

    REQUIRE(children == 1);  // 10 - floor(10 * 0.9)
    CHECK(counters.rankCalls - afterSeeding == children);
}

TEST_CASE("An epoch replaces the bottom slice and refills the world") {
    Counters counters;
    auto engine = MakeEngine(counters);
    engine.setConfig(Config(10));
    engine.setSeed(3);
    Seed(engine, {1, 2, 3, 4, 5});

    engine.runEpoch();
    const auto& world = engine.getWorld();
    CHECK(world.size() == 10);
    CHECK(counters.crossoverCalls == 5);  // grown from 5 up to maxPopulation

    counters.crossoverCalls = 0;
    engine.runEpoch();
    CHECK(counters.crossoverCalls == 1);  // 10 - floor(10 * 0.9)
}

TEST_CASE("Parents come only from the ranked prefix, never from the fresh tail") {
    // runEpoch resizes the world up to maxPopulation with default-constructed
    // organisms. Those carry value 0 and must never be selected as parents.
    Counters counters;
    auto engine = MakeEngine(counters);
    engine.setConfig(Config(30));
    engine.setSeed(99);
    Seed(engine, {5, 6, 7, 8});
    engine.rankPopulation();

    engine.runEpoch();
    REQUIRE_FALSE(counters.parents.empty());
    for (const auto& [mother, father] : counters.parents) {
        CHECK(mother >= 5);
        CHECK(father >= 5);
    }
}

TEST_CASE("The same seed replays the same evolution") {
    const auto evolve = [](const std::uint64_t seed) {
        Counters counters;
        auto engine = MakeEngine(counters);
        engine.setConfig(Config(24));
        engine.setSeed(seed);
        Seed(engine, {1, 2, 3, 4, 5, 6});
        engine.rankPopulation();
        for (int epoch = 0; epoch < 4; ++epoch) {
            engine.runEpoch();
        }
        std::vector<int> result;
        for (const auto& info : engine.getWorld()) {
            result.push_back(info.organism.value);
        }
        return result;
    };

    CHECK(evolve(2026) == evolve(2026));
    CHECK(evolve(2026) != evolve(2027));
}

TEST_CASE("reset empties the world without disturbing the configuration") {
    Counters counters;
    auto engine = MakeEngine(counters);
    engine.setConfig(Config(10));
    Seed(engine, {1, 2, 3});
    engine.reset();

    CHECK(engine.getWorld().empty());
    Seed(engine, {4});
    CHECK(engine.getWorld().size() == 1);
}
