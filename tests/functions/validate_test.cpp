#include <doctest/doctest.h>

#include "core/functions/validate.h"

#include <stdexcept>

namespace {

Genetizer::FunctionsConfig Workable() {
    Genetizer::FunctionsConfig config;
    config.genetizer.maxPopulation = 64;
    config.randomCount = 8;
    return config;
}

}  // namespace

TEST_CASE("Validate accepts the defaults") {
    CHECK_NOTHROW(Genetizer::Validate(Genetizer::FunctionsConfig{}));
    CHECK_NOTHROW(Genetizer::Validate(Workable()));
}

TEST_CASE("Validate names the offending field") {
    auto config = Workable();

    SUBCASE("empty operators") {
        config.mutation.operators.clear();
        CHECK_THROWS_WITH_AS(Genetizer::Validate(config),
                             doctest::Contains("operators"), std::runtime_error);
    }
    SUBCASE("unknown function") {
        config.mutation.functions = {"sin", "sqrt"};
        CHECK_THROWS_WITH_AS(Genetizer::Validate(config),
                             doctest::Contains("sqrt"), std::runtime_error);
    }
    SUBCASE("duplicate function") {
        config.mutation.functions = {"sin", "sin"};
        CHECK_THROWS_WITH_AS(Genetizer::Validate(config),
                             doctest::Contains("duplicate"), std::runtime_error);
    }
    SUBCASE("zero population") {
        config.genetizer.maxPopulation = 0;
        CHECK_THROWS_WITH_AS(Genetizer::Validate(config),
                             doctest::Contains("maxPopulation"), std::runtime_error);
    }
    SUBCASE("zero tournament") {
        config.genetizer.tournamentSize = 0;
        CHECK_THROWS_WITH_AS(Genetizer::Validate(config),
                             doctest::Contains("tournamentSize"), std::runtime_error);
    }
    SUBCASE("zero epochs") {
        config.epochs = 0;
        CHECK_THROWS_WITH_AS(Genetizer::Validate(config),
                             doctest::Contains("epochs"), std::runtime_error);
    }
    SUBCASE("random organisms without depth") {
        config.randomDepth = 0;
        CHECK_THROWS_WITH_AS(Genetizer::Validate(config),
                             doctest::Contains("randomDepth"), std::runtime_error);
    }
    SUBCASE("nothing to seed") {
        config.randomCount = 0;
        config.initialExpressions.clear();
        CHECK_THROWS_WITH_AS(Genetizer::Validate(config),
                             doctest::Contains("nothing to seed"), std::runtime_error);
    }
    SUBCASE("negative weight") {
        config.fitness.complexityWeight = -1;
        CHECK_THROWS_WITH_AS(Genetizer::Validate(config),
                             doctest::Contains("negative"), std::runtime_error);
    }
    SUBCASE("weights sum to zero") {
        config.fitness = {0, 0, 0};
        CHECK_THROWS_WITH_AS(Genetizer::Validate(config),
                             doctest::Contains("positive"), std::runtime_error);
    }
}

TEST_CASE("Validate accepts an empty function set") {
    // Deliberate: no functions means pure arithmetic, unlike an empty operator
    // set, which leaves mutation nothing at all to build with.
    auto config = Workable();
    config.mutation.functions.clear();
    CHECK_NOTHROW(Genetizer::Validate(config));
}

TEST_CASE("Validate accepts seeding by expression alone") {
    auto config = Workable();
    config.randomCount = 0;
    config.initialExpressions = {"x"};
    CHECK_NOTHROW(Genetizer::Validate(config));
}
