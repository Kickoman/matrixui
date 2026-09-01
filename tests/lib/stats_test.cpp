#include <doctest/doctest.h>

#include "core/lib/stats.h"

#include <vector>

TEST_CASE("PearsonOf is 1 for a rising line and -1 for a falling one") {
    const std::vector<double> x{1., 2., 3., 4.};
    CHECK(PearsonOf(x, {2., 4., 6., 8.}) == doctest::Approx(1.));
    CHECK(PearsonOf(x, {8., 6., 4., 2.}) == doctest::Approx(-1.));
}

TEST_CASE("PearsonOf returns zero rather than NaN on degenerate input") {
    // Fewer than two entries: no variance to speak of.
    CHECK(PearsonOf({}, {}) == doctest::Approx(0.));
    CHECK(PearsonOf({1.}, {2.}) == doctest::Approx(0.));

    // One side constant, so its variance -- and the denominator -- is zero.
    CHECK(PearsonOf({5., 5., 5.}, {1., 2., 3.}) == doctest::Approx(0.));
}

TEST_CASE("RanksOf numbers values from 1 in ascending order") {
    const auto ranks = RanksOf({30., 10., 20.});
    REQUIRE(ranks.size() == 3);
    CHECK(ranks[0] == doctest::Approx(3.));
    CHECK(ranks[1] == doctest::Approx(1.));
    CHECK(ranks[2] == doctest::Approx(2.));
}

TEST_CASE("RanksOf averages the ranks a tie spans") {
    // 5 and 5 occupy ranks 2 and 3, so both get 2.5.
    const auto pair = RanksOf({5., 5., 1.});
    REQUIRE(pair.size() == 3);
    CHECK(pair[0] == doctest::Approx(2.5));
    CHECK(pair[1] == doctest::Approx(2.5));
    CHECK(pair[2] == doctest::Approx(1.));

    // A three-way tie at the bottom spans ranks 1..3.
    const auto triple = RanksOf({7., 7., 7., 9.});
    CHECK(triple[0] == doctest::Approx(2.));
    CHECK(triple[1] == doctest::Approx(2.));
    CHECK(triple[2] == doctest::Approx(2.));
    CHECK(triple[3] == doctest::Approx(4.));
}

TEST_CASE("PearsonOf over RanksOf is the Spearman coefficient") {
    const std::vector<double> human{5., 5., 1.};
    const std::vector<double> model{0.984807753012208, 0.8660254037844387, 0.5};

    // Averaged ranks make this sqrt(3)/2; distinct ranks for the tie would
    // wrongly give 1. Same figure the EvaluateSimilarity test pins.
    CHECK(PearsonOf(RanksOf(human), RanksOf(model)) == doctest::Approx(0.8660254037844387));
    CHECK(PearsonOf(human, model) == doctest::Approx(0.9719874013473091));
}

TEST_CASE("RanksOf handles empty and single-element input") {
    CHECK(RanksOf({}).empty());
    const auto one = RanksOf({42.});
    REQUIRE(one.size() == 1);
    CHECK(one[0] == doctest::Approx(1.));
}
