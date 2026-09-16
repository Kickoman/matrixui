#include <doctest/doctest.h>

#include "core/functions/applier.h"
#include "core/lib/text.h"

#include <string>
#include <vector>

namespace {

using Genetizer::FunctionGenetizer;
using Genetizer::FunctionGenetizerApplier;
using Genetizer::OrganismInfo;
using Genetizer::Variable;

// Teaches one variable named "x": point i is (xs[i], expected[i]).
FunctionGenetizerApplier Taught(const std::vector<double>& xs, const std::vector<double>& expected) {
    FunctionGenetizerApplier applier;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        applier.addExpected({Variable{.name = "x", .value = xs[i]}}, expected[i]);
    }
    return applier;
}

double RankOf(FunctionGenetizerApplier& applier, const std::string& expression) {
    return applier.getRankFunction()(OrganismInfo{
        .epochOfBirth = 0,
        .expression = Genetizer::Expression(expression),
    });
}

FunctionGenetizer::OrganismInfo Organism(const std::string& expression, std::size_t birth, double rank) {
    return FunctionGenetizer::OrganismInfo{
        .organism = OrganismInfo{.epochOfBirth = birth, .expression = Genetizer::Expression(expression)},
        .rank = rank,
    };
}

std::vector<double> Triangular(const std::vector<double>& xs) {
    std::vector<double> out;
    for (const auto x : xs) {
        out.push_back(x * (x + 1) / 2);
    }
    return out;
}

const std::vector<double> kZeroToNineteen{0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                          10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

// The tabulator pads cells to the column width, so tests match on what a row
// says rather than on where it says it.
std::vector<std::string> CellsOfRowWith(const std::string& table, const std::string& needle) {
    for (const auto& line : Text::Split(table, "\n")) {
        if (line.find(needle) == std::string::npos) {
            continue;
        }
        auto cells = Text::Split(line, "|");
        for (auto& cell : cells) {
            cell = std::string(Text::Trim(cell));
        }
        return cells;
    }
    return {};
}

}  // namespace

TEST_CASE("Ranking is invariant to a uniform scaling of the data") {
    // Errors are divided by the mean magnitude, so a dataset measured in units
    // and the same dataset measured in tens must rank identically. Only the
    // accuracy term is scale-free, so the compared expressions must be the
    // same strings -- hence the degree-1 candidates, for which e(k*x) = k*e(x).
    const std::vector<double> xs{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<double> scaledXs, small, large;
    for (const auto x : xs) {
        scaledXs.push_back(10 * x);
        small.push_back(3 * x);
        large.push_back(30 * x);
    }

    auto unitScale = Taught(xs, small);
    auto tenfold = Taught(scaledXs, large);

    for (const auto* expression : {"x", "x+x", "x*3"}) {
        CHECK(RankOf(unitScale, expression) == doctest::Approx(RankOf(tenfold, expression)).epsilon(1e-9));
    }
    CHECK(RankOf(unitScale, "x*3") > RankOf(unitScale, "x+x"));
    CHECK(RankOf(unitScale, "x+x") > RankOf(unitScale, "x"));
}

TEST_CASE("Ranking is invariant to the number of points") {
    // Duplicating every point doubles the summed error and the count, so the
    // mean -- and the rank -- must not move.
    const std::vector<double> xs{1, 2, 3, 4};
    std::vector<double> doubledXs, expected, doubledExpected;
    for (const auto x : xs) {
        expected.push_back(x * x);
        doubledXs.insert(doubledXs.end(), {x, x});
        doubledExpected.insert(doubledExpected.end(), {x * x, x * x});
    }

    auto once = Taught(xs, expected);
    auto twice = Taught(doubledXs, doubledExpected);
    CHECK(RankOf(once, "x+1") == doctest::Approx(RankOf(twice, "x+1")).epsilon(1e-9));
}

TEST_CASE("A partial fit outranks the trivial expression on large-valued data") {
    // The regression this file exists for: with an unnormalized error the
    // accuracy term vanished against the complexity term on data of this
    // magnitude, so `x` beat every stepping stone towards the answer.
    auto applier = Taught(kZeroToNineteen, Triangular(kZeroToNineteen));

    CHECK(RankOf(applier, "(x*x+x)/2") > RankOf(applier, "x*x/2"));
    CHECK(RankOf(applier, "x*x/2") > RankOf(applier, "x"));
}

TEST_CASE("Fitness weights decide what wins") {
    auto applier = Taught({1, 2, 3, 4}, {2, 4, 6, 8});

    SUBCASE("accuracy only makes an exact fit worth exactly one") {
        applier.setFitnessOptions({.accuracyWeight = 1, .complexityWeight = 0, .lengthWeight = 0});
        CHECK(RankOf(applier, "x*2") == doctest::Approx(1.0));
        CHECK(RankOf(applier, "x*2") > RankOf(applier, "x"));
    }

    SUBCASE("complexity only makes the shortest expression win regardless of fit") {
        applier.setFitnessOptions({.accuracyWeight = 0, .complexityWeight = 1, .lengthWeight = 0});
        CHECK(RankOf(applier, "x") > RankOf(applier, "x*2"));
    }
}

TEST_CASE("Re-teaching an applier invalidates cached ranks") {
    auto applier = Taught({1, 2, 3}, {2, 4, 6});
    const auto onDoubles = RankOf(applier, "x*2");

    applier.resetExpected();
    for (std::size_t i = 0; i < 3; ++i) {
        applier.addExpected({Variable{.name = "x", .value = static_cast<double>(i + 1)}},
                            static_cast<double>((i + 1) * 5));
    }
    CHECK(RankOf(applier, "x*2") != doctest::Approx(onDoubles));
}

TEST_CASE("PrintWorld collapses duplicate expressions") {
    const FunctionGenetizer::TWorld world{
        Organism("x+x", 5, 0.9),
        Organism("x+x", 2, 0.9),
        Organism("x+x", 7, 0.9),
        Organism("x", 1, 0.4),
    };

    const auto printed = FunctionGenetizerApplier::PrintWorld(world, 2);
    CHECK(printed.find("unique 2 / 4") != std::string::npos);
    CHECK(printed.find("Copies") != std::string::npos);

    const auto collapsed = CellsOfRowWith(printed, "x+x");
    REQUIRE(collapsed.size() == 4);
    // Copies of one expression tie on rank and sorting is not stable, so the
    // row reports the earliest birth in the group, not whichever came first.
    CHECK(collapsed[0] == "2");
    CHECK(collapsed[1] == "x+x");
    CHECK(collapsed[3] == "3");

    const auto single = CellsOfRowWith(printed, "0.4000");  // the lone `x`
    REQUIRE(single.size() == 4);
    CHECK(single[1] == "x");
    CHECK(single[3] == "1");
}

TEST_CASE("CollectDistinct collapses clones and counts the whole world") {
    const FunctionGenetizer::TWorld world{
        Organism("x+x", 5, 0.9),
        Organism("x+x", 2, 0.9),
        Organism("x+x", 7, 0.9),
        Organism("x", 1, 0.4),
    };

    const auto all = FunctionGenetizerApplier::CollectDistinct(world);
    CHECK(all.uniqueCount == 2);
    CHECK(all.totalCount == 4);
    REQUIRE(all.rows.size() == 2);
    CHECK(all.rows[0].representative->organism.getPresentation() == "x+x");
    CHECK(all.rows[0].representative->rank == 0.9);
    CHECK(all.rows[0].birth == 2);  // earliest of the clone group
    CHECK(all.rows[0].copies == 3);
    CHECK(all.rows[1].copies == 1);

    // Truncating the rows still reports how many distinct ones exist.
    const auto topOne = FunctionGenetizerApplier::CollectDistinct(world, 1);
    CHECK(topOne.rows.size() == 1);
    CHECK(topOne.uniqueCount == 2);
    CHECK(topOne.totalCount == 4);

    CHECK(FunctionGenetizerApplier::CollectDistinct(world, 99).rows.size() == 2);

    const auto empty = FunctionGenetizerApplier::CollectDistinct({});
    CHECK(empty.rows.empty());
    CHECK(empty.uniqueCount == 0);
    CHECK(empty.totalCount == 0);
}

TEST_CASE("PrintWorld handles the degenerate worlds") {
    CHECK(FunctionGenetizerApplier::PrintWorld({}) == "<empty world>\n");

    const FunctionGenetizer::TWorld world{Organism("x", 0, 0.4), Organism("x+1", 1, 0.3)};
    const auto all = FunctionGenetizerApplier::PrintWorld(world, 0);
    CHECK(all.find("unique 2 / 2") != std::string::npos);
    CHECK(all.find("x+1") != std::string::npos);

    // Asking for more distinct rows than exist is not an error.
    CHECK(FunctionGenetizerApplier::PrintWorld(world, 99).find("unique 2 / 2") != std::string::npos);
}
