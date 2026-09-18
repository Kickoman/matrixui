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


namespace {

// Ranking needs at least one taught point, or every expression hits the
// "mentions no known variable" floor before it is ever evaluated.
FunctionGenetizerApplier TaughtOnLine() {
    return Taught({1, 2, 3, 4}, {2, 4, 6, 8});
}

Genetizer::OrganismInfo Mutated(FunctionGenetizerApplier& applier,
                                const std::string& expression, const unsigned seed) {
    FunctionGenetizerApplier::SeedThreadRng(seed);
    OrganismInfo organism{.epochOfBirth = 0, .expression = Genetizer::Expression(expression)};
    applier.getMutateFunction()(organism);
    return organism;
}

// setMutationOptions must run before addExpected -- addExpected is what teaches
// the variable list, and resetExpected would wipe the options again.
FunctionGenetizerApplier TaughtWithFunctions(const std::vector<std::string>& functions) {
    FunctionGenetizerApplier applier;
    applier.setMutationOptions({'+', '-', '*', '/', '^'}, functions, 5.0);
    const std::vector<double> xs{1, 2, 3, 4};
    const std::vector<double> expected{2, 4, 6, 8};
    for (std::size_t i = 0; i < xs.size(); ++i) {
        applier.addExpected({Variable{.name = "x", .value = xs[i]}}, expected[i]);
    }
    return applier;
}

std::size_t FunctionUnits(const Genetizer::Expression& expression) {
    std::size_t count = 0;
    for (const auto& unit : expression.getRpn()) {
        if (unit.getType() == Matematyka::rpn::UnitType::Function) {
            ++count;
        }
    }
    return count;
}

bool Runnable(const Genetizer::Expression& expression) {
    Genetizer::VariableHolder holder;
    holder.setVariable("x", 1.5);
    try {
        expression.run(holder);
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace


TEST_CASE("An expression that mentions no known variable gets the floor rank") {
    // The gate short-circuits before any evaluation, and the constant it returns
    // is observable -- it decides where constant expressions sort in the world.
    auto applier = TaughtOnLine();
    CHECK(RankOf(applier, "1+1") == doctest::Approx(0.00001));
    CHECK(RankOf(applier, "sin(1)") == doctest::Approx(0.00001));

    // A name no data point carries is not a known variable either.
    CHECK(RankOf(applier, "y+1") == doctest::Approx(0.00001));

    // Mentioning x is enough; it does not have to matter to the result.
    CHECK(RankOf(applier, "x*0") > 0.00001);
}

TEST_CASE("An expression that cannot be evaluated ranks exactly zero") {
    // Every failure inside the evaluation loop is caught and collapsed to 0.
    // Any faster evaluator has to collapse the same set of failures the same way.
    auto applier = TaughtOnLine();

    Genetizer::Expression::TRpn rpn;
    rpn.push_back(Genetizer::Expression::TUnit::createVariable("x"));
    rpn.push_back(Genetizer::Expression::TUnit::createFunction("sin"));
    rpn.push_back(Genetizer::Expression::TUnit::createScalar(1));
    rpn.push_back(Genetizer::Expression::TUnit::createOperator('+'));

    const auto rank = applier.getRankFunction()(OrganismInfo{
        .epochOfBirth = 0,
        .expression = Genetizer::Expression{std::move(rpn)},
    });
    CHECK(rank == 0.0);
}

TEST_CASE("Ranking the same organism twice gives the same answer") {
    auto applier = TaughtOnLine();
    const auto first = RankOf(applier, "x*2+1");

    // Rank 1500 distinct expressions to push the first one out of the LRU, then
    // ask again: a cache may only ever be an optimisation.
    for (int i = 0; i < 1500; ++i) {
        RankOf(applier, "x+" + std::to_string(i) + ".5");
    }
    CHECK(RankOf(applier, "x*2+1") == doctest::Approx(first));
}

TEST_CASE("Crossover of two runnable parents stays runnable") {
    auto applier = TaughtOnLine();
    const auto crossover = applier.getCrossoverFunction();

    for (unsigned seed = 1; seed <= 300; ++seed) {
        FunctionGenetizerApplier::SeedThreadRng(seed);
        const auto mother = applier.makeRandomOrganism(3, false);
        const auto father = applier.makeRandomOrganism(3, true);
        if (!Runnable(mother.expression) || !Runnable(father.expression)) {
            continue;
        }
        const auto child = crossover(mother, father);
        CHECK(Runnable(child.expression));
    }
}

TEST_CASE("Crossover dates the child one epoch after its later parent") {
    auto applier = TaughtOnLine();
    FunctionGenetizerApplier::SeedThreadRng(1);
    const auto child = applier.getCrossoverFunction()(
        OrganismInfo{.epochOfBirth = 4, .expression = Genetizer::Expression("x+1")},
        OrganismInfo{.epochOfBirth = 9, .expression = Genetizer::Expression("x*2")});
    CHECK(child.epochOfBirth == 10);
}

TEST_CASE("Mutation leaves an expression that can still be evaluated") {
    // Three of the four mutations used to be able to leave an expression that
    // threw on its first data point, and such an organism ranks zero -- so a
    // fifth to a third of every generation was stillborn, depending on whether
    // the parent contained a function:
    //   insertUnary  spliced in a function unit with no '#' to apply it, always;
    //   wrapBinary   could land inside a [fn, arg, '#'] triple and split it;
    //   pointMutate  could rewrite a '#' into an ordinary operator, which the
    //                configured operator set never contains, orphaning the
    //                function to its left.
    // The base expression has to contain a function, or only the first of the
    // three is exercised.
    auto applier = TaughtOnLine();
    for (const auto* base : {"x+1", "1+sin(x)", "1+sin(x)*cos(x)"}) {
        unsigned broken = 0;
        for (unsigned seed = 1; seed <= 500; ++seed) {
            if (!Runnable(Mutated(applier, base, seed).expression)) {
                ++broken;
            }
        }
        CHECK(broken == 0);
    }
}

TEST_CASE("Mutation actually changes the expression") {
    // Guarding '#' against pointMutate must not turn that draw into a no-op for
    // every expression that happens to contain a function application.
    auto applier = TaughtOnLine();
    unsigned changed = 0;
    for (unsigned seed = 1; seed <= 500; ++seed) {
        const auto before = Genetizer::Expression("1+sin(x)").toString();
        if (Mutated(applier, "1+sin(x)", seed).expression.toString() != before) {
            ++changed;
        }
    }
    CHECK(changed > 450);
}

TEST_CASE("Naming every function explicitly is the same as the default") {
    // The default set is the whole table in table order, so pinning it must not
    // move the RNG stream -- this is what keeps the seeded goldens valid.
    auto byDefault = TaughtOnLine();
    auto explicitly = TaughtWithFunctions(Genetizer::AllFunctionNames());
    const auto sample = [](FunctionGenetizerApplier& applier) {
        std::vector<std::string> out;
        FunctionGenetizerApplier::SeedThreadRng(77);
        for (int i = 0; i < 20; ++i) {
            out.push_back(applier.makeRandomOrganism(3, i % 2 == 0).getPresentation());
        }
        for (unsigned seed = 1; seed <= 20; ++seed) {
            out.push_back(Mutated(applier, "1+sin(x)", seed).expression.toString());
        }
        return out;
    };
    CHECK(sample(byDefault) == sample(explicitly));
}

TEST_CASE("Random organisms use only the configured functions") {
    auto applier = TaughtWithFunctions({"sin"});
    FunctionGenetizerApplier::SeedThreadRng(11);
    for (int i = 0; i < 200; ++i) {
        const auto text = applier.makeRandomOrganism(4, i % 2 == 0).getPresentation();
        for (const auto* absent : {"cos(", "tan(", "exp(", "log(", "floor(", "ceil("}) {
            CHECK(text.find(absent) == std::string::npos);
        }
    }
}

TEST_CASE("An empty function set grows pure arithmetic") {
    auto applier = TaughtWithFunctions({});
    FunctionGenetizerApplier::SeedThreadRng(5);
    for (int i = 0; i < 200; ++i) {
        CHECK(FunctionUnits(applier.makeRandomOrganism(4, i % 2 == 0).expression) == 0);
    }
}

TEST_CASE("An empty function set never introduces a function") {
    // A seeded expression may still carry one (the set only limits what evolution
    // adds), so the count may never *grow* -- it is not required to be zero.
    auto applier = TaughtWithFunctions({});
    for (const auto* base : {"x+1", "1+sin(x)", "1+sin(x)*cos(x)"}) {
        const auto before = FunctionUnits(Genetizer::Expression(base));
        for (unsigned seed = 1; seed <= 500; ++seed) {
            const auto mutated = Mutated(applier, base, seed);
            CHECK(FunctionUnits(mutated.expression) <= before);
            CHECK(Runnable(mutated.expression));
        }
    }
}

TEST_CASE("An empty function set still changes the expression") {
    // The guards must degrade to another mutation, not quietly do nothing:
    // a no-op draw is the bug the default-set case above already guards against.
    auto applier = TaughtWithFunctions({});
    unsigned changed = 0;
    for (unsigned seed = 1; seed <= 500; ++seed) {
        const auto before = Genetizer::Expression("1+sin(x)").toString();
        if (Mutated(applier, "1+sin(x)", seed).expression.toString() != before) {
            ++changed;
        }
    }
    CHECK(changed > 450);
}

TEST_CASE("BUG: mutating an organism does not invalidate its printed form") {
    // getPresentation() memoises into a mutable member that mutateFunction never
    // clears. Nothing reads the stale value today, because crossover always hands
    // mutation a fresh organism -- but anything that caches per organism inherits
    // this trap, so the hazard is recorded rather than left to be rediscovered.
    auto applier = TaughtOnLine();
    FunctionGenetizerApplier::SeedThreadRng(5);
    OrganismInfo organism{.epochOfBirth = 0, .expression = Genetizer::Expression("x+1")};

    const auto before = organism.getPresentation();
    applier.getMutateFunction()(organism);
    CHECK(organism.getPresentation() == before);
    CHECK(organism.expression.toString() != before);
}

TEST_CASE("Random organisms are reproducible under a seed") {
    auto applier = TaughtOnLine();
    const auto sample = [&applier] {
        std::vector<std::string> out;
        FunctionGenetizerApplier::SeedThreadRng(77);
        for (int i = 0; i < 20; ++i) {
            out.push_back(applier.makeRandomOrganism(3, i % 2 == 0).getPresentation());
        }
        return out;
    };
    CHECK(sample() == sample());
}


TEST_CASE("A broken organism cannot inherit or hand out another one's rank") {
    // Ranks are memoised by printed form. While an ill-formed RPN could print
    // exactly like a well-formed one, whichever of the pair was ranked first
    // decided the other's fitness: a valid expression could be killed by a
    // neighbouring wreck, or a wreck could be promoted into the parent pool.
    // Roughly a quarter of all children are ill-formed, so this was not rare.
    Genetizer::Expression::TRpn leftover;
    leftover.push_back(Genetizer::Expression::TUnit::createScalar(3));
    leftover.push_back(Genetizer::Expression::TUnit::createVariable("x"));
    leftover.push_back(Genetizer::Expression::TUnit::createScalar(1));
    leftover.push_back(Genetizer::Expression::TUnit::createOperator('+'));

    const auto broken = [&leftover] {
        return OrganismInfo{.epochOfBirth = 0, .expression = Genetizer::Expression{Genetizer::Expression::TRpn(leftover)}};
    };

    SUBCASE("the good one is ranked first") {
        auto applier = TaughtOnLine();
        const auto good = RankOf(applier, "x+1");
        CHECK(good > 0.0);
        CHECK(applier.getRankFunction()(broken()) == 0.0);
    }

    SUBCASE("the broken one is ranked first") {
        auto applier = TaughtOnLine();
        CHECK(applier.getRankFunction()(broken()) == 0.0);
        CHECK(RankOf(applier, "x+1") > 0.0);
    }
}


TEST_CASE("Constants that print the same are still ranked apart") {
    // Ranks are memoised, and the memo used to be keyed on the printed
    // expression -- but numbers print rounded to four decimals, so two organisms
    // whose constants agreed to four decimals and differed beyond shared a key,
    // and the second was handed the first's rank. jitterConstant moves constants
    // by continuous amounts, so near-duplicates arise constantly. The rank error
    // was small, but it was enough to make a whole run depend on how much cache
    // traffic there had been, which is no basis for a reproducible search.
    const auto scaled = [](const double constant) {
        Genetizer::Expression::TRpn rpn;
        rpn.push_back(Genetizer::Expression::TUnit::createVariable("x"));
        rpn.push_back(Genetizer::Expression::TUnit::createScalar(constant));
        rpn.push_back(Genetizer::Expression::TUnit::createOperator('*'));
        return OrganismInfo{.epochOfBirth = 0, .expression = Genetizer::Expression{std::move(rpn)}};
    };

    REQUIRE(scaled(2.000001).getPresentation() == scaled(2.000004).getPresentation());

    auto alone = TaughtOnLine();
    auto also = TaughtOnLine();
    const auto rankOfFirst = alone.getRankFunction()(scaled(2.000001));
    const auto rankOfSecond = also.getRankFunction()(scaled(2.000004));
    REQUIRE(rankOfFirst != rankOfSecond);

    // Both orders, through one applier, must give each organism its own rank.
    auto shared = TaughtOnLine();
    CHECK(shared.getRankFunction()(scaled(2.000001)) == rankOfFirst);
    CHECK(shared.getRankFunction()(scaled(2.000004)) == rankOfSecond);

    auto reversed = TaughtOnLine();
    CHECK(reversed.getRankFunction()(scaled(2.000004)) == rankOfSecond);
    CHECK(reversed.getRankFunction()(scaled(2.000001)) == rankOfFirst);
}
