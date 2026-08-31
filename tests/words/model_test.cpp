#include <doctest/doctest.h>

#include "core/lib/random.h"
#include "core/words/model.h"
#include "core/words/negativesampler.h"
#include "core/words/vocabulary.h"
#include "tests/support/fixtures.h"

#include <algorithm>
#include <cmath>

using namespace Words;

namespace {

constexpr std::size_t kTableSize = 100'000;

Vocabulary ManyWordVocabulary(const Tests::TempDir& dir, const std::size_t distinct = 300) {
    const auto path = dir.write("zipf.txt", Tests::ZipfCorpusText(distinct, 60'000));
    return Vocabulary::Build(path, 1);
}

// --- gradient-check helpers -------------------------------------------------
//
// applyUpdate mutates the centre row and every touched output row, so each
// probe has to run against a saved copy and then restore it.

using RowSnapshot = std::vector<std::vector<TFloat>>;

RowSnapshot SaveRows(const SGNSModel& model, const Pair& pair, const std::vector<TWordId>& negatives) {
    const auto dim = model.getConfig().dim;
    RowSnapshot saved;
    saved.reserve(negatives.size() + 2);

    const auto* center = model.getInput().row(pair.center);
    saved.emplace_back(center, center + dim);
    const auto* context = model.getOutput().row(pair.context);
    saved.emplace_back(context, context + dim);
    for (const auto id : negatives) {
        const auto* negative = model.getOutput().row(id);
        saved.emplace_back(negative, negative + dim);
    }
    return saved;
}

void RestoreRows(SGNSModel& model, const Pair& pair, const std::vector<TWordId>& negatives, const RowSnapshot& saved) {
    std::copy(saved[0].begin(), saved[0].end(), model.getInputMutable().row(pair.center));
    std::copy(saved[1].begin(), saved[1].end(), model.getOutputMutable().row(pair.context));
    for (std::size_t i = 0; i < negatives.size(); ++i) {
        std::copy(saved[i + 2].begin(), saved[i + 2].end(), model.getOutputMutable().row(negatives[i]));
    }
}

// The step the optimiser actually takes, divided by the learning rate.
double AnalyticGradient(
    SGNSModel& model, const Pair& pair, const std::vector<TWordId>& negatives,
    std::vector<TFloat>& gradient, const bool inInput, const TWordId row, const std::size_t component
) {
    constexpr double step = 1e-3;
    const auto saved = SaveRows(model, pair, negatives);

    const auto before = inInput ? model.getInput().row(row)[component]
                                : model.getOutput().row(row)[component];
    model.applyUpdate(pair, negatives, step, gradient);
    const auto after = inInput ? model.getInput().row(row)[component]
                               : model.getOutput().row(row)[component];

    RestoreRows(model, pair, negatives, saved);
    return (static_cast<double>(before) - static_cast<double>(after)) / step;
}

// Central difference of computeLoss.
double NumericGradient(
    SGNSModel& model, const Pair& pair, const std::vector<TWordId>& negatives,
    const bool inInput, const TWordId row, const std::size_t component
) {
    constexpr double eps = 1e-3;
    auto* cell = inInput ? model.getInputMutable().row(row) + component
                         : model.getOutputMutable().row(row) + component;
    const auto original = *cell;

    *cell = static_cast<TFloat>(static_cast<double>(original) + eps);
    const double plus = model.computeLoss(pair, negatives);
    *cell = static_cast<TFloat>(static_cast<double>(original) - eps);
    const double minus = model.computeLoss(pair, negatives);
    *cell = original;

    return (plus - minus) / (2. * eps);
}

}  // namespace

TEST_CASE("A freshly built model has zeroed output and varied input") {
    const Tests::TempDir dir;
    const auto vocabulary = ManyWordVocabulary(dir);

    ModelConfig config;
    config.dim = 32;
    XorShift rng(1234);
    const SGNSModel model(vocabulary, config, rng);

    const auto dim = config.dim;
    for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
        for (std::size_t i = 0; i < dim; ++i) {
            REQUIRE(model.getOutput().row(id)[i] == 0.f);
        }
    }
    CHECK(!std::equal(model.getInput().row(0), model.getInput().row(0) + dim, model.getInput().row(1)));
    CHECK(model.getBytes() == 2 * vocabulary.getSize() * dim * sizeof(TFloat));
}

TEST_CASE("The initial score of any pair is exactly zero") {
    const Tests::TempDir dir;
    const auto vocabulary = ManyWordVocabulary(dir);

    ModelConfig config;
    config.dim = 32;
    XorShift rng(1234);
    const SGNSModel model(vocabulary, config, rng);

    // Output starts at zero, so every dot product must vanish.
    CHECK(dot(model.getInput().row(0), model.getOutput().row(1), config.dim) == 0.);
}

TEST_CASE("The initial loss is (negatives + 1) * ln 2") {
    const Tests::TempDir dir;
    const auto vocabulary = ManyWordVocabulary(dir);

    ModelConfig config;
    config.dim = 16;
    config.negatives = 5;
    XorShift rng(2025);
    const SGNSModel model(vocabulary, config, rng);

    const std::vector<TWordId> negatives{10, 20, 30, 40, 50};
    const double loss = model.computeLoss(Pair{5, 11}, negatives);

    CHECK(loss == doctest::Approx((config.negatives + 1) * std::log(2.)).epsilon(1e-12));
}

TEST_CASE("The learning rate decays linearly to the configured floor") {
    const Tests::TempDir dir;
    const auto vocabulary = ManyWordVocabulary(dir);

    ModelConfig config;
    config.dim = 8;
    config.initialLearningRate = 0.025;
    config.minLearningRateFactor = 1e-4;
    XorShift rng(1);
    const SGNSModel model(vocabulary, config, rng);

    constexpr std::size_t total = 1'000'000;
    CHECK(model.getLearningRateForStep(0, total) == doctest::Approx(0.025));
    CHECK(model.getLearningRateForStep(total, total)
          == doctest::Approx(0.025 * config.minLearningRateFactor));
    CHECK(model.getLearningRateForStep(total / 2, total) == doctest::Approx(0.025 * 0.50005));

    // Never increases, never runs past the floor.
    double previous = model.getLearningRateForStep(0, total);
    for (std::size_t done = 0; done <= total; done += total / 20) {
        const double rate = model.getLearningRateForStep(done, total);
        CHECK(rate <= previous + 1e-12);
        CHECK(rate > 0.);
        previous = rate;
    }
    // Clamped past the end.
    CHECK(model.getLearningRateForStep(total * 2, total)
          == doctest::Approx(model.getLearningRateForStep(total, total)));
}

TEST_CASE("The analytic gradient matches a numeric central difference") {
    const Tests::TempDir dir;
    const auto vocabulary = ManyWordVocabulary(dir);

    ModelConfig config;
    config.dim = 16;
    config.negatives = 3;
    XorShift rng(777);
    SGNSModel model(vocabulary, config, rng);
    const NegativeSampler sampler(vocabulary, kTableSize);

    std::vector<TFloat> gradient(config.dim, TFloat{0});
    WorkerContext context(config, 555);

    const Pair pair{7, 42};
    const std::vector<TWordId> negatives{13, 20, 99};

    // Move off the all-zero initialisation, where every gradient is degenerate.
    for (std::size_t i = 0; i < 30; ++i) {
        model.applyUpdate(pair, negatives, 0.05, gradient);
    }
    for (std::size_t i = 0; i < 200; ++i) {
        const Pair warmup{
            static_cast<TWordId>(rng.nextInteger(vocabulary.getSize())),
            static_cast<TWordId>(rng.nextInteger(vocabulary.getSize()))
        };
        model.trainPair(warmup, 0.05, sampler, context);
    }

    struct Target { bool isInput; TWordId row; const char* name; };
    const std::vector<Target> targets{
        {true,  pair.center,  "W_in[centre]"},
        {false, pair.context, "W_out[context]"},
        {false, negatives[0], "W_out[negative 0]"},
        {false, negatives[2], "W_out[negative 2]"},
    };

    double worst = 0.;
    for (const auto& target : targets) {
        CAPTURE(target.name);
        for (std::size_t component = 0; component < 3; ++component) {
            CAPTURE(component);
            const double analytic = AnalyticGradient(
                model, pair, negatives, gradient, target.isInput, target.row, component);
            const double numeric = NumericGradient(
                model, pair, negatives, target.isInput, target.row, component);

            const double scale = std::max({std::abs(analytic), std::abs(numeric), 1e-8});
            const double relative = std::abs(analytic - numeric) / scale;
            worst = std::max(worst, relative);
            CHECK(relative < 1e-2);
        }
    }
    // The residual is dominated by the 1000-entry sigmoid lookup table used by
    // the update path, while computeLoss evaluates log-sigmoid exactly.
    CHECK(worst < 1e-2);
}

TEST_CASE("Overfitting a single pair drives its loss down monotonically") {
    const Tests::TempDir dir;
    const auto vocabulary = ManyWordVocabulary(dir);

    ModelConfig config;
    config.dim = 50;
    config.negatives = 5;
    XorShift rng(2025);
    SGNSModel model(vocabulary, config, rng);

    const Pair pair{5, 11};
    const std::vector<TWordId> negatives{100, 200, 250, 40, 50};

    const double initial = model.computeLoss(pair, negatives);
    double previous = initial;

    std::vector<TFloat> gradient(config.dim, TFloat{0});
    for (std::size_t step = 1; step <= 50; ++step) {
        model.applyUpdate(pair, negatives, 0.05, gradient);
        const double loss = model.computeLoss(pair, negatives);
        CAPTURE(step);
        CHECK(loss <= previous);
        previous = loss;
    }
    CHECK(previous < initial);
}

TEST_CASE("trainPair never picks the context word as its own negative") {
    const Tests::TempDir dir;
    const auto vocabulary = ManyWordVocabulary(dir);

    ModelConfig config;
    config.dim = 8;
    config.negatives = 5;
    XorShift rng(9);
    const SGNSModel model(vocabulary, config, rng);
    const NegativeSampler sampler(vocabulary, kTableSize);

    WorkerContext context(config, 12345);
    const Pair pair{3, 4};
    for (std::size_t i = 0; i < 200; ++i) {
        model.trainPair(pair, 0.01, sampler, context);
    }
    // The negatives buffer holds the most recent draw.
    for (const auto id : context.negatives) {
        CHECK(id < vocabulary.getSize());
    }
}
