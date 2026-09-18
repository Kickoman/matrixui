#include <doctest/doctest.h>

#include "core/lib/random.h"
#include "core/words/train/model.h"
#include "core/words/train/negativesampler.h"
#include "core/words/data/vocabulary.h"
#include "tests/support/fixtures.h"

#include <algorithm>
#include <cmath>

using namespace Words;

namespace {

constexpr std::size_t kTableSize = 100'000;

Vocabulary ManyWordVocabulary(const std::size_t distinct = 300) {
    return Tests::VocabularyFromText(Tests::ZipfCorpusText(distinct, 60'000), 1);
}

// Gradient-check helpers. applyUpdate mutates the centre row and every touched
// output row, so each probe has to run against a saved copy and then restore it.

using RowSnapshot = std::vector<std::vector<TFloat>>;

// Every row applyUpdate is allowed to touch: the centre word, the centre's
// n-gram rows, the context and the negatives. Missing any of the n-gram rows
// would let them drift between probes.
RowSnapshot SaveRows(const SGNSModel& model, const Pair& pair, const std::vector<TWordId>& negatives) {
    const auto dim = model.getConfig().dim;
    const auto buckets = model.getSubwordTable().getSubwords(pair.center);
    RowSnapshot saved;
    saved.reserve(negatives.size() + buckets.size() + 2);

    const auto* center = model.getInput().row(pair.center);
    saved.emplace_back(center, center + dim);
    const auto* context = model.getOutput().row(pair.context);
    saved.emplace_back(context, context + dim);
    for (const auto id : negatives) {
        const auto* negative = model.getOutput().row(id);
        saved.emplace_back(negative, negative + dim);
    }
    for (const auto bucket : buckets) {
        const auto* row = model.getSubwordInput().row(bucket);
        saved.emplace_back(row, row + dim);
    }
    return saved;
}

void RestoreRows(SGNSModel& model, const Pair& pair, const std::vector<TWordId>& negatives, const RowSnapshot& saved) {
    std::copy(saved[0].begin(), saved[0].end(), model.getInputMutable().row(pair.center));
    std::copy(saved[1].begin(), saved[1].end(), model.getOutputMutable().row(pair.context));
    for (std::size_t i = 0; i < negatives.size(); ++i) {
        std::copy(saved[i + 2].begin(), saved[i + 2].end(), model.getOutputMutable().row(negatives[i]));
    }
    const auto buckets = model.getSubwordTable().getSubwords(pair.center);
    for (std::size_t i = 0; i < buckets.size(); ++i) {
        const auto& row = saved[negatives.size() + 2 + i];
        std::copy(row.begin(), row.end(), model.getSubwordInputMutable().row(buckets[i]));
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
    const auto vocabulary = ManyWordVocabulary();

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
    const auto vocabulary = ManyWordVocabulary();

    ModelConfig config;
    config.dim = 32;
    XorShift rng(1234);
    const SGNSModel model(vocabulary, config, rng);

    // Output starts at zero, so every dot product must vanish.
    CHECK(dot(model.getInput().row(0), model.getOutput().row(1), config.dim) == 0.);
}

TEST_CASE("The initial loss is (negatives + 1) * ln 2") {
    const auto vocabulary = ManyWordVocabulary();

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
    const auto vocabulary = ManyWordVocabulary();

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
    const auto vocabulary = ManyWordVocabulary();

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
    const auto vocabulary = ManyWordVocabulary();

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
    const auto vocabulary = ManyWordVocabulary();

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

namespace {

constexpr std::size_t kBuckets = 1000;

ModelConfig SubwordConfig(const std::size_t dim, const std::size_t negatives = 5) {
    ModelConfig config;
    config.dim = dim;
    config.negatives = negatives;
    config.minN = 3;
    config.maxN = 6;
    config.buckets = kBuckets;
    return config;
}

enum class Matrix { Input, Output, Subword };

const TFloat* RowOf(const SGNSModel& model, const Matrix matrix, const TWordId row) {
    switch (matrix) {
        case Matrix::Input: return model.getInput().row(row);
        case Matrix::Output: return model.getOutput().row(row);
        case Matrix::Subword: return model.getSubwordInput().row(row);
    }
    return nullptr;
}

// The step the optimiser takes on one row, divided by the learning rate, for
// any of the three matrices.
double AnalyticStep(
    SGNSModel& model, const Pair& pair, const std::vector<TWordId>& negatives,
    std::vector<TFloat>& gradient, const Matrix matrix, const TWordId row, const std::size_t component
) {
    constexpr double step = 1e-3;
    const auto saved = SaveRows(model, pair, negatives);

    const auto before = RowOf(model, matrix, row)[component];
    model.applyUpdate(pair, negatives, step, gradient);
    const auto after = RowOf(model, matrix, row)[component];

    RestoreRows(model, pair, negatives, saved);
    return (static_cast<double>(before) - static_cast<double>(after)) / step;
}

// The rows that make up one centre vector, each listed once however many times
// it appears in the n-gram list.
std::vector<TFloat*> CentreRows(SGNSModel& model, const TWordId center) {
    std::vector<TFloat*> rows{model.getInputMutable().row(center)};
    for (const auto bucket : model.getSubwordTable().getSubwords(center)) {
        auto* row = model.getSubwordInputMutable().row(bucket);
        if (std::find(rows.begin(), rows.end(), row) == rows.end()) {
            rows.push_back(row);
        }
    }
    return rows;
}

// d(loss) / d(h[component]): shift every row that feeds h, so the composed
// vector itself moves by eps regardless of how many rows there are.
double NumericGradientOfHidden(
    SGNSModel& model, const Pair& pair, const std::vector<TWordId>& negatives, const std::size_t component
) {
    constexpr double eps = 1e-3;
    const auto rows = CentreRows(model, pair.center);

    std::vector<TFloat> original;
    original.reserve(rows.size());
    for (auto* row : rows) {
        original.push_back(row[component]);
    }

    const auto shift = [&](const double delta) {
        for (std::size_t i = 0; i < rows.size(); ++i) {
            rows[i][component] = static_cast<TFloat>(static_cast<double>(original[i]) + delta);
        }
    };

    shift(eps);
    const double plus = model.computeLoss(pair, negatives);
    shift(-eps);
    const double minus = model.computeLoss(pair, negatives);
    shift(0.);

    return (plus - minus) / (2. * eps);
}

void WarmUp(SGNSModel& model, const Vocabulary& vocabulary, const NegativeSampler& sampler,
            const Pair& pair, const std::vector<TWordId>& negatives, XorShift& rng) {
    // W_out starts at zero, where every gradient is degenerate and a check
    // would compare zeros with zeros.
    std::vector<TFloat> gradient(model.getConfig().dim, TFloat{0});
    WorkerContext context(model.getConfig(), 555);
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
}

}  // namespace

TEST_CASE("The centre vector is the mean of the word row and its n-gram rows") {
    const auto vocabulary = ManyWordVocabulary();
    const auto config = SubwordConfig(8);
    XorShift rng(4242);
    const SGNSModel model(vocabulary, config, rng);

    const TWordId id = 17;
    const auto buckets = model.getSubwordTable().getSubwords(id);
    REQUIRE(buckets.size() > 0);

    std::vector<double> expected(config.dim, 0.);
    for (std::size_t i = 0; i < config.dim; ++i) {
        expected[i] = model.getInput().row(id)[i];
    }
    for (const auto bucket : buckets) {
        for (std::size_t i = 0; i < config.dim; ++i) {
            expected[i] += model.getSubwordInput().row(bucket)[i];
        }
    }
    for (auto& value : expected) {
        value /= 1. + buckets.size();
    }

    std::vector<TFloat> hidden;
    model.composeInto(id, hidden);
    REQUIRE(hidden.size() == config.dim);
    for (std::size_t i = 0; i < config.dim; ++i) {
        CHECK(hidden[i] == doctest::Approx(expected[i]).epsilon(1e-6));
    }
}

TEST_CASE("The analytic gradient matches a numeric central difference with subwords") {
    const auto vocabulary = ManyWordVocabulary();
    const auto config = SubwordConfig(16, 3);
    XorShift rng(777);
    SGNSModel model(vocabulary, config, rng);
    const NegativeSampler sampler(vocabulary, kTableSize);

    const Pair pair{7, 42};
    const std::vector<TWordId> negatives{13, 20, 99};
    const auto buckets = model.getSubwordTable().getSubwords(pair.center);
    REQUIRE(buckets.size() >= 3);

    WarmUp(model, vocabulary, sampler, pair, negatives, rng);

    std::vector<TFloat> gradient(config.dim, TFloat{0});

    SUBCASE("rows that feed the centre") {
        // Every such row takes the whole gradient of h, fastText's convention:
        // the composed vector then moves exactly as a plain word2vec row would.
        struct Target { bool isSubword; TWordId row; const char* name; };
        const std::vector<Target> targets{
            {false, pair.center, "W_in[centre word]"},
            {true, buckets[0], "W_sub[first n-gram]"},
            {true, buckets[buckets.size() - 1], "W_sub[last n-gram]"},
        };

        double worst = 0.;
        double largest = 0.;
        for (const auto& target : targets) {
            CAPTURE(std::string(target.name));
            // A bucket that repeats inside the word takes the gradient once per
            // occurrence, which would scale the step.
            const auto occurrences = target.isSubword
                ? static_cast<double>(std::count(buckets.begin(), buckets.end(), target.row))
                : 1.;

            for (std::size_t component = 0; component < 3; ++component) {
                CAPTURE(component);
                const double analytic = AnalyticStep(
                    model, pair, negatives, gradient,
                    target.isSubword ? Matrix::Subword : Matrix::Input, target.row, component)
                    / occurrences;
                const double numeric = NumericGradientOfHidden(model, pair, negatives, component);

                const double scale = std::max({std::abs(analytic), std::abs(numeric), 1e-8});
                worst = std::max(worst, std::abs(analytic - numeric) / scale);
                largest = std::max(largest, std::abs(numeric));
                CHECK(std::abs(analytic - numeric) / scale < 1e-2);
            }
        }
        // Guard against the check passing because everything is zero.
        CHECK(largest > 1e-6);
        CHECK(worst < 1e-2);
    }

    SUBCASE("output rows") {
        double largest = 0.;
        for (const TWordId row : {pair.context, negatives[0]}) {
            for (std::size_t component = 0; component < 3; ++component) {
                CAPTURE(row);
                CAPTURE(component);
                const double analytic =
                    AnalyticGradient(model, pair, negatives, gradient, false, row, component);
                const double numeric = NumericGradient(model, pair, negatives, false, row, component);

                const double scale = std::max({std::abs(analytic), std::abs(numeric), 1e-8});
                largest = std::max(largest, std::abs(numeric));
                CHECK(std::abs(analytic - numeric) / scale < 1e-2);
            }
        }
        CHECK(largest > 1e-6);
    }
}

TEST_CASE("Every row feeding the centre moves by the same gradient") {
    const auto vocabulary = ManyWordVocabulary();
    const auto config = SubwordConfig(12, 3);
    XorShift rng(31337);
    SGNSModel model(vocabulary, config, rng);
    const NegativeSampler sampler(vocabulary, kTableSize);

    const Pair pair{5, 11};
    const std::vector<TWordId> negatives{101, 202, 250};
    WarmUp(model, vocabulary, sampler, pair, negatives, rng);

    const auto buckets = model.getSubwordTable().getSubwords(pair.center);
    const auto before = SaveRows(model, pair, negatives);

    std::vector<TFloat> gradient(config.dim, TFloat{0});
    model.applyUpdate(pair, negatives, 0.05, gradient);

    const auto wordDelta = [&](const std::size_t i) {
        return static_cast<double>(before[0][i]) - model.getInput().row(pair.center)[i];
    };

    double moved = 0.;
    for (std::size_t i = 0; i < config.dim; ++i) {
        moved = std::max(moved, std::abs(wordDelta(i)));
    }
    CHECK(moved > 1e-9);

    for (std::size_t b = 0; b < buckets.size(); ++b) {
        CAPTURE(b);
        const auto occurrences =
            static_cast<double>(std::count(buckets.begin(), buckets.end(), buckets[b]));
        const auto& saved = before[negatives.size() + 2 + b];
        for (std::size_t i = 0; i < config.dim; ++i) {
            const double delta =
                static_cast<double>(saved[i]) - model.getSubwordInput().row(buckets[b])[i];
            CHECK(delta == doctest::Approx(occurrences * wordDelta(i)).epsilon(1e-5));
        }
    }
}

TEST_CASE("A vocabulary whose words are all shorter than minN trains like no subwords at all") {
    const auto vocabulary = ManyWordVocabulary();

    ModelConfig withSubwords = SubwordConfig(8, 3);
    withSubwords.minN = 40;   // longer than any word in the fixture
    withSubwords.maxN = 44;

    ModelConfig without = withSubwords;
    without.buckets = 0;

    const NegativeSampler sampler(vocabulary, kTableSize);
    const auto run = [&](const ModelConfig& config) {
        XorShift rng(20260918);
        SGNSModel model(vocabulary, config, rng);
        WorkerContext context(config, 99);
        XorShift pairs(7);
        for (std::size_t i = 0; i < 500; ++i) {
            const Pair pair{
                static_cast<TWordId>(pairs.nextInteger(vocabulary.getSize())),
                static_cast<TWordId>(pairs.nextInteger(vocabulary.getSize()))
            };
            model.trainPair(pair, 0.05, sampler, context);
        }
        std::vector<TFloat> flat;
        for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
            flat.insert(flat.end(), model.getInput().row(id), model.getInput().row(id) + config.dim);
        }
        return flat;
    };

    CHECK(run(withSubwords) == run(without));
}

TEST_CASE("composeWords materialises exactly what the hot path composes") {
    const auto vocabulary = ManyWordVocabulary();
    const auto config = SubwordConfig(16, 3);
    XorShift rng(2026);
    SGNSModel model(vocabulary, config, rng);
    const NegativeSampler sampler(vocabulary, kTableSize);

    WorkerContext context(config, 5);
    for (std::size_t i = 0; i < 500; ++i) {
        const Pair pair{
            static_cast<TWordId>(rng.nextInteger(vocabulary.getSize())),
            static_cast<TWordId>(rng.nextInteger(vocabulary.getSize()))
        };
        model.trainPair(pair, 0.05, sampler, context);
    }

    const auto composed = model.composeWords();
    REQUIRE(composed.getWords() == vocabulary.getSize());
    REQUIRE(composed.getDim() == config.dim);

    std::vector<TFloat> hidden;
    for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
        CAPTURE(id);
        model.composeInto(id, hidden);
        // Bit for bit: both paths add the same rows in the same order.
        CHECK(std::equal(hidden.begin(), hidden.end(), composed.row(id)));
    }
}

TEST_CASE("Without buckets the composed matrix is the input matrix unchanged") {
    const auto vocabulary = ManyWordVocabulary();
    ModelConfig config;
    config.dim = 8;
    XorShift rng(11);
    const SGNSModel model(vocabulary, config, rng);

    const auto composed = model.composeWords();
    for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
        CHECK(std::equal(composed.row(id), composed.row(id) + config.dim, model.getInput().row(id)));
    }
    CHECK(model.getSubwordInput().getWords() == 0);
    CHECK(model.getBytes() == 2 * vocabulary.getSize() * config.dim * sizeof(TFloat));
}

TEST_CASE("Enabling subwords leaves the word matrix initialisation untouched") {
    const auto vocabulary = ManyWordVocabulary();

    ModelConfig plain;
    plain.dim = 8;
    ModelConfig subwords = SubwordConfig(8);

    XorShift first(1234);
    const SGNSModel plainModel(vocabulary, plain, first);
    XorShift second(1234);
    const SGNSModel subwordModel(vocabulary, subwords, second);

    // The word rows are drawn before the bucket rows, so the same seed gives
    // the same words either way and runs stay comparable.
    for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
        CHECK(std::equal(plainModel.getInput().row(id), plainModel.getInput().row(id) + plain.dim,
                         subwordModel.getInput().row(id)));
    }

    REQUIRE(subwordModel.getSubwordInput().getWords() == kBuckets);
    CHECK(!std::equal(subwordModel.getSubwordInput().row(0),
                      subwordModel.getSubwordInput().row(0) + subwords.dim,
                      subwordModel.getSubwordInput().row(1)));
    CHECK(subwordModel.getBytes()
          == (2 * vocabulary.getSize() + kBuckets) * subwords.dim * sizeof(TFloat));
}

TEST_CASE("Composing into an undersized buffer grows it instead of running off the end") {
    const auto vocabulary = ManyWordVocabulary();
    const auto config = SubwordConfig(16);
    XorShift rng(808);
    const SGNSModel model(vocabulary, config, rng);

    const Pair pair{3, 9};
    const std::vector<TWordId> negatives{31, 47, 55};
    REQUIRE(model.getSubwordTable().getSubwords(pair.center).size() > 0);

    std::vector<TFloat> gradient(config.dim, TFloat{0});
    std::vector<TFloat> tooSmall;
    model.applyUpdate(pair, negatives, 0.01, gradient, tooSmall);

    CHECK(tooSmall.size() >= config.dim);
}
