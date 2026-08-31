#include "core/words/diagnostics/diagnostics.h"

#include "core/lib/random.h"
#include "core/words/negativesampler.h"
#include "core/words/vocabulary.h"

#include <algorithm>
#include <cmath>

namespace Words::Diagnostics {

namespace {

using RowSnapshot = std::vector<std::vector<TFloat>>;

// applyUpdate touches the centre row and every output row it uses, so each
// probe has to run against a copy and restore it afterwards.
RowSnapshot SaveTouchedRows(
    const SGNSModel& model,
    const Pair& pair,
    const std::vector<TWordId>& negatives
) {
    const auto dim = model.getConfig().dim;
    RowSnapshot saved;
    saved.reserve(negatives.size() + 2);

    const auto* centerVector = model.getInput().row(pair.center);
    saved.emplace_back(centerVector, centerVector + dim);

    const auto* contextVector = model.getOutput().row(pair.context);
    saved.emplace_back(contextVector, contextVector + dim);

    for (const auto id : negatives) {
        const auto* negativeVector = model.getOutput().row(id);
        saved.emplace_back(negativeVector, negativeVector + dim);
    }
    return saved;
}

void RestoreTouchedRows(
    SGNSModel& model,
    const Pair& pair,
    const std::vector<TWordId>& negatives,
    const RowSnapshot& saved
) {
    std::copy(saved[0].begin(), saved[0].end(), model.getInputMutable().row(pair.center));
    std::copy(saved[1].begin(), saved[1].end(), model.getOutputMutable().row(pair.context));

    for (std::size_t i = 0; i < negatives.size(); ++i) {
        std::copy(saved[i + 2].begin(), saved[i + 2].end(), model.getOutputMutable().row(negatives[i]));
    }
}

// The step the optimiser actually takes, divided by the learning rate.
double AnalyticGradientOf(
    SGNSModel& model,
    const Pair& pair,
    const std::vector<TWordId>& negatives,
    std::vector<TFloat>& gradient,
    const bool inInputMatrix,
    const TWordId row,
    const std::size_t component
) {
    constexpr double step = 1e-3;

    const auto saved = SaveTouchedRows(model, pair, negatives);

    const auto before = inInputMatrix
        ? model.getInput().row(row)[component]
        : model.getOutput().row(row)[component];

    model.applyUpdate(pair, negatives, step, gradient);

    const auto after = inInputMatrix
        ? model.getInput().row(row)[component]
        : model.getOutput().row(row)[component];

    RestoreTouchedRows(model, pair, negatives, saved);

    return (static_cast<double>(before) - static_cast<double>(after)) / step;
}

// Central difference of computeLoss.
double NumericGradientOf(
    SGNSModel& model,
    const Pair& pair,
    const std::vector<TWordId>& negatives,
    const bool inInputMatrix,
    const TWordId row,
    const std::size_t component
) {
    constexpr double eps = 1e-3;

    auto* cell = inInputMatrix
        ? model.getInputMutable().row(row) + component
        : model.getOutputMutable().row(row) + component;

    const auto original = *cell;

    *cell = static_cast<TFloat>(static_cast<double>(original) + eps);
    const double lossPlus = model.computeLoss(pair, negatives);

    *cell = static_cast<TFloat>(static_cast<double>(original) - eps);
    const double lossMinus = model.computeLoss(pair, negatives);

    *cell = original;

    return (lossPlus - lossMinus) / (2. * eps);
}

struct Target {
    bool isInput;
    TWordId row;
    const char* name;
};

}  // namespace

GradientReport CheckGradients(const Vocabulary& vocabulary) {
    ModelConfig config;
    config.dim = 16;
    config.negatives = 3;

    XorShift rng(777);
    SGNSModel model(vocabulary, config, rng);
    const NegativeSampler sampler(vocabulary);

    std::vector<TFloat> gradient(config.dim, TFloat{0});
    WorkerContext context(config, 555);

    const Pair pair{7, 42};
    const std::vector<TWordId> negatives = {13, 200, 999};

    // Move away from the all-zero initialisation, where the gradients are
    // degenerate and the comparison would be meaningless.
    for (std::size_t i = 0; i < 30; ++i) {
        model.applyUpdate(pair, negatives, 0.05, gradient);
    }

    for (std::size_t i = 0; i < 200; ++i) {
        const Pair warmup{
            static_cast<TWordId>(rng.nextInteger(1000)),
            static_cast<TWordId>(rng.nextInteger(1000))
        };
        model.trainPair(warmup, 0.05, sampler, context);
    }

    const std::vector<Target> targets = {
        {true,  pair.center,  "W_in[c]"},
        {false, pair.context, "W_out[o]"},
        {false, negatives[0], "W_out[n0]"},
        {false, negatives[2], "W_out[n2]"}
    };

    GradientReport report;

    for (const auto& target : targets) {
        for (std::size_t component = 0; component < 3; ++component) {
            const double analytic = AnalyticGradientOf(
                model, pair, negatives, gradient, target.isInput, target.row, component);
            const double numeric = NumericGradientOf(
                model, pair, negatives, target.isInput, target.row, component);

            const double scale = std::max({std::abs(analytic), std::abs(numeric), 1e-8});
            const double relative = std::abs(analytic - numeric) / scale;
            report.worstRelativeError = std::max(report.worstRelativeError, relative);

            report.samples.push_back({
                target.name, target.row, component, analytic, numeric, relative,
            });
        }
    }

    // The residual is dominated by the 1000-entry sigmoid lookup table the
    // update path uses, while computeLoss evaluates log-sigmoid exactly.
    report.passed = {"worst relative error", report.worstRelativeError < 1e-2};
    return report;
}

}  // namespace Words::Diagnostics
