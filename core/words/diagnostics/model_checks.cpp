#include "core/words/diagnostics/diagnostics.h"

#include "core/lib/random.h"
#include "core/words/vocabulary.h"

#include <algorithm>
#include <cmath>

namespace Words::Diagnostics {

// Note: main_words.cpp used to carry two near-identical copies of this check
// (PerformModelInitChecks and ModelInitCheck), reached by `validate-sgns` and
// `validate-gradients` respectively. They differed only in which annotations
// they printed; this is the merged version and both commands now share it.
ModelInitReport CheckModelInit(const Vocabulary& vocabulary, const ModelConfig& config) {
    XorShift rng(1234);
    const SGNSModel model(vocabulary, config, rng);

    const auto& input = model.getInput();
    const auto& output = model.getOutput();
    const auto dim = config.dim;

    ModelInitReport report;
    report.vocabularySize = vocabulary.getSize();
    report.dim = dim;
    report.bytes = model.getBytes();

    // The output matrix starts at exactly zero.
    bool allZero = true;
    for (TWordId id = 0; id < vocabulary.getSize() && allZero; ++id) {
        const auto* row = output.row(id);
        for (std::size_t i = 0; i < dim; ++i) {
            if (row[i] != 0) {
                allZero = false;
                break;
            }
        }
    }
    report.outputAllZero = {"output all zero", allZero};

    // The input matrix is uniform on (-0.5/dim, 0.5/dim).
    double sum = 0.;
    double sumSquares = 0.;
    double maxAbs = 0.;
    std::size_t count = 0;

    for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
        const auto* row = input.row(id);
        for (std::size_t i = 0; i < dim; ++i) {
            const double value = row[i];
            sum += value;
            sumSquares += value * value;
            maxAbs = std::max(maxAbs, std::abs(value));
            ++count;
        }
    }

    report.inputMean = count > 0 ? sum / static_cast<double>(count) : 0.;
    report.inputStdDev = count > 0
        ? std::sqrt(sumSquares / static_cast<double>(count) - report.inputMean * report.inputMean)
        : 0.;
    report.expectedStdDev = (0.5 / static_cast<double>(dim)) / std::sqrt(3.);
    report.inputMaxAbs = maxAbs;
    report.bound = 0.5 / static_cast<double>(dim);
    report.boundRespected = {"bound respected", maxAbs < report.bound};

    // Rows must not be identical.
    bool distinct = true;
    for (TWordId id = 1; id < 100 && id < vocabulary.getSize(); ++id) {
        if (std::equal(input.row(0), input.row(0) + dim, input.row(id))) {
            distinct = false;
            break;
        }
    }
    report.rowsDistinct = {"rows distinct", distinct};

    // With the output matrix zeroed, every score must vanish.
    report.initialScore = vocabulary.getSize() > 1
        ? dot(input.row(0), output.row(1), dim)
        : 0.;

    for (const double progress : {0., 0.25, 0.5, 0.75, 1.}) {
        const auto processed = static_cast<std::size_t>(progress * 1'000'000);
        report.learningRateSchedule.emplace_back(
            progress, model.getLearningRateForStep(processed, 1'000'000));
    }

    return report;
}

LossBehaviourReport CheckLossBehaviour(const Vocabulary& vocabulary) {
    ModelConfig config;
    config.dim = 50;
    config.negatives = 5;

    XorShift rng(2025);
    SGNSModel model(vocabulary, config, rng);

    const Pair pair{5, 11};
    const std::vector<TWordId> negatives = {100, 200, 300, 400, 500};

    LossBehaviourReport report;
    report.initialLoss = model.computeLoss(pair, negatives);
    report.expectedInitialLoss = static_cast<double>(config.negatives + 1) * std::log(2.);
    report.initialLossMatches = {
        "initial loss",
        std::abs(report.initialLoss - report.expectedInitialLoss) < 1e-9,
    };

    double previous = report.initialLoss;
    bool monotone = true;

    std::vector<TFloat> gradient(config.dim, TFloat{0});
    for (std::size_t step = 1; step <= 50; ++step) {
        model.applyUpdate(pair, negatives, 0.05, gradient);
        const double loss = model.computeLoss(pair, negatives);
        if (loss > previous) {
            monotone = false;
        }
        if (step % 10 == 0) {
            report.lossTrace.emplace_back(step, loss);
        }
        previous = loss;
    }

    report.monotoneDecrease = {"monotone decrease", monotone};
    return report;
}

}  // namespace Words::Diagnostics
