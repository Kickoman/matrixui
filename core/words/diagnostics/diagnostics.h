#pragma once

// Structured self-checks for the words pipeline.
//
// These used to live inside core/main_words.cpp, where they printed straight to
// std::cout and could not be reached from anywhere else -- not from a GUI, and
// not from a test binary, since that translation unit defines main(). Each
// check now computes a report; printing lives in core/words/report.

#include "core/words/corpus.h"
#include "core/words/model.h"
#include "core/words/types.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace Words {

class Vocabulary;
class Subsampler;
class WindowSampler;
class NegativeSampler;

}  // namespace Words

namespace Words::Diagnostics {

// One named pass/fail verdict.
struct Check {
    std::string name;
    bool passed{false};
};

// --- subsampler -------------------------------------------------------------

struct SubsamplerReport {
    struct KeepEntry {
        TWordId id{0};
        double share{0.};   // this word's share of the corpus
        double keep{0.};    // probability the subsampler keeps it
    };

    double sample{0.};
    std::size_t vocabularySize{0};
    std::size_t affectedWords{0};
    std::size_t corpusSize{0};
    double expectedCorpusLength{0.};
    std::size_t actualCorpusLength{0};
    std::vector<KeepEntry> topKeepProbabilities;
    std::vector<TWordId> before;   // first tokens of the corpus
    std::vector<TWordId> after;    // first tokens after subsampling

    bool allPassed() const { return true; }  // informational only
};

SubsamplerReport CheckSubsampler(
    const Vocabulary& vocabulary,
    const TCorpus& corpus,
    double sample = 1e-4,
    std::size_t topWords = 10,
    std::size_t previewTokens = 40
);

// --- window sampler ---------------------------------------------------------

struct WindowSamplerReport {
    std::vector<TWordId> toyChunk;
    std::vector<Pair> toyPairs;

    Check selfPair;    // no token is its own context
    Check coverage;    // every token appears as a centre

    std::size_t pairsGenerated{0};
    std::size_t expectedUpperBound{0};
    std::vector<std::pair<std::size_t, TWordId>> topCenters;

    bool allPassed() const { return selfPair.passed && coverage.passed; }
};

WindowSamplerReport CheckWindowSampler(
    const Vocabulary& vocabulary,
    const TCorpus& corpus,
    std::size_t window = 5,
    double sample = 1e-4
);

// --- negative sampler -------------------------------------------------------

struct NegativeSamplerReport {
    std::size_t tableSize{0};
    std::size_t vocabularySize{0};
    std::size_t reachableWords{0};
    Check coverage;

    std::size_t wordsTested{0};
    double reducedChiSquare{0.};
    double tolerance{0.};
    double maxAbsZ{0.};
    double expectedMaxZ{0.};
    TWordId worstId{0};
    Check distribution;

    double rawCountRatio{0.};
    double flattenedRatio{0.};

    Check exclusion;

    bool allPassed() const { return coverage.passed && distribution.passed && exclusion.passed; }
};

NegativeSamplerReport CheckNegativeSampler(
    const Vocabulary& vocabulary,
    std::size_t coverageDraws = 20'000'000,
    std::size_t distributionDraws = 20'000'000,
    std::size_t exclusionDraws = 1'000'000
);

// --- model initialisation ---------------------------------------------------

struct ModelInitReport {
    std::size_t vocabularySize{0};
    std::size_t dim{0};
    std::size_t bytes{0};

    Check outputAllZero;
    Check boundRespected;
    Check rowsDistinct;

    double inputMean{0.};
    double inputStdDev{0.};
    double expectedStdDev{0.};
    double inputMaxAbs{0.};
    double bound{0.};
    double initialScore{0.};

    std::vector<std::pair<double, double>> learningRateSchedule;  // progress -> rate

    bool allPassed() const {
        return outputAllZero.passed && boundRespected.passed && rowsDistinct.passed;
    }
};

ModelInitReport CheckModelInit(const Vocabulary& vocabulary, const ModelConfig& config);

// --- gradients --------------------------------------------------------------

struct GradientSample {
    std::string matrix;
    TWordId row{0};
    std::size_t component{0};
    double analytic{0.};
    double numeric{0.};
    double relativeError{0.};
};

struct GradientReport {
    std::vector<GradientSample> samples;
    double worstRelativeError{0.};
    Check passed;

    bool allPassed() const { return passed.passed; }
};

GradientReport CheckGradients(const Vocabulary& vocabulary);

// --- loss behaviour ---------------------------------------------------------

struct LossBehaviourReport {
    double initialLoss{0.};
    double expectedInitialLoss{0.};
    Check initialLossMatches;
    Check monotoneDecrease;
    std::vector<std::pair<std::size_t, double>> lossTrace;  // step -> loss

    bool allPassed() const { return initialLossMatches.passed && monotoneDecrease.passed; }
};

LossBehaviourReport CheckLossBehaviour(const Vocabulary& vocabulary);

}  // namespace Words::Diagnostics
