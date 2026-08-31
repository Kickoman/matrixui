// Unit tests for the diagnostics that used to live inside main_words.cpp.
//
// While they sat in a translation unit defining main() they could not be
// linked into a test binary at all; the only safety net was the CLI snapshot
// under tests/golden. Now each check returns a struct and can be asserted on.

#include <doctest/doctest.h>

#include "core/words/corpus.h"
#include "core/words/diagnostics/diagnostics.h"
#include "core/words/report/diagnostics_report.h"
#include "core/words/vocabulary.h"
#include "tests/support/capture.h"
#include "tests/support/fixtures.h"

#include <sstream>

using namespace Words;
using namespace Words::Diagnostics;

namespace {

// The gradient and loss checks address hardcoded word ids (999, 500), so the
// vocabulary has to be at least that large.
struct BigFixture {
    Tests::TempDir dir;
    Vocabulary vocabulary;
    TCorpus corpus;

    BigFixture()
        : vocabulary(Build(dir))
        , corpus(EncodeCorpus(dir.file("big.txt"), vocabulary))
    {}

    static Vocabulary Build(const Tests::TempDir& dir) {
        const auto path = dir.write("big.txt", Tests::ZipfCorpusText(1200, 120'000));
        return Vocabulary::Build(path, 1);
    }
};

}  // namespace

TEST_CASE("CheckSubsampler summarises the effect of subsampling") {
    const BigFixture fixture;
    const auto report = CheckSubsampler(fixture.vocabulary, fixture.corpus, 1e-3);

    CHECK(report.sample == doctest::Approx(1e-3));
    CHECK(report.vocabularySize == fixture.vocabulary.getSize());
    CHECK(report.corpusSize == fixture.corpus.size());
    CHECK(report.topKeepProbabilities.size() == 10);

    // The predicted survival count should track the observed one closely.
    CHECK(static_cast<double>(report.actualCorpusLength)
          == doctest::Approx(report.expectedCorpusLength).epsilon(0.05));
    CHECK(report.actualCorpusLength <= report.corpusSize);

    CHECK(report.before.size() == 40);
    CHECK(report.after.size() == 40);
}

TEST_CASE("CheckSubsampler keeps everything when subsampling is disabled") {
    const BigFixture fixture;
    const auto report = CheckSubsampler(fixture.vocabulary, fixture.corpus, 0.);

    CHECK(report.affectedWords == 0);
    CHECK(report.actualCorpusLength == report.corpusSize);
}

TEST_CASE("CheckWindowSampler confirms the sampler invariants") {
    const BigFixture fixture;
    const auto report = CheckWindowSampler(fixture.vocabulary, fixture.corpus);

    CHECK(report.selfPair.passed);
    CHECK(report.coverage.passed);
    CHECK(report.allPassed());

    CHECK(report.toyChunk.size() == 8);
    CHECK_FALSE(report.toyPairs.empty());
    CHECK(report.pairsGenerated > 0);
    CHECK(report.topCenters.size() == 10);

    // Edge effects mean the real count stays under the estimate.
    CHECK(report.pairsGenerated < report.expectedUpperBound);
}

TEST_CASE("CheckNegativeSampler confirms coverage, distribution and exclusion") {
    const BigFixture fixture;
    // Far fewer draws than the CLI default, which does 20M twice.
    const auto report = CheckNegativeSampler(
        fixture.vocabulary, /*coverage=*/400'000, /*distribution=*/400'000, /*exclusion=*/20'000);

    CHECK(report.vocabularySize == fixture.vocabulary.getSize());
    CHECK(report.reachableWords == fixture.vocabulary.getSize());
    CHECK(report.coverage.passed);
    CHECK(report.exclusion.passed);

    CHECK(report.wordsTested > 0);
    CHECK(report.flattenedRatio < report.rawCountRatio);
}

TEST_CASE("CheckModelInit validates the freshly initialised matrices") {
    const BigFixture fixture;

    ModelConfig config;
    config.dim = 32;
    const auto report = CheckModelInit(fixture.vocabulary, config);

    CHECK(report.outputAllZero.passed);
    CHECK(report.boundRespected.passed);
    CHECK(report.rowsDistinct.passed);
    CHECK(report.allPassed());

    CHECK(report.dim == 32);
    CHECK(report.vocabularySize == fixture.vocabulary.getSize());
    CHECK(report.initialScore == doctest::Approx(0.));
    CHECK(report.bound == doctest::Approx(0.5 / 32.));
    CHECK(report.inputStdDev == doctest::Approx(report.expectedStdDev).epsilon(0.02));
    CHECK(report.learningRateSchedule.size() == 5);

    // The schedule decays.
    CHECK(report.learningRateSchedule.front().second > report.learningRateSchedule.back().second);
}

TEST_CASE("CheckGradients matches analytic against numeric gradients") {
    const BigFixture fixture;
    const auto report = CheckGradients(fixture.vocabulary);

    CHECK(report.samples.size() == 12);  // 4 targets x 3 components
    CHECK(report.passed.passed);
    CHECK(report.worstRelativeError < 1e-2);

    for (const auto& sample : report.samples) {
        CAPTURE(sample.matrix);
        CAPTURE(sample.component);
        CHECK(sample.relativeError < 1e-2);
    }
}

TEST_CASE("CheckLossBehaviour confirms the loss falls monotonically") {
    const BigFixture fixture;
    const auto report = CheckLossBehaviour(fixture.vocabulary);

    CHECK(report.initialLossMatches.passed);
    CHECK(report.monotoneDecrease.passed);
    CHECK(report.allPassed());

    CHECK(report.initialLoss == doctest::Approx(report.expectedInitialLoss).epsilon(1e-9));
    CHECK(report.lossTrace.size() == 5);  // every 10th step of 50
    CHECK(report.lossTrace.back().second < report.initialLoss);
}

TEST_CASE("Every diagnostics printer writes to the stream it is given") {
    const BigFixture fixture;

    ModelConfig config;
    config.dim = 16;

    std::ostringstream out;
    PrintSubsamplerReport(out, fixture.vocabulary,
                          CheckSubsampler(fixture.vocabulary, fixture.corpus, 1e-3));
    PrintWindowSamplerReport(out, fixture.vocabulary,
                             CheckWindowSampler(fixture.vocabulary, fixture.corpus));
    PrintModelInitReport(out, CheckModelInit(fixture.vocabulary, config));
    PrintLossBehaviourReport(out, CheckLossBehaviour(fixture.vocabulary));

    CHECK_FALSE(out.str().empty());
    CHECK(out.str().find("Top-10 keep probabilities") != std::string::npos);
    CHECK(out.str().find("=== Toy chunk ===") != std::string::npos);
    CHECK(out.str().find("=== Model init ===") != std::string::npos);
    CHECK(out.str().find("=== Loss behaviour ===") != std::string::npos);
}

TEST_CASE("Diagnostics printers leave std::cout untouched") {
    const BigFixture fixture;

    ModelConfig config;
    config.dim = 16;

    std::ostringstream out;
    std::string leaked;
    {
        Tests::CoutCapture capture;
        PrintModelInitReport(out, CheckModelInit(fixture.vocabulary, config));
        leaked = capture.str();
    }
    // The whole point of the std::ostream& seam: nothing reaches stdout.
    CHECK(leaked.empty());
    CHECK_FALSE(out.str().empty());
}
