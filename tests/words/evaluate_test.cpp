#include <doctest/doctest.h>

#include "core/words/query/evaluate.h"
#include "core/words/query/similarity.h"
#include "core/words/data/vocabulary.h"
#include "tests/support/fixtures.h"

#include <cmath>
#include <numbers>
#include <sstream>

using namespace Words;

namespace {

// Six unit vectors in the plane at known angles, so every cosine similarity
// is exactly cos(theta_i - theta_j) and correlations can be predicted by hand.
// Word ids follow the toy vocabulary order: a, b, c, d, e, f.
EmbeddingIndex AngledIndex() {
    constexpr std::size_t words = 6;
    constexpr std::size_t dim = 2;
    const double degrees[words] = {0., 10., 30., 60., 90., 120.};

    Embeddings embeddings(words, dim);
    embeddings.initializeZero();
    for (TWordId id = 0; id < words; ++id) {
        const double radians = degrees[id] * std::numbers::pi / 180.;
        embeddings.row(id)[0] = static_cast<TFloat>(std::cos(radians));
        embeddings.row(id)[1] = static_cast<TFloat>(std::sin(radians));
    }

    return EmbeddingIndex(std::move(embeddings));
}

}  // namespace

TEST_CASE("EvaluateSimilarity counts pairs and skips unknown words") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const auto index = AngledIndex();

    std::istringstream file("a b 9.0\n"
        "a c 7.0\n"
        "a zzz 5.0\n"     // unknown -> skipped
        "qqq b 4.0\n"     // unknown -> skipped
        "a d 3.0\n");

    const auto report = EvaluateSimilarity(vocabulary, index, file, /*scoreColumn=*/2);

    CHECK(report.asked == 3);
    CHECK(report.skipped == 2);
}

TEST_CASE("EvaluateSimilarity reports a perfect rank correlation when orders agree") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const auto index = AngledIndex();

    // Model similarity to "a" falls as the angle grows: b(10) > c(30) > d(60).
    // Human scores are given in the same order, so Spearman must be exactly 1.
    std::istringstream file("a b 9.0\n"
        "a c 5.0\n"
        "a d 1.0\n");

    const auto report = EvaluateSimilarity(vocabulary, index, file, 2);

    REQUIRE(report.asked == 3);
    CHECK(report.spearman == doctest::Approx(1.).epsilon(1e-9));
}

TEST_CASE("EvaluateSimilarity reports an inverted rank correlation when orders disagree") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const auto index = AngledIndex();

    std::istringstream file("a b 1.0\n"
        "a c 5.0\n"
        "a d 9.0\n");

    const auto report = EvaluateSimilarity(vocabulary, index, file, 2);

    REQUIRE(report.asked == 3);
    CHECK(report.spearman == doctest::Approx(-1.).epsilon(1e-9));
}

TEST_CASE("EvaluateSimilarity honours the score column") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const auto index = AngledIndex();

    // Real score in column 3; column 2 holds a part-of-speech tag.
    const std::string dataset =
        "a b N 9.0\n"
        "a c N 5.0\n"
        "a d N 1.0\n";

    std::istringstream file(dataset);
    const auto report = EvaluateSimilarity(vocabulary, index, file, /*scoreColumn=*/3);
    REQUIRE(report.asked == 3);
    CHECK(report.spearman == doctest::Approx(1.).epsilon(1e-9));

    // Column 2 parses as a non-number on every line, so nothing is counted.
    std::istringstream reread(dataset);
    const auto broken = EvaluateSimilarity(vocabulary, index, reread, /*scoreColumn=*/2);
    CHECK(broken.asked == 0);
}

TEST_CASE("EvaluateAnalogies groups questions under their categories") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const auto index = AngledIndex();

    std::istringstream file(": first-category\n"
        "a b c d\n"
        "b c d e\n"
        ": gram-second\n"
        "a b c zzz\n"     // unknown -> skipped
        "c d e f\n");

    const auto report = EvaluateAnalogies(vocabulary, index, file, /*restrictTo=*/0, /*threads=*/1);

    REQUIRE(report.categories.size() == 2);
    CHECK(report.categories[0].name == "first-category");
    CHECK(report.categories[0].asked == 2);
    CHECK(report.categories[0].skipped == 0);

    CHECK(report.categories[1].name == "gram-second");
    CHECK(report.categories[1].asked == 1);
    CHECK(report.categories[1].skipped == 1);

    CHECK(report.overall.asked == 3);
    CHECK(report.overall.skipped == 1);
    CHECK(report.overall.total() == 4);
}

TEST_CASE("A category named gram-* counts as syntactic, others as semantic") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const auto index = AngledIndex();

    std::istringstream file(": capital-common-countries\n"
        "a b c d\n"
        ": gram1-adjective-to-adverb\n"
        "b c d e\n");

    const auto report = EvaluateAnalogies(vocabulary, index, file, 0, 1);

    CHECK(report.semantic.asked == 1);
    CHECK(report.syntactic.asked == 1);
    CHECK(report.overall.asked == 2);
}

TEST_CASE("AnalogyStats computes accuracy safely when nothing was asked") {
    AnalogyStats stats;
    CHECK(stats.asked == 0);
    CHECK(stats.accuracyAdd() == doctest::Approx(0.));
    CHECK(stats.accuracyMul() == doctest::Approx(0.));

    stats.asked = 4;
    stats.correctAdd = 1;
    stats.correctMul = 3;
    CHECK(stats.accuracyAdd() == doctest::Approx(0.25));
    CHECK(stats.accuracyMul() == doctest::Approx(0.75));
}

TEST_CASE("EvaluateSimilarity averages ranks over tied human scores") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const auto index = AngledIndex();

    // Human scores tie on the first two pairs; model similarities are all
    // distinct (cos 10 > cos 30 > cos 60). Averaged ranks make the human side
    // [2.5, 2.5, 1] against a model side of [3, 2, 1], which correlates at
    // sqrt(3)/2. Assigning tied entries distinct ranks would give 1 instead.
    std::istringstream file("a b 5.0\n"
        "a c 5.0\n"
        "a d 1.0\n");

    const auto report = EvaluateSimilarity(vocabulary, index, file, 2);

    REQUIRE(report.asked == 3);
    CHECK(report.spearman == doctest::Approx(0.8660254037844387).epsilon(1e-6));
}

TEST_CASE("EvaluateSimilarity reports Pearson over the raw scores") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const auto index = AngledIndex();

    std::istringstream file("a b 5.0\n"
        "a c 5.0\n"
        "a d 1.0\n");

    const auto report = EvaluateSimilarity(vocabulary, index, file, 2);

    REQUIRE(report.asked == 3);
    // Raw human [5, 5, 1] against raw model [cos 10, cos 30, cos 60]. Unlike
    // Spearman this uses the magnitudes, so the two coefficients differ.
    CHECK(report.pearson == doctest::Approx(0.9719874013473091).epsilon(1e-6));
    CHECK(report.pearson != doctest::Approx(report.spearman).epsilon(1e-6));
}

TEST_CASE("EvaluateSimilarity reports zero correlation for a single pair") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const auto index = AngledIndex();

    std::istringstream file("a b 9.0\n");

    const auto report = EvaluateSimilarity(vocabulary, index, file, 2);

    REQUIRE(report.asked == 1);
    CHECK(report.pearson == doctest::Approx(0.));
    CHECK(report.spearman == doctest::Approx(0.));
}

TEST_CASE("EvaluateSimilarity reports zero correlation when one side is constant") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const auto index = AngledIndex();

    // Every human score is identical, so its variance is zero and the
    // correlation denominator vanishes. That must yield 0, not NaN.
    std::istringstream file("a b 5.0\n"
        "a c 5.0\n"
        "a d 5.0\n");

    const auto report = EvaluateSimilarity(vocabulary, index, file, 2);

    REQUIRE(report.asked == 3);
    CHECK(report.pearson == doctest::Approx(0.));
    CHECK(report.spearman == doctest::Approx(0.));
}
