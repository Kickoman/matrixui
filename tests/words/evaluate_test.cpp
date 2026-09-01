#include <doctest/doctest.h>

#include "core/words/query/evaluate.h"
#include "core/words/query/similarity.h"
#include "core/words/data/vocabulary.h"
#include "tests/support/fixtures.h"

#include <cmath>
#include <numbers>

using namespace Words;

namespace {

// Six unit vectors in the plane at known angles, so every cosine similarity
// is exactly cos(theta_i - theta_j) and correlations can be predicted by hand.
// Word ids follow the toy vocabulary order: a, b, c, d, e, f.
EmbeddingIndex AngledIndex(const Tests::TempDir& dir) {
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

    const auto path = dir.file("angled.emb");
    Embeddings::Save(embeddings, path);
    return EmbeddingIndex::Load(path);
}

}  // namespace

TEST_CASE("EvaluateSimilarity counts pairs and skips unknown words") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const auto index = AngledIndex(dir);

    const auto path = dir.write("wordsim.txt",
        "a b 9.0\n"
        "a c 7.0\n"
        "a zzz 5.0\n"     // unknown -> skipped
        "qqq b 4.0\n"     // unknown -> skipped
        "a d 3.0\n");

    const auto report = EvaluateSimilarity(vocabulary, index, path, /*scoreColumn=*/2);

    CHECK(report.asked == 3);
    CHECK(report.skipped == 2);
}

TEST_CASE("EvaluateSimilarity reports a perfect rank correlation when orders agree") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const auto index = AngledIndex(dir);

    // Model similarity to "a" falls as the angle grows: b(10) > c(30) > d(60).
    // Human scores are given in the same order, so Spearman must be exactly 1.
    const auto path = dir.write("agree.txt",
        "a b 9.0\n"
        "a c 5.0\n"
        "a d 1.0\n");

    const auto report = EvaluateSimilarity(vocabulary, index, path, 2);

    REQUIRE(report.asked == 3);
    CHECK(report.spearman == doctest::Approx(1.).epsilon(1e-9));
}

TEST_CASE("EvaluateSimilarity reports an inverted rank correlation when orders disagree") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const auto index = AngledIndex(dir);

    const auto path = dir.write("disagree.txt",
        "a b 1.0\n"
        "a c 5.0\n"
        "a d 9.0\n");

    const auto report = EvaluateSimilarity(vocabulary, index, path, 2);

    REQUIRE(report.asked == 3);
    CHECK(report.spearman == doctest::Approx(-1.).epsilon(1e-9));
}

TEST_CASE("EvaluateSimilarity honours the score column") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const auto index = AngledIndex(dir);

    // Real score in column 3; column 2 holds a part-of-speech tag.
    const auto path = dir.write("col3.txt",
        "a b N 9.0\n"
        "a c N 5.0\n"
        "a d N 1.0\n");

    const auto report = EvaluateSimilarity(vocabulary, index, path, /*scoreColumn=*/3);
    REQUIRE(report.asked == 3);
    CHECK(report.spearman == doctest::Approx(1.).epsilon(1e-9));

    // Column 2 parses as a non-number on every line, so nothing is counted.
    const auto broken = EvaluateSimilarity(vocabulary, index, path, /*scoreColumn=*/2);
    CHECK(broken.asked == 0);
}

TEST_CASE("EvaluateSimilarity throws when the dataset is missing") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const auto index = AngledIndex(dir);
    CHECK_THROWS_AS(EvaluateSimilarity(vocabulary, index, dir.file("absent.txt"), 2), std::runtime_error);
}

TEST_CASE("EvaluateAnalogies groups questions under their categories") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const auto index = AngledIndex(dir);

    const auto path = dir.write("analogies.txt",
        ": first-category\n"
        "a b c d\n"
        "b c d e\n"
        ": gram-second\n"
        "a b c zzz\n"     // unknown -> skipped
        "c d e f\n");

    const auto report = EvaluateAnalogies(vocabulary, index, path, /*restrictTo=*/0, /*threads=*/1);

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
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const auto index = AngledIndex(dir);

    const auto path = dir.write("split.txt",
        ": capital-common-countries\n"
        "a b c d\n"
        ": gram1-adjective-to-adverb\n"
        "b c d e\n");

    const auto report = EvaluateAnalogies(vocabulary, index, path, 0, 1);

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

TEST_CASE("EvaluateAnalogies throws when the dataset is missing") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const auto index = AngledIndex(dir);
    CHECK_THROWS_AS(EvaluateAnalogies(vocabulary, index, dir.file("absent.txt"), 0, 1), std::runtime_error);
}

TEST_CASE("EvaluateSimilarity averages ranks over tied human scores") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const auto index = AngledIndex(dir);

    // Human scores tie on the first two pairs; model similarities are all
    // distinct (cos 10 > cos 30 > cos 60). Averaged ranks make the human side
    // [2.5, 2.5, 1] against a model side of [3, 2, 1], which correlates at
    // sqrt(3)/2. Assigning tied entries distinct ranks would give 1 instead.
    const auto path = dir.write("ties.txt",
        "a b 5.0\n"
        "a c 5.0\n"
        "a d 1.0\n");

    const auto report = EvaluateSimilarity(vocabulary, index, path, 2);

    REQUIRE(report.asked == 3);
    CHECK(report.spearman == doctest::Approx(0.8660254037844387).epsilon(1e-6));
}

TEST_CASE("EvaluateSimilarity reports Pearson over the raw scores") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const auto index = AngledIndex(dir);

    const auto path = dir.write("pearson.txt",
        "a b 5.0\n"
        "a c 5.0\n"
        "a d 1.0\n");

    const auto report = EvaluateSimilarity(vocabulary, index, path, 2);

    REQUIRE(report.asked == 3);
    // Raw human [5, 5, 1] against raw model [cos 10, cos 30, cos 60]. Unlike
    // Spearman this uses the magnitudes, so the two coefficients differ.
    CHECK(report.pearson == doctest::Approx(0.9719874013473091).epsilon(1e-6));
    CHECK(report.pearson != doctest::Approx(report.spearman).epsilon(1e-6));
}

TEST_CASE("EvaluateSimilarity reports zero correlation for a single pair") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const auto index = AngledIndex(dir);

    const auto path = dir.write("single.txt", "a b 9.0\n");

    const auto report = EvaluateSimilarity(vocabulary, index, path, 2);

    REQUIRE(report.asked == 1);
    CHECK(report.pearson == doctest::Approx(0.));
    CHECK(report.spearman == doctest::Approx(0.));
}

TEST_CASE("EvaluateSimilarity reports zero correlation when one side is constant") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const auto index = AngledIndex(dir);

    // Every human score is identical, so its variance is zero and the
    // correlation denominator vanishes. That must yield 0, not NaN.
    const auto path = dir.write("constant.txt",
        "a b 5.0\n"
        "a c 5.0\n"
        "a d 5.0\n");

    const auto report = EvaluateSimilarity(vocabulary, index, path, 2);

    REQUIRE(report.asked == 3);
    CHECK(report.pearson == doctest::Approx(0.));
    CHECK(report.spearman == doctest::Approx(0.));
}
