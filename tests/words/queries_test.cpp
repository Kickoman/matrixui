// Tests for the algorithms that used to exist only inside print functions.

#include <doctest/doctest.h>

#include "core/words/queries.h"
#include "core/words/similarity.h"
#include "core/words/vocabulary.h"
#include "tests/support/fixtures.h"

#include <cmath>
#include <numbers>

using namespace Words;

namespace {

struct Fixture {
    Tests::TempDir dir;
    Vocabulary vocabulary;
    EmbeddingIndex index;

    Fixture() : vocabulary(Tests::ToyVocabulary(dir, 1)), index(Build()) {}

    // Unit vectors at known angles; ids follow the toy vocabulary
    // (a, b, c, d, f, e -- the e/f tie resolves to f -> 4, e -> 5).
    static EmbeddingIndex Build() {
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
};

}  // namespace

TEST_CASE("An index can be built in memory without touching the filesystem") {
    const Fixture fixture;
    CHECK(fixture.index.getWords() == 6);
    CHECK(fixture.index.getDim() == 2);
    // Rows are normalised by the constructor.
    CHECK(fixture.index.similarity(0, 0) == doctest::Approx(1.).epsilon(1e-6));
}

TEST_CASE("QueryNeighbours reports a missing word as data, not an exception") {
    const Fixture fixture;
    const auto report = QueryNeighbours(fixture.vocabulary, fixture.index, "zzz", 3);

    CHECK_FALSE(report.status.ok);
    CHECK(report.status.message == "'zzz' is not in the vocabulary");
    CHECK(report.neighbours.empty());
}

TEST_CASE("QueryNeighbours returns the closest words in order") {
    const Fixture fixture;
    const auto report = QueryNeighbours(fixture.vocabulary, fixture.index, "a", 3);

    REQUIRE(report.status.ok);
    REQUIRE(report.neighbours.size() == 3);
    CHECK(report.word == "a");
    CHECK(report.neighbours[0].similarity == doctest::Approx(std::cos(10. * std::numbers::pi / 180.)).epsilon(1e-4));
    for (std::size_t i = 1; i < report.neighbours.size(); ++i) {
        CHECK(report.neighbours[i].similarity <= report.neighbours[i - 1].similarity);
    }
}

TEST_CASE("QueryAnalogy reports missing words as data") {
    const Fixture fixture;
    const auto report = QueryAnalogy(fixture.vocabulary, fixture.index, "a", "zzz", "c", 3);
    CHECK_FALSE(report.status.ok);
    CHECK(report.neighbours.empty());
}

TEST_CASE("QueryExpression computes 3CosMul only for analogy-shaped input") {
    const Fixture fixture;

    const auto analogy = QueryExpression(fixture.vocabulary, fixture.index, "b - a + c", 3);
    REQUIRE(analogy.status.ok);
    CHECK(analogy.analogyShape);
    CHECK_FALSE(analogy.cosAdd.empty());
    CHECK_FALSE(analogy.cosMul.empty());

    // A single term is not an analogy.
    const auto single = QueryExpression(fixture.vocabulary, fixture.index, "a", 3);
    REQUIRE(single.status.ok);
    CHECK_FALSE(single.analogyShape);
    CHECK(single.cosMul.empty());
}

TEST_CASE("QueryExpression surfaces a parse failure through its status") {
    const Fixture fixture;
    const auto report = QueryExpression(fixture.vocabulary, fixture.index, "b + zzz", 3);
    CHECK_FALSE(report.status.ok);
    CHECK(report.cosAdd.empty());
}

TEST_CASE("RankByCosMul returns descending scores and honours exclusions") {
    const Fixture fixture;
    const std::array<TWordId, 3> exclude{0, 1, 2};

    const auto ranked = RankByCosMul(fixture.index, 0, 1, 2, exclude, 3);
    REQUIRE_FALSE(ranked.empty());

    for (const auto& entry : ranked) {
        CHECK(entry.id != 0);
        CHECK(entry.id != 1);
        CHECK(entry.id != 2);
    }
    for (std::size_t i = 1; i < ranked.size(); ++i) {
        CHECK(ranked[i].score <= ranked[i - 1].score);
    }
}

TEST_CASE("Centroid of a single word is that word's unit vector") {
    const Fixture fixture;
    const std::array<TWordId, 1> ids{2};
    const auto centroid = Centroid(fixture.index, ids);

    REQUIRE(centroid.size() == fixture.index.getDim());
    const auto* row = fixture.index.getNormalized().row(2);
    for (std::size_t i = 0; i < centroid.size(); ++i) {
        CHECK(centroid[i] == doctest::Approx(row[i]).epsilon(1e-5));
    }
}

TEST_CASE("QueryOddOneOut needs at least three words") {
    const Fixture fixture;
    const auto report = QueryOddOneOut(fixture.vocabulary, fixture.index, "a b");
    CHECK_FALSE(report.status.ok);
    CHECK(report.status.message == "need at least three words");
}

TEST_CASE("QueryOddOneOut names the least central word") {
    const Fixture fixture;
    const auto report = QueryOddOneOut(fixture.vocabulary, fixture.index, "a b c d");

    REQUIRE(report.status.ok);
    REQUIRE(report.scored.size() == 4);
    // Sorted by descending similarity to the centroid, so the outlier is last.
    CHECK(report.oddOne == report.scored.back().id);
    for (std::size_t i = 1; i < report.scored.size(); ++i) {
        CHECK(report.scored[i].score <= report.scored[i - 1].score);
    }
}

TEST_CASE("QueryAxis with an explicit word list ranks just those words") {
    const Fixture fixture;
    const auto report = QueryAxis(fixture.vocabulary, fixture.index, "a - e", "b c d", 0, 3);

    REQUIRE(report.status.ok);
    CHECK(report.explicitWordList);
    CHECK(report.ranked.size() == 3);
    CHECK(report.positive.empty());
    CHECK(report.negative.empty());
}

TEST_CASE("QueryAxis without a word list scans and returns both ends") {
    const Fixture fixture;
    const auto report = QueryAxis(fixture.vocabulary, fixture.index, "a - e", "", 0, 2);

    REQUIRE(report.status.ok);
    CHECK_FALSE(report.explicitWordList);
    REQUIRE(report.positive.size() == 2);
    REQUIRE(report.negative.size() == 2);

    // The ends are opposite: the most positive outranks the most negative.
    CHECK(report.positive.front().score > report.negative.front().score);
    // Each end is itself ordered.
    CHECK(report.positive[0].score >= report.positive[1].score);
    CHECK(report.negative[0].score <= report.negative[1].score);
}

TEST_CASE("DefaultBatteryWords is the list the battery actually queries") {
    const Fixture fixture;
    const auto report = RunDefaultBattery(fixture.vocabulary, fixture.index);

    CHECK(report.neighbours.size() == DefaultBatteryWords().size());
    CHECK(report.analogies.size() == 3);

    // None of the battery words are in the toy vocabulary, so every lookup
    // should fail cleanly rather than throw.
    for (const auto& neighbours : report.neighbours) {
        CHECK_FALSE(neighbours.status.ok);
    }
}

TEST_CASE("SplitWords lowercases and splits on whitespace") {
    const auto words = SplitWords("  King   MAN\twoman ");
    REQUIRE(words.size() == 3);
    CHECK(words[0] == "king");
    CHECK(words[1] == "man");
    CHECK(words[2] == "woman");
}

TEST_CASE("Normalized scales a vector to unit length") {
    const auto unit = Normalized(std::vector<TFloat>{3.f, 4.f});
    REQUIRE(unit.size() == 2);
    CHECK(unit[0] == doctest::Approx(0.6));
    CHECK(unit[1] == doctest::Approx(0.8));

    // A zero vector is left alone rather than producing NaNs.
    const auto zero = Normalized(std::vector<TFloat>{0.f, 0.f});
    CHECK(zero[0] == 0.f);
    CHECK(zero[1] == 0.f);
}
