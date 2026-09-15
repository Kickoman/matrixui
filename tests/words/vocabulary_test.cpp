#include <doctest/doctest.h>

#include "core/words/data/vocabulary.h"
#include "tests/support/fixtures.h"
#include <cstdint>
#include <sstream>

using namespace Words;

namespace {

std::uint64_t ReadU64At(const std::string& bytes, const std::size_t offset) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[offset + i]))
                 << (8 * i);
    }
    return value;
}

}

TEST_CASE("Vocabulary::Build counts tokens and orders ids by frequency") {
    const auto vocabulary = Tests::ToyVocabulary(1);

    REQUIRE(vocabulary.getSize() == 6);

    // Fixture frequencies: a=32, b=16, c=8, d=4, e=2, f=2.
    CHECK(vocabulary.getWord(0) == "a");
    CHECK(vocabulary.getCount(0) == 32);
    CHECK(vocabulary.getWord(1) == "b");
    CHECK(vocabulary.getCount(1) == 16);
    CHECK(vocabulary.getWord(2) == "c");
    CHECK(vocabulary.getCount(2) == 8);
    CHECK(vocabulary.getWord(3) == "d");
    CHECK(vocabulary.getCount(3) == 4);

    // Counts are non-increasing across ids.
    for (TWordId id = 1; id < vocabulary.getSize(); ++id) {
        CHECK(vocabulary.getCount(id) <= vocabulary.getCount(id - 1));
    }
}

TEST_CASE("Vocabulary::Build drops words below minCount") {
    const auto vocabulary = Tests::ToyVocabulary(5);

    // e and f occur twice, d four times -- all below 5.
    CHECK(vocabulary.getSize() == 3);
    CHECK(vocabulary.getId("d") == std::nullopt);
    CHECK(vocabulary.getId("e") == std::nullopt);
    CHECK(vocabulary.getId("a").has_value());
}

TEST_CASE("Vocabulary::getId returns nullopt for an unknown word") {
    const auto vocabulary = Tests::ToyVocabulary();

    CHECK(vocabulary.getId("definitely-not-present") == std::nullopt);
    REQUIRE(vocabulary.getId("a").has_value());
    CHECK(vocabulary.getWord(*vocabulary.getId("a")) == "a");
}

TEST_CASE("Vocabulary token accounting distinguishes raw from kept") {
    const auto all = Tests::ToyVocabulary(1);
    CHECK(all.getRawTokens() == 64);
    CHECK(all.getKeptTokens() == 64);

    const auto pruned = Tests::ToyVocabulary(5);
    CHECK(pruned.getRawTokens() == 64);
    // a + b + c survive: 32 + 16 + 8.
    CHECK(pruned.getKeptTokens() == 56);
}

TEST_CASE("Vocabulary frequencies sum to one") {
    const auto vocabulary = Tests::ToyVocabulary(1);

    double total = 0.;
    for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
        total += vocabulary.getFrequency(id);
    }
    CHECK(total == doctest::Approx(1.).epsilon(1e-9));
}

TEST_CASE("Vocabulary survives a Save/Load round-trip") {
    const auto original = Tests::ToyVocabulary(1);

    std::stringstream stream;
    Vocabulary::Save(stream, original);
    const auto loaded = Vocabulary::Load(stream);

    REQUIRE(loaded.getSize() == original.getSize());
    CHECK(loaded.getRawTokens() == original.getRawTokens());
    CHECK(loaded.getKeptTokens() == original.getKeptTokens());
    for (TWordId id = 0; id < original.getSize(); ++id) {
        CHECK(loaded.getWord(id) == original.getWord(id));
        CHECK(loaded.getCount(id) == original.getCount(id));
    }
}

TEST_CASE("The .voc byte layout is frozen: size, raw, kept, then length-prefixed records") {
    // Fixture frequencies: a=32, b=16, c=8, d=4, e=2, f=2; raw = kept = 64.
    const auto vocabulary = Tests::ToyVocabulary(1);

    std::ostringstream stream;
    Vocabulary::Save(stream, vocabulary);
    const auto bytes = stream.str();

    REQUIRE(bytes.size() == 24 + 6 * (8 + 1 + 8));
    CHECK(ReadU64At(bytes, 0) == 6);
    CHECK(ReadU64At(bytes, 8) == 64);
    CHECK(ReadU64At(bytes, 16) == 64);
    CHECK(ReadU64At(bytes, 24) == 1);
    CHECK(bytes[32] == 'a');
    CHECK(ReadU64At(bytes, 33) == 32);
    CHECK(ReadU64At(bytes, 41) == 1);
    CHECK(bytes[49] == 'b');
    CHECK(ReadU64At(bytes, 50) == 16);
}

TEST_CASE("Words with equal counts are ordered alphabetically") {
    // Four words at the same count, written in an order that is neither
    // alphabetical nor its reverse, so a stable sort alone cannot pass this.
    const auto vocabulary =
        Tests::VocabularyFromText("pear apple fig banana pear apple fig banana", 1);

    REQUIRE(vocabulary.getSize() == 4);
    for (TWordId id = 0; id < 4; ++id) {
        CHECK(vocabulary.getCount(id) == 2);
    }

    CHECK(vocabulary.getWord(0) == "apple");
    CHECK(vocabulary.getWord(1) == "banana");
    CHECK(vocabulary.getWord(2) == "fig");
    CHECK(vocabulary.getWord(3) == "pear");
}

TEST_CASE("Count still outranks the alphabetical tiebreak") {
    const auto vocabulary = Tests::VocabularyFromText("zebra zebra zebra apple apple mango", 1);

    REQUIRE(vocabulary.getSize() == 3);
    CHECK(vocabulary.getWord(0) == "zebra");   // 3
    CHECK(vocabulary.getWord(1) == "apple");   // 2
    CHECK(vocabulary.getWord(2) == "mango");   // 1
}

TEST_CASE("A huge prune threshold leaves the build identical") {
    const auto text = Tests::ZipfCorpusText(200, 5'000);
    std::istringstream plain(text);
    std::istringstream guarded(text);

    const auto expected = Vocabulary::Build(plain, 1);
    VocabularyBuildStats stats;
    const auto actual = Vocabulary::Build(guarded, 1, 1'000'000, &stats);

    CHECK(stats.pruneRuns == 0);
    CHECK(stats.finalMinReduce == 0);
    REQUIRE(actual.getSize() == expected.getSize());
    CHECK(actual.getRawTokens() == expected.getRawTokens());
    CHECK(actual.getKeptTokens() == expected.getKeptTokens());
    for (TWordId id = 0; id < expected.getSize(); ++id) {
        CHECK(actual.getWord(id) == expected.getWord(id));
        CHECK(actual.getCount(id) == expected.getCount(id));
    }
}

TEST_CASE("Pruning evicts words that stay rare") {
    // After w4 the table holds 5 distinct words, crossing the threshold of 4:
    // everything with count <= 1 (w1..w4) is evicted; w5 and w6 arrive later.
    std::istringstream dump("a a a a a a a a a a w1 w2 w3 w4 w5 w6");
    VocabularyBuildStats stats;
    const auto vocabulary = Vocabulary::Build(dump, 1, 4, &stats);

    CHECK(stats.pruneRuns == 1);
    CHECK(stats.finalMinReduce == 1);
    REQUIRE(vocabulary.getSize() == 3);
    CHECK(vocabulary.getWord(0) == "a");
    CHECK(vocabulary.getCount(0) == 10);
    CHECK(vocabulary.getId("w1") == std::nullopt);
    CHECK(vocabulary.getId("w5").has_value());
    CHECK(vocabulary.getId("w6").has_value());
    CHECK(vocabulary.getRawTokens() == 16);
    CHECK(vocabulary.getKeptTokens() == 12);
}

TEST_CASE("The reduce counter grows with every prune run") {
    // First prune drops count <= 1 (w1..w4); the second drops count <= 2,
    // taking b (seen twice) with it. Only "a" and the late "x4" remain.
    std::istringstream dump("a a a a a a a a a a w1 w2 w3 w4 b b x1 x2 x3 x4");
    VocabularyBuildStats stats;
    const auto vocabulary = Vocabulary::Build(dump, 1, 4, &stats);

    CHECK(stats.pruneRuns == 2);
    CHECK(stats.finalMinReduce == 2);
    REQUIRE(vocabulary.getSize() == 2);
    CHECK(vocabulary.getWord(0) == "a");
    CHECK(vocabulary.getId("x4").has_value());
    CHECK(vocabulary.getId("b") == std::nullopt);
}

TEST_CASE("Pruning composes with the final min-count filter") {
    std::istringstream dump("a a a a a a a a a a w1 w2 w3 w4 w5 w6");
    const auto vocabulary = Vocabulary::Build(dump, 2, 4);

    REQUIRE(vocabulary.getSize() == 1);
    CHECK(vocabulary.getWord(0) == "a");
}

TEST_CASE("A zero prune threshold disables pruning") {
    std::istringstream dump("a a a a a a a a a a w1 w2 w3 w4 w5 w6");
    VocabularyBuildStats stats;
    const auto vocabulary = Vocabulary::Build(dump, 1, 0, &stats);

    CHECK(stats.pruneRuns == 0);
    CHECK(vocabulary.getSize() == 7);
}

TEST_CASE("A pruned vocabulary survives a Save/Load round-trip") {
    std::istringstream dump("a a a a a a a a a a w1 w2 w3 w4 w5 w6");
    const auto original = Vocabulary::Build(dump, 1, 4);

    std::stringstream stream;
    Vocabulary::Save(stream, original);
    const auto loaded = Vocabulary::Load(stream);

    REQUIRE(loaded.getSize() == original.getSize());
    CHECK(loaded.getRawTokens() == original.getRawTokens());
    CHECK(loaded.getKeptTokens() == original.getKeptTokens());
    for (TWordId id = 0; id < original.getSize(); ++id) {
        CHECK(loaded.getWord(id) == original.getWord(id));
        CHECK(loaded.getCount(id) == original.getCount(id));
    }
}

TEST_CASE("Building the same dump twice gives identical ids") {
    const std::string text = "one two three one two three four five";

    const auto first = Tests::VocabularyFromText(text, 1);
    const auto second = Tests::VocabularyFromText(text, 1);

    REQUIRE(first.getSize() == second.getSize());
    for (TWordId id = 0; id < first.getSize(); ++id) {
        CHECK(first.getWord(id) == second.getWord(id));
    }
}
