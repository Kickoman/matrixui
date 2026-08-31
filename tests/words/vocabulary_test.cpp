#include <doctest/doctest.h>

#include "core/words/vocabulary.h"
#include "tests/support/fixtures.h"
#include "tests/support/temp_dir.h"

using namespace Words;

TEST_CASE("Vocabulary::Build counts tokens and orders ids by frequency") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);

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
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 5);

    // e and f occur twice, d four times -- all below 5.
    CHECK(vocabulary.getSize() == 3);
    CHECK(vocabulary.getId("d") == std::nullopt);
    CHECK(vocabulary.getId("e") == std::nullopt);
    CHECK(vocabulary.getId("a").has_value());
}

TEST_CASE("Vocabulary::getId returns nullopt for an unknown word") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir);

    CHECK(vocabulary.getId("definitely-not-present") == std::nullopt);
    REQUIRE(vocabulary.getId("a").has_value());
    CHECK(vocabulary.getWord(*vocabulary.getId("a")) == "a");
}

TEST_CASE("Vocabulary token accounting distinguishes raw from kept") {
    const Tests::TempDir dir;
    const auto all = Tests::ToyVocabulary(dir, 1);
    CHECK(all.getRawTokens() == 64);
    CHECK(all.getKeptTokens() == 64);

    const auto pruned = Tests::ToyVocabulary(dir, 5);
    CHECK(pruned.getRawTokens() == 64);
    // a + b + c survive: 32 + 16 + 8.
    CHECK(pruned.getKeptTokens() == 56);
}

TEST_CASE("Vocabulary frequencies sum to one") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);

    double total = 0.;
    for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
        total += vocabulary.getFrequency(id);
    }
    CHECK(total == doctest::Approx(1.).epsilon(1e-9));
}

TEST_CASE("Vocabulary survives a Save/Load round-trip") {
    const Tests::TempDir dir;
    const auto original = Tests::ToyVocabulary(dir, 1);

    const auto path = dir.file("round.voc");
    Vocabulary::Save(original, path);
    const auto loaded = Vocabulary::Load(path);

    REQUIRE(loaded.getSize() == original.getSize());
    CHECK(loaded.getRawTokens() == original.getRawTokens());
    CHECK(loaded.getKeptTokens() == original.getKeptTokens());
    for (TWordId id = 0; id < original.getSize(); ++id) {
        CHECK(loaded.getWord(id) == original.getWord(id));
        CHECK(loaded.getCount(id) == original.getCount(id));
    }
}
