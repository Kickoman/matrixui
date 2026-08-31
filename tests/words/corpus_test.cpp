#include <doctest/doctest.h>

#include "core/words/corpus.h"
#include "core/words/vocabulary.h"
#include "tests/support/fixtures.h"
#include "tests/support/temp_dir.h"

using namespace Words;

TEST_CASE("EncodeCorpus maps every token to its id") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const auto text = dir.write("encode.txt", "a b a c");

    const auto corpus = EncodeCorpus(text, vocabulary);

    REQUIRE(corpus.size() == 4);
    CHECK(corpus[0] == *vocabulary.getId("a"));
    CHECK(corpus[1] == *vocabulary.getId("b"));
    CHECK(corpus[2] == *vocabulary.getId("a"));
    CHECK(corpus[3] == *vocabulary.getId("c"));
}

TEST_CASE("EncodeCorpus silently drops out-of-vocabulary tokens") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const auto text = dir.write("oov.txt", "a zzz b qqq");

    const auto corpus = EncodeCorpus(text, vocabulary);

    REQUIRE(corpus.size() == 2);
    CHECK(corpus[0] == *vocabulary.getId("a"));
    CHECK(corpus[1] == *vocabulary.getId("b"));
}

TEST_CASE("EncodeCorpus of the source text has exactly keptTokens entries") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 5);
    const auto text = dir.write("full.txt", Tests::ToyCorpusText());

    const auto corpus = EncodeCorpus(text, vocabulary);
    CHECK(corpus.size() == vocabulary.getKeptTokens());
}

TEST_CASE("EncodeCorpus throws when the input is missing") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    CHECK_THROWS_AS(EncodeCorpus(dir.file("nope.txt"), vocabulary), std::runtime_error);
}

TEST_CASE("Corpus survives a Save/Load round-trip") {
    const Tests::TempDir dir;
    const TCorpus original{4, 0, 1, 9, 2, 2, 7};

    const auto path = dir.file("round.cor");
    SaveCorpus(path, original);
    const auto loaded = LoadCorpus(path);

    CHECK(loaded == original);
}

TEST_CASE("An empty corpus round-trips") {
    const Tests::TempDir dir;
    const auto path = dir.file("empty.cor");
    SaveCorpus(path, TCorpus{});
    CHECK(LoadCorpus(path).empty());
}

TEST_CASE("LoadCorpus rejects a file that is not a corpus") {
    const Tests::TempDir dir;
    const auto path = dir.write("garbage.cor", "this is definitely not a corpus file");
    CHECK_THROWS_AS(LoadCorpus(path), std::runtime_error);
}

TEST_CASE("LoadCorpus throws when the file is missing") {
    const Tests::TempDir dir;
    CHECK_THROWS_AS(LoadCorpus(dir.file("absent.cor")), std::runtime_error);
}
