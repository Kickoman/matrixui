#include <doctest/doctest.h>

#include "core/words/data/corpus.h"
#include "core/words/data/vocabulary.h"
#include "tests/support/fixtures.h"

#include <sstream>

using namespace Words;

TEST_CASE("EncodeCorpus maps every token to its id") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const auto corpus = Tests::CorpusFromText("a b a c", vocabulary);

    REQUIRE(corpus.size() == 4);
    CHECK(corpus[0] == *vocabulary.getId("a"));
    CHECK(corpus[1] == *vocabulary.getId("b"));
    CHECK(corpus[2] == *vocabulary.getId("a"));
    CHECK(corpus[3] == *vocabulary.getId("c"));
}

TEST_CASE("EncodeCorpus silently drops out-of-vocabulary tokens") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const auto corpus = Tests::CorpusFromText("a zzz b qqq", vocabulary);

    REQUIRE(corpus.size() == 2);
    CHECK(corpus[0] == *vocabulary.getId("a"));
    CHECK(corpus[1] == *vocabulary.getId("b"));
}

TEST_CASE("EncodeCorpus of the source text has exactly keptTokens entries") {
    const auto vocabulary = Tests::ToyVocabulary(5);
    const auto corpus = Tests::CorpusFromText(Tests::ToyCorpusText(), vocabulary);
    CHECK(corpus.size() == vocabulary.getKeptTokens());
}

TEST_CASE("Corpus survives a Save/Load round-trip") {
    const TCorpus original{4, 0, 1, 9, 2, 2, 7};

    std::stringstream stream;
    SaveCorpus(stream, original);
    const auto loaded = LoadCorpus(stream);

    CHECK(loaded == original);
}

TEST_CASE("An empty corpus round-trips") {
    std::stringstream stream;
    SaveCorpus(stream, TCorpus{});
    CHECK(LoadCorpus(stream).empty());
}

TEST_CASE("LoadCorpus rejects a stream that does not hold a corpus") {
    std::istringstream stream("this is definitely not a corpus file");
    CHECK_THROWS_AS(LoadCorpus(stream), std::runtime_error);
}

TEST_CASE("LoadCorpus rejects a truncated header") {
    std::istringstream stream("ab");
    CHECK_THROWS_AS(LoadCorpus(stream), std::runtime_error);
}
