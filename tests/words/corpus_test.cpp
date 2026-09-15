#include <doctest/doctest.h>

#include "core/words/error.h"
#include "core/words/data/corpus.h"
#include "core/words/data/vocabulary.h"
#include "tests/support/fixtures.h"
#include "tests/support/temp_dir.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <utility>

using namespace Words;

namespace {

std::uint64_t ReadLittleEndianAt(const std::string& bytes, const std::size_t offset,
                                 const std::size_t width) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < width; ++i) {
        value |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[offset + i]))
                 << (8 * i);
    }
    return value;
}

}

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

TEST_CASE("The .cor byte layout is frozen: magic, version, count, then raw ids") {
    const TCorpus corpus{7, 0x01020304, 42};

    std::ostringstream stream;
    SaveCorpus(stream, corpus);
    const auto bytes = stream.str();

    REQUIRE(bytes.size() == 16 + corpus.size() * sizeof(TWordId));
    CHECK(ReadLittleEndianAt(bytes, 0, 4) == 0x57435250);
    CHECK(ReadLittleEndianAt(bytes, 4, 4) == 1);
    CHECK(ReadLittleEndianAt(bytes, 8, 8) == corpus.size());
    CHECK(ReadLittleEndianAt(bytes, 16, 4) == 7);
    CHECK(ReadLittleEndianAt(bytes, 20, 4) == 0x01020304);
    CHECK(ReadLittleEndianAt(bytes, 24, 4) == 42);
}

TEST_CASE("A corpus larger than the writer's buffer round-trips") {
    TCorpus original(300'000);
    for (std::size_t i = 0; i < original.size(); ++i) {
        original[i] = static_cast<TWordId>(i * 2654435761u);
    }

    std::stringstream stream;
    SaveCorpus(stream, original);
    CHECK(LoadCorpus(stream) == original);
}

TEST_CASE("LoadCorpus rejects a body shorter than the declared count") {
    const TCorpus original{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::ostringstream out;
    SaveCorpus(out, original);

    std::istringstream truncated(out.str().substr(0, 16 + 4 * sizeof(TWordId)));
    CHECK_THROWS_AS(LoadCorpus(truncated), IoError);
}

TEST_CASE("LoadCorpus rejects a stream that does not hold a corpus") {
    std::istringstream stream("this is definitely not a corpus file");
    CHECK_THROWS_AS(LoadCorpus(stream), std::runtime_error);
}

TEST_CASE("LoadCorpus rejects a truncated header") {
    std::istringstream stream("ab");
    CHECK_THROWS_AS(LoadCorpus(stream), std::runtime_error);
}

TEST_CASE("EncodeCorpusToStream writes the same bytes as the in-memory path") {
    const auto vocabulary = Tests::ToyVocabulary(2);
    const auto text = Tests::ToyCorpusText();

    std::ostringstream reference;
    SaveCorpus(reference, Tests::CorpusFromText(text, vocabulary));

    std::istringstream dump(text);
    std::stringstream streamed;
    const auto written = EncodeCorpusToStream(dump, vocabulary, streamed);

    CHECK(streamed.str() == reference.str());
    CHECK(written == vocabulary.getKeptTokens());
}

TEST_CASE("EncodeCorpusToStream flushes correctly at buffer boundaries") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const auto text = Tests::ToyCorpusText();

    std::ostringstream reference;
    SaveCorpus(reference, Tests::CorpusFromText(text, vocabulary));

    // 64 tokens: an even divisor, an odd remainder, and one-by-one writes.
    for (const std::size_t bufferTokens : {1, 3, 8, 64, 1000}) {
        std::istringstream dump(text);
        std::stringstream streamed;
        const auto written = EncodeCorpusToStream(dump, vocabulary, streamed, bufferTokens);

        CHECK(streamed.str() == reference.str());
        CHECK(written == 64);
    }
}

TEST_CASE("EncodeCorpusToStream with only unknown tokens writes an empty corpus") {
    const auto vocabulary = Tests::ToyVocabulary(2);

    std::ostringstream reference;
    SaveCorpus(reference, TCorpus{});

    std::istringstream dump("zzz qqq www");
    std::stringstream streamed;
    CHECK(EncodeCorpusToStream(dump, vocabulary, streamed) == 0);
    CHECK(streamed.str() == reference.str());
}

TEST_CASE("A streamed corpus loads back with the right tokens") {
    const auto vocabulary = Tests::ToyVocabulary(1);

    std::istringstream dump("a b a c");
    std::stringstream streamed;
    EncodeCorpusToStream(dump, vocabulary, streamed, 2);

    const auto loaded = LoadCorpus(streamed);
    REQUIRE(loaded.size() == 4);
    CHECK(loaded[0] == *vocabulary.getId("a"));
    CHECK(loaded[3] == *vocabulary.getId("c"));
}

namespace {

std::filesystem::path WriteCorpusFile(const Tests::TempDir& dir, const TCorpus& corpus) {
    std::ostringstream stream;
    SaveCorpus(stream, corpus);
    return dir.write("corpus.cor", stream.str());
}

}

TEST_CASE("Corpus::Open serves identical tokens in loaded and mapped modes") {
    const TCorpus original{4, 0, 1, 9, 2, 2, 7};
    const Tests::TempDir dir;
    const auto path = WriteCorpusFile(dir, original);

    const auto loaded = Corpus::Open(path, CorpusStorage::Loaded);
    const auto mapped = Corpus::Open(path, CorpusStorage::Mapped);

    REQUIRE(loaded.size() == original.size());
    REQUIRE(mapped.size() == original.size());
    CHECK(loaded.getStorage() == CorpusStorage::Loaded);
    if (Io::MappedFile::IsSupported()) {
        CHECK(mapped.getStorage() == CorpusStorage::Mapped);
    }
    for (std::size_t i = 0; i < original.size(); ++i) {
        CHECK(loaded[i] == original[i]);
        CHECK(mapped[i] == original[i]);
    }
    CHECK(std::equal(loaded.begin(), loaded.end(), mapped.begin()));
}

TEST_CASE("Corpus::Open in auto mode loads a tiny file into memory") {
    const Tests::TempDir dir;
    const auto path = WriteCorpusFile(dir, TCorpus{1, 2, 3});

    const auto corpus = Corpus::Open(path);
    CHECK(corpus.getStorage() == CorpusStorage::Loaded);
    CHECK(corpus.size() == 3);
}

TEST_CASE("Corpus::Open rejects a truncated file in both modes") {
    const TCorpus original{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::ostringstream stream;
    SaveCorpus(stream, original);
    const Tests::TempDir dir;
    const auto path = dir.write("trunc.cor", stream.str().substr(0, 16 + 4 * sizeof(TWordId)));

    CHECK_THROWS_AS(Corpus::Open(path, CorpusStorage::Loaded), IoError);
    CHECK_THROWS_AS(Corpus::Open(path, CorpusStorage::Mapped), IoError);
}

TEST_CASE("Corpus::Open rejects a file that is not a corpus") {
    const Tests::TempDir dir;
    const auto path = dir.write("garbage.cor", "this is definitely not a corpus file");

    CHECK_THROWS_AS(Corpus::Open(path), std::runtime_error);
}

TEST_CASE("Corpus::Open handles an empty corpus in both modes") {
    const Tests::TempDir dir;
    const auto path = WriteCorpusFile(dir, TCorpus{});

    CHECK(Corpus::Open(path, CorpusStorage::Loaded).size() == 0);
    CHECK(Corpus::Open(path, CorpusStorage::Mapped).size() == 0);
}

TEST_CASE("Corpus wraps an in-memory buffer and survives a move") {
    Corpus corpus{TCorpus{5, 6, 7}};
    REQUIRE(corpus.size() == 3);
    CHECK(corpus.getStorage() == CorpusStorage::Loaded);
    CHECK(corpus[0] == 5);

    const Corpus moved(std::move(corpus));
    REQUIRE(moved.size() == 3);
    CHECK(moved[2] == 7);
    CHECK(std::equal(moved.begin(), moved.end(), TCorpus{5, 6, 7}.begin()));
}
