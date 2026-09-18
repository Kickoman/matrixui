#include <doctest/doctest.h>

#include "core/words/data/embeddings.h"
#include "core/words/data/subwords.h"
#include "core/words/data/vocabulary.h"
#include "core/words/error.h"
#include "tests/support/fixtures.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace Words;

namespace {

constexpr std::size_t kBuckets = 100'000;

std::vector<TBucketId> BucketsOf(const std::vector<std::string>& ngrams, const std::size_t buckets = kBuckets) {
    std::vector<TBucketId> expected;
    expected.reserve(ngrams.size());
    for (const auto& ngram : ngrams) {
        expected.push_back(static_cast<TBucketId>(HashSubword(ngram) % buckets));
    }
    return expected;
}

}  // namespace

TEST_CASE("Subwords of a Cyrillic word are whole characters, not bytes") {
    // "кот" wrapped is "<кот>": five characters, ten bytes. Over n = 3..6 that
    // is 3 + 2 + 1 + 0 = 6 n-grams; a byte-wise split would report 15.
    const auto actual = ComputeSubwords("кот", 3, 6, kBuckets);
    const auto expected = BucketsOf({
        "<ко", "кот", "от>",
        "<кот", "кот>",
        "<кот>",
    });

    CHECK(actual.size() == 6);
    CHECK(actual == expected);
}

TEST_CASE("Subword order is by length, then by position") {
    const auto actual = ComputeSubwords("abc", 2, 3, kBuckets);
    const auto expected = BucketsOf({"<a", "ab", "bc", "c>", "<ab", "abc", "bc>"});
    CHECK(actual == expected);
}

TEST_CASE("The whole wrapped word is a subword when it fits the range") {
    const auto five = ComputeSubwords("кот", 5, 5, kBuckets);
    CHECK(five == BucketsOf({"<кот>"}));

    const auto tooLong = ComputeSubwords("кот", 6, 6, kBuckets);
    CHECK(tooLong.empty());
}

TEST_CASE("A word shorter than minN yields no subwords") {
    CHECK(ComputeSubwords("я", 3, 6, kBuckets) == BucketsOf({"<я>"}));
    CHECK(ComputeSubwords("я", 4, 6, kBuckets).empty());
    CHECK(ComputeSubwords("", 3, 6, kBuckets).empty());
    CHECK(ComputeSubwords("он", 3, 6, kBuckets) == BucketsOf({"<он", "он>", "<он>"}));
}

TEST_CASE("Repeated n-grams inside one word are kept, not collapsed") {
    // "<аааа>" contains "ааа" twice.
    const auto actual = ComputeSubwords("аааа", 3, 3, kBuckets);
    REQUIRE(actual.size() == 4);
    CHECK(actual[1] == actual[2]);
}

TEST_CASE("Mixed-width characters keep their boundaries") {
    const auto actual = ComputeSubwords("a€б", 3, 3, kBuckets);
    CHECK(actual == BucketsOf({"<a€", "a€б", "€б>"}));
}

TEST_CASE("Zero buckets disable subwords entirely") {
    CHECK(ComputeSubwords("кот", 3, 6, 0).empty());
}

TEST_CASE("An empty or inverted n range yields nothing") {
    CHECK(ComputeSubwords("кот", 0, 6, kBuckets).empty());
    CHECK(ComputeSubwords("кот", 5, 3, kBuckets).empty());
}

TEST_CASE("Every bucket stays inside the table") {
    for (const auto* word : {"кот", "котёнок", "слово-другое", "mixed слово"}) {
        for (const auto bucket : ComputeSubwords(word, 3, 6, 97)) {
            CHECK(bucket < 97);
        }
    }
}

TEST_CASE("HashSubword is deterministic and distinguishes its input") {
    CHECK(HashSubword("кот") == HashSubword("кот"));
    CHECK(HashSubword("кот") != HashSubword("ток"));
    CHECK(HashSubword("") == 2166136261u);
}

TEST_CASE("SubwordTable holds what ComputeSubwords returns for every word") {
    const auto vocabulary = Tests::VocabularyFromText("кот кота котом кот кота кот", 1);
    const auto table = SubwordTable::Build(vocabulary, 3, 6, kBuckets);

    REQUIRE(table.isEnabled());
    REQUIRE(table.getWords() == vocabulary.getSize());
    CHECK(table.getMinN() == 3);
    CHECK(table.getMaxN() == 6);
    CHECK(table.getBuckets() == kBuckets);

    std::size_t total = 0;
    for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
        const auto expected = ComputeSubwords(vocabulary.getWord(id), 3, 6, kBuckets);
        const auto actual = table.getSubwords(id);
        REQUIRE(actual.size() == expected.size());
        CHECK(std::equal(actual.begin(), actual.end(), expected.begin()));
        total += expected.size();
    }
    CHECK(table.getReferences() == total);
    CHECK(table.getAverageSubwords() == doctest::Approx(1. * total / vocabulary.getSize()));
    CHECK(table.getBytes() > 0);
}

TEST_CASE("A table built with zero buckets is empty for every word") {
    const auto vocabulary = Tests::VocabularyFromText("кот кота котом", 1);
    const auto table = SubwordTable::Build(vocabulary, 3, 6, 0);

    CHECK_FALSE(table.isEnabled());
    CHECK(table.getWords() == 0);
    CHECK(table.getReferences() == 0);
    CHECK(table.getAverageSubwords() == doctest::Approx(0.));
    for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
        CHECK(table.getSubwords(id).empty());
    }
}

TEST_CASE("A default-constructed table is empty and disabled") {
    const SubwordTable table;
    CHECK_FALSE(table.isEnabled());
    CHECK(table.getSubwords(0).empty());
    CHECK(table.getSubwords(1'000'000).empty());
}

TEST_CASE("Word forms of the same stem share most of their subwords") {
    // The point of the whole exercise: "кот", "кота" and "котом" are separate
    // vocabulary entries that nonetheless overlap in n-gram space.
    const auto base = ComputeSubwords("кот", 3, 6, kBuckets);
    const auto genitive = ComputeSubwords("кота", 3, 6, kBuckets);

    std::size_t shared = 0;
    for (const auto bucket : base) {
        if (std::find(genitive.begin(), genitive.end(), bucket) != genitive.end()) {
            ++shared;
        }
    }
    CHECK(shared >= 2);
}

namespace {

Embeddings BucketMatrix(const std::size_t buckets, const std::size_t dim) {
    Embeddings matrix(buckets, dim);
    matrix.initializeZero();
    for (TWordId bucket = 0; bucket < buckets; ++bucket) {
        for (std::size_t i = 0; i < dim; ++i) {
            matrix.row(bucket)[i] = static_cast<TFloat>(bucket + i * 0.5);
        }
    }
    return matrix;
}

}  // namespace

TEST_CASE("Subword vectors survive a Save/Load round-trip") {
    const auto matrix = BucketMatrix(64, 4);

    std::stringstream stream;
    SubwordVectors::Save(stream, matrix, 3, 6, 64);
    const auto loaded = SubwordVectors::Load(stream);

    CHECK(loaded.getMinN() == 3);
    CHECK(loaded.getMaxN() == 6);
    CHECK(loaded.getBuckets() == 64);
    CHECK(loaded.getDim() == 4);
    REQUIRE(loaded.getVectors().getWords() == 64);
    for (TWordId bucket = 0; bucket < 64; ++bucket) {
        CHECK(std::equal(loaded.getVectors().row(bucket), loaded.getVectors().row(bucket) + 4,
                         matrix.row(bucket)));
    }
}

TEST_CASE("An out-of-vocabulary vector is the mean of its n-gram rows") {
    const std::size_t buckets = 64;
    const std::size_t dim = 4;
    const auto matrix = BucketMatrix(buckets, dim);

    std::stringstream stream;
    SubwordVectors::Save(stream, matrix, 3, 6, buckets);
    const auto loaded = SubwordVectors::Load(stream);

    const auto ngrams = ComputeSubwords("котёнком", 3, 6, buckets);
    REQUIRE(ngrams.size() > 0);

    std::vector<double> expected(dim, 0.);
    for (const auto bucket : ngrams) {
        for (std::size_t i = 0; i < dim; ++i) {
            expected[i] += matrix.row(bucket)[i];
        }
    }
    for (auto& value : expected) {
        value /= static_cast<double>(ngrams.size());
    }

    const auto composed = loaded.compose("котёнком");
    REQUIRE(composed.size() == dim);
    for (std::size_t i = 0; i < dim; ++i) {
        CHECK(composed[i] == doctest::Approx(expected[i]).epsilon(1e-6));
    }
}

TEST_CASE("A word too short for any n-gram composes to nothing") {
    std::stringstream stream;
    SubwordVectors::Save(stream, BucketMatrix(32, 4), 5, 6, 32);
    const auto loaded = SubwordVectors::Load(stream);

    CHECK(loaded.compose("я").empty());
    CHECK_FALSE(loaded.compose("слово").empty());
}

TEST_CASE("Loading refuses a file that is not a subword file") {
    std::stringstream junk("not a subword file at all");
    CHECK_THROWS_AS(SubwordVectors::Load(junk), IoError);

    std::stringstream empty;
    CHECK_THROWS_AS(SubwordVectors::Load(empty), IoError);
}

TEST_CASE("Loading refuses a truncated subword file") {
    std::stringstream stream;
    SubwordVectors::Save(stream, BucketMatrix(32, 4), 3, 6, 32);
    const auto bytes = stream.str();

    std::stringstream cut(bytes.substr(0, bytes.size() / 2));
    CHECK_THROWS_AS(SubwordVectors::Load(cut), IoError);

    std::stringstream header(bytes.substr(0, 10));
    CHECK_THROWS_AS(SubwordVectors::Load(header), IoError);
}

TEST_CASE("Loading refuses a bucket count that does not match the matrix") {
    std::stringstream stream;
    SubwordVectors::Save(stream, BucketMatrix(32, 4), 3, 6, 64);
    CHECK_THROWS_AS(SubwordVectors::Load(stream), IoError);
}

TEST_CASE("Loading refuses an unusable n-gram range") {
    std::stringstream inverted;
    SubwordVectors::Save(inverted, BucketMatrix(8, 2), 6, 3, 8);
    CHECK_THROWS_AS(SubwordVectors::Load(inverted), IoError);

    std::stringstream zero;
    SubwordVectors::Save(zero, BucketMatrix(8, 2), 0, 3, 8);
    CHECK_THROWS_AS(SubwordVectors::Load(zero), IoError);
}

TEST_CASE("Embeddings::Load rejects a subword file handed to it by mistake") {
    // The two files are both matrices of floats; only the tag keeps a .sub out
    // of a query that expects one row per word.
    std::stringstream stream;
    SubwordVectors::Save(stream, BucketMatrix(32, 4), 3, 6, 32);
    CHECK_THROWS_AS(Embeddings::Load(stream), IoError);
}

TEST_CASE("Out-of-range ids give an empty span instead of reading past the table") {
    const auto vocabulary = Tests::VocabularyFromText("кот кота котом", 1);
    const auto table = SubwordTable::Build(vocabulary, 3, 6, kBuckets);

    REQUIRE(table.getWords() == vocabulary.getSize());
    CHECK_FALSE(table.getSubwords(0).empty());
    CHECK(table.getSubwords(static_cast<TWordId>(vocabulary.getSize())).empty());
    // The largest id there is: id + 1 wraps to zero, which a naive bounds check
    // would read as "in range".
    CHECK(table.getSubwords(std::numeric_limits<TWordId>::max()).empty());
}

TEST_CASE("compose reports how many n-grams went into the vector") {
    std::stringstream stream;
    SubwordVectors::Save(stream, BucketMatrix(64, 4), 3, 6, 64);
    const auto loaded = SubwordVectors::Load(stream);

    std::size_t used = 99;
    const auto composed = loaded.compose("котёнком", used);
    CHECK(used == ComputeSubwords("котёнком", 3, 6, 64).size());
    CHECK(composed.size() == 4);

    // Wrapped, "я" is three characters, so it still has one n-gram.
    const auto single = loaded.compose("я", used);
    CHECK(used == 1);
    CHECK(single.size() == 4);

    // Nothing to compose from says zero rather than leaving the previous count
    // in place: "<>" is two characters, below min-n.
    const auto empty = loaded.compose("", used);
    CHECK(used == 0);
    CHECK(empty.empty());
}
