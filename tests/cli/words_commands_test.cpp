#include <doctest/doctest.h>

#include "cli/words/commands.h"
#include "cli/words/options.h"

#include "core/lib/file_stream.h"
#include "core/words/data/embeddings.h"
#include "core/words/data/subwords.h"
#include "tests/support/fixtures.h"
#include "tests/support/temp_dir.h"

#include <filesystem>
#include <sstream>
#include <string>

TEST_CASE("LoadVocabulary maps an unreadable file to the failure code") {
    Tests::TempDir dir;
    WordsCli::LoadVocabularyOptions options;
    options.input = dir.file("absent.voc");

    std::ostringstream out, err;
    CHECK(WordsCli::LoadVocabulary(out, err, options) == WordsCli::kFailure);
    CHECK(err.str().rfind("error: ", 0) == 0);
    CHECK(out.str().empty());
}

TEST_CASE("LoadVocabulary rejects a file that is not a vocabulary") {
    Tests::TempDir dir;
    WordsCli::LoadVocabularyOptions options;
    options.input = dir.write("junk.voc", "junk");

    std::ostringstream out, err;
    CHECK(WordsCli::LoadVocabulary(out, err, options) == WordsCli::kFailure);
    CHECK(err.str().find("error: Not a vocabulary file") != std::string::npos);
}

TEST_CASE("BuildVocabulary reports an unwritable output path") {
    Tests::TempDir dir;
    WordsCli::BuildVocabularyOptions options;
    options.input = dir.write("corpus.txt", "a a a b b c\n");
    options.output = dir.file("no-such-dir") / "built.voc";

    std::ostringstream out, err;
    CHECK(WordsCli::BuildVocabulary(out, err, options) == WordsCli::kFailure);
    CHECK(err.str().rfind("error: ", 0) == 0);
}

TEST_CASE("BuildVocabulary reports pruning when it happens") {
    Tests::TempDir dir;
    WordsCli::BuildVocabularyOptions options;
    options.input = dir.write("dump.txt", "a a a a a a a a a a w1 w2 w3 w4 w5 w6\n");
    options.output = dir.file("built.voc");
    options.minCount = 1;
    options.pruneThreshold = 4;

    std::ostringstream out, err;
    CHECK(WordsCli::BuildVocabulary(out, err, options) == WordsCli::kSuccess);
    CHECK(out.str().find("Pruned 1 times (final min-reduce 1)") != std::string::npos);
}

TEST_CASE("Train honours the corpus storage option") {
    Tests::TempDir dir;
    const auto text = dir.write("dump.txt", Tests::ToyCorpusText());

    WordsCli::BuildVocabularyOptions buildVocabulary;
    buildVocabulary.input = text;
    buildVocabulary.output = dir.file("built.voc");
    buildVocabulary.minCount = 1;

    WordsCli::BuildCorpusOptions buildCorpus;
    buildCorpus.input = text;
    buildCorpus.output = dir.file("built.cor");
    buildCorpus.vocabulary = buildVocabulary.output;

    std::ostringstream out, err;
    REQUIRE(WordsCli::BuildVocabulary(out, err, buildVocabulary) == WordsCli::kSuccess);
    REQUIRE(WordsCli::BuildCorpus(out, err, buildCorpus) == WordsCli::kSuccess);

    WordsCli::TrainOptions train;
    train.vocabulary = buildVocabulary.output;
    train.corpus = buildCorpus.output;
    train.output = dir.file("emb.bin");
    train.corpusStorage = "mmap";
    train.config.model.dim = 4;
    train.config.train.epochs = 1;
    train.config.train.threads = 1;

    std::ostringstream trainOut, trainErr;
    CHECK(WordsCli::Train(trainOut, trainErr, train) == WordsCli::kSuccess);
    CHECK(trainErr.str().empty());
    CHECK(trainOut.str().find("corpus storage: mapped") != std::string::npos);
}

TEST_CASE("Train rejects an unknown corpus storage name") {
    WordsCli::TrainOptions train;
    train.corpusStorage = "bogus";

    std::ostringstream out, err;
    CHECK(WordsCli::Train(out, err, train) == WordsCli::kFailure);
    CHECK(err.str().find("error: unknown corpus storage") != std::string::npos);
}

TEST_CASE("Inspect summarises a small text file") {
    Tests::TempDir dir;
    WordsCli::InspectOptions options;
    options.input = dir.write("corpus.txt", "the cat sat on the mat\n");

    std::ostringstream out, err;
    CHECK(WordsCli::Inspect(out, err, options) == WordsCli::kSuccess);
    CHECK(err.str().empty());
    CHECK_FALSE(out.str().empty());
}

TEST_CASE("Neighbours refuses embeddings whose row count is not the vocabulary size") {
    Tests::TempDir dir;
    const auto text = dir.write("dump.txt", Tests::ToyCorpusText());

    WordsCli::BuildVocabularyOptions buildVocabulary;
    buildVocabulary.input = text;
    buildVocabulary.output = dir.file("built.voc");
    buildVocabulary.minCount = 1;

    std::ostringstream out, err;
    REQUIRE(WordsCli::BuildVocabulary(out, err, buildVocabulary) == WordsCli::kSuccess);

    // A matrix with the wrong number of rows -- what feeding the n-gram half of
    // a subword model to a query would look like.
    Words::Embeddings mismatched(99, 4);
    mismatched.initializeZero();
    std::ostringstream bytes;
    Words::Embeddings::Save(bytes, mismatched);

    WordsCli::NeighboursOptions neighbours;
    neighbours.vocabulary = buildVocabulary.output;
    neighbours.embeddings = dir.write("emb.bin", bytes.str());
    neighbours.word = "a";

    std::ostringstream queryOut, queryErr;
    CHECK(WordsCli::Neighbours(queryOut, queryErr, neighbours) == WordsCli::kFailure);
    CHECK(queryErr.str().find("99 rows but the vocabulary has 6 words") != std::string::npos);
}

TEST_CASE("Train writes n-gram vectors beside the embeddings when buckets are asked for") {
    Tests::TempDir dir;
    const auto text = dir.write("dump.txt", Tests::ToyCorpusText());

    WordsCli::BuildVocabularyOptions buildVocabulary;
    buildVocabulary.input = text;
    buildVocabulary.output = dir.file("built.voc");
    buildVocabulary.minCount = 1;

    WordsCli::BuildCorpusOptions buildCorpus;
    buildCorpus.input = text;
    buildCorpus.output = dir.file("built.cor");
    buildCorpus.vocabulary = buildVocabulary.output;

    std::ostringstream out, err;
    REQUIRE(WordsCli::BuildVocabulary(out, err, buildVocabulary) == WordsCli::kSuccess);
    REQUIRE(WordsCli::BuildCorpus(out, err, buildCorpus) == WordsCli::kSuccess);

    WordsCli::TrainOptions train;
    train.vocabulary = buildVocabulary.output;
    train.corpus = buildCorpus.output;
    train.output = dir.file("emb.bin");
    train.config.model.dim = 4;
    train.config.model.buckets = 500;
    train.config.train.epochs = 1;
    train.config.train.threads = 1;

    std::ostringstream trainOut, trainErr;
    REQUIRE(WordsCli::Train(trainOut, trainErr, train) == WordsCli::kSuccess);
    CHECK(trainOut.str().find("subwords: n 3..6, 500 buckets") != std::string::npos);

    const auto subwordPath = dir.file("emb.bin.sub");
    REQUIRE(std::filesystem::exists(subwordPath));

    // The .emb still holds one row per word; the buckets live in the sidecar.
    const auto embeddings = Io::ReadFile(train.output,
        [](std::istream& in) { return Words::Embeddings::Load(in); }, std::ios::binary);
    CHECK(embeddings.getWords() == 6);

    const auto subwords = Io::ReadFile(subwordPath,
        [](std::istream& in) { return Words::SubwordVectors::Load(in); }, std::ios::binary);
    CHECK(subwords.getBuckets() == 500);
    CHECK(subwords.getDim() == 4);
    CHECK(subwords.getVectors().getWords() == 500);
}

TEST_CASE("Train writes no n-gram file when subwords are off") {
    Tests::TempDir dir;
    const auto text = dir.write("dump.txt", Tests::ToyCorpusText());

    WordsCli::BuildVocabularyOptions buildVocabulary;
    buildVocabulary.input = text;
    buildVocabulary.output = dir.file("built.voc");
    buildVocabulary.minCount = 1;

    WordsCli::BuildCorpusOptions buildCorpus;
    buildCorpus.input = text;
    buildCorpus.output = dir.file("built.cor");
    buildCorpus.vocabulary = buildVocabulary.output;

    std::ostringstream out, err;
    REQUIRE(WordsCli::BuildVocabulary(out, err, buildVocabulary) == WordsCli::kSuccess);
    REQUIRE(WordsCli::BuildCorpus(out, err, buildCorpus) == WordsCli::kSuccess);

    WordsCli::TrainOptions train;
    train.vocabulary = buildVocabulary.output;
    train.corpus = buildCorpus.output;
    train.output = dir.file("emb.bin");
    train.config.model.dim = 4;
    train.config.train.epochs = 1;
    train.config.train.threads = 1;

    std::ostringstream trainOut, trainErr;
    REQUIRE(WordsCli::Train(trainOut, trainErr, train) == WordsCli::kSuccess);
    CHECK(trainOut.str().find("subwords") == std::string::npos);
    CHECK_FALSE(std::filesystem::exists(dir.file("emb.bin.sub")));
}
