#include <doctest/doctest.h>

#include "cli/words/commands.h"
#include "cli/words/options.h"

#include "tests/support/temp_dir.h"

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

TEST_CASE("Inspect summarises a small text file") {
    Tests::TempDir dir;
    WordsCli::InspectOptions options;
    options.input = dir.write("corpus.txt", "the cat sat on the mat\n");

    std::ostringstream out, err;
    CHECK(WordsCli::Inspect(out, err, options) == WordsCli::kSuccess);
    CHECK(err.str().empty());
    CHECK_FALSE(out.str().empty());
}
