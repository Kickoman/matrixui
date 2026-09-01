#include <doctest/doctest.h>

#include "core/lib/text.h"

TEST_CASE("ToLower folds ASCII case and leaves everything else alone") {
    CHECK(ToLower("KiNg") == "king");
    CHECK(ToLower("already") == "already");
    CHECK(ToLower("") == "");
    CHECK(ToLower("Word-2 Two") == "word-2 two");
}

TEST_CASE("SplitWords lowercases and splits on whitespace") {
    const auto words = SplitWords("  King   MAN\twoman ");
    REQUIRE(words.size() == 3);
    CHECK(words[0] == "king");
    CHECK(words[1] == "man");
    CHECK(words[2] == "woman");
}

TEST_CASE("SplitWords returns nothing for blank input") {
    CHECK(SplitWords("").empty());
    CHECK(SplitWords("   \t\n  ").empty());
}
