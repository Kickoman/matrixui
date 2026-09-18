#include <doctest/doctest.h>

#include "core/lib/text.h"

#include <cstddef>
#include <string>
#include <vector>

TEST_CASE("ToLower folds ASCII case and leaves everything else alone") {
    CHECK(Text::ToLower("KiNg") == "king");
    CHECK(Text::ToLower("already") == "already");
    CHECK(Text::ToLower("") == "");
    CHECK(Text::ToLower("Word-2 Two") == "word-2 two");
}

TEST_CASE("Trim drops whitespace on both ends") {
    CHECK(Text::Trim("  king \t\r\n") == "king");
    CHECK(Text::Trim("king") == "king");
    CHECK(Text::Trim("two words") == "two words");
}

TEST_CASE("Trim returns nothing when there is nothing but whitespace") {
    CHECK(Text::Trim("").empty());
    CHECK(Text::Trim(" \t\r\n\f\v").empty());
}

TEST_CASE("Trim takes the set of whitespace characters to cut") {
    CHECK(Text::Trim("--king--", "-") == "king");
    CHECK(Text::Trim(" king ", "-") == " king ");
    CHECK(Text::Trim("\rking\r", " \t") == "\rking\r");
}

TEST_CASE("Split keeps every cell, empty ones included") {
    const auto cells = Text::Split("a,,b", ",");
    REQUIRE(cells.size() == 3);
    CHECK(cells[0] == "a");
    CHECK(cells[1] == "");
    CHECK(cells[2] == "b");
}

TEST_CASE("Split leaves the cells exactly as they are") {
    const auto cells = Text::Split(" a , b ", ",");
    REQUIRE(cells.size() == 2);
    CHECK(cells[0] == " a ");
    CHECK(cells[1] == " b ");
}

TEST_CASE("Split of text without a separator is the text itself") {
    const auto cells = Text::Split("king", ",");
    REQUIRE(cells.size() == 1);
    CHECK(cells[0] == "king");
}

TEST_CASE("Split of an empty text is one empty cell") {
    const auto cells = Text::Split("", ",");
    REQUIRE(cells.size() == 1);
    CHECK(cells[0] == "");
}

TEST_CASE("Split cuts at any of the separators") {
    const auto cells = Text::Split("a|b,c", "|,");
    REQUIRE(cells.size() == 3);
    CHECK(cells[0] == "a");
    CHECK(cells[1] == "b");
    CHECK(cells[2] == "c");
}

TEST_CASE("Split counts leading and trailing separators") {
    const auto cells = Text::Split(",a,", ",");
    REQUIRE(cells.size() == 3);
    CHECK(cells[0] == "");
    CHECK(cells[1] == "a");
    CHECK(cells[2] == "");
}

TEST_CASE("SplitWords lowercases and splits on whitespace") {
    const auto words = Text::SplitWords("  King   MAN\twoman ");
    REQUIRE(words.size() == 3);
    CHECK(words[0] == "king");
    CHECK(words[1] == "man");
    CHECK(words[2] == "woman");
}

TEST_CASE("SplitWords returns nothing for blank input") {
    CHECK(Text::SplitWords("").empty());
    CHECK(Text::SplitWords("   \t\n  ").empty());
}

TEST_CASE("ParseNumber reads a number that fills the whole text") {
    CHECK(Text::ParseNumber<double>("1.5") == doctest::Approx(1.5));
    CHECK(Text::ParseNumber<double>(" \t-0.25\r\n") == doctest::Approx(-0.25));
    CHECK(Text::ParseNumber<double>("1e-3") == doctest::Approx(0.001));
    CHECK(Text::ParseNumber<std::size_t>("42").value() == 42);
}

TEST_CASE("ParseNumber accepts a leading plus, which from_chars does not") {
    CHECK(Text::ParseNumber<double>("+1.5") == doctest::Approx(1.5));
    CHECK(Text::ParseNumber<double>(" +1.5 ") == doctest::Approx(1.5));
    CHECK(Text::ParseNumber<std::size_t>("+42").value() == 42);
}

TEST_CASE("ParseNumber refuses anything that is not exactly one number") {
    CHECK_FALSE(Text::ParseNumber<double>("").has_value());
    CHECK_FALSE(Text::ParseNumber<double>("   ").has_value());
    CHECK_FALSE(Text::ParseNumber<double>("+").has_value());
    CHECK_FALSE(Text::ParseNumber<double>("+-1").has_value());
    CHECK_FALSE(Text::ParseNumber<double>("1 2").has_value());
    CHECK_FALSE(Text::ParseNumber<double>("1abc").has_value());
    CHECK_FALSE(Text::ParseNumber<double>("abc").has_value());
}

TEST_CASE("ParseNumber refuses what does not fit the asked-for type") {
    CHECK_FALSE(Text::ParseNumber<std::size_t>("-1").has_value());
    CHECK_FALSE(Text::ParseNumber<std::size_t>("1.5").has_value());
}
