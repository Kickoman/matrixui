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

TEST_CASE("IsUtf8Continuation recognises the 10xxxxxx bytes") {
    CHECK_FALSE(Text::IsUtf8Continuation(static_cast<unsigned char>('a')));
    CHECK_FALSE(Text::IsUtf8Continuation(0xD0));
    CHECK_FALSE(Text::IsUtf8Continuation(0xE2));
    CHECK_FALSE(Text::IsUtf8Continuation(0xF0));
    CHECK(Text::IsUtf8Continuation(0x80));
    CHECK(Text::IsUtf8Continuation(0xBF));
    CHECK(Text::IsUtf8Continuation(0xBA));
}

TEST_CASE("Utf8CharacterOffsets walks ASCII one byte at a time") {
    const std::vector<std::size_t> expected{0, 1, 2, 3};
    CHECK(Text::Utf8CharacterOffsets("cat") == expected);
}

TEST_CASE("Utf8CharacterOffsets keeps two-byte Cyrillic characters whole") {
    // "кот" is three characters and six bytes.
    const std::vector<std::size_t> expected{0, 2, 4, 6};
    CHECK(Text::Utf8CharacterOffsets("кот") == expected);
    CHECK(std::string("кот").size() == 6);
}

TEST_CASE("Utf8CharacterOffsets handles mixed character widths") {
    // 'a' is one byte, the euro sign three, 'б' two.
    const std::vector<std::size_t> expected{0, 1, 4, 6};
    CHECK(Text::Utf8CharacterOffsets("a€б") == expected);

    // Four-byte characters count once too.
    const std::vector<std::size_t> emoji{0, 4, 5};
    CHECK(Text::Utf8CharacterOffsets("\xF0\x9F\x99\x82x") == emoji);
}

TEST_CASE("Utf8CharacterOffsets ends with the size, so an empty text gives one entry") {
    const std::vector<std::size_t> expected{0};
    CHECK(Text::Utf8CharacterOffsets("") == expected);
}

TEST_CASE("Utf8CharacterOffsets does not run past the end on broken UTF-8") {
    // A stray continuation byte becomes a character of its own rather than
    // attaching to a lead byte that is not there.
    const std::vector<std::size_t> stray{0, 1};
    CHECK(Text::Utf8CharacterOffsets("\x80") == stray);

    // A truncated two-byte sequence still terminates.
    const std::vector<std::size_t> truncated{0, 1};
    CHECK(Text::Utf8CharacterOffsets("\xD0") == truncated);

    const std::vector<std::size_t> mixed{0, 2, 3};
    CHECK(Text::Utf8CharacterOffsets("\xD0\xBA\x41") == mixed);
}

TEST_CASE("Utf8CharacterOffsets reuses the buffer it is handed") {
    std::vector<std::size_t> offsets{99, 99, 99, 99, 99};
    Text::Utf8CharacterOffsets("ab", offsets);
    const std::vector<std::size_t> expected{0, 1, 2};
    CHECK(offsets == expected);
}
