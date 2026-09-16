#include <doctest/doctest.h>

#include "core/functions/expected_csv.h"

#include <sstream>
#include <string>
#include <vector>

using Genetizer::Entry;
using Genetizer::ParseExpectedCsv;
using Genetizer::Variable;
using Genetizer::WriteExpectedCsv;

TEST_CASE("ParseExpectedCsv reads a two-variable file") {
    std::istringstream in("x,y,expected\n1,2,3\n4,5,9\n");
    const auto entries = ParseExpectedCsv(in);

    REQUIRE(entries.size() == 2);
    REQUIRE(entries[0].variables.size() == 2);
    CHECK(entries[0].variables[0].name == "x");
    CHECK(entries[0].variables[0].value == 1);
    CHECK(entries[0].variables[1].name == "y");
    CHECK(entries[0].variables[1].value == 2);
    CHECK(entries[0].expectedResult == 3);
    CHECK(entries[1].variables[0].value == 4);
    CHECK(entries[1].expectedResult == 9);
}

TEST_CASE("ParseExpectedCsv tolerates CRLF and a trailing blank line") {
    std::istringstream in("x,expected\r\n1,2\r\n\r\n");
    const auto entries = ParseExpectedCsv(in);

    REQUIRE(entries.size() == 1);
    CHECK(entries[0].variables[0].name == "x");
    CHECK(entries[0].expectedResult == 2);
}

TEST_CASE("ParseExpectedCsv rejects a header without an expected column") {
    std::istringstream in("x,y\n1,2\n");
    CHECK_THROWS_WITH_AS(ParseExpectedCsv(in),
                         doctest::Contains("'expected'"), std::runtime_error);
}

TEST_CASE("ParseExpectedCsv rejects a header with no variable columns") {
    std::istringstream in("expected\n1\n");
    CHECK_THROWS_WITH_AS(ParseExpectedCsv(in),
                         doctest::Contains("at least one variable column"), std::runtime_error);
}

TEST_CASE("ParseExpectedCsv reports a non-numeric cell with its position") {
    std::istringstream in("x,expected\n1,2\nabc,4\n");
    CHECK_THROWS_WITH_AS(ParseExpectedCsv(in),
                         doctest::Contains("line 3, column 'x'"), std::runtime_error);
}

TEST_CASE("ParseExpectedCsv rejects an empty stream") {
    std::istringstream in("");
    CHECK_THROWS_WITH_AS(ParseExpectedCsv(in), doctest::Contains("empty"), std::runtime_error);
}

TEST_CASE("ParseExpectedCsv rejects a row with the wrong column count") {
    std::istringstream in("x,y,expected\n1,2\n");
    CHECK_THROWS_WITH_AS(ParseExpectedCsv(in), doctest::Contains("line 2"), std::runtime_error);
}

TEST_CASE("ParseExpectedCsv rejects a file with no data rows") {
    std::istringstream in("x,expected\n");
    CHECK_THROWS_WITH_AS(ParseExpectedCsv(in),
                         doctest::Contains("no data rows"), std::runtime_error);
}

TEST_CASE("WriteExpectedCsv writes the header the parser demands") {
    const std::vector<Entry> entries{
        Entry{.variables = {Variable{"x", 1}, Variable{"y", 2}}, .expectedResult = 3},
    };

    std::ostringstream out;
    WriteExpectedCsv(out, {"x", "y"}, entries);
    CHECK(out.str() == "x,y,expected\n1,2,3\n");
}

TEST_CASE("WriteExpectedCsv round-trips through the parser") {
    const std::vector<Entry> entries{
        Entry{.variables = {Variable{"x", 2.2}}, .expectedResult = -0.5},
        Entry{.variables = {Variable{"x", 1e-3}}, .expectedResult = 1.0 / 3.0},
    };

    std::ostringstream out;
    WriteExpectedCsv(out, {"x"}, entries);

    std::istringstream in(out.str());
    const auto parsed = ParseExpectedCsv(in);
    REQUIRE(parsed.size() == entries.size());
    for (std::size_t i = 0; i < parsed.size(); ++i) {
        REQUIRE(parsed[i].variables.size() == 1);
        CHECK(parsed[i].variables[0].value == entries[i].variables[0].value);
        CHECK(parsed[i].expectedResult == entries[i].expectedResult);
    }
}

TEST_CASE("WriteExpectedCsv follows the header order, not the entry order") {
    const std::vector<Entry> entries{
        Entry{.variables = {Variable{"y", 20}, Variable{"x", 10}}, .expectedResult = 30},
    };

    std::ostringstream out;
    WriteExpectedCsv(out, {"x", "y"}, entries);
    CHECK(out.str() == "x,y,expected\n10,20,30\n");
}

TEST_CASE("WriteExpectedCsv names a point that is missing a variable") {
    const std::vector<Entry> entries{
        Entry{.variables = {Variable{"x", 1}}, .expectedResult = 2},
        Entry{.variables = {Variable{"x", 3}}, .expectedResult = 4},
    };

    std::ostringstream out;
    CHECK_THROWS_WITH_AS(WriteExpectedCsv(out, {"x", "y"}, entries),
                         doctest::Contains("point 1 has no value for 'y'"), std::runtime_error);
}
