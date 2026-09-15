#include <doctest/doctest.h>

#include "cli/functions/commands.h"
#include "cli/functions/csv.h"
#include "cli/functions/options.h"
#include "cli/lib/json_config.h"
#include "core/functions/config_json.h"

#include "tests/support/temp_dir.h"

#include <fstream>
#include <sstream>
#include <string>

namespace {

Genetizer::FunctionsConfig SmallConfig() {
    Genetizer::FunctionsConfig config;
    config.genetizer.maxPopulation = 64;
    config.randomCount = 32;
    config.randomDepth = 3;
    config.epochs = 2;
    config.printTop = 3;
    config.printEvery = 1;
    config.seed = 42;
    return config;
}

}  // namespace

TEST_CASE("ParseExpectedCsv reads a two-variable file") {
    std::istringstream in("x,y,expected\n1,2,3\n4,5,9\n");
    const auto entries = FunctionsCli::ParseExpectedCsv(in);

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
    const auto entries = FunctionsCli::ParseExpectedCsv(in);

    REQUIRE(entries.size() == 1);
    CHECK(entries[0].variables[0].name == "x");
    CHECK(entries[0].expectedResult == 2);
}

TEST_CASE("ParseExpectedCsv rejects a header without an expected column") {
    std::istringstream in("x,y\n1,2\n");
    CHECK_THROWS_WITH_AS(FunctionsCli::ParseExpectedCsv(in),
                         doctest::Contains("'expected'"), std::runtime_error);
}

TEST_CASE("ParseExpectedCsv rejects a header with no variable columns") {
    std::istringstream in("expected\n1\n");
    CHECK_THROWS_WITH_AS(FunctionsCli::ParseExpectedCsv(in),
                         doctest::Contains("at least one variable column"), std::runtime_error);
}

TEST_CASE("ParseExpectedCsv reports a non-numeric cell with its position") {
    std::istringstream in("x,expected\n1,2\nabc,4\n");
    CHECK_THROWS_WITH_AS(FunctionsCli::ParseExpectedCsv(in),
                         doctest::Contains("line 3, column 'x'"), std::runtime_error);
}

TEST_CASE("ParseExpectedCsv rejects an empty stream") {
    std::istringstream in("");
    CHECK_THROWS_WITH_AS(FunctionsCli::ParseExpectedCsv(in),
                         doctest::Contains("empty"), std::runtime_error);
}

TEST_CASE("ParseExpectedCsv rejects a row with the wrong column count") {
    std::istringstream in("x,y,expected\n1,2\n");
    CHECK_THROWS_WITH_AS(FunctionsCli::ParseExpectedCsv(in),
                         doctest::Contains("line 2"), std::runtime_error);
}

TEST_CASE("ParseExpectedCsv rejects a file with no data rows") {
    std::istringstream in("x,expected\n");
    CHECK_THROWS_WITH_AS(FunctionsCli::ParseExpectedCsv(in),
                         doctest::Contains("no data rows"), std::runtime_error);
}

TEST_CASE("FunctionsConfig loads from JSON including nested fields") {
    Tests::TempDir dir;
    const auto path = dir.write("config.json", R"json({
        "genetizer": {"maxPopulation": 128, "tournamentSize": 5, "populationDecreaseFactor": 0.8},
        "mutation": {"operators": "+-", "scalarRange": 2.5},
        "initialExpressions": ["x", "sin(x)"],
        "randomCount": 16, "randomDepth": 2,
        "epochs": 7, "printTop": 4, "printEvery": 0, "seed": 99
    })json");

    Genetizer::FunctionsConfig config;
    std::ostringstream err;
    REQUIRE(CliLib::LoadJsonConfig(err, path.string(), config, "functions"));
    CHECK(config.genetizer.maxPopulation == 128);
    CHECK(config.genetizer.tournamentSize == 5);
    CHECK(config.mutation.operators == "+-");
    CHECK(config.mutation.scalarRange == 2.5);
    REQUIRE(config.initialExpressions.size() == 2);
    CHECK(config.initialExpressions[1] == "sin(x)");
    CHECK(config.epochs == 7);
    CHECK(config.seed == 99);
}

TEST_CASE("Malformed config JSON is reported") {
    Tests::TempDir dir;
    const auto path = dir.write("config.json", "{not json");

    Genetizer::FunctionsConfig config;
    std::ostringstream err;
    CHECK_FALSE(CliLib::LoadJsonConfig(err, path.string(), config, "functions"));
    CHECK(err.str().find("Failed to parse functions config") != std::string::npos);
}

TEST_CASE("Run maps a missing data file to the failure code") {
    Tests::TempDir dir;
    FunctionsCli::RunOptions options;
    options.dataPath = dir.file("absent.csv");
    options.config = SmallConfig();

    std::ostringstream out, err;
    CHECK(FunctionsCli::Run(out, err, options) == FunctionsCli::kFailure);
    CHECK(err.str().rfind("error: ", 0) == 0);
    CHECK(out.str().empty());
}

TEST_CASE("Run rejects a CSV with a bad header") {
    Tests::TempDir dir;
    FunctionsCli::RunOptions options;
    options.dataPath = dir.write("data.csv", "x,y\n1,2\n");
    options.config = SmallConfig();

    std::ostringstream out, err;
    CHECK(FunctionsCli::Run(out, err, options) == FunctionsCli::kFailure);
    CHECK(err.str().find("'expected'") != std::string::npos);
}

TEST_CASE("Run rejects an invalid initial expression naming it") {
    Tests::TempDir dir;
    FunctionsCli::RunOptions options;
    options.dataPath = dir.write("data.csv", "x,expected\n1,2\n2,4\n");
    options.config = SmallConfig();
    options.config.initialExpressions = {"sin(x"};

    std::ostringstream out, err;
    CHECK(FunctionsCli::Run(out, err, options) == FunctionsCli::kFailure);
    CHECK(err.str().find("sin(x") != std::string::npos);
}

TEST_CASE("Run rejects a config with no organisms to seed") {
    Tests::TempDir dir;
    FunctionsCli::RunOptions options;
    options.dataPath = dir.write("data.csv", "x,expected\n1,2\n");
    options.config = SmallConfig();
    options.config.randomCount = 0;
    options.config.initialExpressions = {};

    std::ostringstream out, err;
    CHECK(FunctionsCli::Run(out, err, options) == FunctionsCli::kFailure);
    CHECK(err.str().rfind("error: ", 0) == 0);
}

TEST_CASE("Run evolves deterministically under a fixed seed") {
    Tests::TempDir dir;
    FunctionsCli::RunOptions options;
    options.dataPath = dir.write("data.csv", "x,expected\n1,2\n2,4\n3,6\n4,8\n");
    options.config = SmallConfig();
    options.config.initialExpressions = {"x", "x+1"};

    std::ostringstream out1, err1;
    REQUIRE(FunctionsCli::Run(out1, err1, options) == FunctionsCli::kSuccess);
    CHECK(err1.str().empty());
    CHECK(out1.str().find("Start world:") != std::string::npos);
    CHECK(out1.str().find("Expression") != std::string::npos);  // table header
    CHECK(out1.str().find("Epoch 2:") != std::string::npos);
    CHECK(out1.str().find("Best: ") != std::string::npos);

    std::ostringstream out2, err2;
    REQUIRE(FunctionsCli::Run(out2, err2, options) == FunctionsCli::kSuccess);
    CHECK(out1.str() == out2.str());
}

TEST_CASE("Run writes the effective config when asked") {
    Tests::TempDir dir;
    FunctionsCli::RunOptions options;
    options.dataPath = dir.write("data.csv", "x,expected\n1,2\n2,4\n");
    options.saveConfigPath = dir.file("saved.json");
    options.config = SmallConfig();

    std::ostringstream out, err;
    REQUIRE(FunctionsCli::Run(out, err, options) == FunctionsCli::kSuccess);

    Genetizer::FunctionsConfig reloaded;
    std::ostringstream loadErr;
    REQUIRE(CliLib::LoadJsonConfig(loadErr, options.saveConfigPath, reloaded, "functions"));
    CHECK(reloaded.seed == 42);
    CHECK(reloaded.genetizer.maxPopulation == 64);
}
