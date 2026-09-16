#include <doctest/doctest.h>

#include "cli/functions/commands.h"
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

TEST_CASE("FunctionsConfig loads from JSON including nested fields") {
    Tests::TempDir dir;
    const auto path = dir.write("config.json", R"json({
        "genetizer": {"maxPopulation": 128, "tournamentSize": 5, "populationDecreaseFactor": 0.8},
        "mutation": {"operators": "+-", "scalarRange": 2.5},
        "fitness": {"accuracyWeight": 2.0, "complexityWeight": 0.5, "lengthWeight": 0.001},
        "initialExpressions": ["x", "sin(x)"],
        "randomCount": 16, "randomDepth": 2,
        "epochs": 7, "patience": 3, "printTop": 4, "printEvery": 0, "seed": 99
    })json");

    Genetizer::FunctionsConfig config;
    std::ostringstream err;
    REQUIRE(CliLib::LoadJsonConfig(err, path.string(), config, "functions"));
    CHECK(config.genetizer.maxPopulation == 128);
    CHECK(config.genetizer.tournamentSize == 5);
    CHECK(config.mutation.operators == "+-");
    CHECK(config.mutation.scalarRange == 2.5);
    CHECK(config.fitness.accuracyWeight == 2.0);
    CHECK(config.fitness.complexityWeight == 0.5);
    REQUIRE(config.initialExpressions.size() == 2);
    CHECK(config.initialExpressions[1] == "sin(x)");
    CHECK(config.epochs == 7);
    CHECK(config.patience == 3);
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
    CHECK(out1.str().find("Copies") != std::string::npos);
    CHECK(out1.str().find("unique ") != std::string::npos);
    CHECK(out1.str().find("Epoch 1:") != std::string::npos);
    // The last epoch is printed by the final block, not the periodic one.
    CHECK(out1.str().find("Epoch 2:") == std::string::npos);
    CHECK(out1.str().find("Final world (epoch 2):") != std::string::npos);
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
    CHECK(reloaded.patience == options.config.patience);
    CHECK(reloaded.fitness.accuracyWeight == options.config.fitness.accuracyWeight);
}

TEST_CASE("Run stops early when the best rank stagnates") {
    Tests::TempDir dir;
    FunctionsCli::RunOptions options;
    options.dataPath = dir.write("data.csv", "x,expected\n1,2\n2,4\n3,6\n4,8\n");
    options.config = SmallConfig();
    options.config.epochs = 50;
    options.config.patience = 1;
    options.config.printEvery = 0;

    std::ostringstream out, err;
    REQUIRE(FunctionsCli::Run(out, err, options) == FunctionsCli::kSuccess);
    CHECK(out.str().find("Stopped early: no improvement for 1 epochs (patience 1)")
          != std::string::npos);
    CHECK(out.str().find("Epoch 50:") == std::string::npos);
    CHECK(out.str().find("Final world (epoch 50)") == std::string::npos);
}

TEST_CASE("Run uses every epoch when patience is disabled") {
    Tests::TempDir dir;
    FunctionsCli::RunOptions options;
    options.dataPath = dir.write("data.csv", "x,expected\n1,2\n2,4\n3,6\n4,8\n");
    options.config = SmallConfig();
    options.config.epochs = 3;
    options.config.patience = 0;
    options.config.printEvery = 0;

    std::ostringstream out, err;
    REQUIRE(FunctionsCli::Run(out, err, options) == FunctionsCli::kSuccess);
    CHECK(out.str().find("Stopped early") == std::string::npos);
    CHECK(out.str().find("Final world (epoch 3):") != std::string::npos);
}

TEST_CASE("Run rejects unusable fitness weights") {
    Tests::TempDir dir;
    FunctionsCli::RunOptions options;
    options.dataPath = dir.write("data.csv", "x,expected\n1,2\n2,4\n");
    options.config = SmallConfig();

    SUBCASE("negative") {
        options.config.fitness.complexityWeight = -1;
    }
    SUBCASE("all zero") {
        options.config.fitness = {0, 0, 0};
    }

    std::ostringstream out, err;
    CHECK(FunctionsCli::Run(out, err, options) == FunctionsCli::kFailure);
    CHECK(err.str().find("fitness weight") != std::string::npos);
}
