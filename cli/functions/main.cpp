#include "cli/functions/commands.h"
#include "cli/functions/options.h"

#include "cli/lib/argv_scan.h"
#include "cli/lib/json_config.h"
#include "core/functions/config_json.h"

#include <CLI11/CLI11.hpp>

#include <iostream>

int main(int argc, char** argv) {
    CLI::App app{"Genetic symbolic regression: evolve an expression fitting expected values"};
    app.require_subcommand(1);

    int exitCode = FunctionsCli::kSuccess;

    FunctionsCli::RunOptions run;

    // Load-bearing ordering: do not move this block below the registrations.
    // The config file is read straight out of argv, before the per-field
    // options exist, so that a config file supplies the new defaults and a
    // per-field flag still overrides just that field -- and so the
    // capture_default_str() calls below snapshot the loaded values, making
    // --help show what will be used.
    run.configPath = CliLib::FindOptionValue(argc, argv, "--config");
    if (!CliLib::LoadJsonConfig(std::cerr, run.configPath, run.config, "functions")) {
        return FunctionsCli::kFailure;
    }

    auto* runCmd = app.add_subcommand("run", "Evolve an expression fitting a CSV of expected values");
    runCmd->add_option("--data", run.dataPath,
            "CSV data points: header row of variable names, last column 'expected'")
        ->required()->check(CLI::ExistingFile);

    runCmd->add_option("--config", run.configPath,
            "Load settings from a JSON file (as written by --save-config); "
            "flags below override individual fields from the loaded config.")
        ->group("Config")
        ->check(CLI::ExistingFile);
    runCmd->add_option("--save-config", run.saveConfigPath,
            "Write the effective config JSON to this path")
        ->group("Config");

    runCmd->add_option("--epochs", run.config.epochs, "Epochs to evolve")
        ->capture_default_str();
    runCmd->add_option("--patience", run.config.patience,
            "Stop when the best rank has not improved for this many epochs (0 = never)")
        ->capture_default_str();
    runCmd->add_option("--seed", run.config.seed, "Random seed (0 = nondeterministic)")
        ->capture_default_str();
    runCmd->add_option("--max-population", run.config.genetizer.maxPopulation, "World size")
        ->capture_default_str();
    runCmd->add_option("--tournament-size", run.config.genetizer.tournamentSize,
            "Tournament selection size")
        ->capture_default_str();
    runCmd->add_option("--population-decrease", run.config.genetizer.populationDecreaseFactor,
            "Survivor share kept between epochs")
        ->capture_default_str();
    runCmd->add_option("--operators", run.config.mutation.operators,
            "Binary operators available to mutation")
        ->capture_default_str();
    runCmd->add_option("--scalar-range", run.config.mutation.scalarRange,
            "Random constants are drawn from [-range, range]")
        ->capture_default_str();
    runCmd->add_option("--accuracy-weight", run.config.fitness.accuracyWeight,
            "Fitness weight of how well the expression fits the data")
        ->group("Fitness")->capture_default_str();
    runCmd->add_option("--complexity-weight", run.config.fitness.complexityWeight,
            "Fitness weight of how few RPN units the expression uses")
        ->group("Fitness")->capture_default_str();
    runCmd->add_option("--length-weight", run.config.fitness.lengthWeight,
            "Fitness weight of how short the printed expression is")
        ->group("Fitness")->capture_default_str();
    runCmd->add_option("--expression", run.config.initialExpressions,
            "Initial expression, repeatable (replaces the config list)");
    runCmd->add_option("--random-count", run.config.randomCount,
            "Random organisms seeded into the initial population")
        ->capture_default_str();
    runCmd->add_option("--random-depth", run.config.randomDepth,
            "Max tree depth of the random initial organisms")
        ->capture_default_str();
    runCmd->add_option("--print-top", run.config.printTop,
            "Distinct expressions to print (0 = all)")
        ->capture_default_str();
    runCmd->add_option("--print-every", run.config.printEvery,
            "Print the world every N epochs (0 = only start and final)")
        ->capture_default_str();
    runCmd->callback([&] {
        exitCode = FunctionsCli::Run(std::cout, std::cerr, run);
    });

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& error) {
        return app.exit(error);
    }

    return exitCode;
}
