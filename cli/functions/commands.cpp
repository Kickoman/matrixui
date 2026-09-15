#include "cli/functions/commands.h"
#include "cli/functions/csv.h"

#include "core/functions/applier.h"
#include "core/functions/config_json.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace FunctionsCli {

namespace {

void ValidateConfig(const Genetizer::FunctionsConfig& config) {
    if (config.mutation.operators.empty()) {
        throw std::runtime_error("config: operators must not be empty");
    }
    if (config.genetizer.maxPopulation == 0) {
        throw std::runtime_error("config: maxPopulation must be positive");
    }
    if (config.genetizer.tournamentSize == 0) {
        throw std::runtime_error("config: tournamentSize must be positive");
    }
    if (config.epochs == 0) {
        throw std::runtime_error("config: epochs must be positive");
    }
    if (config.randomCount > 0 && config.randomDepth == 0) {
        throw std::runtime_error("config: randomDepth must be positive when randomCount is");
    }
    if (config.randomCount == 0 && config.initialExpressions.empty()) {
        throw std::runtime_error(
            "config: nothing to seed -- need randomCount > 0 or initialExpressions");
    }
}

void SaveConfig(const std::string& path, const Genetizer::FunctionsConfig& config) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write config file: " + path);
    }
    out << nlohmann::json(config).dump(2) << '\n';
}

void RunRun(std::ostream& out, const RunOptions& options) {
    const auto& config = options.config;
    ValidateConfig(config);

    std::ifstream data(options.dataPath);
    if (!data) {
        throw std::runtime_error("cannot open data file: " + options.dataPath);
    }
    auto entries = ParseExpectedCsv(data);
    Genetizer::FunctionGenetizerApplier applier;
    if (config.seed != 0) {
        Genetizer::FunctionGenetizerApplier::SeedThreadRng(config.seed);
    }
    applier.setMutationOptions(
        {config.mutation.operators.begin(), config.mutation.operators.end()},
        config.mutation.scalarRange);
    for (auto& entry : entries) {
        applier.addExpected(std::move(entry.variables), entry.expectedResult);
    }

    Genetizer::FunctionGenetizer genetizer(
        applier.getRankFunction(),
        applier.getMutateFunction(),
        applier.getCrossoverFunction());
    genetizer.setConfig(config.genetizer);
    if (config.seed != 0) {
        genetizer.setSeed(config.seed + 1);  // a stream distinct from the applier rng
    }

    for (const auto& text : config.initialExpressions) {
        try {
            genetizer.addOrganism(Genetizer::OrganismInfo{
                .epochOfBirth = 0,
                .expression = Genetizer::Expression(text),
            });
        } catch (const std::runtime_error& error) {
            throw std::runtime_error(
                "invalid initial expression '" + text + "': " + error.what());
        }
    }
    applier.seedRandom(genetizer, config.randomCount, config.randomDepth);

    if (!options.saveConfigPath.empty()) {
        SaveConfig(options.saveConfigPath, config);
    }

    genetizer.rankPopulation();
    out << "Start world:\n"
        << Genetizer::FunctionGenetizerApplier::PrintWorld(genetizer.getWorld(), config.printTop)
        << '\n';

    bool printed = false;
    for (std::size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        genetizer.runEpoch();
        printed = config.printEvery != 0 && epoch % config.printEvery == 0;
        if (printed) {
            out << "Epoch " << epoch << ":\n"
                << Genetizer::FunctionGenetizerApplier::PrintWorld(
                       genetizer.getWorld(), config.printTop)
                << '\n';
        }
    }

    const auto& world = genetizer.getWorld();
    if (!printed) {
        out << "Final world:\n"
            << Genetizer::FunctionGenetizerApplier::PrintWorld(world, config.printTop) << '\n';
    }
    out << "Best: " << world.front().organism.expression.toString()
        << " (rank " << world.front().rank << ")\n";
}

template <typename Body>
int Guarded(std::ostream& err, Body&& body) {
    try {
        body();
        return kSuccess;
    } catch (const std::runtime_error& error) {
        err << "error: " << error.what() << '\n';
        return kFailure;
    } catch (const std::exception& error) {
        err << "internal error: " << error.what() << '\n';
        return kInternalError;
    }
}

}  // namespace

int Run(std::ostream& out, std::ostream& err, const RunOptions& options) {
    return Guarded(err, [&] { RunRun(out, options); });
}

}  // namespace FunctionsCli
