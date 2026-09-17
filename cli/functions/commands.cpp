#include "cli/functions/commands.h"

#include "core/functions/applier.h"
#include "core/functions/config_json.h"
#include "core/functions/expected_csv.h"
#include "core/functions/validate.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace FunctionsCli {

namespace {

void SaveConfig(const std::string& path, const Genetizer::FunctionsConfig& config) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write config file: " + path);
    }
    out << nlohmann::json(config).dump(2) << '\n';
}

std::vector<std::string> WithoutNoneSentinel(std::vector<std::string> names) {
    const auto found = std::find(names.begin(), names.end(), kNoFunctionsToken);
    if (found == names.end()) {
        return names;
    }
    if (names.size() != 1) {
        throw std::runtime_error("--functions: 'none' cannot be combined with function names");
    }
    return {};
}

void RunRun(std::ostream& out, const RunOptions& options) {
    auto config = options.config;
    config.mutation.functions = WithoutNoneSentinel(std::move(config.mutation.functions));
    Genetizer::Validate(config);

    std::ifstream data(options.dataPath);
    if (!data) {
        throw std::runtime_error("cannot open data file: " + options.dataPath);
    }
    auto entries = Genetizer::ParseExpectedCsv(data);
    Genetizer::FunctionGenetizerApplier applier;
    if (config.seed != 0) {
        Genetizer::FunctionGenetizerApplier::SeedThreadRng(config.seed);
    }
    applier.setMutationOptions(
        {config.mutation.operators.begin(), config.mutation.operators.end()},
        config.mutation.functions,
        config.mutation.scalarRange);
    applier.setFitnessOptions(config.fitness);
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

    double bestRank = genetizer.getWorld().front().rank;
    std::size_t stagnateEpochs = 0;
    std::size_t lastEpoch = 0;
    for (std::size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        genetizer.runEpoch();
        lastEpoch = epoch;

        const auto epochBest = genetizer.getWorld().front().rank;
        bool exhausted = false;
        if (epochBest > bestRank) {
            bestRank = epochBest;
            stagnateEpochs = 0;
        } else {
            exhausted = config.patience != 0 && ++stagnateEpochs >= config.patience;
        }

        const bool isLastEpoch = exhausted || epoch == config.epochs;
        if (!isLastEpoch && config.printEvery != 0 && epoch % config.printEvery == 0) {
            out << "Epoch " << epoch << ":\n"
                << Genetizer::FunctionGenetizerApplier::PrintWorld(
                       genetizer.getWorld(), config.printTop)
                << '\n';
        }

        if (exhausted) {
            out << "Stopped early: no improvement for " << stagnateEpochs
                << " epochs (patience " << config.patience << ")\n";
            break;
        }
    }

    const auto& world = genetizer.getWorld();
    out << "Final world (epoch " << lastEpoch << "):\n"
        << Genetizer::FunctionGenetizerApplier::PrintWorld(world, config.printTop) << '\n';
    out << "Best: " << world.front().organism.getPresentation()
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
