#include "gui/functions/functions_controller.h"

#include "cli/lib/json_config.h"
#include "core/functions/config_json.h"
#include "core/functions/expected_csv.h"
#include "core/functions/validate.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

constexpr auto kSnapshotInterval = std::chrono::milliseconds(100);

constexpr std::size_t kLoggedRowCap = 200;

std::size_t LoggedRows(const std::size_t printTop) {
    return printTop == 0 ? kLoggedRowCap : std::min(printTop, kLoggedRowCap);
}

}  // namespace


FunctionsController::FunctionsController(QObject* parent)
    : ModeController(parent)
    , settings("functions")
{
    qRegisterMetaType<FunctionsSnapshot>("FunctionsSnapshot");
}

FunctionsController::~FunctionsController() {
    requestStop();
    waitUntilFinished();
}

std::ostream& FunctionsController::out() const {
    return logger != nullptr ? *logger : std::cout;
}

void FunctionsController::setLogger(std::ostream* stream) {
    logger = stream;
}

FunctionsController::Info FunctionsController::getInfo() const {
    const auto running = runningFlag.load(std::memory_order_relaxed);
    return Info{
        .running = running,
        .hasSession = session != nullptr,
        .canResume = session != nullptr && session->valid && !sessionDirty && !running,
        .epoch = session != nullptr ? session->epochsDone : 0,
        .config = config,
        .dataPath = dataPath,
        .configPath = configPath,
        .topCurves = topCurves,
    };
}

void FunctionsController::waitUntilFinished() {
    if (internalRunner == nullptr || !internalRunner->isRunning()) {
        return;
    }
    stopRequested.store(true, std::memory_order_relaxed);
    internalRunner->quit();
    internalRunner->wait();
}

void FunctionsController::setConfig(const Genetizer::FunctionsConfig& value) {
    config = value;
    markSessionDirty();
}

void FunctionsController::setPoints(std::vector<Genetizer::Entry> value) {
    points = std::move(value);
    markSessionDirty();
}

void FunctionsController::setTopCurves(const int value) {
    topCurves = value;
}

void FunctionsController::setDataPath(const QString& path) {
    dataPath = path;
    emit infoUpdated();
}

void FunctionsController::setConfigPath(const QString& path) {
    configPath = path;
    emit infoUpdated();
}

void FunctionsController::markSessionDirty() {
    if (sessionDirty) {
        return;
    }
    sessionDirty = true;
    emit infoUpdated();
}

std::vector<Genetizer::Entry> FunctionsController::takeLoadedPoints() {
    return std::exchange(loadedPoints, {});
}

bool FunctionsController::loadConfigFile(const QString& path) {
    Genetizer::FunctionsConfig loaded;
    if (!CliLib::LoadJsonConfig(out(), path.toStdString(), loaded, "functions")) {
        return false;
    }
    config = loaded;
    markSessionDirty();
    emit infoUpdated();
    return true;
}

bool FunctionsController::saveConfigFile(const QString& path) {
    std::ofstream file(path.toStdString());
    if (!file) {
        out() << "Cannot write the config file: " << path.toStdString() << std::endl;
        return false;
    }
    file << nlohmann::json(config).dump(2) << '\n';
    return true;
}

void FunctionsController::loadSettings() {
    if (const auto raw = settings.getValue("config"); !raw.isNull()) {
        try {
            config = nlohmann::json::parse(raw.toString().toStdString())
                         .get<Genetizer::FunctionsConfig>();
        } catch (const nlohmann::json::exception& error) {
            out() << "Ignoring the stored configuration (" << error.what()
                  << "), using defaults." << std::endl;
            config = Genetizer::FunctionsConfig{};
        }
    }

    if (const auto raw = settings.getValue("points_csv"); !raw.isNull()) {
        try {
            std::istringstream in(raw.toString().toStdString());
            loadedPoints = Genetizer::ParseExpectedCsv(in);
        } catch (const std::exception& error) {
            out() << "Ignoring the stored points (" << error.what() << ")." << std::endl;
            loadedPoints.clear();
        }
    }

    dataPath = settings.getValue("data_path").toString();
    configPath = settings.getValue("config_path").toString();
    topCurves = settings.getValue("top_curves", topCurves).toInt();
    emit infoUpdated();
}

void FunctionsController::saveSettings() {
    settings.setValue("config", QString::fromStdString(nlohmann::json(config).dump()));

    QString pointsCsv;
    if (!points.empty()) {
        std::vector<std::string> names;
        for (const auto& variable : points.front().variables) {
            names.push_back(variable.name);
        }
        try {
            std::ostringstream stream;
            Genetizer::WriteExpectedCsv(stream, names, points);
            pointsCsv = QString::fromStdString(stream.str());
        } catch (const std::exception& error) {
            out() << "Not storing the points (" << error.what() << ")." << std::endl;
        }
    }
    settings.setValue("points_csv", pointsCsv);

    settings.setValue("data_path", dataPath);
    settings.setValue("config_path", configPath);
    settings.setValue("top_curves", topCurves);
}

bool FunctionsController::validateForStart() {
    try {
        Genetizer::Validate(config);
    } catch (const std::runtime_error& error) {
        out() << "Configuration invalid: " << error.what() << std::endl;
        return false;
    }

    if (points.empty()) {
        out() << "Add at least one data point before starting." << std::endl;
        return false;
    }

    for (const auto& text : config.initialExpressions) {
        try {
            const Genetizer::Expression parsed(text);
            (void)parsed;
        } catch (const std::runtime_error& error) {
            out() << "Invalid initial expression '" << text << "': " << error.what() << std::endl;
            return false;
        }
    }
    return true;
}

void FunctionsController::start() {
    if (runningFlag.load(std::memory_order_relaxed)) {
        out() << "Already running, ignoring the request." << std::endl;
        return;
    }
    if (!validateForStart()) {
        return;
    }
    session.reset();
    launch(std::make_shared<Session>(), true);
}

void FunctionsController::resume() {
    if (runningFlag.load(std::memory_order_relaxed)) {
        out() << "Already running, ignoring the request." << std::endl;
        return;
    }
    if (session == nullptr || !session->valid || sessionDirty) {
        out() << "Nothing to resume; start a new run." << std::endl;
        return;
    }
    launch(std::make_shared<Session>(std::move(*session)), false);
}

void FunctionsController::requestStop() {
    stopRequested.store(true, std::memory_order_relaxed);
}

FunctionsSnapshot FunctionsController::MakeSnapshot(
    const Genetizer::FunctionGenetizer& genetizer, const std::size_t epoch, const std::size_t top)
{
    const auto distinct =
        Genetizer::FunctionGenetizerApplier::CollectDistinct(genetizer.getWorld(), top);

    FunctionsSnapshot snapshot;
    snapshot.epoch = epoch;
    snapshot.uniqueCount = distinct.uniqueCount;
    snapshot.totalCount = distinct.totalCount;
    snapshot.rows.reserve(distinct.rows.size());
    for (const auto& row : distinct.rows) {
        const auto& organism = row.representative->organism;
        snapshot.rows.push_back(FunctionsSnapshot::Row{
            .expression = QString::fromStdString(organism.getPresentation()),
            .rank = row.representative->rank,
            .copies = row.copies,
            .birth = row.birth,
            .expressionCopy = std::make_shared<const Genetizer::Expression>(organism.expression),
        });
    }
    return snapshot;
}

void FunctionsController::launch(std::shared_ptr<Session> shared, const bool fresh) {
    runningFlag.store(true, std::memory_order_relaxed);
    stopRequested.store(false, std::memory_order_relaxed);
    sessionDirty = false;
    session.reset();
    emit infoUpdated();

    const auto configCopy = config;
    auto pointsCopy = points;
    const auto top = std::max(config.printTop, static_cast<std::size_t>(kMaxCurves));

    internalRunner = new QThread(this);
    connect(internalRunner, &QThread::finished, internalRunner, &QObject::deleteLater);
    connect(internalRunner, &QThread::started,
            [this, shared, fresh, configCopy, pointsCopy = std::move(pointsCopy), top]() mutable {
        const auto publish = [this](FunctionsSnapshot snapshot) {
            QMetaObject::invokeMethod(this, [this, snapshot = std::move(snapshot)] {
                emit snapshotReady(snapshot);
            });
        };

        try {
            if (fresh) {
                shared->applier = std::make_unique<Genetizer::FunctionGenetizerApplier>();
                if (configCopy.seed != 0) {
                    // The engine is thread_local, so it has to be seeded here.
                    Genetizer::FunctionGenetizerApplier::SeedThreadRng(configCopy.seed);
                }
                shared->applier->setMutationOptions(
                    {configCopy.mutation.operators.begin(), configCopy.mutation.operators.end()},
                    configCopy.mutation.scalarRange);
                shared->applier->setFitnessOptions(configCopy.fitness);
                for (auto& entry : pointsCopy) {
                    shared->applier->addExpected(std::move(entry.variables), entry.expectedResult);
                }

                shared->genetizer = std::make_unique<Genetizer::FunctionGenetizer>(
                    shared->applier->getRankFunction(),
                    shared->applier->getMutateFunction(),
                    shared->applier->getCrossoverFunction());
                shared->genetizer->setConfig(configCopy.genetizer);
                if (configCopy.seed != 0) {
                    shared->genetizer->setSeed(configCopy.seed + 1);
                }

                for (const auto& text : configCopy.initialExpressions) {
                    shared->genetizer->addOrganism(Genetizer::OrganismInfo{
                        .epochOfBirth = 0,
                        .expression = Genetizer::Expression(text),
                    });
                }
                shared->applier->seedRandom(
                    *shared->genetizer, configCopy.randomCount, configCopy.randomDepth);
                shared->genetizer->rankPopulation();
                shared->epochsDone = 0;

                publish(MakeSnapshot(*shared->genetizer, 0, top));
                out() << "Start world:\n"
                      << Genetizer::FunctionGenetizerApplier::PrintWorld(
                             shared->genetizer->getWorld(), LoggedRows(configCopy.printTop))
                      << std::endl;
            } else {
                out() << "Resuming at epoch " << shared->epochsDone << std::endl;
            }

            auto lastPublish = std::chrono::steady_clock::now();
            while (!stopRequested.load(std::memory_order_relaxed)) {
                shared->genetizer->runEpoch();
                ++shared->epochsDone;

                const auto now = std::chrono::steady_clock::now();
                if (now - lastPublish >= kSnapshotInterval) {
                    publish(MakeSnapshot(*shared->genetizer, shared->epochsDone, top));
                    lastPublish = now;
                }
            }
            publish(MakeSnapshot(*shared->genetizer, shared->epochsDone, top));

            const auto& world = shared->genetizer->getWorld();
            out() << "Stopped at epoch " << shared->epochsDone << ":\n"
                  << Genetizer::FunctionGenetizerApplier::PrintWorld(world, LoggedRows(configCopy.printTop))
                  << "\nBest: " << world.front().organism.getPresentation()
                  << " (rank " << world.front().rank << ")" << std::endl;
            shared->valid = true;
        } catch (const std::exception& error) {
            out() << "Run failed: " << error.what() << std::endl;
            shared->valid = false;
        }

        QMetaObject::invokeMethod(this, [this, shared] {
            if (shared->valid) {
                session = std::make_unique<Session>(std::move(*shared));
            }
            runningFlag.store(false, std::memory_order_relaxed);
            emit infoUpdated();
        });
        QThread::currentThread()->quit();
    });
    internalRunner->start();
}
