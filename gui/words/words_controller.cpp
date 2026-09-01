#include "gui/words/words_controller.h"

#include "core/words/config_json.h"
#include "core/words/error.h"
#include "core/words/evaluate.h"
#include "core/words/queries.h"
#include "core/words/report/evaluate_report.h"
#include "core/words/report/inspect_report.h"
#include "core/words/report/query_report.h"
#include "core/words/utils/inspect.h"

#include <QFileInfo>

#include <iostream>
#include <utility>

WordsController::WordsController(QObject* parent)
    : ModeController(parent)
    , settings("words")
{
    // The core invokes this on the training thread; hop to the GUI thread so
    // connections to the widget stay direct.
    trainer.setProgressCallback([this](const Words::TrainProgress& progress) {
        QMetaObject::invokeMethod(this, [this, progress] {
            emit trainProgressed(progress);
        });
    });
}

WordsController::~WordsController()
{
    requestStop();
    waitUntilFinished();
}

WordsController::Info WordsController::getInfo() const
{
    Info info;
    info.busy = busyFlag.load(std::memory_order_relaxed);
    info.training = trainingFlag.load(std::memory_order_relaxed);
    info.busyLabel = busyLabel;

    info.hasVocabulary = vocabulary != nullptr;
    info.hasCorpus = corpus != nullptr;
    info.hasEmbeddings = embeddings != nullptr;
    info.hasIndex = index != nullptr;

    if (vocabulary) {
        info.vocabularySize = vocabulary->getSize();
        info.rawTokens = vocabulary->getRawTokens();
        info.keptTokens = vocabulary->getKeptTokens();
    }
    if (corpus) {
        info.corpusTokens = corpus->size();
    }
    if (embeddings) {
        info.embeddingDim = embeddings->getDim();
    }

    info.dumpPath = dumpPath;
    info.vocabularyPath = vocabularyPath;
    info.corpusPath = corpusPath;
    info.embeddingsPath = embeddingsPath;
    info.analogiesPath = analogiesPath;
    info.similarityPath = similarityPath;

    info.minCount = minCount;
    info.config = config;
    info.scoreColumn = scoreColumn;
    info.restrictTo = restrictTo;
    info.evaluateThreads = evaluateThreads;
    return info;
}

void WordsController::waitUntilFinished()
{
    if (!internalRunner || !internalRunner->isRunning()) {
        return;
    }
    internalRunner->quit();
    internalRunner->wait();
}

void WordsController::setLogger(std::ostream* stream)
{
    logger = stream;
}

std::ostream& WordsController::out()
{
    if (logger) {
        return *logger << "[controller] ";
    }
    return std::cerr << "[controller] ";
}

// --- settings ---------------------------------------------------------------

void WordsController::loadSettings()
{
    dumpPath = settings.getValue("dump_path", {}).toString();
    vocabularyPath = settings.getValue("vocabulary_path", {}).toString();
    corpusPath = settings.getValue("corpus_path", {}).toString();
    embeddingsPath = settings.getValue("embeddings_path", {}).toString();
    analogiesPath = settings.getValue("analogies_path", {}).toString();
    similarityPath = settings.getValue("similarity_path", {}).toString();

    minCount = settings.getValue("min_count", 5).toULongLong();
    scoreColumn = settings.getValue("score_column", 2).toULongLong();
    restrictTo = settings.getValue("restrict_to", 30000).toULongLong();
    evaluateThreads = settings.getValue("evaluate_threads", 0).toULongLong();

    if (const auto raw = settings.getValue("words_config"); !raw.isNull()) {
        try {
            config = nlohmann::json::parse(raw.toString().toStdString()).get<Words::WordsConfig>();
        } catch (const nlohmann::json::exception& error) {
            out() << "Ignoring the stored configuration (" << error.what()
                  << "), using defaults." << std::endl;
            config = Words::WordsConfig{};
        }
    }

    emit infoUpdated();
}

void WordsController::saveSettings()
{
    settings.setValue("dump_path", dumpPath);
    settings.setValue("vocabulary_path", vocabularyPath);
    settings.setValue("corpus_path", corpusPath);
    settings.setValue("embeddings_path", embeddingsPath);
    settings.setValue("analogies_path", analogiesPath);
    settings.setValue("similarity_path", similarityPath);

    settings.setValue("min_count", static_cast<quint64>(minCount));
    settings.setValue("score_column", static_cast<quint64>(scoreColumn));
    settings.setValue("restrict_to", static_cast<quint64>(restrictTo));
    settings.setValue("evaluate_threads", static_cast<quint64>(evaluateThreads));

    settings.setValue("words_config", QString::fromStdString(nlohmann::json(config).dump()));
}

// --- trivial state ----------------------------------------------------------

void WordsController::setConfig(const Words::WordsConfig& value) { config = value; }
void WordsController::setMinCount(const std::size_t value) { minCount = value; }

void WordsController::setEvaluateParams(
    const std::size_t newScoreColumn, const std::size_t newRestrictTo, const std::size_t threads)
{
    scoreColumn = newScoreColumn;
    restrictTo = newRestrictTo;
    evaluateThreads = threads;
}

void WordsController::setDumpPath(const QString& path) { dumpPath = path; emit infoUpdated(); }
void WordsController::setVocabularyPath(const QString& path) { vocabularyPath = path; emit infoUpdated(); }
void WordsController::setCorpusPath(const QString& path) { corpusPath = path; emit infoUpdated(); }
void WordsController::setEmbeddingsPath(const QString& path) { embeddingsPath = path; emit infoUpdated(); }
void WordsController::setAnalogiesPath(const QString& path) { analogiesPath = path; emit infoUpdated(); }
void WordsController::setSimilarityPath(const QString& path) { similarityPath = path; emit infoUpdated(); }

// --- the single worker ------------------------------------------------------

void WordsController::runTask(const QString& label, const bool isTraining, std::function<void()> task)
{
    if (busyFlag.load(std::memory_order_relaxed)) {
        out() << "Busy (" << busyLabel.toStdString() << "), ignoring: "
              << label.toStdString() << std::endl;
        return;
    }

    // Set on the GUI thread BEFORE the thread starts, so a double click cannot
    // slip through the gap.
    busyFlag.store(true, std::memory_order_relaxed);
    trainingFlag.store(isTraining, std::memory_order_relaxed);
    busyLabel = label;
    emit infoUpdated();

    internalRunner = new QThread(this);
    connect(internalRunner, &QThread::finished, internalRunner, &QObject::deleteLater);
    connect(internalRunner, &QThread::finished, this, &WordsController::infoUpdated);
    connect(internalRunner, &QThread::started, [this, label, task = std::move(task)] {
        try {
            task();
        } catch (const Words::Error& error) {
            out() << label.toStdString() << " failed: " << error.what() << std::endl;
        } catch (const std::exception& error) {
            out() << label.toStdString() << " failed: " << error.what() << std::endl;
        }
        trainingFlag.store(false, std::memory_order_relaxed);
        busyFlag.store(false, std::memory_order_relaxed);
        QMetaObject::invokeMethod(this, [this] { busyLabel.clear(); });
        QThread::currentThread()->quit();
    });
    internalRunner->start();
}

void WordsController::requestStop()
{
    trainer.requestStop();
}

// --- data pipeline ----------------------------------------------------------

void WordsController::inspectDump()
{
    if (dumpPath.isEmpty()) {
        out() << "No text dump selected." << std::endl;
        return;
    }
    const auto path = dumpPath.toStdString();
    runTask("Inspecting dump", false, [this, path] {
        const auto statistics = Words::InspectDump(path);
        if (logger) {
            Words::PrintCorpusStatistics(*logger, statistics);
        }
    });
}

void WordsController::buildVocabulary()
{
    if (dumpPath.isEmpty() || vocabularyPath.isEmpty()) {
        out() << "Need a text dump and a vocabulary path." << std::endl;
        return;
    }
    const auto dump = dumpPath.toStdString();
    const auto target = vocabularyPath.toStdString();
    const auto count = minCount;
    runTask("Building vocabulary", false, [this, dump, target, count] {
        auto built = std::make_shared<const Words::Vocabulary>(
            Words::Vocabulary::Build(dump, count));
        Words::Vocabulary::Save(*built, target);
        out() << "Vocabulary: " << built->getSize() << " words, kept "
              << built->getKeptTokens() << " of " << built->getRawTokens()
              << " tokens; saved to " << target << std::endl;
        QMetaObject::invokeMethod(this, [this, built] {
            vocabulary = built;
            if (corpus) {
                corpus.reset();
                out() << "The corpus was encoded with the previous vocabulary -- "
                         "rebuild or reload it." << std::endl;
            }
            emit infoUpdated();
        });
    });
}

void WordsController::loadVocabulary()
{
    if (vocabularyPath.isEmpty()) {
        out() << "No vocabulary file selected." << std::endl;
        return;
    }
    const auto path = vocabularyPath.toStdString();
    runTask("Loading vocabulary", false, [this, path] {
        auto loaded = std::make_shared<const Words::Vocabulary>(Words::Vocabulary::Load(path));
        out() << "Vocabulary: " << loaded->getSize() << " words." << std::endl;
        QMetaObject::invokeMethod(this, [this, loaded] {
            vocabulary = loaded;
            if (corpus) {
                corpus.reset();
                out() << "The corpus may not match the newly loaded vocabulary -- "
                         "rebuild or reload it." << std::endl;
            }
            emit infoUpdated();
        });
    });
}

void WordsController::buildCorpus()
{
    if (dumpPath.isEmpty() || corpusPath.isEmpty() || !vocabulary) {
        out() << "Need a text dump, a corpus path and a vocabulary." << std::endl;
        return;
    }
    const auto dump = dumpPath.toStdString();
    const auto target = corpusPath.toStdString();
    auto vocab = vocabulary;
    runTask("Building corpus", false, [this, dump, target, vocab] {
        auto built = std::make_shared<const Words::TCorpus>(Words::EncodeCorpus(dump, *vocab));
        Words::SaveCorpus(target, *built);
        out() << "Corpus: " << built->size() << " tokens; saved to " << target << std::endl;
        QMetaObject::invokeMethod(this, [this, built] {
            corpus = built;
            emit infoUpdated();
        });
    });
}

void WordsController::loadCorpus()
{
    if (corpusPath.isEmpty()) {
        out() << "No corpus file selected." << std::endl;
        return;
    }
    if (!vocabulary) {
        out() << "Load a vocabulary first -- corpus tokens are meaningless without it." << std::endl;
        return;
    }
    const auto path = corpusPath.toStdString();
    runTask("Loading corpus", false, [this, path] {
        auto loaded = std::make_shared<const Words::TCorpus>(Words::LoadCorpus(path));
        out() << "Corpus: " << loaded->size() << " tokens." << std::endl;
        QMetaObject::invokeMethod(this, [this, loaded] {
            corpus = loaded;
            emit infoUpdated();
        });
    });
}

// --- training ---------------------------------------------------------------

void WordsController::startTraining()
{
    if (!vocabulary || !corpus) {
        out() << "Need a vocabulary and a corpus before training." << std::endl;
        return;
    }
    try {
        Words::Validate(config, vocabulary->getSize());
    } catch (const Words::ConfigError& error) {
        out() << "Configuration invalid: " << error.what() << std::endl;
        return;
    }

    trainer.setVocabulary(vocabulary);
    trainer.setCorpus(corpus);
    trainer.setModelConfig(config.model);
    trainer.setSamplingConfig(config.sampling);
    trainer.setVerbose(true);
    trainer.setOutputStream(logger);

    const auto trainConfig = config.train;
    runTask("Training", true, [this, trainConfig] {
        trainer.train(trainConfig);   // prints its own banner/ticks/summary via the logger

        // Keep a raw copy for saving (the index below holds normalized rows
        // only) before any later run overwrites the trainer's model.
        auto raw = std::make_shared<const Words::Embeddings>(trainer.getInputEmbeddings());
        auto built = std::make_shared<const Words::EmbeddingIndex>(Words::EmbeddingIndex(*raw));

        QMetaObject::invokeMethod(this, [this, raw, built] {
            embeddings = raw;
            index = built;
            emit infoUpdated();
        });
    });
}

void WordsController::saveEmbeddings()
{
    if (!embeddings) {
        out() << "Nothing to save -- train or load embeddings first." << std::endl;
        return;
    }
    if (embeddingsPath.isEmpty()) {
        out() << "No embeddings path selected." << std::endl;
        return;
    }
    auto raw = embeddings;
    const auto path = embeddingsPath.toStdString();
    runTask("Saving embeddings", false, [this, raw, path] {
        Words::Embeddings::Save(*raw, path);
        out() << "Saved embeddings to " << path << std::endl;
    });
}

void WordsController::loadEmbeddings()
{
    if (embeddingsPath.isEmpty()) {
        out() << "No embeddings file selected." << std::endl;
        return;
    }
    if (!vocabulary) {
        out() << "Load a vocabulary first -- embeddings are indexed by its word ids." << std::endl;
        return;
    }
    const auto path = embeddingsPath.toStdString();
    auto vocab = vocabulary;
    runTask("Loading embeddings", false, [this, path, vocab] {
        auto raw = std::make_shared<const Words::Embeddings>(Words::Embeddings::Load(path));
        if (raw->getWords() != vocab->getSize()) {
            out() << "Embeddings hold " << raw->getWords() << " words but the vocabulary has "
                  << vocab->getSize() << " -- they do not match." << std::endl;
            return;
        }
        auto built = std::make_shared<const Words::EmbeddingIndex>(Words::EmbeddingIndex(*raw));
        out() << "Embeddings: " << raw->getWords() << " words, dim " << raw->getDim() << std::endl;
        QMetaObject::invokeMethod(this, [this, raw, built] {
            embeddings = raw;
            index = built;
            emit infoUpdated();
        });
    });
}

// --- queries (GUI thread) ---------------------------------------------------

bool WordsController::requireIndex()
{
    if (!vocabulary || !index) {
        out() << "No embeddings available -- train or load them first." << std::endl;
        return false;
    }
    return true;
}

void WordsController::queryNeighbours(const QString& word, const int count)
{
    if (!requireIndex()) {
        return;
    }
    const auto report = Words::QueryNeighbours(
        *vocabulary, *index, word.trimmed().toStdString(), static_cast<std::size_t>(count));
    Words::PrintNeighbourReport(out(), *vocabulary, report);
}

void WordsController::runDefaultBattery()
{
    if (!requireIndex()) {
        return;
    }
    Words::PrintBatteryReport(out(), *vocabulary, Words::RunDefaultBattery(*vocabulary, *index));
}

void WordsController::queryExpression(const QString& expression, const int count)
{
    if (!requireIndex()) {
        return;
    }
    const auto report = Words::QueryExpression(
        *vocabulary, *index, expression.trimmed().toStdString(), static_cast<std::size_t>(count));
    Words::PrintExpressionReport(out(), *vocabulary, report);
}

void WordsController::queryOddOneOut(const QString& words)
{
    if (!requireIndex()) {
        return;
    }
    const auto report = Words::QueryOddOneOut(*vocabulary, *index, words.toStdString());
    Words::PrintOddOneOutReport(out(), *vocabulary, report);
}

void WordsController::queryAxis(
    const QString& axis, const QString& words, const int axisRestrictTo, const int count)
{
    if (!requireIndex()) {
        return;
    }
    const auto report = Words::QueryAxis(
        *vocabulary, *index, axis.trimmed().toStdString(), words.toStdString(),
        static_cast<std::size_t>(axisRestrictTo), static_cast<std::size_t>(count));
    Words::PrintAxisReport(out(), *vocabulary, report);
}

// --- evaluation (worker thread) ---------------------------------------------

void WordsController::evaluateAnalogies()
{
    if (!requireIndex()) {
        return;
    }
    if (analogiesPath.isEmpty()) {
        out() << "No analogies file selected." << std::endl;
        return;
    }
    auto vocab = vocabulary;   // refcounted snapshots: replacing the index
    auto idx = index;          // mid-run is safe by construction
    const auto path = analogiesPath.toStdString();
    const auto restrict = restrictTo;
    const auto threads = evaluateThreads;

    out() << "Analogy evaluation can take minutes and cannot be cancelled." << std::endl;
    runTask("Evaluating analogies", false, [this, vocab, idx, path, restrict, threads] {
        const auto report = Words::EvaluateAnalogies(*vocab, *idx, path, restrict, threads);
        if (logger) {
            Words::PrintAnalogyReport(*logger, report);
        }
    });
}

void WordsController::evaluateSimilarity()
{
    if (!requireIndex()) {
        return;
    }
    if (similarityPath.isEmpty()) {
        out() << "No similarity file selected." << std::endl;
        return;
    }
    auto vocab = vocabulary;
    auto idx = index;
    const auto path = similarityPath.toStdString();
    const auto column = scoreColumn;
    const auto name = QFileInfo(similarityPath).fileName().toStdString();

    runTask("Evaluating similarity", false, [this, vocab, idx, path, column, name] {
        const auto report = Words::EvaluateSimilarity(*vocab, *idx, path, column);
        if (logger) {
            Words::PrintSimilarityReport(*logger, name, report);
        }
    });
}
