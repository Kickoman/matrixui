#pragma once

#include "core/words/config.h"
#include "core/words/data/corpus.h"
#include "core/words/data/embeddings.h"
#include "core/words/query/similarity.h"
#include "core/words/train/trainer.h"
#include "core/words/data/vocabulary.h"

#include "gui/lib/mode_controller.h"
#include "gui/lib/mode_settings.h"

#include <QPointer>
#include <QThread>

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <ostream>

Q_DECLARE_METATYPE(Words::TrainProgress)

// Threading contract: every member is written on the GUI thread only. Worker
// lambdas capture immutable snapshots (shared_ptr, copies of paths) up front
// and publish results back through QMetaObject::invokeMethod. One worker
// thread at a time serves all long operations.
class WordsController final : public ModeController
{
    Q_OBJECT
public:
    struct Info {
        bool busy = false;              // any worker operation in flight
        bool training = false;          // ... and that operation is training
        QString busyLabel;

        bool hasVocabulary = false;
        bool hasCorpus = false;
        bool hasEmbeddings = false;     // raw vectors held (can be saved)
        bool hasIndex = false;          // query index held (can explore/evaluate)

        std::size_t vocabularySize = 0;
        std::size_t rawTokens = 0;
        std::size_t keptTokens = 0;
        std::size_t corpusTokens = 0;
        std::size_t embeddingDim = 0;

        QString dumpPath;
        QString vocabularyPath;
        QString corpusPath;
        QString embeddingsPath;
        QString analogiesPath;
        QString similarityPath;

        std::size_t minCount = 5;
        Words::WordsConfig config;
        std::size_t scoreColumn = 2;
        std::size_t restrictTo = 30000;
        std::size_t evaluateThreads = 0;
    };

    explicit WordsController(QObject* parent = nullptr);
    ~WordsController() override;

    Info getInfo() const;

    void waitUntilFinished() override;
    void loadSettings();
    void saveSettings();
    void setLogger(std::ostream* stream);

    // Read from the widgets at click time, never on valueChanged: that would
    // loop through setValue -> valueChanged -> infoUpdated. See docs/gui.md.
    void setConfig(const Words::WordsConfig& value);
    void setMinCount(std::size_t value);
    void setEvaluateParams(std::size_t scoreColumn, std::size_t restrictTo, std::size_t threads);

public slots:
    void requestStop() override;

    void setDumpPath(const QString& path);
    void setVocabularyPath(const QString& path);
    void setCorpusPath(const QString& path);
    void setEmbeddingsPath(const QString& path);
    void setAnalogiesPath(const QString& path);
    void setSimilarityPath(const QString& path);

    // Data & Training tab -- worker thread.
    void inspectDump();
    void buildVocabulary();
    void loadVocabulary();
    void buildCorpus();
    void loadCorpus();
    void startTraining();
    void saveEmbeddings();
    void loadEmbeddings();

    // Explore tab -- synchronous on the GUI thread (milliseconds).
    void queryNeighbours(const QString& word, int count);
    void runDefaultBattery();
    void queryExpression(const QString& expression, int count);
    void queryOddOneOut(const QString& words);
    void queryAxis(const QString& axis, const QString& words, int restrictTo, int count);

    // Evaluate tab -- worker thread.
    void evaluateAnalogies();
    void evaluateSimilarity();

signals:
    void infoUpdated();
    void trainProgressed(const Words::TrainProgress& progress);

private:
    std::ostream& out();
    void runTask(const QString& label, bool isTraining, std::function<void()> task);
    bool requireIndex();

    QPointer<QThread> internalRunner;
    std::atomic<bool> busyFlag{false};
    std::atomic<bool> trainingFlag{false};
    QString busyLabel;   // GUI thread only

    ModeSettings settings;
    std::ostream* logger = nullptr;

    QString dumpPath;
    QString vocabularyPath;
    QString corpusPath;
    QString embeddingsPath;
    QString analogiesPath;
    QString similarityPath;

    std::size_t minCount = 5;
    Words::WordsConfig config;
    std::size_t scoreColumn = 2;
    std::size_t restrictTo = 30000;
    std::size_t evaluateThreads = 0;

    // train() resets its own stop flag, so the trainer survives reruns.
    Words::Trainer trainer;

    std::shared_ptr<const Words::Vocabulary> vocabulary;
    std::shared_ptr<const Words::TCorpus> corpus;
    std::shared_ptr<const Words::Embeddings> embeddings;   // raw: the only thing worth saving
    std::shared_ptr<const Words::EmbeddingIndex> index;    // normalized: queries only
};
