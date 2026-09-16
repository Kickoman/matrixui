#pragma once

#include "core/functions/applier.h"
#include "core/functions/config.h"

#include "gui/lib/mode_controller.h"
#include "gui/lib/mode_settings.h"

#include <QPointer>
#include <QString>
#include <QThread>

#include <atomic>
#include <memory>
#include <ostream>
#include <vector>


struct FunctionsSnapshot {
    struct Row {
        QString expression;
        double rank = 0;
        std::size_t copies = 0;
        std::size_t birth = 0;
        std::shared_ptr<const Genetizer::Expression> expressionCopy;
    };

    std::size_t epoch = 0;
    std::size_t uniqueCount = 0;
    std::size_t totalCount = 0;
    std::vector<Row> rows;
};
Q_DECLARE_METATYPE(FunctionsSnapshot)


class FunctionsController : public ModeController
{
    Q_OBJECT
public:
    static constexpr int kMaxCurves = 10;

    struct Info {
        bool running = false;
        bool hasSession = false;
        bool canResume = false;
        std::size_t epoch = 0;
        Genetizer::FunctionsConfig config;
        QString dataPath;
        QString configPath;
        int topCurves = 3;
    };

    explicit FunctionsController(QObject* parent = nullptr);
    ~FunctionsController() override;

    Info getInfo() const;
    void waitUntilFinished() override;

    void setLogger(std::ostream* stream);
    void loadSettings();
    void saveSettings();

    void setConfig(const Genetizer::FunctionsConfig& value);
    void setPoints(std::vector<Genetizer::Entry> value);
    void setTopCurves(int value);
    void setDataPath(const QString& path);
    void setConfigPath(const QString& path);

    bool loadConfigFile(const QString& path);
    bool saveConfigFile(const QString& path);

    std::vector<Genetizer::Entry> takeLoadedPoints();

public slots:
    void start();
    void resume();
    void requestStop() override;
    void markSessionDirty();

signals:
    void infoUpdated();
    void snapshotReady(const FunctionsSnapshot& snapshot);

private:
    struct Session {
        std::unique_ptr<Genetizer::FunctionGenetizerApplier> applier;
        std::unique_ptr<Genetizer::FunctionGenetizer> genetizer;
        std::size_t epochsDone = 0;
        bool valid = false;
    };

    std::ostream& out() const;
    bool validateForStart();
    void launch(std::shared_ptr<Session> session, bool fresh);
    static FunctionsSnapshot MakeSnapshot(const Genetizer::FunctionGenetizer& genetizer, std::size_t epoch, std::size_t top);

    QPointer<QThread> internalRunner;
    std::atomic<bool> runningFlag{false};
    std::atomic<bool> stopRequested{false};

    ModeSettings settings;
    std::ostream* logger = nullptr;

    Genetizer::FunctionsConfig config;
    std::vector<Genetizer::Entry> points;
    std::vector<Genetizer::Entry> loadedPoints;
    int topCurves = 3;
    QString dataPath;
    QString configPath;

    std::unique_ptr<Session> session;
    bool sessionDirty = false;
};
