#pragma once

#include "core/lib/learning_config.h"
#include "core/lib/neural_network.h"
#include "core/classifier/trainer.h"

#include "gui/lib/mode_settings.h"
#include "gui/lib/mode_controller.h"

#include <QThread>
#include <QPointer>

Q_DECLARE_METATYPE(Neural::TestResult);
Q_DECLARE_METATYPE(Neural::LearningConfig);
Q_DECLARE_METATYPE(Neural::NeuralNetworkConfiguration);


class DigitsClassifierController : public ModeController
{
    Q_OBJECT
public:
    struct Info {
        bool initialized;
        bool running;
        std::string pathToTrainingDataset;
        std::string pathToTestingDataset;
        std::string networkName;
        std::vector<std::size_t> layersConfiguration;
        Neural::LearningConfig learningConfig;
    };

    explicit DigitsClassifierController(QObject* parent = nullptr);
    ~DigitsClassifierController();

    Info getInfo() const;

    void waitUntilFinished() override;
    void loadSettings();
    void saveSettings();
    void setLogger(std::ostream* stream);

public slots:
    void run(const Neural::LearningConfig& config);
    void requestStop() override;
    void testOnce();
    void loadNetwork(const QString& networkName, Neural::NeuralNetworkConfiguration config = {});
    bool setTrainingDataset(const QString& pathToDataset);
    bool setTestingDataset(const QString& pathToDataset);


signals:
    void infoUpdated();
    void updatedStatistics(const Neural::TestResult& result) const;

private:
    void updateStatistic(const Neural::TestResult& result) const;

    QPointer<QThread> internalRunner;

    ModeSettings settings;
    QString networkName;
    QString pathToTrainingDataset;
    QString pathToTestingDataset;
    Neural::LearningConfig learningConfig;
    Neural::Trainer recognizer;
};
