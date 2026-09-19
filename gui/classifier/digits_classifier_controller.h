#pragma once

#include "core/nn/neural_network.h"

#include "core/classifier/learning_config.h"
#include "core/classifier/trainer.h"

#include "gui/lib/mode_settings.h"
#include "gui/lib/mode_controller.h"

#include <QThread>
#include <QPointer>
#include <filesystem>

Q_DECLARE_METATYPE(Neural::Classifier::TestResult);
Q_DECLARE_METATYPE(Neural::Classifier::LearningConfig);
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
        std::size_t imageWidth;
        std::size_t imageHeight;
        std::vector<std::size_t> layersConfiguration;
        Neural::Classifier::LearningConfig learningConfig;
    };

    explicit DigitsClassifierController(QObject* parent = nullptr);
    ~DigitsClassifierController();

    Info getInfo() const;

    void waitUntilFinished() override;
    void loadSettings();
    void saveSettings();
    void setLogger(std::ostream* stream);

public slots:
    void run(const Neural::Classifier::LearningConfig& config);
    void requestStop() override;
    void testOnce();
    void loadNetwork(const QString& networkName, Neural::NeuralNetworkConfiguration config = {});
    bool setTrainingDataset(const QString& pathToDataset);
    bool setTestingDataset(const QString& pathToDataset);
    void setImageWidth(std::size_t width) { imageWidth = width; }
    void setImageHeight(std::size_t height) { imageHeight = height; }


signals:
    void infoUpdated();
    void updatedStatistics(const Neural::Classifier::TestResult& result) const;

private:
    void updateStatistic(const Neural::Classifier::TestResult& result) const;
    Matrix readCached(const std::filesystem::path& path) const;
    std::vector<std::size_t> getDefaultLayers() const;
    void adoptNetwork(Neural::NeuralNetwork&& network, const QString& name);

    QPointer<QThread> internalRunner;

    ModeSettings settings;
    QString networkName;
    QString pathToTrainingDataset;
    QString pathToTestingDataset;
    std::size_t imageWidth;
    std::size_t imageHeight;
    Neural::Classifier::LearningConfig learningConfig;
    Neural::Classifier::Trainer recognizer;
};
