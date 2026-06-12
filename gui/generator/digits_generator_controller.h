#pragma once

#include "gui/lib/mode_controller.h"
#include "gui/lib/mode_settings.h"

#include "core/generator/gan_config.h"
#include "core/lib/neural_network.h"
#include "matrix/matrix.h"

#include <QThread>
#include <QPointer>
#include <QImage>
#include <optional>
#include <vector>
#include <atomic>

namespace Neural { namespace GAN { class GanTrainer; } }

Q_DECLARE_METATYPE(Neural::GAN::GanConfig)

class DigitsGeneratorController : public ModeController
{
    Q_OBJECT
public:
    struct Info {
        bool running = false;
        bool canRun = false;
        std::string classifierPath;
        std::string datasetPath;
        std::string generatorPath;
        std::string discriminatorPath;
    };

    explicit DigitsGeneratorController(QObject* parent = nullptr);
    ~DigitsGeneratorController();

    Info getInfo() const;

    // Generate a sample using the current generator weights (not during training).
    QImage generateSample(std::size_t label) const;

    void waitUntilFinished() override;
    void loadSettings();
    void saveSettings();
    void setLogger(std::ostream* stream);

public slots:
    void run(const Neural::GAN::GanConfig& config);
    void requestStop() override;
    bool loadClassifier(const QString& path);
    bool loadDataset(const QString& path);
    void loadGenerator(const QString& path, Neural::NeuralNetworkConfiguration config = {});
    void loadDiscriminator(const QString& path, Neural::NeuralNetworkConfiguration config = {});

signals:
    void infoUpdated();
    void epochCompleted(std::size_t epoch, double avgDiscScore, double avgGenScore, double emaReal, double emaGen);

private:
    bool canRunTraining() const;

    QPointer<QThread> internalRunner;
    std::atomic<Neural::GAN::GanTrainer*> activeTrainer{nullptr};
    std::atomic<bool> trainingRunning{false};

    ModeSettings settings;
    std::ostream* logger = nullptr;

    QString classifierPath;
    QString datasetPath;
    QString generatorPath{"generator.wgt"};
    QString discriminatorPath{"discriminator.wgt"};

    std::optional<Neural::NeuralNetwork> classifierNet;
    std::optional<Neural::NeuralNetwork> generatorNet;
    std::optional<Neural::NeuralNetwork> discriminatorNet;
    std::vector<Matrix> realSamples;
};
