#pragma once

#include "gui/lib/mode_controller.h"
#include "gui/lib/mode_settings.h"

#include "core/generator/learning_config.h"
#include "core/lib/neural_network.h"
#include "matrix/matrix.h"

#include <QThread>
#include <QPointer>
#include <QImage>
#include <optional>
#include <random>
#include <vector>
#include <atomic>

namespace Neural { namespace GAN { class GanTrainer; } }

Q_DECLARE_METATYPE(Neural::GAN::LearningConfig)

class DigitsGeneratorController final : public ModeController
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
        std::size_t imageWidth;
        std::size_t imageHeight;
        std::size_t numClasses = 0;  // 0 = classifier not loaded yet
    };

    explicit DigitsGeneratorController(QObject* parent = nullptr);
    ~DigitsGeneratorController();

    Info getInfo() const;
    QImage generateSample(std::size_t label) const;

    void waitUntilFinished() override;
    void loadSettings();
    void saveSettings();
    void setLogger(std::ostream* stream);

public slots:
    void run(const Neural::GAN::LearningConfig& config);
    void requestStop() override;
    void setImageWidth(std::size_t width) { imageWidth = width; }
    void setImageHeight(std::size_t height) { imageHeight = height; }

    void setClassifierPath(const QString& path);
    void setGeneratorPath(const QString& path, Neural::NeuralNetworkConfiguration config = {});
    void setDiscriminatorPath(const QString& path, Neural::NeuralNetworkConfiguration config = {});
    void setDatasetPath(const QString& path);

signals:
    void infoUpdated();
    void epochCompleted(std::size_t epoch, double avgDiscScore, double avgGenScore, double emaReal, double emaGen);

private:
    std::ostream& out();

    bool loadClassifier();
    bool loadDataset();
    bool loadGenerator();
    bool loadDiscriminator();

    void resetProject();
    bool loadProject();


    Matrix readCached(const std::filesystem::path& path) const;
    std::function<Matrix(const std::filesystem::path& path)> reader;

    QPointer<QThread> internalRunner;
    std::atomic<Neural::GAN::GanTrainer*> activeTrainer{nullptr};
    std::atomic<bool> trainingRunning{false};

    ModeSettings settings;
    std::ostream* logger = nullptr;

    QString classifierPath;
    QString datasetPath;
    QString generatorPath{"generator.wgt"};
    QString discriminatorPath{"discriminator.wgt"};

    Neural::NeuralNetworkConfiguration generatorConfiguration;
    Neural::NeuralNetworkConfiguration discriminatorConfiguration;

    std::size_t imageWidth;
    std::size_t imageHeight;
    std::size_t latentDim = 0;

    mutable std::mt19937 rng{std::random_device{}()};

    std::optional<Neural::NeuralNetwork> classifierNet;
    std::optional<Neural::NeuralNetwork> generatorNet;
    std::optional<Neural::NeuralNetwork> discriminatorNet;
    std::vector<Matrix> realSamples;
};
