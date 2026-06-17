#include "gui/classifier/digits_classifier_controller.h"

#include "core/classifier//learning_config.h"
#include "core/classifier/trainer.h"

#include "core/lib/directory_dataset.h"
#include "core/lib/neural_network.h"
#include "core/lib/neural_network_loader.h"
#include "core/lib/cache.h"
#include "core/lib/matrix_cache.h"

#include "gui/lib/mode_controller.h"
#include "png/pngreader.h"

#include <QDir>
#include <QSettings>
#include <filesystem>


namespace {

std::unique_ptr<Neural::DirectoryDataset> LoadDataset(
    const QString& pathToDataset,
    std::function<Matrix(const std::filesystem::path&)>&& reader)
{
    const QDir path(pathToDataset);
    for (int i = 0; i < 10; ++i) {
        const QDir subdirectory(path.filePath(QString::number(i)));
        if (!subdirectory.exists() || !subdirectory.isReadable()) {
            return {};
        }
    }
    auto dataset = std::make_unique<Neural::DirectoryDataset>(pathToDataset.toStdString());
    dataset->setFileReader(reader);
    return dataset;
}

}


DigitsClassifierController::DigitsClassifierController(QObject* parent)
    : ModeController(parent)
    , settings("digits_classifier")
{
    recognizer.setEpochCallback([this](const Neural::Classifier::EpochLog&){
        this->testOnce();
        Neural::SaveNetwork(this->recognizer.getNetwork(), this->networkName.toStdString());
    });
}

DigitsClassifierController::~DigitsClassifierController() {
    qDebug() << "Digits classifier controller destructor";
    waitUntilFinished();
}

void DigitsClassifierController::loadSettings() {
    loadNetwork(settings.getValue("last_network_name", "network.wgt").toString());
    if (auto d = settings.getValue("last_training_dataset_path"); d.isValid()) {
        setTrainingDataset(d.toString());
    }
    if (auto d = settings.getValue("last_testing_dataset_path"); d.isValid()) {
        setTestingDataset(d.toString());
    }
    imageHeight = settings.getValue("last_image_height", 28).toULongLong();
    imageWidth = settings.getValue("last_image_width", 28).toULongLong();
}

void DigitsClassifierController::saveSettings() {
    settings.setValue("last_network_name", networkName);
    settings.setValue("last_training_dataset_path", pathToTrainingDataset);
    settings.setValue("last_testing_dataset_path", pathToTestingDataset);
    settings.setValue("last_image_width", static_cast<quint64>(imageWidth));
    settings.setValue("last_image_height", static_cast<quint64>(imageHeight));
}

void DigitsClassifierController::run(const Neural::Classifier::LearningConfig& config) {
    if (recognizer.isRunning()) {
        throw std::runtime_error("Can't start learning while learning in progress");
    }
    internalRunner = new QThread(this);
    connect(internalRunner, &QThread::started, [this, config]{
        QMetaObject::invokeMethod(this, &DigitsClassifierController::infoUpdated);
        recognizer.train(config);
        QMetaObject::invokeMethod(this, &DigitsClassifierController::infoUpdated);
    });
    connect(internalRunner, &QThread::finished, internalRunner, &QObject::deleteLater);
    internalRunner->start();
}

void DigitsClassifierController::requestStop() {
    qDebug() << "Digits classifier controller: Requesting stop";
    recognizer.requestStop();
}

void DigitsClassifierController::waitUntilFinished() {
    qDebug() << "Digits classifier controller: Gracefully waiting";
    if (!internalRunner || !internalRunner->isRunning()) {
        return;
    }
    internalRunner->quit();
    internalRunner->wait();
    qDebug() << "Digits classifier controller: Finished";
}

void DigitsClassifierController::testOnce() {
    const auto result = recognizer.test();
    emit updatedStatistics(result);
}

void DigitsClassifierController::loadNetwork(const QString& network, Neural::NeuralNetworkConfiguration config) {
    if (recognizer.isRunning()) {
        throw std::runtime_error("Can't load network, while learning is running");
    }
    if (config.layersSizes.empty()) {
        config.layersSizes = getDefaultLayers();
    }

    std::optional<Neural::NeuralNetwork> loadedNetwork;
    if (QFile::exists(network)) {
        try {
            loadedNetwork = Neural::LoadNetwork(network.toStdString());
        } catch (const std::runtime_error& e) { }
    }
    if (!loadedNetwork) {
        loadedNetwork = Neural::CreateNetwork(config);
    }
    assert(loadedNetwork);
    recognizer.setNetwork(loadedNetwork.value());
    networkName = network;

    emit infoUpdated();
}

bool DigitsClassifierController::setTestingDataset(const QString& pathToDataset) {
    if (recognizer.isRunning()) {
        throw std::runtime_error("Can't change datasets while learning is running");
    }
    auto dataset = LoadDataset(pathToDataset, [this](const std::filesystem::path& path) { return readCached(path); });
    if (!dataset) {
        return false;
    }
    pathToTestingDataset = pathToDataset;
    recognizer.setTestingDataset(std::move(dataset));
    emit infoUpdated();
    return true;
}

bool DigitsClassifierController::setTrainingDataset(const QString& pathToDataset) {
    if (recognizer.isRunning()) {
        throw std::runtime_error("Can't change datasets while learning is running");
    }
    auto dataset = LoadDataset(pathToDataset, [this](const std::filesystem::path& path) { return readCached(path); });
    if (!dataset) {
        return false;
    }
    pathToTrainingDataset = pathToDataset;
    recognizer.setTrainingDataset(std::move(dataset));
    emit infoUpdated();
    return true;
}

DigitsClassifierController::Info DigitsClassifierController::getInfo() const {
    return {
        .initialized = true,
        .running = recognizer.isRunning(),
        .pathToTrainingDataset = pathToTrainingDataset.toStdString(),
        .pathToTestingDataset = pathToTestingDataset.toStdString(),
        .networkName = networkName.toStdString(),
        .imageWidth = imageWidth,
        .imageHeight = imageHeight,
        .layersConfiguration = recognizer.getNetwork().config.layersSizes,
        .learningConfig = learningConfig,
    };
}

void DigitsClassifierController::setLogger(std::ostream* stream) {
    recognizer.setOutputStream(stream);
}

void DigitsClassifierController::updateStatistic(const Neural::Classifier::TestResult& result) const {
    emit updatedStatistics(result);
}

Matrix DigitsClassifierController::readCached(const std::filesystem::path& path) const {
    static cache::LRUCache<std::tuple<std::filesystem::path, std::size_t, std::size_t>, Matrix, ArbitraryCache::TupleHash> pngCache;
    return ArbitraryCache::DoCached(
        pngCache,
        [](const std::filesystem::path& path, std::size_t height, std::size_t width) -> Matrix {
            return PngUtils::fromImage(path.string(), height, width).transform(1, height * width);
        },
        path, imageHeight, imageWidth
    );
}

std::vector<std::size_t> DigitsClassifierController::getDefaultLayers() const {
    return {imageHeight * imageWidth, 256, 10};
}
