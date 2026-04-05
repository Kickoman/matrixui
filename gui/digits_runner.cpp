#include "digits_runner.h"
#include "directory_dataset.h"
#include "learning_config.h"
#include "neural_network.h"
#include "trainer.h"
#include "neural_network_loader.h"
#include "pngreader.h"
#include <qmessagebox.h>
#include <stdexcept>
#include <QThreadPool>
#include <QDir>


namespace {

std::unique_ptr<Neural::DirectoryDataset> LoadDataset(const QString& pathToDataset) {
    static PngUtils::Cache pngCache;
    static const auto reader = [](const std::filesystem::path& path) {
        return PngUtils::fromImage(path, 28, 28, pngCache).transform(1, 28 * 28);
    };
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


static const std::vector<std::size_t> DEFAULT_LAYERS = {
    28 * 28, 50, 20, 10
};

}


DigitsRecognizerController::DigitsRecognizerController(Neural::Trainer* recognizer)
    : recognizer(recognizer)
{
    this->recognizer->setEpochCallback([this]{
        this->testOnce();
        Neural::SaveNetwork(this->recognizer->getNetwork(), this->networkName.toStdString());
    });
}

DigitsRecognizerController::~DigitsRecognizerController() {
    QThreadPool::globalInstance()->waitForDone();
}

void DigitsRecognizerController::run(const Neural::LearningConfig& config) {
    if (recognizer->isRunning()) {
        throw std::runtime_error("Can't start learning while learning in progress");
    }
    QThreadPool::globalInstance()->start([this, config]{
        QMetaObject::invokeMethod(this, &DigitsRecognizerController::infoUpdated);
        recognizer->train(config);
        QMetaObject::invokeMethod(this, &DigitsRecognizerController::infoUpdated);
    });
}

void DigitsRecognizerController::requestStop() {
    recognizer->requestStop();
}

void DigitsRecognizerController::testOnce() {
    const auto result = recognizer->test();
    emit updatedStatistics(result);
}

void DigitsRecognizerController::loadNetwork(const QString& network, Neural::NeuralNetworkConfiguration config) {
    if (recognizer->isRunning()) {
        throw std::runtime_error("Can't load network, while learning is running");
    }
    if (config.layersSizes.empty()) {
        config.layersSizes = DEFAULT_LAYERS;
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
    recognizer->setNetwork(loadedNetwork.value());
    networkName = network;

    if (recognizer->getTestingDataset()) {
        testOnce();
    }

    emit infoUpdated();
}

bool DigitsRecognizerController::setTestingDataset(const QString& pathToDataset) {
    if (recognizer->isRunning()) {
        throw std::runtime_error("Can't change datasets while learning is running");
    }
    auto dataset = LoadDataset(pathToDataset);
    if (!dataset) {
        return false;
    }
    pathToTestingDataset = pathToDataset;
    recognizer->setTestingDataset(std::move(dataset));
    emit infoUpdated();
    return true;
}

bool DigitsRecognizerController::setTrainingDataset(const QString& pathToDataset) {
    if (recognizer->isRunning()) {
        throw std::runtime_error("Can't change datasets while learning is running");
    }
    auto dataset = LoadDataset(pathToDataset);
    if (!dataset) {
        return false;
    }
    pathToTrainingDataset = pathToDataset;
    recognizer->setTrainingDataset(std::move(dataset));
    emit infoUpdated();
    return true;
}

DigitsRecognizerController::Info DigitsRecognizerController::getInfo() const {
    return {
        .initialized = true,
        .running = recognizer->isRunning(),
        .pathToTrainingDataset = pathToTrainingDataset.toStdString(),
        .pathToTestingDataset = pathToTestingDataset.toStdString(),
        .networkName = networkName.toStdString(),
        .layersConfiguration = recognizer->getNetwork().config.layersSizes,
        .learningConfig = learningConfig,
    };
}

void DigitsRecognizerController::updateStatistic(const Neural::TestResult& result) const {
    emit updatedStatistics(result);
}
