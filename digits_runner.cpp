#include "digits_runner.h"
#include "digits_recognizer.h"
#include "directory_dataset.h"
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

}


DigitsRecognizerController::DigitsRecognizerController(DigitsRecognizer* recognizer)
    : recognizer(recognizer)
{
    this->recognizer->setResultCallback([this](const TestResult& result){
        QMetaObject::invokeMethod(this, [this, result]() {
            emit updatedStatistics(result);
        });
    });
}

DigitsRecognizerController::~DigitsRecognizerController() {
    QThreadPool::globalInstance()->waitForDone();
}

void DigitsRecognizerController::run() {
    if (recognizer->isRunning()) {
        throw std::runtime_error("Can't start learning while learning in progress");
    }
    QThreadPool::globalInstance()->start([this]{
        QMetaObject::invokeMethod(this, &DigitsRecognizerController::infoUpdated);
        recognizer->doLearning();
        QMetaObject::invokeMethod(this, &DigitsRecognizerController::infoUpdated);
    });
}

void DigitsRecognizerController::requestStop() {
    recognizer->requestStop();
}

void DigitsRecognizerController::loadNetwork(const QString& network, const QVector<unsigned>& layers) {
    if (recognizer->isRunning()) {
        throw std::runtime_error("Can't load network, while learning is running");
    }
    std::vector<unsigned long> layersSizes;
    std::transform(layers.begin(), layers.end(), std::back_inserter(layersSizes), [](unsigned size) {
        return static_cast<unsigned long>(size);
    });
    if (layersSizes.size()) {
        recognizer->loadNetwork(network.toStdString(), layersSizes);
    } else {
        recognizer->loadNetwork(network.toStdString());
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

void DigitsRecognizerController::setSamplesLimit(const unsigned limit) {
    recognizer->setDatasetFileLimit(limit);
}

void DigitsRecognizerController::setTestingLimit(const unsigned limit) {
    recognizer->setTestingFileLimit(limit);
}

DigitsRecognizerController::Info DigitsRecognizerController::getInfo() const {
    return {
        .initialized = recognizer->isInitialized(),
        .running = recognizer->isRunning(),
        .pathToTrainingDataset = pathToTrainingDataset.toStdString(),
        .pathToTestingDataset = pathToTestingDataset.toStdString(),
        .networkName = recognizer->getNetworkName(),
        .layersConfiguration = recognizer->getLayersConfiguration(),
    };
}

void DigitsRecognizerController::updateStatistic(const TestResult& result) const {
    emit updatedStatistics(result);
}
