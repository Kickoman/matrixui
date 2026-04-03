#include "digits_runner.h"
#include "digits_recognizer.h"
#include <stdexcept>
#include <QThreadPool>
#include <QDir>


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

void DigitsRecognizerController::loadNetwork(const QString& network) {
    if (recognizer->isRunning()) {
        throw std::runtime_error("Can't load network, while learning is running");
    }
    recognizer->loadNetwork(network.toStdString());
    emit infoUpdated();
}

bool DigitsRecognizerController::setDataset(const QString& pathToDataset) {
    if (recognizer->isRunning()) {
        throw std::runtime_error("Can't change datasets while learning is running");
    }
    const QDir path(pathToDataset);
    for (int i = 0; i < 10; ++i) {
        const QDir subdirectory(path.filePath(QString::number(i)));
        if (!subdirectory.exists() || !subdirectory.isReadable()) {
            return false;
        }
    }
    recognizer->setDataset(pathToDataset.toStdString());
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
        .pathToDataset = recognizer->getPathToDataset(),
        .networkName = recognizer->getNetworkName(),
        .layersConfiguration = recognizer->getLayersConfiguration(),
    };
}

void DigitsRecognizerController::updateStatistic(const TestResult& result) const {
    emit updatedStatistics(result);
}
