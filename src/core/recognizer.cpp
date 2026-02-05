#include "core/recognizer.h"

namespace recognition {

NeuralNetwork& Recognizer::getNetwork() {
    return network;
}

const NeuralNetwork& Recognizer::getNetwork() const {
    return network;
}

bool Recognizer::isStopRequested() const {
    return stopRequested;
}

PngUtils::Cache& Recognizer::getPngCache() const {
    return pngCache;
}

Recognizer::TSize Recognizer::GetPredictionFast(const Matrix& prediction) {
    assert(prediction.getRows() == 1);
    assert(prediction.getCols() > 0);
    TSize result = 0;
    double maxProbability = -1;
    for (TSize i = 0; i < prediction.getCols(); ++i) {
        if (prediction(0, i) > maxProbability) {
            maxProbability = prediction(0, i);
            result = i;
        }
    }
    return result;
}

Matrix Recognizer::GenerateExpectedResult(const TSize position, const TSize count) {
    auto result = Matrix::zeros(1, count);
    result(0, position) = 1;
    return result;
}

void Recognizer::setLogger(std::ostream* stream) {
    this->stream = stream;
}

void Recognizer::loadNetwork(const std::string& networkName) {
    loadNetwork(networkName, getDefaultLayersConfiguration());
}

void Recognizer::loadNetwork(const std::string& networkName, const TLayers& layers) {
    this->networkName = networkName;
    try {
        if (layers != getDefaultLayersConfiguration()) {
            log() << "Specified layers were ignored, using the default ones" << std::endl;
        }
        network.loadWeights(networkName);
    } catch (const std::runtime_error& e) {
        log() << "Cannot load weights: " << e.what() << std::endl;
        log() << "Initialized with default values." << std::endl;
        std::mt19937 generator;
        network = NeuralNetwork(getDefaultLayersConfiguration());
        network.initializeWeights(generator);
    } catch (const std::exception& e) {
        log() << "Failed to load network: " << e.what() << std::endl;
        throw std::runtime_error("Failed to load network");
    }
}

void Recognizer::loadNetwork(const NeuralNetwork& network, const std::string& name) {
    this->network = network;
    this->networkName = name;
}

void Recognizer::setDataset(const std::filesystem::path& pathToDataset) {
    this->pathToDataset = pathToDataset;
}

const std::filesystem::path& Recognizer::getPathToDataset() const {
    return pathToDataset;
}

const std::string& Recognizer::getNetworkName() const {
    return networkName;
}

const Recognizer::TLayers& Recognizer::getLayersConfiguration() const {
    return network.getLayerSizes();
}

void Recognizer::setResultCallback(std::function<void(const TestResult&)> callback) {
    this->resultCallback = callback;
}

void Recognizer::setSaveOnEachIteration(const bool save) {
    this->saveOnEachIteration = save;
}

void Recognizer::requestStop() {
    stopRequested = true;
}

void Recognizer::doLearning() {
    running = true;
    stopRequested = false;
    while (!stopRequested) {
        learnNetwork();
        doTest();
    }
}

void Recognizer::doTest() const {
    const auto result = testNetwork();
    if (resultCallback.has_value()) {
        (*resultCallback)(result);
    }
}

void Recognizer::saveNetwork() const {
    network.saveWeights(networkName);
}

bool Recognizer::isRunning() const {
    return running;
}

bool Recognizer::isInitialized() const {
    return network.isInitialized();
}

bool Recognizer::isSaveOnEachIteration() const {
    return saveOnEachIteration;
}

std::ostream& Recognizer::log() const {
    return *stream;
}

}
