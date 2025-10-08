#include "digits_recognizer.h"
#include "directory_lister.h"
#include "pngreader.h"

#include <cassert>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>


const DigitsRecognizer::TLayers DigitsRecognizer::DEFAULT_LAYERS = {DigitsRecognizer::IMAGE_H * DigitsRecognizer::IMAGE_W, 100, 100, 100, 100, 10};

void DigitsRecognizer::setLogger(std::ostream* stream) {
    this->stream = stream;
}

void DigitsRecognizer::loadNetwork(const TString& networkName) {
    this->networkName = networkName;
    try {
        network.loadWeights(networkName);
    } catch (const std::runtime_error& e) {
        log() << "Cannot load weights: " << e.what() << std::endl;
        log() << "Initialized with default values." << std::endl;
        std::mt19937 generator;
        network = NeuralNetwork(DEFAULT_LAYERS);
        network.initializeWeights(generator);
    }
}

void DigitsRecognizer::setDataset(const TString& pathToDataset) {
    this->pathToDataset = pathToDataset;
}

TestResult DigitsRecognizer::testNetwork() const {
    TestResult testResult;
    for (TDigit digitToCheck = 0; digitToCheck < 10 && !stopRequested; ++digitToCheck) {
        log() << "[test] Testing digit " << digitToCheck << std::flush;
        const TString& directory = pathToDataset + std::to_string(digitToCheck);
        const auto files = DirectoryLister::listFilesWithExtensions(directory, {".png", ".PNG"});
        RecognitionStatistics statistics;

        for (const auto& file : files) {
            log() << "\r[test] Testing digit " << digitToCheck << "; " << std::flush;
            const auto image = PngUtils::fromPNG(file).transform(1, 64*64);
            const auto prediction = network.predict(image);
            const auto result = getPredictionFast(prediction);
            ++statistics.totalTests;
            statistics.passedTests += result == digitToCheck;
            log() << "Total rate: " << 100.0 * testResult.getTotal().passedTests / testResult.getTotal().totalTests << "%                    ";
        }

        testResult.setDigit(digitToCheck, statistics);
    }
    log() << std::endl;
    return testResult;
}

void DigitsRecognizer::doTest() const {
    const auto result = testNetwork();
    printTestResult(result);
    if (resultCallback.has_value()) {
        (*resultCallback)(result);
    }
}

void DigitsRecognizer::setResultCallback(std::function<void(const TestResult&)> callback) {
    resultCallback = callback;
}

void DigitsRecognizer::printTestResult(const TestResult& result) const {
    log() << "Test result per digit:" << std::endl;
    for (unsigned digit = 0; digit < 10; ++digit) {
        const auto& r = result.digits[digit];
        if (!r.has_value()) {
            log() << "\t- " << digit << " is missing!" << std::endl;
            continue;
        }
        log() << "\t- " << digit << ": " << r->passedTests << "/" << r->totalTests << " tests passed. ";
        log() << "(" << 100.0 * r->passedTests / r->totalTests << "%)" << std::endl;
    }
    log() << "-------------" << std::endl;
    log() << "Total result: " << std::endl;
    const auto r = result.getTotal();
    log() << r.passedTests << "/" << r.totalTests << " tests passed. ";
    log() << "(" << 100.0 * r.passedTests / r.totalTests << "%)" << std::endl;
}

void DigitsRecognizer::doLearning() {
    stopRequested = false;
    TDigit digitToTrain = 0;
    while (digitToTrain < 10 && !stopRequested) {
        if (!trainDigit(digitToTrain)) {
            continue;
        }
        digitToTrain = validateAndFindNextDigit(digitToTrain);
        if (saveOnEachDigit) {
            network.saveWeights(networkName);
        }
        doTest();
    }
}

void DigitsRecognizer::requestStop() {
    stopRequested = true;
}

DigitsRecognizer::TDigit DigitsRecognizer::getPredictionFast(const Matrix& prediction) {
    assert(prediction.getRows() == 1);
    assert(prediction.getCols() == 10);
    TDigit result = 10;
    double maxProbability = -1;
    for (size_t i = 0; i < 10; ++i) {
        const auto currentProbability = prediction(0, i);
        if (currentProbability > maxProbability) {
            maxProbability = currentProbability;
            result = i;
        }
    }
    assert(result != 10);
    return result;
}

Matrix DigitsRecognizer::generateExpectedResult(const TDigit digit) {
    assert(digit < 10);
    auto res = Matrix::zeros(1, 10);
    res(0, digit) = 1;
    return res;
}

DigitsRecognizer::TSamplesList DigitsRecognizer::getBadSamples(
    const TDigit expected,
    const TString& datasetDir,
    const bool fastCircuit
) const {
    TSamplesList badSamples;
    const auto files = DirectoryLister::listFilesWithExtensions(datasetDir, {".png", ".PNG"});
    for (TSize i = 0; i < std::min(DATASET_FILE_LIMIT, files.size()); ++i) {
        log() << "\r[validation] Checking digit " << expected << " for file #" << i + 1 << "                     " << std::flush;
        const auto image = PngUtils::fromPNG(files[i]).transform(1, IMAGE_H * IMAGE_W);
        const auto prediction = network.predict(image);
        const unsigned result = getPredictionFast(prediction);
        if (result != expected) {
            log() << "\n[validation] Found bad result. Expected " << expected << ", got " << result << std::endl;
            log() << "Bad filename: " << files[i] << std::endl;
            badSamples.push_back(files[i]);
            if (fastCircuit) {
                break;
            }
        }
    }
    if (badSamples.empty()) {
        log() << "\n[validation] Digit " << expected << " is completely okay!" << std::endl;
    }
    return badSamples;
}

bool DigitsRecognizer::trainSample(const TString& sample, const auto& expectedResult, const TDigit digit) {
    log() << "Training for sample " << sample << std::endl;

    const auto image = PngUtils::fromPNG(sample).transform(1, IMAGE_H * IMAGE_W);
    unsigned int epochs = 25;
    double learningRate = LEARNING_RATE;

    do {
        network.train(image, expectedResult, epochs, learningRate, log());
        const auto prediction = getPredictionFast(network.predict(image));
        const bool passed = (prediction == digit);

        if (passed) {
            return true;
        }

        epochs = std::min(epochs * 2, EPOCHS);
        learningRate = std::min(learningRate + 0.05, 0.8);
        log() << "Bad training, continue with rate " << learningRate << std::endl;
    } while (!stopRequested);

    return false;
}

bool DigitsRecognizer::trainDigit(const TDigit digit) {
    log() << "=== Training for " << digit << std::endl;

    const auto expectedResult = generateExpectedResult(digit);
    const auto directory = pathToDataset + std::to_string(digit);
    const auto badSamples = getBadSamples(digit, directory);

    if (badSamples.empty()) {
        log() << "No bad samples found for digit " << digit << std::endl;
        return true;
    }

    for (const auto& sample : badSamples) {
        if (!trainSample(sample, expectedResult, digit)) {
            return false;
        }
    }

    return true;
}

DigitsRecognizer::TDigit DigitsRecognizer::validateAndFindNextDigit(const TDigit currentDigit) {
    for (TDigit digit = 0; digit < 10; ++digit) {
        const auto directoryToCheck = pathToDataset + std::to_string(digit);
        const auto badSamples = getBadSamples(digit, directoryToCheck, true);

        if (!badSamples.empty()) {
            log() << "Bad digit " << digit << std::endl;
            return digit;
        }

        log() << "[validation] Digit " << digit << " is completely OK!" << std::endl;
    }

    return currentDigit + 1;
}

std::ostream& DigitsRecognizer::log() const {
    return *stream;
}
