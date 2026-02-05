#include "core/digits_recognizer.h"
#include "utils/directory_lister.h"
#include "core/neural_network.h"
#include "utils/pngreader.h"

#include <cassert>
#include <ostream>
#include <string>


const DigitsRecognizer::TLayers DEFAULT_LAYERS = {
    DigitsRecognizer::IMAGE_H * DigitsRecognizer::IMAGE_W,
    1000, 1000, 100, 100,  100, 100,
    10
};

void DigitsRecognizer::setTestingFileLimit(const TSize limit) {
    testingFileLimit = limit;
}

void DigitsRecognizer::setDatasetFileLimit(const TSize limit) {
    datasetFileLimit = limit;
}

DigitsRecognizer::TestResult DigitsRecognizer::testNetwork() const {
    TestResult testResult(10);
    for (TDigit digitToCheck = 0; digitToCheck < 10 && !isStopRequested(); ++digitToCheck) {
        log() << "[test] Testing digit " << digitToCheck << std::flush;
        const TString& directory = getPathToDataset() / std::to_string(digitToCheck);
        const auto files = DirectoryLister::listFilesWithExtensions(
            directory,
            {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"},
            testingFileLimit
        );
        RecognitionStatistics statistics;

        for (const auto& file : files) {
            if (isStopRequested()) {
                break;
            }
            log() << "\r[test] Testing digit " << digitToCheck << "; " << std::flush;
            const auto image = PngUtils::fromImage(
                file,
                IMAGE_H,
                IMAGE_W,
                getPngCache()
            ).transform(1, IMAGE_W*IMAGE_H);
            const auto prediction = getNetwork().predict(image);
            const auto result = GetPredictionFast(prediction);
            ++statistics.totalTests;
            statistics.passedTests += result == digitToCheck;
            log() << "Total rate: " << 100.0 * testResult.getTotal().passedTests / testResult.getTotal().totalTests << "%                    ";
        }

        testResult.setPosition(digitToCheck, statistics);
    }
    log() << std::endl;
    return testResult;
}

void DigitsRecognizer::printTestResult(const TestResult& result) const {
    log() << "Test result per digit:" << std::endl;
    for (unsigned digit = 0; digit < 10; ++digit) {
        const auto& r = result.positions[digit];
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

void DigitsRecognizer::learnNetwork() {
    if (lastTrainedDigitIsOk) {
        currentDigit = validateAndFindNextDigit(currentDigit);
    }
    lastTrainedDigitIsOk = trainDigit(currentDigit);
    if (isSaveOnEachIteration() && !isStopRequested()) {
        getNetwork().saveWeights(getNetworkName());
    }
    if (currentDigit >= 10) {
        requestStop();
    }
}

DigitsRecognizer::TSamplesList DigitsRecognizer::getBadSamples(
    const TDigit expected,
    const TString& datasetDir,
    const bool fastCircuit
) const {
    TSamplesList badSamples;
    const auto files = DirectoryLister::listFilesWithExtensions(datasetDir, {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"});
    for (TSize i = 0; i < std::min(datasetFileLimit, files.size()); ++i) {
        log() << "\r[validation] Checking digit " << expected << " for file #" << i + 1 << "                     " << std::flush;
        const auto image = PngUtils::fromImage(
            files[i],
            IMAGE_H,
            IMAGE_W,
            getPngCache()
        ).transform(1, IMAGE_H * IMAGE_W);
        const auto prediction = getNetwork().predict(image);
        const unsigned result = GetPredictionFast(prediction);
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

void DigitsRecognizer::filterBadSamples(
    TSamplesList& samples,
    const TDigit digit
) const {
    for (TSize i = 0; i < samples.size();) {
        const auto image = PngUtils::fromImage(
            samples[i],
            IMAGE_W,
            IMAGE_H,
            getPngCache()
        ).transform(1, IMAGE_H * IMAGE_W);
        const auto prediction = getNetwork().predict(image);
        const unsigned result = GetPredictionFast(prediction);
        if (result == digit) {
            std::swap(samples[i], samples.back());
            samples.pop_back();
        } else {
            ++i;
        }
    }
}

bool DigitsRecognizer::trainSample(const TString& sample, const auto& expectedResult, const TDigit digit) {
    log() << "Training for sample " << sample << std::endl;

    const auto image = PngUtils::fromImage(
        sample,
        IMAGE_W,
        IMAGE_H,
        getPngCache()
    ).transform(1, IMAGE_H * IMAGE_W);
    unsigned int epochs = 1;
    double learningRate = LEARNING_RATE;

    do {
        getNetwork().train(image, expectedResult, epochs, learningRate, log());
        const auto prediction = GetPredictionFast(getNetwork().predict(image));
        const bool passed = (prediction == digit);

        if (passed) {
            return true;
        }

        epochs = std::min(epochs * 2, EPOCHS);
        learningRate = std::min(learningRate + 0.05, 1.0);
        log() << "Bad training, continue with rate " << learningRate << std::endl;
    } while (!isStopRequested());

    return false;
}

bool DigitsRecognizer::trainDigit(const TDigit digit) {
    log() << "=== Training for " << digit << std::endl;

    const auto expectedResult = GenerateExpectedResult(digit, 10);
    const auto directory = getPathToDataset() / std::to_string(digit);
    auto badSamples = getBadSamples(digit, directory);

    if (badSamples.empty()) {
        log() << "No bad samples found for digit " << digit << std::endl;
        return true;
    }

    while (!badSamples.empty() && !isStopRequested()) {
        const auto& sample = badSamples.front();
        if (!trainSample(sample, expectedResult, digit) || isStopRequested()) {
            return false;
        }
        filterBadSamples(badSamples, digit);
    }

    return true;
}

DigitsRecognizer::TDigit DigitsRecognizer::validateAndFindNextDigit(const TDigit currentDigit) {
    for (TDigit digit = 0; digit < 10; ++digit) {
        const auto directoryToCheck = getPathToDataset() / std::to_string(digit);
        const auto badSamples = getBadSamples(digit, directoryToCheck, true);

        if (!badSamples.empty()) {
            log() << "Bad digit " << digit << std::endl;
            return digit;
        }

        log() << "[validation] Digit " << digit << " is completely OK!" << std::endl;
    }

    return currentDigit + 1;
}

const DigitsRecognizer::TLayers& DigitsRecognizer::getDefaultLayersConfiguration() const {
    return DEFAULT_LAYERS;
}
