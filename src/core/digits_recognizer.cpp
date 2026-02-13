#include "core/digits_recognizer.h"
#include "utils/directory_lister.h"
#include "core/neural_network.h"
#include "utils/pngreader.h"

#include <cassert>
#include <ostream>
#include <string>
#include <algorithm>


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

void DigitsRecognizer::setBatchTrainingMode(BatchTrainingMode mode) {
    trainingMode = mode;
}

BatchTrainingMode DigitsRecognizer::getBatchTrainingMode() const {
    return trainingMode;
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
            const auto& total = testResult.getTotal();
            if (total.totalTests > 0) {
                log() << "Total rate: " << 100.0 * total.passedTests / total.totalTests << "%                    ";
            }
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
        if (r->totalTests > 0) {
            log() << "(" << 100.0 * r->passedTests / r->totalTests << "%)" << std::endl;
        } else {
            log() << "(N/A)" << std::endl;
        }
    }
    log() << "-------------" << std::endl;
    log() << "Total result: " << std::endl;
    const auto r = result.getTotal();
    log() << r.passedTests << "/" << r.totalTests << " tests passed. ";
    if (r.totalTests > 0) {
        log() << "(" << 100.0 * r.passedTests / r.totalTests << "%)" << std::endl;
    } else {
        log() << "(N/A)" << std::endl;
    }
}

void DigitsRecognizer::learnNetwork() {
    log() << "=== Learning iteration";
    log() << " [mode: " << (trainingMode == BatchTrainingMode::TrueBatch ? "TrueBatch" : "SampleBySample") << "]" << std::endl;

    // Collect bad samples from all digits
    auto badSamples = getAllBadSamples();

    if (badSamples.empty()) {
        log() << "All samples are correctly classified! Stopping." << std::endl;
        requestStop();
        return;
    }

    // Shuffle to mix digits together
    shuffleSamples(badSamples);

    log() << "Found " << badSamples.size() << " bad samples across all digits" << std::endl;

    // Train on mixed samples
    trainMixedSamples(badSamples);

    if (isSaveOnEachIteration() && !isStopRequested()) {
        getNetwork().saveWeights(getNetworkName());
    }
}

DigitsRecognizer::TLabeledSamplesList DigitsRecognizer::getAllBadSamples() const {
    TLabeledSamplesList allBadSamples;

    for (TDigit digit = 0; digit < 10 && !isStopRequested(); ++digit) {
        const auto directory = getPathToDataset() / std::to_string(digit);
        const auto files = DirectoryLister::listFilesWithExtensions(
            directory,
            {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"}
        );

        for (TSize i = 0; i < std::min(datasetFileLimit, files.size()); ++i) {
            log() << "\r[validation] Checking digit " << digit << " file #" << i + 1 << "          " << std::flush;

            const auto image = PngUtils::fromImage(
                files[i],
                IMAGE_H,
                IMAGE_W,
                getPngCache()
            ).transform(1, IMAGE_H * IMAGE_W);

            const auto prediction = getNetwork().predict(image);
            const unsigned result = GetPredictionFast(prediction);

            if (result != digit) {
                allBadSamples.push_back({files[i], digit});
            }
        }
    }

    log() << std::endl;
    return allBadSamples;
}

void DigitsRecognizer::filterBadSamples(TLabeledSamplesList& samples) const {
    for (TSize i = 0; i < samples.size();) {
        const auto& sample = samples[i];
        const auto image = PngUtils::fromImage(
            sample.path,
            IMAGE_H,
            IMAGE_W,
            getPngCache()
        ).transform(1, IMAGE_H * IMAGE_W);

        const auto prediction = getNetwork().predict(image);
        const unsigned result = GetPredictionFast(prediction);

        if (result == sample.digit) {
            // Correctly classified now, remove it
            std::swap(samples[i], samples.back());
            samples.pop_back();
        } else {
            ++i;
        }
    }
}

void DigitsRecognizer::shuffleSamples(TLabeledSamplesList& samples) {
    std::shuffle(samples.begin(), samples.end(), rng);
}

bool DigitsRecognizer::trainMixedSamples(TLabeledSamplesList& badSamples) {
    if (trainingMode == BatchTrainingMode::TrueBatch) {
        return trainMixedTrueBatch(badSamples);
    } else {
        return trainMixedSampleBySample(badSamples);
    }
}

bool DigitsRecognizer::trainMixedSampleBySample(TLabeledSamplesList& badSamples) {
    unsigned int epochs = 1;
    double learningRate = LEARNING_RATE;

    while (!badSamples.empty() && !isStopRequested()) {
        // Re-shuffle each round to vary the order
        shuffleSamples(badSamples);

        log() << "Training " << badSamples.size() << " samples (epochs=" << epochs << ", lr=" << learningRate << ")" << std::endl;

        for (const auto& sample : badSamples) {
            if (isStopRequested()) {
                return false;
            }

            const auto image = PngUtils::fromImage(
                sample.path,
                IMAGE_H,
                IMAGE_W,
                getPngCache()
            ).transform(1, IMAGE_H * IMAGE_W);

            const auto expectedResult = GenerateExpectedResult(sample.digit, 10);

            log() << "\rTraining digit " << sample.digit << "          " << std::flush;
            getNetwork().train(image, expectedResult, epochs, learningRate, log());
        }
        log() << std::endl;

        // Filter out samples that are now correct
        const size_t beforeFilter = badSamples.size();
        filterBadSamples(badSamples);
        const size_t fixed = beforeFilter - badSamples.size();

        log() << "Fixed " << fixed << " samples, " << badSamples.size() << " remaining" << std::endl;

        if (!badSamples.empty()) {
            epochs = std::min(epochs * 2, EPOCHS);
            learningRate = std::min(learningRate + 0.05, 1.0);
        }
    }

    return badSamples.empty();
}

bool DigitsRecognizer::trainMixedTrueBatch(TLabeledSamplesList& badSamples) {
    unsigned int epochs = 50;
    double learningRate = LEARNING_RATE;
    const size_t checks = 10;

    // while (!badSamples.empty() && !isStopRequested()) {
    for (size_t i = 0; i < checks && !isStopRequested(); ++i) {
        // Re-shuffle each round
        shuffleSamples(badSamples);

        log() << "Training batch of " << badSamples.size() << " mixed samples (epochs=" << epochs << ", lr=" << learningRate << ")" << std::endl;

        // Build batch matrices with mixed digits
        Matrix batchInputs(badSamples.size(), IMAGE_H * IMAGE_W);
        Matrix batchTargets(badSamples.size(), 10, 0.0);

        for (size_t i = 0; i < badSamples.size(); ++i) {
            const auto& sample = badSamples[i];

            const auto image = PngUtils::fromImage(
                sample.path,
                IMAGE_H,
                IMAGE_W,
                getPngCache()
            ).transform(1, IMAGE_H * IMAGE_W);

            // Copy image into row of batch matrix
            for (size_t col = 0; col < IMAGE_H * IMAGE_W; ++col) {
                batchInputs(i, col) = image(0, col);
            }

            // Set the target for this sample's digit (one-hot)
            batchTargets(i, sample.digit) = 1.0;
        }

        // Train on entire mixed batch at once
        getNetwork().train(batchInputs, batchTargets, epochs, learningRate, log());

        // Filter out samples that are now correct
        const size_t beforeFilter = badSamples.size();
        // filterBadSamples(badSamples);
        const size_t fixed = beforeFilter - badSamples.size();

        log() << "Fixed " << fixed << " samples, " << badSamples.size() << " remaining" << std::endl;

        // if (!badSamples.empty()) {
        //     epochs = std::min(epochs * 2, EPOCHS);
        //     learningRate = std::min(learningRate + 0.05, 1.0);
        // }
    }

    return badSamples.empty();
}

DigitsRecognizer::TSamplesList DigitsRecognizer::getBadSamplesForDigit(
    const TDigit expected,
    const TString& datasetDir
) const {
    TSamplesList badSamples;
    const auto files = DirectoryLister::listFilesWithExtensions(
        datasetDir,
        {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"}
    );

    for (TSize i = 0; i < std::min(datasetFileLimit, files.size()); ++i) {
        const auto image = PngUtils::fromImage(
            files[i],
            IMAGE_H,
            IMAGE_W,
            getPngCache()
        ).transform(1, IMAGE_H * IMAGE_W);

        const auto prediction = getNetwork().predict(image);
        const unsigned result = GetPredictionFast(prediction);

        if (result != expected) {
            badSamples.push_back(files[i]);
        }
    }

    return badSamples;
}

const DigitsRecognizer::TLayers& DigitsRecognizer::getDefaultLayersConfiguration() const {
    return DEFAULT_LAYERS;
}
