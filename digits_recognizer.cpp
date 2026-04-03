#include "digits_recognizer.h"
#include "directory_lister.h"
#include "neural_network.h"
#include "pngreader.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <numeric>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>


const DigitsRecognizer::TLayers DigitsRecognizer::DEFAULT_LAYERS = {
    DigitsRecognizer::IMAGE_H * DigitsRecognizer::IMAGE_W, 128, 10
};

void DigitsRecognizer::setLogger(std::ostream* stream) {
    this->stream = stream;
}

void DigitsRecognizer::loadNetwork(const TString& networkName, const TLayers& layers) {
    this->networkName = networkName;
    if (std::filesystem::exists(networkName)){
    try {
            network.loadWeights(networkName);
            log() << "Loaded weights from " << networkName << std::endl;
            if (network.getLayerSizes() != layers) {
            log() << "Specified layers were ignored, using the loaded ones" << std::endl;
        }
            return;
    } catch (const std::runtime_error& e) {
        log() << "Cannot load weights: " << e.what() << std::endl;
        }
    }
        std::mt19937 generator;
        network = NeuralNetwork(layers);
        network.initializeWeights(generator);
    log() << "Initialized with default values." << std::endl;
}

void DigitsRecognizer::loadNetwork(const NeuralNetwork& network, const TString& name) {
    this->networkName = name;
    this->network = network;
}

void DigitsRecognizer::setDataset(const std::filesystem::path& pathToDataset) {
    this->pathToDataset = pathToDataset;
}

void DigitsRecognizer::setDatasetFileLimit(const TSize limit) {
    datasetFileLimit = limit;
}

void DigitsRecognizer::setTestingFileLimit(const TSize limit) {
    testingFileLimit = limit;
}

// ==================== Загрузка данных ====================

std::vector<DigitsRecognizer::Sample> DigitsRecognizer::loadSamplesForDigit(
    TDigit digit, TSize limit
) const {
    std::vector<Sample> samples;
    const auto directory = pathToDataset / std::to_string(digit);
    const auto files = DirectoryLister::listFilesWithExtensions(
        directory,
        {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"},
        limit
    );

    samples.reserve(files.size());
    for (const auto& file : files) {
        if (stopRequested) break;
        auto image = PngUtils::fromImage(
            file, IMAGE_H, IMAGE_W, pngCache
        ).transform(1, IMAGE_H * IMAGE_W);
        samples.push_back({std::move(image), digit});
    }
    return samples;
}

std::vector<DigitsRecognizer::Sample> DigitsRecognizer::loadAllSamples() const {
    std::vector<Sample> allSamples;
    for (TDigit digit = 0; digit < 10 && !stopRequested; ++digit) {
        log() << "[load] Loading digit " << digit << "..." << std::flush;
        auto digitSamples = loadSamplesForDigit(digit, datasetFileLimit);
        log() << " " << digitSamples.size() << " samples" << std::endl;
        allSamples.insert(
            allSamples.end(),
            std::make_move_iterator(digitSamples.begin()),
            std::make_move_iterator(digitSamples.end())
        );
    }
    log() << "[load] Total: " << allSamples.size() << " samples loaded" << std::endl;
    return allSamples;
}

// ==================== Вспомогательные методы ====================

void DigitsRecognizer::shuffleSamples(
    std::vector<Sample>& samples, std::mt19937& rng
) const {
    std::shuffle(samples.begin(), samples.end(), rng);
}

double DigitsRecognizer::evaluateAccuracy(const std::vector<Sample>& samples) const {
    if (samples.empty()) return 0.0;

    size_t correct = 0;
    for (size_t i = 0; i < samples.size() && !stopRequested; ++i) {
        const auto prediction = network.predict(samples[i].image);
        const auto predicted = getPredictionFast(prediction);
        if (predicted == samples[i].label) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / samples.size();
}

double DigitsRecognizer::trainEpoch(
    std::vector<Sample>& samples, double learningRate, std::mt19937& rng
) {
    shuffleSamples(samples, rng);

    size_t correct = 0;
    // Количество внутренних эпох на каждый пример —
    // даём сети несколько итераций градиентного спуска на каждый пример
    constexpr unsigned INNER_EPOCHS = 5;

    for (size_t i = 0; i < samples.size() && !stopRequested; ++i) {
        const auto& sample = samples[i];
        const auto expected = generateExpectedResult(sample.label);

        // Обучаем на текущем примере несколько итераций
        network.train(sample.image, expected, INNER_EPOCHS, learningRate, log());

        // Проверяем результат
        const auto prediction = network.predict(sample.image);
        const auto predicted = getPredictionFast(prediction);
        if (predicted == sample.label) {
            ++correct;
        }

        if ((i + 1) % 10 == 0 || i + 1 == samples.size()) {
            log() << "\r[train] " << (i + 1) << "/" << samples.size()
                  << " samples, current accuracy: "
                  << 100.0 * correct / (i + 1) << "%          " << std::flush;
        }
    }

    log() << std::endl;
    return samples.empty() ? 0.0 : static_cast<double>(correct) / samples.size();
}

// ==================== Основной цикл обучения ====================

void DigitsRecognizer::doLearning() {
    running = true;
    stopRequested = false;

    // 1. Загружаем весь датасет в память один раз
    log() << "========== Loading dataset ==========" << std::endl;
    auto allSamples = loadAllSamples();

    if (allSamples.empty() || stopRequested) {
        log() << "No training data available!" << std::endl;
        running = false;
        return;
    }

    std::mt19937 rng(
        static_cast<unsigned>(
            std::chrono::steady_clock::now().time_since_epoch().count()
        )
    );

    log() << "Training samples: " << allSamples.size() << std::endl;

    // 2. Обучение эпохами на всех данных одновременно
    double learningRate = INITIAL_LEARNING_RATE;
    double bestTrainAccuracy = 0.0;
    NeuralNetwork bestNetwork = network;
    size_t epochsWithoutImprovement = 0;

    for (size_t epoch = 0; epoch < MAX_EPOCHS && !stopRequested; ++epoch) {
        log() << "========== Epoch " << (epoch + 1) << "/" << MAX_EPOCHS
              << " (lr=" << learningRate << ") ==========" << std::endl;

        // Обучение: проход по всем примерам в случайном порядке
        double trainAccuracy = trainEpoch(allSamples, learningRate, rng);
        log() << "[train] Train accuracy after pass: " << trainAccuracy * 100.0 << "%" << std::endl;

        // Проверяем реальную точность на всём наборе (без обучения)
        double realAccuracy = evaluateAccuracy(allSamples);
        log() << "[eval] Real accuracy on training set: " << realAccuracy * 100.0 << "%" << std::endl;

        // Отслеживаем лучшую модель
        if (realAccuracy > bestTrainAccuracy + 0.001) {
            bestTrainAccuracy = realAccuracy;
            bestNetwork = network;
            epochsWithoutImprovement = 0;
            log() << "[best] New best accuracy: " << bestTrainAccuracy * 100.0 << "%" << std::endl;

            if (saveOnEachDigit) {
                network.saveWeights(networkName);
            }
        } else {
            ++epochsWithoutImprovement;
            log() << "[patience] No improvement for "
                  << epochsWithoutImprovement << "/" << PATIENCE
                  << " epochs" << std::endl;
        }

        // Полный тест
        doTest();

        // Если достигли очень высокой точности — можно остановиться
        if (realAccuracy > 0.99) {
            log() << "[converged] Reached 99%+ accuracy, stopping." << std::endl;
            break;
        }

        // Early stopping
        if (epochsWithoutImprovement >= PATIENCE) {
            // Вместо остановки — уменьшаем learning rate и сбрасываем patience
            learningRate *= LR_DECAY_ON_PLATEAU;
            epochsWithoutImprovement = 0;
            log() << "[lr reduce] Reducing learning rate to " << learningRate << std::endl;

            if (learningRate < MIN_LEARNING_RATE) {
                log() << "[stopping] Learning rate too small, stopping." << std::endl;
                break;
            }

            // Восстанавливаем лучшую модель перед продолжением с меньшим lr
            network = bestNetwork;
            log() << "[restore] Restored best model before continuing." << std::endl;
        }
    }

    // Восстанавливаем лучшую модель
    network = bestNetwork;
    log() << "Restored best model with training accuracy: "
          << bestTrainAccuracy * 100.0 << "%" << std::endl;

    if (saveOnEachDigit) {
        network.saveWeights(networkName);
    }

    doTest();
    running = false;
}

// ==================== Тестирование ====================

TestResult DigitsRecognizer::testNetwork() const {
    TestResult testResult;
    for (TDigit digitToCheck = 0; digitToCheck < 10 && !stopRequested; ++digitToCheck) {
        log() << "[test] Testing digit " << digitToCheck << std::flush;
        const TString directory = pathToDataset / std::to_string(digitToCheck);
        const auto files = DirectoryLister::listFilesWithExtensions(
            directory,
            {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"},
            testingFileLimit
        );
        RecognitionStatistics statistics;

        for (const auto& file : files) {
            if (stopRequested) break;

            const auto image = PngUtils::fromImage(
                file, IMAGE_H, IMAGE_W, pngCache
            ).transform(1, IMAGE_H * IMAGE_W);

            const auto prediction = network.predict(image);
            const auto result = getPredictionFast(prediction);
            ++statistics.totalTests;
            statistics.passedTests += (result == digitToCheck);
        }

        testResult.setDigit(digitToCheck, statistics);
        const auto total = testResult.getTotal();
        if (total.totalTests > 0) {
            log() << " — cumulative: "
                  << 100.0 * total.passedTests / total.totalTests << "%" << std::endl;
        }
    }
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

void DigitsRecognizer::setSaveOnEachDigit(const bool save) {
    saveOnEachDigit = save;
}

void DigitsRecognizer::printTestResult(const TestResult& result) const {
    log() << "Test result per digit:" << std::endl;
    for (unsigned digit = 0; digit < 10; ++digit) {
        const auto& r = result.digits[digit];
        if (!r.has_value()) {
            log() << "\t- " << digit << " is missing!" << std::endl;
            continue;
        }
        double pct = r->totalTests > 0
            ? 100.0 * r->passedTests / r->totalTests
            : 0.0;
        log() << "\t- " << digit << ": " << r->passedTests
              << "/" << r->totalTests << " tests passed. "
              << "(" << pct << "%)" << std::endl;
    }
    log() << "-------------" << std::endl;
    log() << "Total result: " << std::endl;
    const auto r = result.getTotal();
    if (r.totalTests > 0) {
        log() << r.passedTests << "/" << r.totalTests << " tests passed. "
              << "(" << 100.0 * r.passedTests / r.totalTests << "%)" << std::endl;
    } else {
        log() << "No tests performed." << std::endl;
    }
}

// ==================== Утилиты ====================

bool DigitsRecognizer::isRunning() const {
    return running;
}

bool DigitsRecognizer::isInitialized() const {
    return network.isInitialized();
}

const std::filesystem::path& DigitsRecognizer::getPathToDataset() const {
    return pathToDataset;
}

const DigitsRecognizer::TString& DigitsRecognizer::getNetworkName() const {
    return networkName;
}

const DigitsRecognizer::TLayers& DigitsRecognizer::getLayersConfiguration() const {
    return network.getLayerSizes();
}

void DigitsRecognizer::requestStop() {
    stopRequested = true;
}

void DigitsRecognizer::saveNetwork() const {
    network.saveWeights(networkName);
}

DigitsRecognizer::TDigit DigitsRecognizer::getPredictionFast(const Matrix& prediction) {
    assert(prediction.getRows() == 1);
    assert(prediction.getCols() == 10);
    TDigit result = 0;
    double maxProbability = prediction(0, 0);
    for (size_t i = 1; i < 10; ++i) {
        const auto currentProbability = prediction(0, i);
        if (currentProbability > maxProbability) {
            maxProbability = currentProbability;
            result = static_cast<TDigit>(i);
        }
    }
    return result;
}

Matrix DigitsRecognizer::generateExpectedResult(const TDigit digit) {
    assert(digit < 10);
    auto res = Matrix::zeros(1, 10);
    res(0, digit) = 1;
    return res;
}

// Legacy методы — сохранены для совместимости
DigitsRecognizer::TSamplesList DigitsRecognizer::getBadSamples(
    const TDigit expected,
    const TString& datasetDir,
    const bool fastCircuit
) const {
    TSamplesList badSamples;
    const auto files = DirectoryLister::listFilesWithExtensions(
        datasetDir, {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"}
    );
    for (TSize i = 0; i < std::min(datasetFileLimit, files.size()); ++i) {
        const auto image = PngUtils::fromImage(
            files[i], IMAGE_H, IMAGE_W, pngCache
        ).transform(1, IMAGE_H * IMAGE_W);
        const auto prediction = network.predict(image);
        const unsigned result = getPredictionFast(prediction);
        if (result != expected) {
            badSamples.push_back(files[i]);
            if (fastCircuit) break;
        }
    }
    return badSamples;
}

void DigitsRecognizer::filterBadSamples(
    TSamplesList& samples,
    const TDigit digit,
    const TString& datasetDir,
    const bool fastCircuit
) const {
    for (TSize i = 0; i < samples.size();) {
        const auto image = PngUtils::fromImage(
            samples[i], IMAGE_H, IMAGE_W, pngCache
        ).transform(1, IMAGE_H * IMAGE_W);
        const auto prediction = network.predict(image);
        const unsigned result = getPredictionFast(prediction);
        if (result == digit) {
            std::swap(samples[i], samples.back());
            samples.pop_back();
        } else {
            ++i;
        }
    }
}

bool DigitsRecognizer::trainSample(
    const TString& sample, const auto& expectedResult, const TDigit digit
) {
    const auto image = PngUtils::fromImage(
        sample, IMAGE_H, IMAGE_W, pngCache
    ).transform(1, IMAGE_H * IMAGE_W);
    network.train(image, expectedResult, 1, LEARNING_RATE, log());
    const auto prediction = getPredictionFast(network.predict(image));
    return prediction == digit;
}

bool DigitsRecognizer::trainDigit(const TDigit digit) {
    return true;
}

DigitsRecognizer::TDigit DigitsRecognizer::validateAndFindNextDigit(const TDigit currentDigit) {
    return currentDigit + 1;
}

std::ostream& DigitsRecognizer::log() const {
    return *stream;
}
