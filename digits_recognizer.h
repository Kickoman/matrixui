#pragma once

#include "neural_network.h"
#include "pngreader.h"

#include <atomic>
#include <filesystem>
#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

struct RecognitionStatistics {
    size_t passedTests = 0;
    size_t totalTests = 0;
};

struct TestResult {
    std::optional<RecognitionStatistics> digits[10];

    void setDigit(unsigned digit, const RecognitionStatistics& stats) {
        digits[digit] = stats;
    }

    RecognitionStatistics getTotal() const {
        RecognitionStatistics total;
        for (const auto& d : digits) {
            if (d.has_value()) {
                total.passedTests += d->passedTests;
                total.totalTests += d->totalTests;
            }
        }
        return total;
    }
};

class DigitsRecognizer {
public:
    using TString = std::string;
    using TSize = size_t;
    using TDigit = unsigned;
    using TLayers = std::vector<size_t>;
    using TSamplesList = std::vector<TString>;

    static constexpr size_t IMAGE_H = 28;
    static constexpr size_t IMAGE_W = 28;
    static const TLayers DEFAULT_LAYERS;

    void setLogger(std::ostream* stream);
    void loadNetwork(const TString& networkName, const TLayers& layers = DEFAULT_LAYERS);
    void loadNetwork(const NeuralNetwork& network, const TString& name);
    void setDataset(const std::filesystem::path& pathToDataset);
    void setDatasetFileLimit(TSize limit);
    void setTestingFileLimit(TSize limit);

    TestResult testNetwork() const;
    void doTest() const;
    void doLearning();

    void setResultCallback(std::function<void(const TestResult&)> callback);
    void setSaveOnEachDigit(bool save);

    bool isRunning() const;
    bool isInitialized() const;
    const std::filesystem::path& getPathToDataset() const;
    const TString& getNetworkName() const;
    const TLayers& getLayersConfiguration() const;

    void requestStop();
    void saveNetwork() const;

    static TDigit getPredictionFast(const Matrix& prediction);
    static Matrix generateExpectedResult(TDigit digit);

private:
    struct Sample {
        Matrix image;
        TDigit label;
    };

    // Параметры обучения
    static constexpr double INITIAL_LEARNING_RATE = 0.5;
    static constexpr double MIN_LEARNING_RATE = 0.001;
    static constexpr double LR_DECAY_ON_PLATEAU = 0.5;  // уменьшаем lr вдвое при стагнации
    static constexpr size_t BATCH_SIZE = 32;
    static constexpr size_t MAX_EPOCHS = 200;
    static constexpr size_t PATIENCE = 10;
    static constexpr double LEARNING_RATE = 0.1; // legacy

    NeuralNetwork network;
    TString networkName;
    std::filesystem::path pathToDataset;
    TSize datasetFileLimit = std::numeric_limits<TSize>::max();
    TSize testingFileLimit = std::numeric_limits<TSize>::max();
    std::ostream* stream = &std::cerr;
    mutable PngUtils::Cache pngCache;
    std::atomic<bool> stopRequested{false};
    std::atomic<bool> running{false};
    bool saveOnEachDigit = false;
    std::optional<std::function<void(const TestResult&)>> resultCallback;
    void printTestResult(const TestResult& result) const;

    // Новые методы
    std::vector<Sample> loadAllSamples() const;
    std::vector<Sample> loadSamplesForDigit(TDigit digit, TSize limit) const;
    void shuffleSamples(std::vector<Sample>& samples, std::mt19937& rng) const;
    double trainEpoch(std::vector<Sample>& samples, double learningRate, std::mt19937& rng);
    double evaluateAccuracy(const std::vector<Sample>& samples) const;

    // Legacy методы для совместимости
    TSamplesList getBadSamples(TDigit expected, const TString& datasetDir, bool fastCircuit = false) const;
    void filterBadSamples(TSamplesList& samples, TDigit digit, const TString& datasetDir, bool fastCircuit = false) const;
    bool trainSample(const TString& sample, const auto& expectedResult, TDigit digit);
    bool trainDigit(TDigit digit);
    TDigit validateAndFindNextDigit(TDigit currentDigit);

    std::ostream& log() const;
};
