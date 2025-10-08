#pragma once

#include <cstddef>
#include <functional>
#include <vector>
#include <string>
#include <optional>
#include <algorithm>

#include "neural_network.h"


struct RecognitionStatistics {
    unsigned passedTests = 0;
    unsigned totalTests = 0;
};

struct TestResult {
    std::array<std::optional<RecognitionStatistics>, 11> digits;  // 10 == total

    TestResult() {
        reset();
    }

    void reset() {
        std::fill(digits.begin(), digits.end(), std::nullopt);
        digits[10] = RecognitionStatistics{};
    }

    RecognitionStatistics& getTotalRef() {
        return digits[10].value();
    }

    const RecognitionStatistics& getTotal() const {
        return digits[10].value();
    }

    void setDigit(const std::uint8_t digit, const RecognitionStatistics& results) {
        auto& total = getTotalRef();
        if (digits[digit].has_value()) {
            auto& value = digits[digit].value();
            total.totalTests -= value.totalTests;
            total.passedTests -= total.passedTests;
        }
        digits[digit] = results;
        total.totalTests += results.totalTests;
        total.passedTests += results.passedTests;
    }
};

class DigitsRecognizer
{
public:
    using TDigit = std::uint8_t;
    using TString = std::string;
    template<class T>
    using TVector = std::vector<T>;
    using TSamplesList = TVector<TString>;
    using TSize = std::size_t;
    using TLayers = TVector<TSize>;
    static constexpr TSize IMAGE_W = 64;
    static constexpr TSize IMAGE_H = 64;
    static constexpr TSize DATASET_FILE_LIMIT = 10;
    static constexpr double LEARNING_RATE = 0.2;
    static constexpr unsigned EPOCHS = 1000;
    static const TLayers DEFAULT_LAYERS;

    void setLogger(std::ostream* stream);
    void loadNetwork(const TString& networkName);
    void setDataset(const TString& pathToDataset);
    void setResultCallback(std::function<void(const TestResult&)> callback);

    TestResult testNetwork() const;
    void doTest() const;
    void doLearning();
    void requestStop();
private:
    static TDigit getPredictionFast(const Matrix& prediction);
    static Matrix generateExpectedResult(const TDigit digit);
    TSamplesList getBadSamples(
        const TDigit expected,
        const TString& datasetDir,
        const bool fastCircuit = false
    ) const;
    bool trainSample(const TString& sample, const auto& expectedResult, const TDigit digit);
    bool trainDigit(const TDigit digit);


    TDigit validateAndFindNextDigit(const TDigit currentDigit);
    void performTesting();

    void initializeNetworkWeights(NeuralNetwork& network, const TString);
    void printTestResult(const TestResult& result) const;

    std::ostream& log() const;

    NeuralNetwork network;
    TString networkName;
    TString pathToDataset;
    std::ostream* stream = &std::cout;
    bool saveOnEachDigit = true;
    bool stopRequested = false;
    std::optional<std::function<void(const TestResult& result)>> resultCallback;
};
