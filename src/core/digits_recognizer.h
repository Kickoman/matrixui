#pragma once

#include "core/recognizer.h"

class DigitsRecognizer : public recognition::Recognizer
{
public:
    using Base = recognition::Recognizer;
    using TestResult = recognition::TestResult;
    using RecognitionStatistics = recognition::RecognitionStatistics;

    using TDigit = unsigned;
    template<class T>
    using TVector = std::vector<T>;
    using TSamplesList = TVector<TString>;
    using TSize = std::size_t;
    using TLayers = TVector<TSize>;
    static constexpr TSize IMAGE_W = 28;
    static constexpr TSize IMAGE_H = 28;
    static constexpr double LEARNING_RATE = 0.05;
    static constexpr unsigned EPOCHS = 1000;

    void setTestingFileLimit(const TSize limit);
    void setDatasetFileLimit(const TSize limit);

    TestResult testNetwork() const override;
    void learnNetwork() override;
    const TLayers& getDefaultLayersConfiguration() const override;
private:
    TSamplesList getBadSamples(
        const TDigit expected,
        const TString& datasetDir,
        const bool fastCircuit = false
    ) const;
    void filterBadSamples(TSamplesList& samples, const TDigit expected) const;
    bool trainSample(const TString& sample, const auto& expectedResult, const TDigit digit);
    bool trainDigit(const TDigit digit);


    TDigit validateAndFindNextDigit(const TDigit currentDigit);
    void performTesting();

    void initializeNetworkWeights(NeuralNetwork& network, const TString);
    void printTestResult(const TestResult& result) const;


    TSize datasetFileLimit = 10;
    TSize testingFileLimit = 150;

    // state
    TDigit currentDigit = 0;
    bool lastTrainedDigitIsOk = true;
};
