#pragma once

#include "core/recognizer.h"
#include <random>

using recognition::BatchTrainingMode;


// A sample with its expected digit label
struct LabeledSample {
    std::string path;
    unsigned digit;
};


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
    using TLabeledSamplesList = TVector<LabeledSample>;
    using TSize = std::size_t;
    using TLayers = TVector<TSize>;
    static constexpr TSize IMAGE_W = 28;
    static constexpr TSize IMAGE_H = 28;
    static constexpr double LEARNING_RATE = 0.05;
    static constexpr unsigned EPOCHS = 1000;

    void setTestingFileLimit(const TSize limit);
    void setDatasetFileLimit(const TSize limit);
    void setBatchTrainingMode(BatchTrainingMode mode);
    BatchTrainingMode getBatchTrainingMode() const;

    TestResult testNetwork() const override;
    void learnNetwork() override;
    const TLayers& getDefaultLayersConfiguration() const override;
private:
    // Collect bad samples from all digits
    TLabeledSamplesList getAllBadSamples() const;
    
    // Filter out samples that are now correctly predicted
    void filterBadSamples(TLabeledSamplesList& samples) const;
    
    // Shuffle samples randomly
    void shuffleSamples(TLabeledSamplesList& samples);
    
    // Training methods for mixed-digit batches
    bool trainMixedSamples(TLabeledSamplesList& badSamples);
    bool trainMixedSampleBySample(TLabeledSamplesList& badSamples);
    bool trainMixedTrueBatch(TLabeledSamplesList& badSamples);
    
    // Legacy single-digit methods (kept for reference but not used)
    TSamplesList getBadSamplesForDigit(const TDigit expected, const TString& datasetDir) const;

    void printTestResult(const TestResult& result) const;


    TSize datasetFileLimit = 10;
    TSize testingFileLimit = 150;
    BatchTrainingMode trainingMode = BatchTrainingMode::SampleBySample;
    std::mt19937 rng{std::random_device{}()};
};
