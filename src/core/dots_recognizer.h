#pragma once

#include "core/recognizer.h"
#include <span>


struct DatasetEntry {
    std::filesystem::path path;
    size_t index;
    size_t dotsCount;
};


struct Dataset {
    static constexpr size_t BATCH_SIZE = 50;

    bool loaded = false;
    std::vector<DatasetEntry> entries;

    static std::pair<size_t, size_t> getIndexAndDotsCount(const std::filesystem::path& path);
    bool load(const std::filesystem::path& path);
    std::span<const DatasetEntry> getBatch(const size_t batchIndex) const;
};


class DotsRecognizer : public recognition::Recognizer
{
public:
    using Base = recognition::Recognizer;
    using TestResult = recognition::TestResult;
    using RecognitionStatistics = recognition::RecognitionStatistics;
    using Batch = std::span<const DatasetEntry>;


    TestResult testNetwork() const override;
    void learnNetwork() override;
    const TLayers& getDefaultLayersConfiguration() const override;

private:
    void trainBatch(const Batch& batch);
    bool testImage(const Matrix& image, const size_t expectedIndex) const;
    bool testBatch(const Batch& batch) const;
    Batch getBatch(const size_t batchIndex) const;
    size_t getFirstBadBatchIndex() const;

    mutable Dataset dataset;
    size_t currentBatchIndex = 0;
};
