#pragma once

#include <cstddef>
#include <functional>
#include <vector>
#include <string>
#include <optional>
#include <algorithm>
#include <filesystem>

#include "core/neural_network.h"
#include "utils/pngreader.h"

namespace recognition {

enum class BatchTrainingMode {
    SampleBySample,  // Train on each sample individually (old behavior)
    TrueBatch        // Train on all samples in batch simultaneously
};

struct RecognitionStatistics {
    unsigned passedTests = 0;
    unsigned totalTests = 0;
};

struct TestResult {
    std::vector<std::optional<RecognitionStatistics>> positions;

    TestResult(const size_t positionsCount) {
        positions.resize(positionsCount + 1);
        reset();
    }

    size_t getPositionsCount() const {
        return positions.size() - 1;
    }

    void reset() {
        std::fill(positions.begin(), positions.end(), std::nullopt);
        positions.back() = RecognitionStatistics{};
    }

    RecognitionStatistics& getTotalRef() {
        return positions.back().value();
    }

    const RecognitionStatistics& getTotal() const {
        return positions.back().value();
    }

    void setPosition(const unsigned position, const RecognitionStatistics& results) {
        auto& total = getTotalRef();
        if (positions[position].has_value()) {
            auto& value = positions[position].value();
            total.totalTests -= value.totalTests;
            total.passedTests -= value.passedTests;
        }
        positions[position] = results;
        total.totalTests += results.totalTests;
        total.passedTests += results.passedTests;
    }

    RecognitionStatistics& getPositionRef(const unsigned position) {
        if (!positions[position].has_value()) {
            positions[position] = RecognitionStatistics{};
        }
        return positions[position].value();
    }
};

class Recognizer {
public:
    using TSize = std::size_t;
    using TLayers = std::vector<TSize>;
    using TString = std::string;

    static TSize GetPredictionFast(const Matrix& prediction);
    static Matrix GenerateExpectedResult(const TSize position, const TSize count);

    virtual ~Recognizer() noexcept = default;

    void setLogger(std::ostream* stream);
    void loadNetwork(const std::string& networkName);
    void loadNetwork(const std::string& networkName, const TLayers& layers);
    void loadNetwork(const NeuralNetwork& network, const std::string& name = "unnamed");
    void setDataset(const std::filesystem::path& pathToDataset);
    void setResultCallback(std::function<void(const TestResult&)> callback);
    void setSaveOnEachIteration(const bool save);
    void requestStop();
    void saveNetwork() const;
    void doTest() const;
    void doLearning();

    bool isRunning() const;
    bool isInitialized() const;
    bool isSaveOnEachIteration() const;
    bool isStopRequested() const;

    const std::filesystem::path& getPathToDataset() const;
    const std::string& getNetworkName() const;
    const TLayers& getLayersConfiguration() const;
    PngUtils::Cache& getPngCache() const;

    virtual TestResult testNetwork() const = 0;
    virtual void learnNetwork() = 0;
    virtual const TLayers& getDefaultLayersConfiguration() const = 0;

protected:
    std::ostream& log() const;
    NeuralNetwork& getNetwork();
    const NeuralNetwork& getNetwork() const;

private:
    NeuralNetwork network;
    TString networkName;
    std::ostream* stream = &std::cout;
    bool saveOnEachIteration = true;
    std::filesystem::path pathToDataset;
    std::optional<std::function<void(const TestResult& result)>> resultCallback;
    bool running = false;
    bool stopRequested = false;
    mutable PngUtils::Cache pngCache = PngUtils::Cache({
        .max_size = 10000
    });
};

}
