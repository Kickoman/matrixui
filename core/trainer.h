#pragma once

#include <cstddef>
#include <memory>
#include <atomic>
#include <ostream>

#include "learning_config.h"
#include "neural_network_applier.h"
#include "dataset.h"

namespace Neural {

struct TestStatistics {
    std::size_t passedTests = 0;
    std::size_t totalTests = 0;
};


struct TestResult {
    std::vector<TestStatistics> stats;

    explicit TestResult(const std::size_t indicesCount) : stats(indicesCount) {}

    void setResult(std::size_t outputIndex, const TestStatistics& stats) {
        this->stats[outputIndex] = stats;
    }

    TestStatistics getTotal() const {
        TestStatistics total;
        for (const auto& s : stats) {
            total.passedTests += s.passedTests;
            total.totalTests += s.totalTests;
        }
        return total;
    }

    std::size_t size() const {
        return stats.size();
    }
};


class Trainer {
public:

    void setNetwork(const NeuralNetwork& network);
    void setTrainingDataset(std::unique_ptr<Neural::Dataset>&& dataset);
    void setTestingDataset(std::unique_ptr<Neural::Dataset>&& dataset);
    void setEpochCallback(const std::function<void()> callback);
    void setVerbose(const bool verbose);
    void setOutputStream(std::ostream* stream);

    TestResult test(std::size_t samplesPerLabelLimit = 0);
    void train(const LearningConfig& config = {});

    bool isRunning() const;
    void requestStop();

    const NeuralNetwork& getNetwork() const;
    const Neural::Dataset* getTestingDataset() const;
    const Neural::Dataset* getTrainingDataset() const;

private:
    double trainEpoch(std::vector<Sample>& samples, const double learningRate, const LearningConfig& config);
    std::ostream& log() const;

    NeuralNetworkApplier network;
    std::unique_ptr<Neural::Dataset> trainingDataset;
    std::unique_ptr<Neural::Dataset> testingDataset;

    std::atomic<bool> stopRequested{false};
    std::atomic<bool> running{false};
    bool verbose = true;

    std::function<void()> epochCallback;
    std::ostream* stream = nullptr;
};


}
