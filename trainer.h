#pragma once

#include <cstddef>
#include <array>

#include "neural_network.h"

namespace Neural {

struct TestStatistics {
    std::size_t passedTests = 0;
    std::size_t totalTests = 0;
};

template<std::size_t OutputSize>
struct TestResult {
    std::array<TestStatistics, OutputSize> stats;

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
};


template<std::size_t OutputSize>
class Trainer {
public:

    void setNetwork(const NeuralNetwork& network);

};


}
