#include "perfbench/registry.h"
#include "perftest/include/perfbench/benchmark.h"

#include "core/neural_network.h"

#include <random>

static const long DEFAULT_SEED = 1337;
static const size_t BENCHMARK_IMAGE_H = 10;
static const size_t BENCHMARK_IMAGE_W = 10;
static const std::vector<size_t> DEFAULT_SIZES = {
    BENCHMARK_IMAGE_H * BENCHMARK_IMAGE_W,
    100,
    100,
    100,
    70,
    70,
    50,
    10,
    10,
};
static const Matrix DEFAULT_IMAGE{
    BENCHMARK_IMAGE_H,
    BENCHMARK_IMAGE_W,
    [](size_t, size_t) -> double {
        static std::mt19937 generator(DEFAULT_SEED);
        static std::uniform_real_distribution<double> distribution(50, 25);
        return distribution(generator);
    }
};


class NeuralNetworkRecognitionBenchmark : public perfbench::Benchmark {
public:
    NeuralNetworkRecognitionBenchmark() : Benchmark("NeuralNetworkRecognition") {
        setDescription("Recognition of an 100x100 image");
    }

    void setUp() override {
        std::mt19937 generator(DEFAULT_SEED);
        network = NeuralNetwork(DEFAULT_SIZES);
        network.initializeWeights(generator);
    }

    void run() override {
        auto result = network.predict(DEFAULT_IMAGE);
        volatile double dummy = result(0, 0) * 2;
    }

private:
    Matrix defaultInput;
    NeuralNetwork network;
};


static const perfbench::BenchmarkRegistrar<NeuralNetworkRecognitionBenchmark> neural_network_recognition_registrar;
