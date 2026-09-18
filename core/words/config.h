#pragma once

#include <cstddef>
#include <cstdint>

namespace Words {

class Vocabulary;

struct ModelConfig {
    std::size_t dim{100};
    std::size_t negatives{5};
    double initialLearningRate{0.025};
    double minLearningRateFactor{1e-4};
    std::size_t minN{3};
    std::size_t maxN{6};
    std::size_t buckets{0};
};

struct SamplingConfig {
    std::size_t window{5};
    double sample{1e-4};
    std::size_t negativeTableSize{10'000'000};
    double negativePower{0.75};
};

struct TrainConfig {
    std::size_t epochs{5};
    std::size_t threads{0};
    std::size_t chunkSize{1000};
    std::size_t syncEvery{10000};
    std::size_t reportEveryMs{3000};
    std::size_t probePairs{500};
    std::uint64_t seed{20260831};
};

struct WordsConfig {
    ModelConfig model;
    SamplingConfig sampling;
    TrainConfig train;
};

void Validate(const WordsConfig& config, std::size_t vocabularySize);

}  // namespace Words
