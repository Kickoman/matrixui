#pragma once

// Every knob of the words pipeline, in one place.
//
// ModelConfig used to live in model.h and TrainConfig in trainer.h, while
// `window` and `sample` were passed around as loose arguments belonging to no
// struct at all. They are grouped here so a caller -- a CLI subcommand, or a
// GUI config panel -- can bind to one object.
//
// JSON bindings live in config_json.h so that including this header does not
// drag nlohmann into every translation unit of the module.

#include <cstddef>
#include <cstdint>

namespace Words {

class Vocabulary;

// Shape of the model itself.
struct ModelConfig {
    std::size_t dim{100};
    std::size_t negatives{5};
    double initialLearningRate{0.025};
    double minLearningRateFactor{1e-4};
};

// Which (centre, context) pairs exist, and how negatives are drawn.
struct SamplingConfig {
    std::size_t window{5};
    double sample{1e-4};
    std::size_t negativeTableSize{10'000'000};
    double negativePower{0.75};
};

// How the training run is scheduled.
struct TrainConfig {
    std::size_t epochs{5};
    std::size_t threads{0};          // 0 = hardware_concurrency
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

// Throws Words::ConfigError when a setting cannot produce a usable run.
//
// Pass the vocabulary size to also catch a negative-sampling table smaller than
// the vocabulary, which otherwise surfaces from deep inside NegativeSampler's
// constructor -- in a GUI, only after a worker thread has already started.
void Validate(const WordsConfig& config, std::size_t vocabularySize);

}  // namespace Words
