#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/words/corpus.h"
#include "core/words/model.h"

namespace Words {

class Vocabulary;
class Subsampler;
class WindowSampler;
class NegativeSampler;

struct TrainConfig {
    std::size_t epochs{5};
    std::size_t threads{0};
    std::size_t chunkSize{1000};
    std::size_t syncEvery{10000};
    std::size_t reportEveryMs{3000};
    std::size_t probePairs{500};
    std::uint64_t seed{20260831};
};

struct Probe {
    Pair pair;
    std::vector<TWordId> negatives;
};

std::vector<Probe> BuildProbeSet(
    const TCorpus& corpus,
    const Subsampler& subsampler,
    const WindowSampler& windowSampler,
    const NegativeSampler& negativeSampler,
    const ModelConfig& modelConfig,
    std::size_t count,
    std::uint64_t seed
);

double MeanProbeLoss(const SGNSModel& model, const std::vector<Probe>& probes);

std::size_t EstimateTotalPairs(
    const Vocabulary& vocabulary,
    const Subsampler& subsampler,
    const WindowSampler& windowSampler,
    std::size_t epochs
);

void Train(
    SGNSModel& model,
    const TCorpus& corpus,
    const Subsampler& subsampler,
    const WindowSampler& windowSampler,
    const NegativeSampler& negativeSampler,
    const Vocabulary& vocabulary,
    const TrainConfig& trainConfig
);

}
