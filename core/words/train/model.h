#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/lib/random.h"
#include "core/words/config.h"
#include "core/words/data/embeddings.h"

namespace Words {

class Vocabulary;
class NegativeSampler;
struct Pair;

struct WorkerContext {
    WorkerContext(const ModelConfig& config, const std::uint64_t seed)
        : gradient(config.dim, TFloat{0})
        , negatives(config.negatives, TWordId{0})
        , rng(seed)
    {}

    std::vector<TFloat> gradient;
    std::vector<TWordId> negatives;
    XorShift rng;
};

class SkipGramNegativeSamplingModel {
public:
    SkipGramNegativeSamplingModel(const Vocabulary& vocabulary, ModelConfig config, XorShift& rng);

    void trainPair(const Pair& pair, double learningRate, const NegativeSampler& sampler, WorkerContext& context) const;
    void applyUpdate(const Pair& pair, std::span<const TWordId> negatives, double learningRate, std::vector<TFloat>& gradient) const;
    double computeLoss(const Pair& pair, std::span<const TWordId> negatives) const;
    double getLearningRateForStep(std::size_t processed, std::size_t total) const;

    const Embeddings& getInput() const { return input; }
    Embeddings& getInputMutable() { return input; }

    const Embeddings& getOutput() const { return output; }
    Embeddings& getOutputMutable() { return output; }

    const ModelConfig& getConfig() const { return config; }
    std::size_t getBytes() const { return input.getBytes() + output.getBytes(); }

private:
    void accumulate(const TFloat* centerVector, TFloat* outputVector, double label, double learningRate, std::vector<TFloat>& gradient) const;

    ModelConfig config;
    mutable Embeddings input;
    mutable Embeddings output;
};

using SGNSModel = SkipGramNegativeSamplingModel;

}
