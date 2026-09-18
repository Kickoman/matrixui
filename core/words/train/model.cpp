#include "core/words/train/model.h"
#include "core/lib/random.h"
#include "core/words/train/negativesampler.h"
#include "core/words/data/types.h"
#include "core/words/data/vocabulary.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace {

class SigmoidTable {
public:
    static constexpr std::size_t kSize = 1000;
    static constexpr double kMaxExp = 6.;

    SigmoidTable() {
        for (std::size_t i = 0; i < kSize; ++i) {
            const double x = 2. * kMaxExp * static_cast<double>(i) / kSize - kMaxExp;
            values[i] = 1. / (1. + std::exp(-x));
        }
    }

    double operator()(const double score) const {
        if (score >= kMaxExp) {
            return 1.;
        }
        if (score <= -kMaxExp) {
            return 0.;
        }
        const auto index = static_cast<std::size_t>(
            (score + kMaxExp) * (static_cast<double>(kSize) / (2. * kMaxExp)));
        return values[index];
    }

private:
    std::array<double, kSize> values{};
};

const SigmoidTable sigmoid;

double LogSigmoid(const double x) {
    return x >= 0. ? -std::log1p(std::exp(-x)) : x - std::log1p(std::exp(x));
}

}  // namespace

namespace Words {

SkipGramNegativeSamplingModel::SkipGramNegativeSamplingModel(const Vocabulary& vocabulary, ModelConfig modelConfig, XorShift& rng)
    : config(modelConfig)
    , subwords(SubwordTable::Build(vocabulary, modelConfig.minN, modelConfig.maxN, modelConfig.buckets))
    , input(vocabulary.getSize(), modelConfig.dim)
    , output(vocabulary.getSize(), modelConfig.dim)
    , subwordInput(modelConfig.buckets, modelConfig.dim)
{
    input.initializeUniform(rng);
    output.initializeZero();
    subwordInput.initializeUniform(rng);
}

void SkipGramNegativeSamplingModel::trainPair(const Pair& pair, const double learningRate, const NegativeSampler& sampler, WorkerContext& context) const {
    for (auto& negative : context.negatives) {
        negative = sampler.sampleExcluding(pair.context, context.rng);
    }
    applyUpdate(pair, context.negatives, learningRate, context.gradient, context.hidden);
}

void SkipGramNegativeSamplingModel::applyUpdate(const Pair& pair, const std::span<const TWordId> negativeIds, const double learningRate, std::vector<TFloat>& gradient) const {
    std::vector<TFloat> hidden;
    if (subwords.isEnabled()) {
        hidden.resize(config.dim);
    }
    applyUpdate(pair, negativeIds, learningRate, gradient, hidden);
}

void SkipGramNegativeSamplingModel::applyUpdate(const Pair& pair, const std::span<const TWordId> negativeIds, const double learningRate, std::vector<TFloat>& gradient, std::vector<TFloat>& hidden) const {
    const auto buckets = subwords.getSubwords(pair.center);
    const TFloat* centerVector = composeCenter(pair.center, buckets, hidden);
    std::fill(gradient.begin(), gradient.end(), TFloat{0});

    accumulate(centerVector, output.row(pair.context), 1., learningRate, gradient);
    for (const auto id : negativeIds) {
        accumulate(centerVector, output.row(id), 0., learningRate, gradient);
    }

    applyCenterGradient(pair.center, buckets, gradient);
}

double SkipGramNegativeSamplingModel::computeLoss(const Pair& pair, const std::span<const TWordId> negativeIds) const {
    const auto dim = config.dim;
    const auto buckets = subwords.getSubwords(pair.center);
    std::vector<TFloat> hidden;
    if (!buckets.empty()) {
        hidden.resize(dim);
    }
    const TFloat* centerVector = composeCenter(pair.center, buckets, hidden);

    double loss = -LogSigmoid(dot(centerVector, output.row(pair.context), dim));
    for (const auto id : negativeIds) {
        loss -= LogSigmoid(-dot(centerVector, output.row(id), dim));
    }
    return loss;
}

double SkipGramNegativeSamplingModel::getLearningRateForStep(const std::size_t processed, const std::size_t total) const {
    const double progress = total > 0 ? std::min(1., 1. * processed / total) : 0.;
    const double factor = 1. - progress * (1. - config.minLearningRateFactor);
    return config.initialLearningRate * factor;
}

const TFloat* SkipGramNegativeSamplingModel::composeCenter(const TWordId id, const std::span<const TBucketId> buckets, std::vector<TFloat>& hidden) const {
    TFloat* wordVector = input.row(id);
    if (buckets.empty()) {
        return wordVector;
    }

    const auto dim = config.dim;
    if (hidden.size() < dim) {
        hidden.resize(dim);
    }

    TFloat* composed = hidden.data();
    std::copy(wordVector, wordVector + dim, composed);
    for (const auto bucket : buckets) {
        const TFloat* row = subwordInput.row(bucket);
        for (std::size_t i = 0; i < dim; ++i) {
            composed[i] += row[i];
        }
    }

    const auto scale = static_cast<TFloat>(1. / (1. + static_cast<double>(buckets.size())));
    for (std::size_t i = 0; i < dim; ++i) {
        composed[i] *= scale;
    }
    return composed;
}

void SkipGramNegativeSamplingModel::applyCenterGradient(const TWordId id, const std::span<const TBucketId> buckets, const std::vector<TFloat>& gradient) const {
    const auto dim = config.dim;

    TFloat* wordVector = input.row(id);
    for (std::size_t i = 0; i < dim; ++i) {
        wordVector[i] -= gradient[i];
    }

    for (const auto bucket : buckets) {
        TFloat* row = subwordInput.row(bucket);
        for (std::size_t i = 0; i < dim; ++i) {
            row[i] -= gradient[i];
        }
    }
}

void SkipGramNegativeSamplingModel::composeInto(const TWordId id, std::vector<TFloat>& hidden) const {
    const auto dim = config.dim;
    hidden.resize(dim);
    const TFloat* composed = composeCenter(id, subwords.getSubwords(id), hidden);
    if (composed != hidden.data()) {
        std::copy(composed, composed + dim, hidden.data());
    }
}

Embeddings SkipGramNegativeSamplingModel::composeWords() const {
    const auto dim = config.dim;
    const auto words = input.getWords();

    Embeddings result(words, dim);
    std::vector<TFloat> hidden(dim, TFloat{0});
    for (TWordId id = 0; id < words; ++id) {
        const TFloat* composed = composeCenter(id, subwords.getSubwords(id), hidden);
        std::copy(composed, composed + dim, result.row(id));
    }
    return result;
}

void SkipGramNegativeSamplingModel::accumulate(const TFloat* centerVector, TFloat* outputVector, const double label, const double learningRate, std::vector<TFloat>& gradient) const {
    const auto dim = config.dim;

    const double score = dot(centerVector, outputVector, dim);
    const double g = (sigmoid(score) - label) * learningRate;

    for (std::size_t i = 0; i < dim; ++i) {
        gradient[i] += static_cast<TFloat>(g * static_cast<double>(outputVector[i]));
    }
    for (std::size_t i = 0; i < dim; ++i) {
        outputVector[i] -= static_cast<TFloat>(g * static_cast<double>(centerVector[i]));
    }
}

}
