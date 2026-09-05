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
    , input(vocabulary.getSize(), modelConfig.dim)
    , output(vocabulary.getSize(), modelConfig.dim)
{
    input.initializeUniform(rng);
    output.initializeZero();
}

void SkipGramNegativeSamplingModel::trainPair(const Pair& pair, const double learningRate, const NegativeSampler& sampler, WorkerContext& context) const {
    for (auto& negative : context.negatives) {
        negative = sampler.sampleExcluding(pair.context, context.rng);
    }
    applyUpdate(pair, context.negatives, learningRate, context.gradient);
}

void SkipGramNegativeSamplingModel::applyUpdate(const Pair& pair, const std::span<const TWordId> negativeIds, const double learningRate, std::vector<TFloat>& gradient) const {
    TFloat* centerVector = input.row(pair.center);
    std::fill(gradient.begin(), gradient.end(), TFloat{0});

    accumulate(centerVector, output.row(pair.context), 1., learningRate, gradient);
    for (const auto id : negativeIds) {
        accumulate(centerVector, output.row(id), 0., learningRate, gradient);
    }

    for (std::size_t i = 0; i < config.dim; ++i) {
        centerVector[i] -= gradient[i];
    }
}

double SkipGramNegativeSamplingModel::computeLoss(const Pair& pair, const std::span<const TWordId> negativeIds) const {
    const auto dim = config.dim;
    const TFloat* centerVector = input.row(pair.center);

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
