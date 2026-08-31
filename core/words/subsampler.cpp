#include "core/words/subsampler.h"
#include "core/words/vocabulary.h"

#include "core/lib/random.h"

#include <cmath>

namespace Words {

Subsampler::Subsampler(const Vocabulary& vocabulary, const double sample)
    : enabled(sample > 0.0)
{
    if (!enabled) {
        return;
    }

    keepProbability.resize(vocabulary.getSize());
    for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
        const double frequency = vocabulary.getFrequency(id);
        const double ratio = sample / frequency;
        const double shouldKeep = std::sqrt(ratio) + ratio;
        keepProbability[id] = static_cast<float>(
            shouldKeep < 1.0 ? shouldKeep : 1.0
        );
    }
}

bool Subsampler::shouldKeep(const TWordId id, ::XorShift& rng) const {
    if (!enabled) {
        return true;
    }

    const float probability = keepProbability[id];
    return probability > 1.f || rng.nextDouble() < probability;
}

float Subsampler::getKeepProbability(const TWordId id) const {
    return enabled ? keepProbability[id] : 1.f;
}

std::size_t Subsampler::getAffectedWordsCount() const {
    std::size_t n = 0;
    for (const auto probability : keepProbability) {
        if (probability < 1.f) {
            ++n;
        }
    }
    return n;
}

TCorpus Subsample(
    const TCorpus& corpus,
    const std::size_t from,
    const std::size_t to,
    const Subsampler& subsampler,
    ::XorShift& rng
) {
    TCorpus result;
    result.reserve(to - from);
    for (std::size_t i = from; i < to; ++i) {
        if (subsampler.shouldKeep(corpus[i], rng)) {
            result.push_back(corpus[i]);
        }
    }
    return result;
}

double Subsampler::getExpectedCorpusLength(const Vocabulary& vocabulary) const {
    // With subsampling switched off, keepProbability is empty and the loop
    // below would report zero surviving tokens. That matters beyond the
    // reported figure: EstimateTotalPairs feeds the learning-rate schedule, and
    // a total of zero pins progress at 0 so the rate never decays.
    if (!enabled) {
        return static_cast<double>(vocabulary.getKeptTokens());
    }

    double total = 0;
    for (TWordId id = 0; id < keepProbability.size(); ++id) {
        total += 1. * vocabulary.getCount(id) * keepProbability[id];
    }
    return total;
}

}
