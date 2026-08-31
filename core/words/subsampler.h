#pragma once

#include "core/words/types.h"

#include <vector>

class XorShift;

namespace Words {

class Vocabulary;

class Subsampler {
public:
    explicit Subsampler(const Vocabulary& vocabulary, double sample = 1e-4);

    bool shouldKeep(TWordId id, ::XorShift& rng) const;
    float getKeepProbability(TWordId id) const;

    // Exploration only?
    std::size_t getAffectedWordsCount() const;
    double getExpectedCorpusLength(const Vocabulary& vocabulary) const;

private:
    const bool enabled;
    std::vector<float> keepProbability;
};


}
