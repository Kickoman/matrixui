#pragma once

#include "core/words/corpus.h"
#include "core/words/types.h"

#include <cstddef>
#include <vector>

class XorShift;

namespace Words {

class Vocabulary;

class Subsampler {
public:
    explicit Subsampler(const Vocabulary& vocabulary, double sample = 1e-4);

    bool shouldKeep(TWordId id, ::XorShift& rng) const;
    float getKeepProbability(TWordId id) const;

    // How many words the threshold actually affects; getExpectedCorpusLength
    // additionally feeds EstimateTotalPairs and thus the learning-rate
    // schedule.
    std::size_t getAffectedWordsCount() const;
    double getExpectedCorpusLength(const Vocabulary& vocabulary) const;

private:
    const bool enabled;
    std::vector<float> keepProbability;
};

}
