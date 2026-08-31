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

    // Used by EstimateTotalPairs and the subsampler diagnostics.
    std::size_t getAffectedWordsCount() const;
    double getExpectedCorpusLength(const Vocabulary& vocabulary) const;

private:
    const bool enabled;
    std::vector<float> keepProbability;
};

// Applies the subsampler to corpus[from, to) and returns the surviving tokens.
TCorpus Subsample(
    const TCorpus& corpus,
    std::size_t from,
    std::size_t to,
    const Subsampler& subsampler,
    ::XorShift& rng
);


}
