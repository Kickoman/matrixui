#pragma once

#include <vector>

class XorShift;

namespace Words {

class Vocabulary;

class Subsampler {
public:
    explicit Subsampler(const Vocabulary& vocabulary, double sample = 1e-4);

    bool shouldKeep(std::size_t id, ::XorShift& rng) const;
    float getKeepProbability(std::size_t id) const;

    // Exploration only?
    std::size_t getAffectedWordsCount() const;
    double getExpectedCorpusLength(const Vocabulary& vocabulary) const;

private:
    const bool enabled;
    std::vector<float> keepProbability;
};


}
