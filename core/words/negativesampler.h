#pragma once

#include "core/words/types.h"

#include <vector>


class XorShift;

namespace Words {

class Vocabulary;

class NegativeSampler {
public:
    explicit NegativeSampler(const Vocabulary& vocabulary, std::size_t tableSize = 10'000'000, double power = 0.75);

    TWordId sample(XorShift& rng) const;
    TWordId sampleExcluding(TWordId wordId, XorShift& rng) const;
    std::size_t getTableSize() const { return table.size(); }

    // Exploration ???
    double getProbability(TWordId wordId) const;

private:
    std::vector<TWordId> table;
};

}
