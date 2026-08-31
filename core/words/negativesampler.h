#pragma once

#include <vector>


class XorShift;

namespace Words {

class Vocabulary;

class NegativeSampler {
public:
    explicit NegativeSampler(const Vocabulary& vocabulary, std::size_t tableSize = 10'000'000, double power = 0.75);

    std::size_t sample(XorShift& rng) const;
    std::size_t sampleExcluding(std::size_t wordId, XorShift& rng) const;
    std::size_t getTableSize() const { return table.size(); }

    // Exploration ???
    double getProbability(std::size_t wordId) const;

private:
    std::vector<std::size_t> table;
};

}
