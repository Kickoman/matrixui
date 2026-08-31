#include "core/words/negativesampler.h"

#include "core/lib/random.h"
#include "core/words/vocabulary.h"

#include <cmath>
#include <stdexcept>


namespace Words {

NegativeSampler::NegativeSampler(const Vocabulary& vocabulary, const std::size_t tableSize, const double power) {
    const auto size = vocabulary.getSize();
    if (size == 0) {
        throw std::runtime_error("empty vocabulary");
    }
    if (tableSize < size) {
        throw std::runtime_error("table smaller than vocabulary");
    }

    double total = 0.;
    std::vector<double> weights(size);
    for (std::size_t id = 0; id < size; ++id) {
        weights[id] = std::pow(1. * vocabulary.getCount(id), power);
        total += weights[id];
    }

    table.resize(tableSize);
    std::size_t id = 0;
    double covered = weights[0] / total;

    for (std::size_t slot = 0; slot < tableSize; ++slot) {
        table[slot] = id;
        if (1. * slot / tableSize > covered) {
            ++id;
            if (id < size) {
                covered += weights[id] / total;
            } else {
                id = size - 1;
            }
        }
    }
}

std::size_t NegativeSampler::sample(XorShift& rng) const {
    return table[rng.nextInteger(table.size())];
}

std::size_t NegativeSampler::sampleExcluding(const std::size_t wordId, XorShift& rng) const {
    constexpr std::size_t maxAttempts = 8;
    for (std::size_t attempt = 0; attempt < maxAttempts; ++attempt) {
        const auto candidate = sample(rng);
        if (candidate != wordId) {
            return candidate;
        }
    }
    return sample(rng);
}

double NegativeSampler::getProbability(const std::size_t wordId) const {
    std::size_t slots = 0;
    for (const auto slot : table) {
        if (slot == wordId) {
            ++slots;
        }
    }
    return 1. * slots / table.size();
}

}
