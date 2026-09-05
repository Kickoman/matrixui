#pragma once

#include "core/words/data/types.h"
#include <cstddef>
#include <iosfwd>
#include <vector>

class XorShift;

namespace Words {

using TFloat = float;

class Embeddings {
public:
    Embeddings() = default;

    Embeddings(const std::size_t words, const std::size_t dimension);
    TFloat* row(const TWordId id);
    const TFloat* row(const TWordId id) const;

    std::size_t getWords() const { return words; }
    std::size_t getDim() const { return dim; }
    std::size_t getBytes() const { return data.size() * sizeof(TFloat); }

    void initializeUniform(XorShift& rng);
    void initializeZero();

    static void Save(std::ostream& out, const Embeddings& embeddings);
    static Embeddings Load(std::istream& in);

private:
    std::size_t words{0};
    std::size_t dim{0};
    std::vector<TFloat> data;
};

inline double dot(const TFloat* __restrict a, const TFloat* __restrict b, const std::size_t dim) {
    double sum = 0.;
    for (std::size_t i = 0; i < dim; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

inline void addScaled(TFloat* __restrict a, const TFloat* __restrict b, const double scale, const std::size_t dim) {
    for (std::size_t i = 0; i < dim; ++i) {
        a[i] += static_cast<TFloat>(scale * static_cast<double>(b[i]));
    }
}

}
