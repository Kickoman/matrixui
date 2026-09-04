#pragma once

#include <cstddef>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

#include "core/words/data/embeddings.h"
#include "core/words/data/types.h"

namespace Words {

class Vocabulary;

struct Neighbour {
    TWordId id;
    double similarity;
};

class EmbeddingIndex {
public:
    explicit EmbeddingIndex(Embeddings embeddings);

    static EmbeddingIndex Load(std::istream& in);

    std::vector<Neighbour> nearest(TWordId id, std::size_t count) const;
    std::vector<Neighbour> nearestToVector(std::span<const TFloat> query, std::span<const TWordId> exclude, std::size_t count) const;
    std::vector<TFloat> analogyVector(TWordId a, TWordId b, TWordId c) const;

    double similarity(TWordId first, TWordId second) const;

    const Embeddings& getNormalized() const { return normalized; }
    std::size_t getDim() const { return normalized.getDim(); }
    std::size_t getWords() const { return normalized.getWords(); }

private:
    Embeddings normalized;
};

}
