#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "core/words/embeddings.h"
#include "core/words/types.h"

namespace Words {

class Vocabulary;

struct Neighbour {
    TWordId id;
    double similarity;
};

class EmbeddingIndex {
public:
    // Public so an index can be built from a just-trained model without a
    // round-trip through the filesystem. Normalises the rows it is given.
    explicit EmbeddingIndex(Embeddings embeddings);

    static EmbeddingIndex Load(const std::filesystem::path& path);

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
