#pragma once

#include "core/words/data/embeddings.h"
#include "core/words/data/types.h"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <span>
#include <string_view>
#include <vector>

namespace Words {

class Vocabulary;

inline constexpr std::size_t DefaultMinN = 3;
inline constexpr std::size_t DefaultMaxN = 6;
inline constexpr std::size_t DefaultBuckets = 2'000'000;

inline constexpr char SubwordPrefix = '<';
inline constexpr char SubwordSuffix = '>';

std::uint32_t HashSubword(std::string_view ngram);

std::vector<TBucketId> ComputeSubwords(
    std::string_view word, std::size_t minN, std::size_t maxN, std::size_t buckets);

class SubwordVectors {
public:
    SubwordVectors() = default;
    SubwordVectors(Embeddings vectors, std::size_t minN, std::size_t maxN, std::size_t buckets);

    static void Save(std::ostream& out, const Embeddings& vectors,
                     std::size_t minN, std::size_t maxN, std::size_t buckets);
    static SubwordVectors Load(std::istream& in);

    // empty when the word has no n-grams at all
    std::vector<TFloat> compose(std::string_view word) const;
    std::vector<TFloat> compose(std::string_view word, std::size_t& subwords) const;

    const Embeddings& getVectors() const { return vectors; }
    std::size_t getMinN() const { return minN; }
    std::size_t getMaxN() const { return maxN; }
    std::size_t getBuckets() const { return buckets; }
    std::size_t getDim() const { return vectors.getDim(); }

private:
    Embeddings vectors;
    std::size_t minN{0};
    std::size_t maxN{0};
    std::size_t buckets{0};
};

class SubwordTable {
public:
    SubwordTable() = default;

    static SubwordTable Build(const Vocabulary& vocabulary,
                              std::size_t minN, std::size_t maxN, std::size_t buckets);

    std::span<const TBucketId> getSubwords(TWordId id) const {
        // Not "id + 1 >= offsets.size()": TWordId is 32 bits and wraps.
        if (static_cast<std::size_t>(id) >= getWords()) {
            return {};
        }
        return {ids.data() + offsets[id], offsets[id + 1] - offsets[id]};
    }

    std::size_t getWords() const { return offsets.empty() ? 0 : offsets.size() - 1; }
    std::size_t getMinN() const { return minN; }
    std::size_t getMaxN() const { return maxN; }
    std::size_t getBuckets() const { return buckets; }
    bool isEnabled() const { return buckets > 0; }

    std::size_t getReferences() const { return ids.size(); }
    double getAverageSubwords() const;
    std::size_t getBytes() const;

private:
    std::vector<TBucketId> ids;
    std::vector<std::uint64_t> offsets;
    std::size_t minN{0};
    std::size_t maxN{0};
    std::size_t buckets{0};
};

}  // namespace Words
