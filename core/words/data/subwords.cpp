#include "core/words/data/subwords.h"

#include "core/lib/text.h"
#include "core/lib/write.h"
#include "core/words/data/vocabulary.h"
#include "core/words/error.h"

#include <istream>
#include <ostream>
#include <string>

namespace Words {

namespace {

constexpr std::uint32_t SubwordMagic = 0x57535542;
constexpr std::uint32_t SubwordVersion = 1;

constexpr std::uint32_t FnvOffsetBasis = 2166136261u;
constexpr std::uint32_t FnvPrime = 16777619u;

void Wrap(const std::string_view word, std::string& wrapped) {
    wrapped.clear();
    wrapped.push_back(SubwordPrefix);
    wrapped.append(word);
    wrapped.push_back(SubwordSuffix);
}

void Extract(
    const std::string_view word,
    const std::size_t minN,
    const std::size_t maxN,
    const std::size_t buckets,
    std::string& wrapped,
    std::vector<std::size_t>& offsets,
    std::vector<TBucketId>& result
) {
    result.clear();
    if (buckets == 0 || minN == 0 || maxN < minN) {
        return;
    }

    Wrap(word, wrapped);
    Text::Utf8CharacterOffsets(wrapped, offsets);

    const std::size_t characters = offsets.size() - 1;
    for (std::size_t n = minN; n <= maxN && n <= characters; ++n) {
        for (std::size_t start = 0; start + n <= characters; ++start) {
            const std::string_view ngram(
                wrapped.data() + offsets[start], offsets[start + n] - offsets[start]);
            result.push_back(static_cast<TBucketId>(HashSubword(ngram) % buckets));
        }
    }
}

}  // namespace

std::uint32_t HashSubword(const std::string_view ngram) {
    std::uint32_t hash = FnvOffsetBasis;
    for (const char symbol : ngram) {
        hash ^= static_cast<unsigned char>(symbol);
        hash *= FnvPrime;
    }
    return hash;
}

std::vector<TBucketId> ComputeSubwords(
    const std::string_view word,
    const std::size_t minN,
    const std::size_t maxN,
    const std::size_t buckets
) {
    std::string wrapped;
    std::vector<std::size_t> offsets;
    std::vector<TBucketId> result;
    Extract(word, minN, maxN, buckets, wrapped, offsets, result);
    return result;
}

SubwordVectors::SubwordVectors(Embeddings loaded, const std::size_t minimum, const std::size_t maximum, const std::size_t count)
    : vectors(std::move(loaded))
    , minN(minimum)
    , maxN(maximum)
    , buckets(count)
{ }

void SubwordVectors::Save(
    std::ostream& out,
    const Embeddings& vectors,
    const std::size_t minN,
    const std::size_t maxN,
    const std::size_t buckets
) {
    WriteBinaryLE(out, SubwordMagic);
    WriteBinaryLE(out, SubwordVersion);
    WriteBinaryLE(out, static_cast<std::uint64_t>(minN));
    WriteBinaryLE(out, static_cast<std::uint64_t>(maxN));
    WriteBinaryLE(out, static_cast<std::uint64_t>(buckets));
    Embeddings::Save(out, vectors);
}

SubwordVectors SubwordVectors::Load(std::istream& in) {
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    ReadBinaryLE(in, magic);
    ReadBinaryLE(in, version);
    if (!in || magic != SubwordMagic) {
        throw IoError("Not a subword file (rebuild it with train --buckets)");
    }
    if (version != SubwordVersion) {
        throw IoError("Unsupported subword file version");
    }

    std::uint64_t minN = 0;
    std::uint64_t maxN = 0;
    std::uint64_t buckets = 0;
    ReadBinaryLE(in, minN);
    ReadBinaryLE(in, maxN);
    ReadBinaryLE(in, buckets);
    if (!in) {
        throw IoError("Subword file header is truncated");
    }
    if (buckets == 0 || minN == 0 || maxN < minN) {
        throw IoError("Subword file declares an unusable n-gram range");
    }

    auto vectors = Embeddings::Load(in);
    if (vectors.getWords() != buckets) {
        throw IoError(
            "Subword file declares " + std::to_string(buckets) + " buckets but holds "
            + std::to_string(vectors.getWords()) + " rows");
    }
    return SubwordVectors(std::move(vectors), minN, maxN, buckets);
}

std::vector<TFloat> SubwordVectors::compose(const std::string_view word) const {
    std::size_t ignored = 0;
    return compose(word, ignored);
}

std::vector<TFloat> SubwordVectors::compose(const std::string_view word, std::size_t& subwords) const {
    const auto ngrams = ComputeSubwords(word, minN, maxN, buckets);
    subwords = ngrams.size();
    if (ngrams.empty()) {
        return {};
    }

    const auto dim = vectors.getDim();
    std::vector<TFloat> composed(dim, TFloat{0});
    for (const auto bucket : ngrams) {
        const TFloat* row = vectors.row(bucket);
        for (std::size_t i = 0; i < dim; ++i) {
            composed[i] += row[i];
        }
    }

    const auto scale = static_cast<TFloat>(1. / static_cast<double>(ngrams.size()));
    for (auto& value : composed) {
        value *= scale;
    }
    return composed;
}

SubwordTable SubwordTable::Build(
    const Vocabulary& vocabulary,
    const std::size_t minN,
    const std::size_t maxN,
    const std::size_t buckets
) {
    SubwordTable table;
    table.minN = minN;
    table.maxN = maxN;
    table.buckets = buckets;
    if (buckets == 0) {
        return table;
    }

    const auto size = vocabulary.getSize();
    table.offsets.reserve(static_cast<std::size_t>(size) + 1);
    table.offsets.push_back(0);

    std::string wrapped;
    std::vector<std::size_t> offsets;
    std::vector<TBucketId> current;
    for (TWordId id = 0; id < size; ++id) {
        Extract(vocabulary.getWord(id), minN, maxN, buckets, wrapped, offsets, current);
        table.ids.insert(table.ids.end(), current.begin(), current.end());
        table.offsets.push_back(table.ids.size());
    }
    return table;
}

double SubwordTable::getAverageSubwords() const {
    const auto words = getWords();
    return words > 0 ? static_cast<double>(ids.size()) / static_cast<double>(words) : 0.;
}

std::size_t SubwordTable::getBytes() const {
    return ids.size() * sizeof(TBucketId) + offsets.size() * sizeof(std::uint64_t);
}

}  // namespace Words
