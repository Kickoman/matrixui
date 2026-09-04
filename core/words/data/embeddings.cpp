#include "core/words/data/embeddings.h"

#include "core/lib/random.h"
#include "core/lib/write.h"
#include "core/words/error.h"

#include <algorithm>
#include <istream>
#include <ostream>
#include <string>

namespace Words {

Embeddings::Embeddings(const std::size_t words, const std::size_t dimension)
    : words(words)
    , dim(dimension)
    , data(words * dimension, TFloat{0})
{ }

TFloat* Embeddings::row(const TWordId id) {
    return data.data() + static_cast<std::size_t>(id) * dim;
}

const TFloat* Embeddings::row(const TWordId id) const {
    return data.data() + static_cast<std::size_t>(id) * dim;
}

void Embeddings::initializeUniform(XorShift& rng) {
    for (auto& value : data) {
        value = static_cast<TFloat>((rng.nextDouble() - 0.5) / static_cast<double>(dim));
    }
}

void Embeddings::initializeZero() {
    std::fill(data.begin(), data.end(), TFloat{0});
}

void Embeddings::Save(std::ostream &out, const Embeddings &embeddings) {
    const auto w = static_cast<std::uint64_t>(embeddings.words);
    const auto d = static_cast<std::uint64_t>(embeddings.dim);
    WriteBinaryLE(out, w);
    WriteBinaryLE(out, d);
    WriteBulkLE(out, embeddings.data);

    out.flush();
    if (!out) {
        throw IoError("Failed while writing embeddings");
    }
}

Embeddings Embeddings::Load(std::istream &in) {
    std::uint64_t w = 0;
    std::uint64_t d = 0;
    ReadBinaryLE(in, w);
    ReadBinaryLE(in, d);
    if (!in) {
        throw IoError("Not an embeddings file (header is truncated)");
    }

    if (w == 0 || d == 0) {
        throw IoError("Embeddings file declares an empty matrix");
    }

    // Refuse to allocate for a header that the payload cannot back. Skipped
    // when the stream cannot be positioned -- ReadBulkLE still catches it.
    const auto expectedBytes = static_cast<std::uintmax_t>(w) * d * sizeof(TFloat);
    if (const auto position = in.tellg(); position >= 0) {
        in.seekg(0, std::ios::end);
        const auto available = in.tellg() - position;
        in.seekg(position);
        if (available >= 0 && static_cast<std::uintmax_t>(available) < expectedBytes) {
            throw IoError(
                "Embeddings file is shorter than its header claims (expected "
                + std::to_string(expectedBytes) + " bytes of data, found "
                + std::to_string(available) + ")");
        }
    }

    Embeddings result(w, d);
    ReadBulkLE(in, result.data);
    if (!in) {
        throw IoError("Embeddings file is truncated");
    }
    return result;
}

}
