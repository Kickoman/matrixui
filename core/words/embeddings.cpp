#include "embeddings.h"

#include "core/lib/random.h"
#include "core/lib/write.h"
#include <fstream>

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

void Embeddings::Save(const Embeddings &embeddings, const std::filesystem::path &path) {
    std::ofstream file(path, std::ios::binary);

    const auto w = static_cast<std::uint64_t>(embeddings.words);
    const auto d = static_cast<std::uint64_t>(embeddings.dim);
    WriteBinaryLE(file, w);
    WriteBinaryLE(file, d);
    WriteBulkLE(file, embeddings.data);
}

Embeddings Embeddings::Load(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    std::uint64_t w = 0;
    std::uint64_t d = 0;

    ReadBinaryLE(file, w);
    ReadBinaryLE(file, d);

    Embeddings result(w, d);
    ReadBulkLE(file, result.data);
    return result;
}

}
