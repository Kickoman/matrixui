#include "core/words/data/embeddings.h"

#include "core/lib/random.h"
#include "core/lib/write.h"
#include "core/words/error.h"

#include <algorithm>
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
    if (!file) {
        throw IoError("Can't open file for writing: " + path.string());
    }

    const auto w = static_cast<std::uint64_t>(embeddings.words);
    const auto d = static_cast<std::uint64_t>(embeddings.dim);
    WriteBinaryLE(file, w);
    WriteBinaryLE(file, d);
    WriteBulkLE(file, embeddings.data);

    file.flush();
    if (!file) {
        throw IoError("Failed while writing embeddings to " + path.string());
    }
}

Embeddings Embeddings::Load(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw IoError("Can't open file for reading: " + path.string());
    }

    std::uint64_t w = 0;
    std::uint64_t d = 0;
    ReadBinaryLE(file, w);
    ReadBinaryLE(file, d);
    if (!file) {
        throw IoError("Not an embeddings file (header is truncated): " + path.string());
    }

    if (w == 0 || d == 0) {
        throw IoError("Embeddings file declares an empty matrix: " + path.string());
    }

    const auto expectedBytes = static_cast<std::uintmax_t>(w) * d * sizeof(TFloat);
    std::error_code ec;
    const auto actualBytes = std::filesystem::file_size(path, ec);
    if (!ec && actualBytes < expectedBytes + 2 * sizeof(std::uint64_t)) {
        throw IoError(
            "Embeddings file is shorter than its header claims: " + path.string()
            + " (expected " + std::to_string(expectedBytes) + " bytes of data, file is "
            + std::to_string(actualBytes) + " bytes)");
    }

    Embeddings result(w, d);
    ReadBulkLE(file, result.data);
    if (!file) {
        throw IoError("Embeddings file is truncated: " + path.string());
    }
    return result;
}

}
