#include "core/words/similarity.h"

#include "core/words/vocabulary.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace Words {

EmbeddingIndex::EmbeddingIndex(Embeddings embeddings)
    : normalized(std::move(embeddings))
{
    const auto dim = normalized.getDim();
    for (TWordId id = 0; id < normalized.getWords(); ++id) {
        auto* row = normalized.row(id);
        double norm = 0.;
        for (std::size_t i = 0; i < dim; ++i) {
            norm += 1. * row[i] * row[i];
        }
        norm = std::sqrt(norm);
        if (norm <= 0.) {
            continue;
        }
        for (std::size_t i = 0; i < dim; ++i) {
            row[i] = static_cast<TFloat>(row[i] / norm);
        }
    }
}

EmbeddingIndex EmbeddingIndex::Load(const std::filesystem::path& path) {
    return EmbeddingIndex(Embeddings::Load(path));
}

std::vector<Neighbour> EmbeddingIndex::nearestToVector(
    const std::span<const TFloat> query,
    const std::span<const TWordId> exclude,
    const std::size_t count
) const {
    const auto dim = normalized.getDim();

    std::vector<TFloat> unit(query.begin(), query.end());
    double norm = 0.;
    for (const auto value : unit) {
        norm += 1. * value * value;
    }
    norm = std::sqrt(norm);
    if (norm > 0.) {
        for (auto& value : unit) {
            value = static_cast<TFloat>(value / norm);
        }
    }

    std::vector<Neighbour> all;
    all.reserve(normalized.getWords());

    for (TWordId id = 0; id < normalized.getWords(); ++id) {
        if (std::find(exclude.begin(), exclude.end(), id) != exclude.end()) {
            continue;
        }
        all.push_back({id, dot(unit.data(), normalized.row(id), dim)});
    }

    const auto take = std::min(count, all.size());
    std::partial_sort(all.begin(), all.begin() + take, all.end(),
        [](const Neighbour& lhs, const Neighbour& rhs) {
            return lhs.similarity > rhs.similarity;
        });
    all.resize(take);
    return all;
}

std::vector<Neighbour> EmbeddingIndex::nearest(const TWordId id, const std::size_t count) const {
    const std::array<TWordId, 1> exclude{id};
    return nearestToVector(
        std::span<const TFloat>(normalized.row(id), normalized.getDim()),
        exclude,
        count
    );
}

std::vector<TFloat> EmbeddingIndex::analogyVector(
    const TWordId a,
    const TWordId b,
    const TWordId c
) const {
    const auto dim = normalized.getDim();
    std::vector<TFloat> result(dim);
    const auto* first = normalized.row(a);
    const auto* second = normalized.row(b);
    const auto* third = normalized.row(c);

    for (std::size_t i = 0; i < dim; ++i) {
        result[i] = static_cast<TFloat>(1. * second[i] - first[i] + third[i]);
    }
    return result;
}

double EmbeddingIndex::similarity(const TWordId first, const TWordId second) const {
    return dot(normalized.row(first), normalized.row(second), normalized.getDim());
}

void PrintNeighbours(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& word,
    const std::size_t count
) {
    const auto id = vocabulary.getId(word);
    if (!id.has_value()) {
        std::cout << "  '" << word << "' is not in the vocabulary\n";
        return;
    }

    std::cout << word << " (id " << *id << ", count " << vocabulary.getCount(*id) << "):\n";
    for (const auto& neighbour : index.nearest(*id, count)) {
        std::cout << "    " << std::setw(18) << std::left << vocabulary.getWord(neighbour.id)
                  << std::right << std::fixed << std::setprecision(4) << neighbour.similarity << '\n';
    }
    std::cout << '\n';
}

void PrintAnalogy(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& a,
    const std::string& b,
    const std::string& c,
    const std::size_t count
) {
    const auto idA = vocabulary.getId(a);
    const auto idB = vocabulary.getId(b);
    const auto idC = vocabulary.getId(c);

    if (!idA.has_value() || !idB.has_value() || !idC.has_value()) {
        std::cout << "  some of '" << a << "', '" << b << "', '" << c
                  << "' are not in the vocabulary\n\n";
        return;
    }

    const auto query = index.analogyVector(*idA, *idB, *idC);
    const std::array<TWordId, 3> exclude{*idA, *idB, *idC};

    std::cout << b << " - " << a << " + " << c << ":\n";
    for (const auto& neighbour : index.nearestToVector(query, exclude, count)) {
        std::cout << "    " << std::setw(18) << std::left << vocabulary.getWord(neighbour.id)
                  << std::right << std::fixed << std::setprecision(4) << neighbour.similarity << '\n';
    }
    std::cout << '\n';
}

void RunDefaultBattery(const Vocabulary& vocabulary, const EmbeddingIndex& index) {
    const std::vector<std::string> words = {
        "one", "king", "france", "computer", "water", "music", "red", "war"
    };

    std::cout << "=== Nearest neighbours ===\n\n";
    for (const auto& word : words) {
        PrintNeighbours(vocabulary, index, word, 8);
    }

    std::cout << "=== Analogies ===\n\n";
    PrintAnalogy(vocabulary, index, "man", "king", "woman", 5);
    PrintAnalogy(vocabulary, index, "paris", "france", "rome", 5);
    PrintAnalogy(vocabulary, index, "good", "better", "bad", 5);
}

}
