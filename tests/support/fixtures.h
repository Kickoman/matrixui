#pragma once

#include "core/lib/random.h"
#include "core/words/corpus.h"
#include "core/words/embeddings.h"
#include "core/words/vocabulary.h"
#include "tests/support/temp_dir.h"

#include <cmath>
#include <string>
#include <vector>

namespace Tests {

// A tiny corpus with hand-controlled frequencies.
//
// "a" appears far more often than the rest so subsampling has something to
// bite on; every word clears a min-count of 2. Token counts are exact:
//   a=32, b=16, c=8, d=4, e=2, f=2   (total 64)
inline std::string ToyCorpusText() {
    std::string text;
    const auto repeat = [&text](const char* word, const int times) {
        for (int i = 0; i < times; ++i) {
            text += word;
            text += ' ';
        }
    };
    // Interleaved rather than blocked, so window sampling sees mixed contexts.
    for (int round = 0; round < 2; ++round) {
        repeat("a", 8);
        repeat("b", 4);
        repeat("a", 8);
        repeat("c", 4);
        repeat("b", 4);
        repeat("d", 2);
        repeat("e", 1);
        repeat("f", 1);
    }
    return text;
}

inline Words::Vocabulary ToyVocabulary(const TempDir& dir, const std::size_t minCount = 2) {
    const auto path = dir.write("toy.txt", ToyCorpusText());
    return Words::Vocabulary::Build(path, minCount);
}

// A larger synthetic corpus for tests that need >N distinct words.
// Word i is emitted with a Zipf-like frequency, so ids are ordered and
// frequencies are strictly decreasing.
inline std::string ZipfCorpusText(const std::size_t distinctWords, const std::size_t tokens) {
    XorShift rng(20260831);
    std::vector<double> cumulative(distinctWords);
    double total = 0.;
    for (std::size_t i = 0; i < distinctWords; ++i) {
        total += 1. / std::pow(static_cast<double>(i + 1), 0.9);
        cumulative[i] = total;
    }

    std::string text;
    text.reserve(tokens * 6);
    for (std::size_t t = 0; t < tokens; ++t) {
        const double target = rng.nextDouble() * total;
        std::size_t lo = 0;
        std::size_t hi = distinctWords - 1;
        while (lo < hi) {
            const std::size_t mid = (lo + hi) / 2;
            if (cumulative[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        text += "w" + std::to_string(lo);
        text += ' ';
    }
    return text;
}

// Embeddings whose nearest neighbours are known analytically.
//
// Every row is a unit vector; rows 0 and 1 are deliberately close (a small
// rotation apart), the rest are spread over distinct axes. So the nearest
// neighbour of 0 is always 1 and vice versa.
inline Words::Embeddings AxisAlignedEmbeddings(const std::size_t words, const std::size_t dim) {
    Words::Embeddings embeddings(words, dim);
    embeddings.initializeZero();
    for (Words::TWordId id = 0; id < words; ++id) {
        auto* row = embeddings.row(id);
        row[id % dim] = Words::TFloat{1};
    }
    // Pull row 1 towards row 0 so their cosine similarity is high but < 1.
    if (words > 1 && dim > 1) {
        auto* row = embeddings.row(1);
        row[0] = Words::TFloat{2};
        row[1] = Words::TFloat{1};
    }
    return embeddings;
}

}  // namespace Tests
