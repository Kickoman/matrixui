#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

#include "core/words/query/similarity.h"

namespace Words {

class Vocabulary;

struct AnalogyStats {
    std::string name;
    std::size_t asked{0};
    std::size_t skipped{0};
    std::size_t correctAdd{0};
    std::size_t correctMul{0};

    double accuracyAdd() const { return asked > 0 ? 1. * correctAdd / asked : 0.; }
    double accuracyMul() const { return asked > 0 ? 1. * correctMul / asked : 0.; }
    std::size_t total() const { return asked + skipped; }
};

struct AnalogyReport {
    std::vector<AnalogyStats> categories;
    AnalogyStats semantic;
    AnalogyStats syntactic;
    AnalogyStats overall;
};

AnalogyReport EvaluateAnalogies(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    std::istream& file,
    std::size_t restrictTo,
    std::size_t threads
);

struct SimilarityReport {
    std::size_t asked{0};
    std::size_t skipped{0};
    double spearman{0.};
    double pearson{0.};
};

SimilarityReport EvaluateSimilarity(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    std::istream& file,
    std::size_t scoreColumn
);

}
