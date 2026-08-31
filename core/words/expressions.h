#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "core/words/similarity.h"
#include "core/words/types.h"

namespace Words {

class Vocabulary;

struct ExpressionTerm {
    TWordId id;
    double sign;
};

std::vector<ExpressionTerm> ParseExpression(
    const Vocabulary& vocabulary,
    const std::string& expression,
    std::string& error
);

std::vector<TFloat> BuildExpressionVector(
    const EmbeddingIndex& index,
    const std::vector<ExpressionTerm>& terms
);

void RunExpression(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& expression,
    std::size_t count
);

void RunOddOne(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& words
);

void RunAxis(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& axisExpression,
    const std::string& words,
    std::size_t restrictTo,
    std::size_t count
);

}
