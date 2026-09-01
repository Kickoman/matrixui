#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "core/words/query/similarity.h"
#include "core/words/data/types.h"

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

}
