#include "core/words/expressions.h"

#include "core/words/vocabulary.h"

#include <algorithm>
#include <cctype>

namespace Words {

namespace {

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}





}  // namespace


std::vector<ExpressionTerm> ParseExpression(
    const Vocabulary& vocabulary,
    const std::string& expression,
    std::string& error
) {
    std::vector<ExpressionTerm> terms;
    std::string token;
    double sign = 1.;

    auto flush = [&]() -> bool {
        if (token.empty()) {
            return true;
        }
        const auto id = vocabulary.getId(ToLower(token));
        if (!id.has_value()) {
            error = "'" + token + "' is not in the vocabulary";
            return false;
        }
        terms.push_back(ExpressionTerm{*id, sign});
        token.clear();
        return true;
    };

    for (const char symbol : expression) {
        if (symbol == '+' || symbol == '-') {
            if (!flush()) {
                return {};
            }
            sign = symbol == '+' ? 1. : -1.;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(symbol))) {
            if (!flush()) {
                return {};
            }
            continue;
        }
        token.push_back(symbol);
    }

    if (!flush()) {
        return {};
    }
    if (terms.empty()) {
        error = "empty expression";
    }
    return terms;
}


std::vector<TFloat> BuildExpressionVector(
    const EmbeddingIndex& index,
    const std::vector<ExpressionTerm>& terms
) {
    const auto dim = index.getDim();
    std::vector<TFloat> result(dim, TFloat{0});

    for (const auto& term : terms) {
        const auto* row = index.getNormalized().row(term.id);
        for (std::size_t i = 0; i < dim; ++i) {
            result[i] = static_cast<TFloat>(result[i] + term.sign * row[i]);
        }
    }
    return result;
}

}
