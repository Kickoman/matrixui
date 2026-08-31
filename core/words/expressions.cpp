#include "core/words/expressions.h"

#include "core/words/vocabulary.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace Words {

namespace {

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::vector<std::string> SplitWords(const std::string& text) {
    std::vector<std::string> result;
    std::istringstream stream(text);
    std::string word;
    while (stream >> word) {
        result.push_back(ToLower(word));
    }
    return result;
}

std::vector<TWordId> ResolveWords(
    const Vocabulary& vocabulary,
    const std::vector<std::string>& words,
    std::string& error
) {
    std::vector<TWordId> ids;
    ids.reserve(words.size());
    for (const auto& word : words) {
        const auto id = vocabulary.getId(word);
        if (!id.has_value()) {
            error = "'" + word + "' is not in the vocabulary";
            return {};
        }
        ids.push_back(*id);
    }
    return ids;
}

std::vector<TFloat> Normalized(std::vector<TFloat> vector) {
    double norm = 0.;
    for (const auto value : vector) {
        norm += 1. * value * value;
    }
    norm = std::sqrt(norm);
    if (norm > 0.) {
        for (auto& value : vector) {
            value = static_cast<TFloat>(value / norm);
        }
    }
    return vector;
}

bool IsAnalogyShape(const std::vector<ExpressionTerm>& terms) {
    return terms.size() == 3 && terms[0].sign > 0. && terms[1].sign < 0. && terms[2].sign > 0.;
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


void RunExpression(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& expression,
    const std::size_t count
) {
    std::string error;
    const auto terms = ParseExpression(vocabulary, expression, error);
    if (terms.empty()) {
        std::cout << "  " << error << '\n';
        return;
    }

    std::vector<TWordId> exclude;
    exclude.reserve(terms.size());
    for (const auto& term : terms) {
        exclude.push_back(term.id);
    }

    std::cout << expression << "\n\n";
    std::cout << "  3CosAdd:\n";
    for (const auto& neighbour : index.nearestToVector(BuildExpressionVector(index, terms), exclude, count)) {
        std::cout << "    " << std::setw(18) << std::left << vocabulary.getWord(neighbour.id)
                  << std::right << std::fixed << std::setprecision(4) << neighbour.similarity << '\n';
    }

    if (!IsAnalogyShape(terms)) {
        return;
    }

    const auto dim = index.getDim();
    const auto& embeddings = index.getNormalized();
    const auto* vectorB = embeddings.row(terms[0].id);
    const auto* vectorA = embeddings.row(terms[1].id);
    const auto* vectorC = embeddings.row(terms[2].id);

    std::vector<Neighbour> scored;
    scored.reserve(index.getWords());

    for (TWordId id = 0; id < index.getWords(); ++id) {
        if (std::find(exclude.begin(), exclude.end(), id) != exclude.end()) {
            continue;
        }
        const auto* candidate = embeddings.row(id);
        const double shiftedA = (dot(candidate, vectorA, dim) + 1.) / 2.;
        const double shiftedB = (dot(candidate, vectorB, dim) + 1.) / 2.;
        const double shiftedC = (dot(candidate, vectorC, dim) + 1.) / 2.;
        scored.push_back({id, shiftedB * shiftedC / (shiftedA + 0.001)});
    }

    const auto take = std::min(count, scored.size());
    std::partial_sort(scored.begin(), scored.begin() + take, scored.end(),
        [](const Neighbour& lhs, const Neighbour& rhs) { return lhs.similarity > rhs.similarity; });

    std::cout << "\n  3CosMul:\n";
    for (std::size_t i = 0; i < take; ++i) {
        std::cout << "    " << std::setw(18) << std::left << vocabulary.getWord(scored[i].id)
                  << std::right << std::fixed << std::setprecision(4) << scored[i].similarity << '\n';
    }
}


void RunOddOne(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& words
) {
    const auto tokens = SplitWords(words);
    if (tokens.size() < 3) {
        std::cout << "  need at least three words\n";
        return;
    }

    std::string error;
    const auto ids = ResolveWords(vocabulary, tokens, error);
    if (ids.empty()) {
        std::cout << "  " << error << '\n';
        return;
    }

    const auto dim = index.getDim();
    std::vector<TFloat> centroid(dim, TFloat{0});
    for (const auto id : ids) {
        const auto* row = index.getNormalized().row(id);
        for (std::size_t i = 0; i < dim; ++i) {
            centroid[i] = static_cast<TFloat>(centroid[i] + row[i]);
        }
    }
    centroid = Normalized(std::move(centroid));

    std::vector<Neighbour> scored;
    scored.reserve(ids.size());
    for (const auto id : ids) {
        scored.push_back({id, dot(centroid.data(), index.getNormalized().row(id), dim)});
    }

    std::sort(scored.begin(), scored.end(),
        [](const Neighbour& lhs, const Neighbour& rhs) { return lhs.similarity > rhs.similarity; });

    std::cout << "  similarity to the centroid:\n";
    for (const auto& entry : scored) {
        std::cout << "    " << std::setw(18) << std::left << vocabulary.getWord(entry.id)
                  << std::right << std::fixed << std::setprecision(4) << entry.similarity << '\n';
    }
    std::cout << "\n  odd one out: " << vocabulary.getWord(scored.back().id) << '\n';
}


void RunAxis(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& axisExpression,
    const std::string& words,
    const std::size_t restrictTo,
    const std::size_t count
) {
    std::string error;
    const auto terms = ParseExpression(vocabulary, axisExpression, error);
    if (terms.empty()) {
        std::cout << "  " << error << '\n';
        return;
    }

    const auto axis = Normalized(BuildExpressionVector(index, terms));
    const auto dim = index.getDim();

    std::cout << "axis: " << axisExpression << "\n\n";

    if (!words.empty()) {
        const auto tokens = SplitWords(words);
        const auto ids = ResolveWords(vocabulary, tokens, error);
        if (ids.empty()) {
            std::cout << "  " << error << '\n';
            return;
        }

        std::vector<Neighbour> scored;
        scored.reserve(ids.size());
        for (const auto id : ids) {
            scored.push_back({id, dot(axis.data(), index.getNormalized().row(id), dim)});
        }
        std::sort(scored.begin(), scored.end(),
            [](const Neighbour& lhs, const Neighbour& rhs) { return lhs.similarity > rhs.similarity; });

        for (const auto& entry : scored) {
            std::cout << "  " << std::setw(6) << std::right << std::showpos << std::fixed
                      << std::setprecision(4) << entry.similarity << std::noshowpos
                      << "  " << vocabulary.getWord(entry.id) << '\n';
        }
        return;
    }

    const std::size_t limit = restrictTo > 0
        ? std::min<std::size_t>(restrictTo, index.getWords())
        : index.getWords();

    std::vector<Neighbour> scored;
    scored.reserve(limit);
    for (TWordId id = 0; id < limit; ++id) {
        scored.push_back({id, dot(axis.data(), index.getNormalized().row(id), dim)});
    }
    std::sort(scored.begin(), scored.end(),
        [](const Neighbour& lhs, const Neighbour& rhs) { return lhs.similarity > rhs.similarity; });

    const auto take = std::min(count, scored.size());

    std::cout << "  positive end:\n";
    for (std::size_t i = 0; i < take; ++i) {
        std::cout << "  " << std::setw(6) << std::right << std::showpos << std::fixed
                  << std::setprecision(4) << scored[i].similarity << std::noshowpos
                  << "  " << vocabulary.getWord(scored[i].id) << '\n';
    }

    std::cout << "\n  negative end:\n";
    for (std::size_t i = 0; i < take; ++i) {
        const auto& entry = scored[scored.size() - 1 - i];
        std::cout << "  " << std::setw(6) << std::right << std::showpos << std::fixed
                  << std::setprecision(4) << entry.similarity << std::noshowpos
                  << "  " << vocabulary.getWord(entry.id) << '\n';
    }
}

}
