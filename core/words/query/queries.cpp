#include "core/words/query/queries.h"

#include "core/lib/text.h"
#include "core/words/data/vocabulary.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace Words {

namespace {

std::vector<TWordId> ResolveWords(
    const Vocabulary& vocabulary,
    const std::vector<std::string>& words,
    QueryStatus& status
) {
    std::vector<TWordId> ids;
    ids.reserve(words.size());
    for (const auto& word : words) {
        const auto id = vocabulary.getId(word);
        if (!id.has_value()) {
            status = {false, "'" + word + "' is not in the vocabulary"};
            return {};
        }
        ids.push_back(*id);
    }
    return ids;
}

void SortDescending(std::vector<ScoredWord>& scored) {
    std::sort(scored.begin(), scored.end(),
        [](const ScoredWord& lhs, const ScoredWord& rhs) { return lhs.score > rhs.score; });
}

}  // namespace

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

std::vector<ScoredWord> RankByCosMul(
    const EmbeddingIndex& index,
    const TWordId a,
    const TWordId b,
    const TWordId c,
    const std::span<const TWordId> exclude,
    const std::size_t count
) {
    const auto dim = index.getDim();
    const auto& embeddings = index.getNormalized();
    const auto* vectorA = embeddings.row(a);
    const auto* vectorB = embeddings.row(b);
    const auto* vectorC = embeddings.row(c);

    std::vector<ScoredWord> scored;
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
        [](const ScoredWord& lhs, const ScoredWord& rhs) { return lhs.score > rhs.score; });
    scored.resize(take);
    return scored;
}

std::vector<TFloat> Centroid(const EmbeddingIndex& index, const std::span<const TWordId> ids) {
    const auto dim = index.getDim();
    std::vector<TFloat> centroid(dim, TFloat{0});
    for (const auto id : ids) {
        const auto* row = index.getNormalized().row(id);
        for (std::size_t i = 0; i < dim; ++i) {
            centroid[i] = static_cast<TFloat>(centroid[i] + row[i]);
        }
    }
    return Normalized(std::move(centroid));
}

std::vector<ScoredWord> ProjectOntoAxis(
    const EmbeddingIndex& index,
    const std::span<const TFloat> axis,
    const std::span<const TWordId> ids
) {
    const auto dim = index.getDim();
    std::vector<ScoredWord> scored;
    scored.reserve(ids.size());
    for (const auto id : ids) {
        scored.push_back({id, dot(axis.data(), index.getNormalized().row(id), dim)});
    }
    SortDescending(scored);
    return scored;
}

std::vector<ScoredWord> ProjectVocabularyOntoAxis(
    const EmbeddingIndex& index,
    const std::span<const TFloat> axis,
    const std::size_t restrictTo
) {
    const auto dim = index.getDim();
    const std::size_t limit = restrictTo > 0
        ? std::min<std::size_t>(restrictTo, index.getWords())
        : index.getWords();

    std::vector<ScoredWord> scored;
    scored.reserve(limit);
    for (TWordId id = 0; id < limit; ++id) {
        scored.push_back({id, dot(axis.data(), index.getNormalized().row(id), dim)});
    }
    SortDescending(scored);
    return scored;
}

NeighbourReport QueryNeighbours(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& word,
    const std::size_t count
) {
    NeighbourReport report;
    report.word = word;
    report.count = count;

    const auto id = vocabulary.getId(word);
    if (!id.has_value()) {
        report.status = {false, "'" + word + "' is not in the vocabulary"};
        return report;
    }

    report.id = *id;
    report.neighbours = index.nearest(*id, count);
    return report;
}

AnalogyQueryReport QueryAnalogy(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& a,
    const std::string& b,
    const std::string& c,
    const std::size_t count
) {
    AnalogyQueryReport report;
    report.a = a;
    report.b = b;
    report.c = c;

    const auto idA = vocabulary.getId(a);
    const auto idB = vocabulary.getId(b);
    const auto idC = vocabulary.getId(c);

    if (!idA.has_value() || !idB.has_value() || !idC.has_value()) {
        report.status = {
            false,
            "some of '" + a + "', '" + b + "', '" + c + "' are not in the vocabulary",
        };
        return report;
    }

    const auto query = index.analogyVector(*idA, *idB, *idC);
    const std::array<TWordId, 3> exclude{*idA, *idB, *idC};
    report.neighbours = index.nearestToVector(query, exclude, count);
    return report;
}

ExpressionReport QueryExpression(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& expression,
    const std::size_t count
) {
    ExpressionReport report;
    report.expression = expression;

    std::string error;
    report.terms = ParseExpression(vocabulary, expression, error);
    if (report.terms.empty()) {
        report.status = {false, error};
        return report;
    }

    std::vector<TWordId> exclude;
    exclude.reserve(report.terms.size());
    for (const auto& term : report.terms) {
        exclude.push_back(term.id);
    }

    report.cosAdd = index.nearestToVector(
        BuildExpressionVector(index, report.terms), exclude, count);

    report.analogyShape = IsAnalogyShape(report.terms);
    if (report.analogyShape) {
        report.cosMul = RankByCosMul(
            index, report.terms[1].id, report.terms[0].id, report.terms[2].id, exclude, count);
    }
    return report;
}

OddOneOutReport QueryOddOneOut(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& words
) {
    OddOneOutReport report;

    const auto tokens = Text::SplitWords(words);
    if (tokens.size() < 3) {
        report.status = {false, "need at least three words"};
        return report;
    }

    const auto ids = ResolveWords(vocabulary, tokens, report.status);
    if (ids.empty()) {
        return report;
    }

    const auto centroid = Centroid(index, ids);
    report.scored = ProjectOntoAxis(index, centroid, ids);
    report.oddOne = report.scored.back().id;
    return report;
}

AxisReport QueryAxis(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& axisExpression,
    const std::string& words,
    const std::size_t restrictTo,
    const std::size_t count
) {
    AxisReport report;
    report.axis = axisExpression;

    std::string error;
    const auto terms = ParseExpression(vocabulary, axisExpression, error);
    if (terms.empty()) {
        report.status = {false, error};
        return report;
    }

    const auto axis = Normalized(BuildExpressionVector(index, terms));

    const auto tokens = Text::SplitWords(words);
    if (!tokens.empty()) {
        const auto ids = ResolveWords(vocabulary, tokens, report.status);
        if (ids.empty()) {
            return report;
        }
        report.explicitWordList = true;
        report.ranked = ProjectOntoAxis(index, axis, ids);
        return report;
    }

    const auto scored = ProjectVocabularyOntoAxis(index, axis, restrictTo);
    const auto take = std::min(count, scored.size());

    report.positive.assign(scored.begin(), scored.begin() + take);
    report.negative.reserve(take);
    for (std::size_t i = 0; i < take; ++i) {
        report.negative.push_back(scored[scored.size() - 1 - i]);
    }
    return report;
}

std::span<const std::string_view> DefaultBatteryWords() {
    static constexpr std::array<std::string_view, 8> words{
        "one", "king", "france", "computer", "water", "music", "red", "war",
    };
    return words;
}

BatteryReport RunDefaultBattery(const Vocabulary& vocabulary, const EmbeddingIndex& index) {
    BatteryReport report;

    for (const auto word : DefaultBatteryWords()) {
        report.neighbours.push_back(QueryNeighbours(vocabulary, index, std::string(word), 8));
    }

    report.analogies.push_back(QueryAnalogy(vocabulary, index, "man", "king", "woman", 5));
    report.analogies.push_back(QueryAnalogy(vocabulary, index, "paris", "france", "rome", 5));
    report.analogies.push_back(QueryAnalogy(vocabulary, index, "good", "better", "bad", 5));
    return report;
}

}  // namespace Words
