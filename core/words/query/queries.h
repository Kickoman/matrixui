#pragma once

#include "core/words/data/subwords.h"
#include "core/words/query/expressions.h"
#include "core/words/query/similarity.h"
#include "core/words/data/types.h"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Words {

class Vocabulary;

struct QueryStatus {
    bool ok{true};
    std::string message;
};

struct ScoredWord {
    TWordId id{0};
    double score{0.};
};

struct NeighbourReport {
    QueryStatus status;
    TWordId id{0};
    std::string word;
    std::size_t count{0};
    std::vector<Neighbour> neighbours;
};

NeighbourReport QueryNeighbours(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& word,
    std::size_t count
);

struct SubwordNeighbourReport {
    QueryStatus status;
    std::string word;
    std::size_t subwords{0};
    std::vector<Neighbour> neighbours;
};

SubwordNeighbourReport QuerySubwordNeighbours(
    const EmbeddingIndex& index,
    const SubwordVectors& subwords,
    const std::string& word,
    std::size_t count
);

struct AnalogyQueryReport {
    QueryStatus status;
    std::string a;
    std::string b;
    std::string c;
    std::vector<Neighbour> neighbours;
};

AnalogyQueryReport QueryAnalogy(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& a,
    const std::string& b,
    const std::string& c,
    std::size_t count
);

struct ExpressionReport {
    QueryStatus status;
    std::string expression;
    std::vector<ExpressionTerm> terms;
    std::vector<Neighbour> cosAdd;
    std::vector<ScoredWord> cosMul;
    bool analogyShape{false};
};

ExpressionReport QueryExpression(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& expression,
    std::size_t count
);

struct OddOneOutReport {
    QueryStatus status;
    std::vector<ScoredWord> scored;
    TWordId oddOne{0};
};

OddOneOutReport QueryOddOneOut(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& words
);

struct AxisReport {
    QueryStatus status;
    std::string axis;
    bool explicitWordList{false};
    std::vector<ScoredWord> ranked;
    std::vector<ScoredWord> positive;
    std::vector<ScoredWord> negative;
};

AxisReport QueryAxis(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& axisExpression,
    const std::string& words,
    std::size_t restrictTo,
    std::size_t count
);

struct BatteryReport {
    std::vector<NeighbourReport> neighbours;
    std::vector<AnalogyQueryReport> analogies;
};

std::span<const std::string_view> DefaultBatteryWords();

BatteryReport RunDefaultBattery(const Vocabulary& vocabulary, const EmbeddingIndex& index);

std::vector<TFloat> Normalized(std::vector<TFloat> vector);
bool IsAnalogyShape(const std::vector<ExpressionTerm>& terms);

std::vector<ScoredWord> RankByCosMul(
    const EmbeddingIndex& index,
    TWordId a,
    TWordId b,
    TWordId c,
    std::span<const TWordId> exclude,
    std::size_t count
);

std::vector<TFloat> Centroid(const EmbeddingIndex& index, std::span<const TWordId> ids);

std::vector<ScoredWord> ProjectOntoAxis(
    const EmbeddingIndex& index,
    std::span<const TFloat> axis,
    std::span<const TWordId> ids
);

std::vector<ScoredWord> ProjectVocabularyOntoAxis(
    const EmbeddingIndex& index,
    std::span<const TFloat> axis,
    std::size_t restrictTo
);

}  // namespace Words
