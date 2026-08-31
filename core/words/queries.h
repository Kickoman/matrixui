#pragma once

// Read-only queries against a trained embedding space.
//
// Each entry point returns a report struct; nothing here prints. This is the
// same shape evaluate.h already had (EvaluateAnalogies -> AnalogyReport), now
// applied to the neighbour, expression, odd-one-out and axis queries, whose
// algorithms previously existed only inside print functions in
// similarity.cpp and expressions.cpp.
//
// Bad user input -- a word that is not in the vocabulary, an expression that
// does not parse -- is reported through QueryStatus rather than thrown, so a
// UI can show it inline while someone is still typing.

#include "core/words/expressions.h"
#include "core/words/similarity.h"
#include "core/words/types.h"

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

// A score that is not a cosine similarity (3CosMul, axis projection).
struct ScoredWord {
    TWordId id{0};
    double score{0.};
};

// --- neighbours -------------------------------------------------------------

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

// --- analogy (b - a + c) ----------------------------------------------------

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

// --- free-form expression ---------------------------------------------------

struct ExpressionReport {
    QueryStatus status;
    std::string expression;
    std::vector<ExpressionTerm> terms;
    std::vector<Neighbour> cosAdd;
    std::vector<ScoredWord> cosMul;   // empty unless the terms form an analogy
    bool analogyShape{false};
};

ExpressionReport QueryExpression(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& expression,
    std::size_t count
);

// --- odd one out ------------------------------------------------------------

struct OddOneOutReport {
    QueryStatus status;
    std::vector<ScoredWord> scored;   // descending similarity to the centroid
    TWordId oddOne{0};
};

OddOneOutReport QueryOddOneOut(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& words
);

// --- semantic axis ----------------------------------------------------------

struct AxisReport {
    QueryStatus status;
    std::string axis;
    bool explicitWordList{false};     // true => `ranked` holds the caller's words
    std::vector<ScoredWord> ranked;
    std::vector<ScoredWord> positive; // top `count`
    std::vector<ScoredWord> negative; // bottom `count`, most negative first
};

AxisReport QueryAxis(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    const std::string& axisExpression,
    const std::string& words,
    std::size_t restrictTo,
    std::size_t count
);

// --- the default battery ----------------------------------------------------

struct BatteryReport {
    std::vector<NeighbourReport> neighbours;
    std::vector<AnalogyQueryReport> analogies;
};

// The word list the CLI battery uses; was hardcoded inside the print function.
std::span<const std::string_view> DefaultBatteryWords();

BatteryReport RunDefaultBattery(const Vocabulary& vocabulary, const EmbeddingIndex& index);

// --- building blocks, exposed so they can be reused and tested --------------

std::vector<std::string> SplitWords(const std::string& text);
std::string ToLower(std::string value);
std::vector<TFloat> Normalized(std::vector<TFloat> vector);
bool IsAnalogyShape(const std::vector<ExpressionTerm>& terms);

// Levy & Goldberg's multiplicative analogy reranking.
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
