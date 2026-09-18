# `core/words/query` — using trained embeddings

Everything you do with a trained model. All of it reads; nothing here trains or
writes. The entry point is `EmbeddingIndex`, a normalised copy of an embedding
matrix; every other file in the folder is built on top of it.

Evaluation lives here rather than in a folder of its own because it is exactly
the same operation as a query — nearest-neighbour search over `EmbeddingIndex` —
run in bulk against an answer key.

| File | Contains |
|---|---|
| `similarity.h/.cpp` | `Neighbour`, `class EmbeddingIndex` — normalised vectors and kNN |
| `expressions.h/.cpp` | `ExpressionTerm`, `ParseExpression`, `BuildExpressionVector` |
| `queries.h/.cpp` | The five query entry points, their report structs and the vector helpers |
| `evaluate.h/.cpp` | `EvaluateAnalogies`, `EvaluateSimilarity` and their reports |

## Errors: `QueryStatus`, not exceptions

Every query returns a struct carrying a `QueryStatus { bool ok; std::string message; }`.
An unknown word, an unparseable expression or too few words sets `ok = false`
and fills `message`; it does not throw.

That split is deliberate. A GUI expression box would otherwise throw on every
half-typed word. `QueryStatus` renders as an inline message; a `Words::Error`
renders as a dialog. Exceptions are reserved for a broken file or an impossible
config — see [`../README.md`](../README.md).

## `similarity.h` / `similarity.cpp`

```cpp
struct Neighbour { TWordId id; double similarity; };

explicit EmbeddingIndex(Embeddings embeddings);         // normalises in place
static EmbeddingIndex Load(std::istream& in);

std::vector<Neighbour> nearest(TWordId id, std::size_t count) const;
std::vector<Neighbour> nearestToVector(std::span<const TFloat> query,
                                       std::span<const TWordId> exclude,
                                       std::size_t count) const;
std::vector<TFloat>    analogyVector(TWordId a, TWordId b, TWordId c) const;
double                 similarity(TWordId first, TWordId second) const;

const Embeddings& getNormalized() const;
std::size_t getDim() const;
std::size_t getWords() const;
```

**The constructor normalises the rows it is given**, taking the matrix by value
and scaling it in place. Once every row is a unit vector a cosine similarity is
just a dot product, which is what makes a full-vocabulary scan cheap enough to
do per query.

A zero row is left alone rather than divided by zero, so an untrained or padded
row yields similarity 0 instead of NaN.

Because the index owns a *normalised* copy, it is not what you save. Keep the
raw `Embeddings` for that — the GUI holds both for exactly this reason.

`nearestToVector` takes an explicit `exclude` list, which is how analogy queries
keep the three input words out of their own answers. `nearest(id, count)` is a
thin wrapper over it that excludes `id`, so a word is never its own neighbour.

## `expressions.h` / `expressions.cpp`

```cpp
struct ExpressionTerm { TWordId id; double sign; };

std::vector<ExpressionTerm> ParseExpression(const Vocabulary&, const std::string& expression,
                                            std::string& error);
std::vector<TFloat>         BuildExpressionVector(const EmbeddingIndex&,
                                                  const std::vector<ExpressionTerm>&);
```

Parses `king - man + woman` into signed terms. Splitting on `+`/`-` and
whitespace, each token is lowercased and looked up; an unknown word writes into
`error` and returns an empty vector, as does an expression with no terms at all.

`BuildExpressionVector` is then the signed sum of the corresponding **normalised**
rows — it is not renormalised, because `nearestToVector` only cares about
direction.

## `queries.h` / `queries.cpp`

Five entry points, each returning its own report struct.

| Function | Report | What it answers |
|---|---|---|
| `QueryNeighbours` | `NeighbourReport` | nearest words to one word |
| `QuerySubwordNeighbours` | `SubwordNeighbourReport` | nearest words to a word the vocabulary never saw |
| `QueryAnalogy` | `AnalogyQueryReport` | *a is to b as c is to ?* |
| `QueryExpression` | `ExpressionReport` | nearest words to an arbitrary `a - b + c` |
| `QueryOddOneOut` | `OddOneOutReport` | which of these words does not belong |
| `QueryAxis` | `AxisReport` | rank words along a direction |
| `RunDefaultBattery` | `BatteryReport` | a fixed smoke-test set, via `DefaultBatteryWords()` |

Field notes that are not obvious from the names:

- `ScoredWord` exists alongside `Neighbour` to mark a score that is **not** a
  cosine similarity — a 3CosMul score or an axis projection. Same shape,
  different meaning, and keeping the types apart stops a printer labelling a
  column wrongly.
- `ExpressionReport::cosMul` is empty unless the terms form an analogy;
  `analogyShape` says whether they did.
- `OddOneOutReport::scored` is the words projected onto their own centroid, in
  descending order, so the odd one out is the last entry — `oddOne` is literally
  `scored.back().id`. Needs at least three words.
- `AxisReport::explicitWordList` is true when `ranked` holds the caller's own
  word list rather than a slice of the vocabulary. `positive` is the top
  `count`; `negative` is the bottom `count`, most negative first.

`QuerySubwordNeighbours` is the out-of-vocabulary path and the reason subword
training exists:

```cpp
SubwordNeighbourReport QuerySubwordNeighbours(const EmbeddingIndex&, const SubwordVectors&,
                                              const std::string& word, std::size_t count);
```

It takes no `Vocabulary` — the word by definition is not in one. The query
vector is the mean of the word's n-gram rows (`SubwordVectors::compose`), and
the search itself is the ordinary `nearestToVector` over the same composed
index every other query uses, with nothing excluded. A word too short for any
n-gram, or a `.sub` whose dimension does not match the index, comes back as a
`QueryStatus` rather than an exception or a wrong answer.

`SubwordNeighbourReport` carries no id and no count, because an unknown word has
neither; it carries `subwords`, how many n-grams the vector was built from,
which is the one number that tells you how much evidence the answer rests on.

Shared helpers, exposed for tests and for the report layer:

```cpp
std::vector<TFloat> Normalized(std::vector<TFloat> vector);   // zero vector stays zero
bool                IsAnalogyShape(const std::vector<ExpressionTerm>& terms);
std::vector<ScoredWord> RankByCosMul(index, a, b, c, exclude, count);
std::vector<TFloat>     Centroid(index, ids);                 // normalised
std::vector<ScoredWord> ProjectOntoAxis(index, axis, ids);
std::vector<ScoredWord> ProjectVocabularyOntoAxis(index, axis, restrictTo);
```

`RankByCosMul` is Levy & Goldberg's multiplicative analogy reranking. It shifts
each cosine from `[-1, 1]` into `[0, 1]` before taking the ratio, and adds
`0.001` to the denominator — without the shift a negative cosine flips the sign
of the whole ratio, and without the epsilon a near-orthogonal `a` blows it up.

**Analogy term order is a trap.** An analogy parses as terms `b, -a, c`, i.e.
`b - a + c`. So `QueryExpression` calls `RankByCosMul(index, terms[1].id,
terms[0].id, terms[2].id, ...)` — the first two are swapped relative to how the
expression reads.

`ToLower` and `SplitWords` used to live in this header. They are now in
`core/lib/text.h`.

## `evaluate.h` / `evaluate.cpp`

```cpp
AnalogyReport    EvaluateAnalogies(vocabulary, index, file, restrictTo, threads);
SimilarityReport EvaluateSimilarity(vocabulary, index, file, scoreColumn);
```

`file` is an open `std::istream&` — the caller opens the dataset, normally via
`Io::ReadFile` (`core/lib/file_stream.h`), which is what turns a missing file
into an `Io::Error` naming it.

### Analogies

The Google analogy-set format: `: category-name` lines open a section, every
other line is four words. Each question is scored twice, by 3CosAdd and by
3CosMul, giving `correctAdd` and `correctMul` in the same pass.

A question whose four words are not all in the vocabulary counts as `skipped`,
not wrong — so `accuracyAdd()` and `accuracyMul()` divide by `asked`, and
`total()` is `asked + skipped`. Reporting them separately is what stops a small
vocabulary from looking like a bad model.

Categories are rolled up into `semantic` and `syntactic` by name: **a category
counts as syntactic when its name starts with `gram`**, semantic otherwise.

`restrictTo` caps candidate answers to the N most frequent words (ids are
frequency-ordered, so this is `id < N`); 0 means the whole vocabulary. The CLI
defaults to 30000 because a full scan over a large vocabulary dominates the
runtime. `threads` of 0 means `hardware_concurrency`.

### Similarity

A WordSim-353-style file: two words and a human score per line, the score read
from `scoreColumn` (default 2, i.e. the third field). A line whose score does
not parse is ignored entirely; a line whose words are unknown counts as
`skipped`.

The report carries both coefficients: `pearson` over the raw scores and
`spearman` over their ranks. They come from `core/lib/stats.h` —
`PearsonOf(RanksOf(human), RanksOf(model))` is the Spearman figure, which works
because `RanksOf` averages the ranks a tie spans. Both are 0 rather than NaN
when fewer than two pairs survive or either side is constant.
