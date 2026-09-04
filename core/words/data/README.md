# `core/words/data` — the raw text and what is built from it

Everything with a representation on disk, plus the survey tool you run before
building any of it. All three artifacts serialise through
`core/lib/write.h` (`WriteBinaryLE` / `ReadBinaryLE` / `WriteBulkLE` /
`ReadBulkLE`), so every file below is little-endian regardless of host.

Order of use: `InspectDump` to pick a `--min-count`, `Vocabulary::Build` to fix
the word list, `EncodeCorpus` to turn the same dump into ids, and finally
`Embeddings` to hold what training produces.

| File | Contains |
|---|---|
| `types.h` | `TWordId`, `Pair` |
| `inspect.h/.cpp` | `CorpusStatistics`, `InspectDump` — statistics over a raw dump |
| `vocabulary.h/.cpp` | `class Vocabulary` — the word ↔ id table with counts |
| `corpus.h/.cpp` | `TCorpus`, `EncodeCorpus`, `SaveCorpus`, `LoadCorpus` |
| `embeddings.h/.cpp` | `TFloat`, `class Embeddings`, `dot`, `addScaled` |

## `types.h`

```cpp
using TWordId = std::uint32_t;
struct Pair { TWordId center; TWordId context; };
```

`TWordId` being 32-bit is the cap on vocabulary size; `Vocabulary` throws
`VocabularyError` rather than wrapping around. `Pair` is one skip-gram training
example, produced by `train/windowsampler.h`.

## `inspect.h` / `inspect.cpp`

A quick look at a raw text dump **before any vocabulary is built** — the only
header in `core/words` with no dependency on the rest of the subtree.

```cpp
CorpusStatistics InspectDump(std::istream& dump, std::size_t topN = 15);
```

| `CorpusStatistics` field | Meaning |
|---|---|
| `symbols` | bytes in the file |
| `totalWords` | whitespace-separated tokens |
| `uniqueWords` | distinct tokens |
| `survivorsByMinCount` | pairs of *(minCount, how many distinct words would survive it)* |
| `topByFrequency` | the `topN` most frequent, each with its `count` and its `share` of `totalWords` |

The thresholds in `survivorsByMinCount` are fixed at 5, 10 and 50. That is the
point of the tool: it answers "what does `--min-count` cost me?" without making
you build three vocabularies to find out.

Tokenisation is `ifstream >> word` — whitespace only, no case folding, no
punctuation stripping. It matches `Vocabulary::Build` and `EncodeCorpus`
exactly, which is what makes the numbers predictive rather than indicative.

## `vocabulary.h` / `vocabulary.cpp`

```cpp
static Vocabulary Build(std::istream& dump, std::size_t minCount = 5);
static Vocabulary Load(std::istream& in);
static void       Save(std::ostream& out, const Vocabulary& vocabulary);

std::optional<TWordId> getId(const std::string& word) const;   // nullopt when absent
const std::string&     getWord(TWordId id) const;              // unchecked
std::size_t            getCount(TWordId id) const;             // unchecked
TWordId                getSize() const;
std::size_t            getRawTokens() const;                   // tokens in the dump
std::size_t            getKeptTokens() const;                  // tokens that cleared minCount
double                 getFrequency(TWordId id) const;         // count / keptTokens
```

`getId` is the only lookup that reports failure; `getWord` and `getCount` index
straight into a vector and assume a valid id.

Ids are assigned by descending frequency, so id 0 is the most common word. That
ordering is what lets `query/` restrict a scan to "the N most frequent words"
with a simple `id < N`.

Words with **equal** counts break alphabetically, so the ordering is total and
`Build` is reproducible: the same dump always yields the same ids.

That tiebreak matters more than it looks. On the synthetic corpus behind the
golden tests, 1194 of 1400 words share a count with at least one other — 85% of
the vocabulary. Without a tiebreak their ids would come out in `unordered_map`
iteration order, and every embedding trained from the corpus would shift with it.

`rawTokens` and `keptTokens` are both retained because their ratio is how much
of the dump the vocabulary actually covers. `getFrequency` divides by
`keptTokens`, so frequencies over the surviving vocabulary sum to 1.

**Traps:**

- `Build` reserves `1 << 20` hash buckets up front so a wiki-scale dump (~1M
  distinct words) never rehashes mid-stream. That costs a flat ~10 MB even on a
  tiny corpus. It is a deliberate trade, not an oversight.
- `Load` opens in binary mode to match `Save`. Text mode mangles the payload on
  Windows.
- A vocabulary larger than `TWordId` can address, or a file that declares zero
  words, throws `VocabularyError`; a truncated file throws `IoError`.

### `.voc` file format

All integers little-endian, no magic number and no version field.

```
u64  size          number of words
u64  rawTokens
u64  keptTokens
size × {
    u64  length    bytes in the word
    u8[length]     the word, not NUL-terminated
    u64  count
}
```

## `corpus.h` / `corpus.cpp`

```cpp
using TCorpus = std::vector<TWordId>;

TCorpus EncodeCorpus(std::istream& dump, const Vocabulary& vocabulary);
void    SaveCorpus(std::ostream& out, const TCorpus& corpus);
TCorpus LoadCorpus(std::istream& in);
```

The dump reduced to a flat stream of ids. Words absent from the vocabulary are
dropped, not replaced with a sentinel, so the corpus is shorter than the dump by
exactly the tokens `minCount` removed.

**The ids belong to one specific vocabulary.** A `.cor` and the `.voc` it was
built against must always travel together; nothing in the format detects a
mismatch, and the result of pairing the wrong two is a model that trains happily
and means nothing.

### `.cor` file format

```
u32  magic     0x57435250 ('WCRP')
u32  version   1
u64  count     number of ids
u32[count]     the ids, bulk little-endian
```

Unlike `.voc` this one is tagged, so a wrong file is rejected rather than
misread.

## `embeddings.h` / `embeddings.cpp`

```cpp
using TFloat = float;

Embeddings(std::size_t words, std::size_t dimension);
TFloat*       row(TWordId id);
const TFloat* row(TWordId id) const;
std::size_t   getWords() const;
std::size_t   getDim() const;
std::size_t   getBytes() const;
void          initializeUniform(XorShift& rng);
void          initializeZero();

static void       Save(std::ostream& out, const Embeddings&);
static Embeddings Load(std::istream& in);
```

A row-major `words × dim` matrix of `float`, nothing more. `row(id)` hands back
a raw pointer because the training inner loop wants one; there is no bounds
check.

`initializeUniform` is how the input matrix starts a run (the output matrix
starts at zero, per word2vec). `XorShift` comes from `core/lib/random.h` and is
only forward-declared here, so the header stays cheap.

Two free functions sit alongside, used by the model's inner loop:

```cpp
inline double dot(const TFloat* __restrict a, const TFloat* __restrict b, std::size_t dim);
inline void   addScaled(TFloat* __restrict a, const TFloat* __restrict b, double scale, std::size_t dim);
```

Both accumulate in `double` over `float` storage. They overlap in intent with
`core/matrix/` and Eigen but are kept here on purpose: they are the hot path, they
take raw `__restrict` pointers, and folding them into a general matrix type is a
performance question rather than a layout one.

### Embeddings file format

```
u64  words
u64  dim
f32[words × dim]   bulk little-endian, row-major
```

`Load` refuses a header declaring an empty matrix, and cross-checks the declared
size against the actual file length before allocating — otherwise a corrupt
header asking for terabytes would be honoured as an allocation request. A
genuinely truncated payload throws `IoError` too.
