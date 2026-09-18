# `core/words/train` — turning a corpus into embeddings

Everything a training run needs and nothing else uses. Reading order matches the
data flow: the corpus is thinned by `Subsampler`, cut into (centre, context)
pairs by `WindowSampler`, each pair gets negatives from `NegativeSampler`, and
`SGNSModel` applies one gradient step per pair. `Trainer` drives all of it across
threads and reports progress.

| File | Contains |
|---|---|
| `subsampler.h/.cpp` | `class Subsampler` — drops over-frequent tokens |
| `windowsampler.h` | `class WindowSampler`, `GeneratePairs`, `GeneratePairsWhile` (header-only) |
| `negativesampler.h/.cpp` | `class NegativeSampler` — the unigram^0.75 alias table |
| `model.h/.cpp` | `WorkerContext`, `SkipGramNegativeSamplingModel` (alias `SGNSModel`) |
| `trainer.h/.cpp` | `Probe`, `TrainProgress`, `TrainSummary`, `class Trainer` and its free helpers |

## `subsampler.h` / `subsampler.cpp`

```cpp
explicit Subsampler(const Vocabulary& vocabulary, double sample = 1e-4);

bool        shouldKeep(TWordId id, XorShift& rng) const;
float       getKeepProbability(TWordId id) const;
std::size_t getAffectedWordsCount() const;
double      getExpectedCorpusLength(const Vocabulary& vocabulary) const;
```

Mikolov's frequent-word subsampling: the more common a word, the more likely a
given occurrence is thrown away. `getAffectedWordsCount` is how many words the
threshold actually touches — a diagnostic for choosing `sample`, since a value
that affects nothing is doing nothing.

**`sample = 0` disables subsampling, and that path needs care.** With
subsampling off there are no keep-probabilities to sum, so a naive token
estimate would come back as zero. `getExpectedCorpusLength` special-cases it and
reports the whole corpus instead. The figure feeds `EstimateTotalPairs`, which
feeds the learning-rate schedule: a total of zero would pin progress at 0 and
the rate would never decay.

## `windowsampler.h`

Header-only, because both entry points are templates on the emit callback.

```cpp
explicit WindowSampler(std::size_t window = 5);

template<typename Fn>
void   forEachPair(const TCorpus& chunk, XorShift& rng, Fn&& emit) const;
std::size_t getWindow() const;
double getPairsPerToken() const;
```

For each token a radius is drawn uniformly from `[1, window]` and every other
token within it becomes a context — the standard word2vec trick that weights
near contexts more heavily without weighting anything explicitly.

`getPairsPerToken()` returns `window + 1`. That is the mean pairs emitted per
token away from the chunk edges: with the radius uniform on `[1, window]` each
token emits `2r` pairs on average, and averaging `2r` over `r ∈ [1, window]`
gives `window + 1`.

```cpp
template<typename Fn, typename Predicate>
void GeneratePairsWhile(corpus, subsampler, windowSampler, rng, emit, keepGoing,
                        chunkSize = 1000, from = 0, to = SIZE_MAX);

template<typename Fn>
void GeneratePairs(corpus, subsampler, windowSampler, rng, emit,
                   chunkSize = 1000, from = 0, to = SIZE_MAX);
```

`from`/`to` slice the corpus, which is how the trainer gives each worker its own
region. `GeneratePairs` is `GeneratePairsWhile` with a predicate that is always
true.

**`keepGoing()` is checked once per chunk, not per pair.** `forEachPair` is the
innermost loop in the whole program, and a cancellation test inside it would
cost a branch there for no user-visible benefit — a chunk is ~1000 tokens, small
enough that cancellation still feels immediate.

## `negativesampler.h` / `negativesampler.cpp`

```cpp
explicit NegativeSampler(const Vocabulary& vocabulary,
                         std::size_t tableSize = 10'000'000,
                         double power = 0.75);

TWordId     sample(XorShift& rng) const;
TWordId     sampleExcluding(TWordId wordId, XorShift& rng) const;
std::size_t getTableSize() const;
double      getProbability(TWordId wordId) const;
```

A precomputed table of `tableSize` entries in which each word appears in
proportion to `count^power`. Drawing a negative is then one random index — O(1),
which matters because it happens `negatives` times per pair.

`power = 0.75` is word2vec's value: it flattens the distribution so rare words
are sampled more often than their raw frequency would allow.

`sampleExcluding` keeps the true context out of the negative set — but only on
a best-effort basis. It retries at most 8 times and then returns whatever the
next draw gives, so it **can** return `wordId`. That is intentional: on a
vocabulary where one word dominates the table, an unbounded retry loop would be
the thing that stalls.

`getProbability` scans the entire table to compute what share a word occupies.
It exists for the sampler's own tests — **do not call it from training**, it is
O(tableSize) per call.

The constructor throws `VocabularyError` on an empty vocabulary and
`ConfigError` when `tableSize < vocabulary.getSize()`. `Validate` in
`core/words/config.h` catches the latter first, where the setting is still
visible to the user rather than surfacing from deep inside a constructor.

## `model.h` / `model.cpp`

```cpp
struct WorkerContext {                       // one per thread, reused per pair
    std::vector<TFloat> gradient;
    std::vector<TFloat> hidden;              // the composed centre vector
    std::vector<TWordId> negatives;
    XorShift rng;
};

SkipGramNegativeSamplingModel(const Vocabulary&, ModelConfig, XorShift& rng);

void   trainPair(const Pair&, double learningRate, const NegativeSampler&, WorkerContext&) const;
void   applyUpdate(const Pair&, std::span<const TWordId> negatives, double lr, std::vector<TFloat>& gradient) const;
void   applyUpdate(const Pair&, std::span<const TWordId> negatives, double lr,
                   std::vector<TFloat>& gradient, std::vector<TFloat>& hidden) const;
double computeLoss(const Pair&, std::span<const TWordId> negatives) const;
double getLearningRateForStep(std::size_t processed, std::size_t total) const;

const Embeddings& getInput()        const;   Embeddings& getInputMutable();
const Embeddings& getOutput()       const;   Embeddings& getOutputMutable();
const Embeddings& getSubwordInput() const;   Embeddings& getSubwordInputMutable();
const SubwordTable& getSubwordTable() const;

Embeddings composeWords() const;
void       composeInto(TWordId id, std::vector<TFloat>& hidden) const;

const ModelConfig& getConfig() const;
std::size_t getBytes() const;

using SGNSModel = SkipGramNegativeSamplingModel;
```

Three matrices: `input` holds one row per word, `output` the auxiliary context
vectors discarded after training, and `subwordInput` one row per hash bucket
(`ModelConfig::buckets` rows, **zero rows when subwords are off**). `input` and
`subwordInput` start uniform, `output` at zero.

**Word ids and bucket ids never meet.** `input.row()` and `output.row()` take a
`TWordId`, `subwordInput.row()` a `TBucketId`, and nothing computes `V + b`. That
is what keeps a bucket id from being printed as a word: the two spaces are
separate matrices, not two halves of one.

### Subwords

With `ModelConfig::buckets > 0` the centre vector is composed on the fly:

```
h = (input[word] + Σ subwordInput[b] for b in n-grams(word)) / (1 + |n-grams|)
```

and **the whole accumulated gradient is applied to every one of those rows**,
not a `1 / (1 + k)` share of it. That is fastText's convention for skipgram
(`normalizeGradient_` is only set for its supervised mode): the composed vector
then moves exactly as far as a plain word2vec row would, so `--lr` means the
same thing with and without `--buckets` and the two runs stay comparable. A
bucket that appears twice in one word takes the gradient twice.

`composeCenter` returns a pointer straight into `input` when the word has no
n-grams, so with `--buckets 0` the hot path is byte for byte the code that was
there before subwords existed — no copy, no scaling, no second loop. That is
checked by training the golden corpus and comparing the output file bit for bit.

`composeInto` and `composeWords` go through the same composition as the hot
path, in the same order, so the materialised matrix matches what training used
down to the last bit rather than approximately.

`WorkerContext` exists so the per-pair scratch buffers are allocated once per
thread rather than once per pair — `hidden` included, which is why the five-
argument `applyUpdate` is the one the trainer calls. The four-argument overload
allocates its own buffer and is there for tests and one-off calls.

**`computeLoss` composes into a local buffer, never a member.** The monitor
thread calls it while workers are running, so a shared scratch vector on the
model would be a real race rather than the benign one below. It runs on a few
hundred probes every few seconds; the allocation does not matter.

**`trainPair` is `const` while mutating both matrices** — they are declared
`mutable`. This is deliberate: word2vec's Hogwild-style updates are lock-free,
and workers writing overlapping rows is the design, not a defect. See the race
note under `Trainer` below.

`getLearningRateForStep` decays linearly from `initialLearningRate` towards
`initialLearningRate * minLearningRateFactor` as `processed / total` goes to 1 —
so a wrong `total` distorts the whole schedule, which is why
`EstimateTotalPairs` is careful about the `sample = 0` case.

`trainPair` reads sigmoid from a 1000-entry lookup table clamped at ±6, the
word2vec approximation. `computeLoss` does **not** use it — it computes
`LogSigmoid` exactly with `log1p`, in the numerically stable branch form. So the
reported loss is a true log-loss, not the table's approximation of one.

## `trainer.h` / `trainer.cpp`

### Free helpers

```cpp
struct Probe { Pair pair; std::vector<TWordId> negatives; };

std::vector<Probe> BuildProbeSet(corpus, subsampler, windowSampler, negativeSampler,
                                 modelConfig, count, seed);
double             MeanProbeLoss(const SGNSModel&, const std::vector<Probe>&);
std::size_t        EstimateTotalPairs(vocabulary, subsampler, windowSampler, epochs);
```

A probe set is a **fixed** sample of (pair, negatives) drawn once, before
training. Because the pairs and their negatives never change, `MeanProbeLoss` is
comparable across ticks — a loss averaged over freshly drawn pairs would move
with the sample as much as with the model.

`EstimateTotalPairs` is `expected corpus length × pairs per token × epochs`. It
is an estimate, not a count: the actual number varies with the random radii and
subsampling draws.

### Progress and summary

`TrainProgress` — one sample, emitted every `TrainConfig::reportEveryMs`:
`progress` (0..1), `pairsDone`, `pairsTotal`, `learningRate`, `loss`,
`pairsPerSecond`, `elapsedSeconds`, `etaSeconds`.

`TrainSummary` — the closing figures: `threads`, `pairsDone`, `pairsEstimated`,
`elapsedSeconds`, `pairsPerSecond`, `initialLoss`, `finalLoss`, `probeCount`,
plus `stopped` (true when `requestStop()` cut the run short) and `history`, one
`TrainProgress` per tick — that is what the GUI's charts are drawn from.

### `class Trainer`

```cpp
void setVocabulary(std::shared_ptr<const Vocabulary>);
void setCorpus(std::shared_ptr<const Corpus>);
void setModelConfig(const ModelConfig&);
void setSamplingConfig(const SamplingConfig&);
void setProgressCallback(std::function<void(const TrainProgress&)>);
void setVerbose(bool);
void setOutputStream(std::ostream*);

TrainSummary train(const TrainConfig& config = {});
bool         isRunning() const;
void         requestStop();

const SGNSModel*  getModel() const;
const Embeddings& getInputEmbeddings() const;
const Embeddings& getWordEmbeddings() const;
```

`getInputEmbeddings()` is the raw word matrix — the rows the optimiser touches.
`getWordEmbeddings()` is what you save and query: with subwords off it **is**
`getInputEmbeddings()` (same object, no copy), and with subwords on it is the
`V × N` matrix composed once at the end of `train()`, including after
`requestStop()`. Composing is `V × (1 + k) × dim` additions — seconds even on a
600k-word vocabulary — and doing it here rather than in the CLI is what lets the
GUI build an `EmbeddingIndex` straight from the trainer with no file in between.

Shaped after `Neural::Classifier::Trainer` so a GUI controller drives it the
same way: configure, attach a progress callback and an output stream, run on a
worker thread, cancel with `requestStop()`. It builds its own `Subsampler`,
`WindowSampler` and `NegativeSampler` from the vocabulary and `SamplingConfig`.

`train()` resets its own stop flag, so one `Trainer` survives repeated runs.

### Traps

- **The progress callback runs on the thread that called `train()`, not the
  caller's UI thread.** A Qt controller must marshal it with
  `QMetaObject::invokeMethod`.
- **`setVocabulary`/`setCorpus` share rather than take ownership.** They are
  `shared_ptr<const>` because a GUI keeps showing vocabulary and corpus
  statistics after the run ends, and the query panel needs the same
  `Vocabulary`.
- **`train()` blocks** until training finishes or `requestStop()` is honoured,
  and rethrows any exception raised on a worker thread. Worker bodies catch
  everything and stash it in an `exception_ptr`, because an exception escaping a
  `std::thread` body calls `std::terminate`.
- **`setVerbose(false)` means silent.** This differs from
  `Neural::Classifier::Trainer`, which falls back to `std::cerr` when not
  verbose and therefore still prints.
- The monitor thread sleeps in slices rather than one `reportEveryMs` sleep,
  which would otherwise stall `requestStop()` for up to that long — 3 s by
  default.
- **Known race.** The monitor calls `MeanProbeLoss` while workers mutate the
  same rows. It is formally a data race and ThreadSanitizer will report it. It
  is left as-is on purpose: the reported loss is a progress indicator, and
  contention-free updates are the point of the design.
- **Subwords sharpen that race.** With n-grams every pair updates `1 + k` input
  rows instead of one, and a few buckets (`<th`, `ing>`, `ого>`) are touched by
  almost every pair on every thread. Hogwild assumes updates are sparse, and for
  those rows they are not: expect lost updates and cache-line ping-pong. fastText
  lives with the same trade; measure `--threads` scaling against `--buckets 0`
  rather than assuming it holds.
- `trainer.cpp` is the one place in `core/words` where computation calls into
  `report/` — it prints its own banner, ticks and summary. `trainer.h` does not.
