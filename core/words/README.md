# `core/words` — word embeddings

Skip-gram with negative sampling (SGNS), trained from a plain text dump. This
folder is the whole pipeline; `cli/words/` and `gui/words/` are two front
ends over it, and neither adds logic of its own.

Narrative documentation — pipeline diagram, CLI reference, the error model, how
training behaves — lives in [`docs/words.md`](../../docs/words.md). The README
files here are the file-level reference: what each unit exposes, what it
guarantees, and what will bite you.

## Layout

| Folder | Stage | README |
|---|---|---|
| *(this root)* | Configuration and the exception taxonomy — needed by all four folders | below |
| `data/` | Survey a raw dump; build and persist the vocabulary, the corpus, the embedding matrix and the character n-grams | [`data/README.md`](data/README.md) |
| `train/` | Turn a corpus into embeddings: samplers, the SGNS model, the trainer | [`train/README.md`](train/README.md) |
| `query/` | Use trained embeddings: nearest neighbours, expressions, benchmarks | [`query/README.md`](query/README.md) |
| `report/` | Render the structs the other three return, to a `std::ostream` | [`report/README.md`](report/README.md) |

```
raw text ──inspect───────────────> CorpusStatistics       (data)
    │
    ├──Vocabulary::Build─> .voc ─┐
    │                            ├──EncodeCorpus─> .cor ─┐
    └────────────────────────────┘                       │
                                                         ▼
                              Subsampler ─> WindowSampler ─> NegativeSampler
                                                         │          (train)
                                                         ▼
                                            SGNSModel <── Trainer ──> embeddings
                                                                         │
                                        EmbeddingIndex <────────────────┘
                                                │                    (query)
                     neighbours │ expression │ oddone │ axis │ evaluate
```

## Dependency direction

Includes only ever point down this list, never up. `report/` sits outside it:
everything may be printed, nothing may call a printer — with one deliberate
exception, `train/trainer.cpp`, which prints its own banner and progress ticks.

```
config.h  error.h  data/types.h                       no words dependencies
data/inspect.h
data/corpus.h  data/embeddings.h  data/vocabulary.h
data/subwords.h
train/negativesampler.h  train/subsampler.h  train/windowsampler.h
query/similarity.h  train/model.h
query/expressions.h  query/evaluate.h  train/trainer.h
query/queries.h
```

`Vocabulary` is forward-declared rather than included by eleven headers, so
`data/vocabulary.h` is pulled in only by `.cpp` files. Keep it that way — it is
the widest header in the subtree.

## The one rule

> No `#include <iostream>` anywhere under `core/words/`.

```bash
grep -rln 'include <iostream>' core/words/    # must print nothing
```

Everything that produces output takes an explicit `std::ostream&`. That is what
lets the same code serve the CLI and a GUI: a Qt tab can point the stream at
`ThreadSafeTerminalOStream` (`gui_common/advanced_terminal.h`), or ignore
`report/` entirely and render the report structs into widgets.

The rule used to carve out `report/`, which needed `std::cerr` for a default log
stream. That plumbing now lives in `core/lib/stream_format.h`, so the carve-out
is gone and the grep above is exact.

## Files in this root

### `config.h` / `config.cpp`

Four structs and one validator. Split three ways so that a caller can name the
part it cares about:

| Struct | Governs | Fields |
|---|---|---|
| `ModelConfig` | the shape of the model itself | `dim` (100), `negatives` (5), `initialLearningRate` (0.025), `minLearningRateFactor` (1e-4), `minN` (3), `maxN` (6), `buckets` (0 — subwords off) |
| `SamplingConfig` | which (centre, context) pairs exist and how negatives are drawn | `window` (5), `sample` (1e-4), `negativeTableSize` (10'000'000), `negativePower` (0.75) |
| `TrainConfig` | how the run is scheduled | `epochs` (5), `threads` (0), `chunkSize` (1000), `syncEvery` (10000), `reportEveryMs` (3000), `probePairs` (500), `seed` (20260831) |
| `WordsConfig` | all three together — what the CLI and the GUI pass around | `model`, `sampling`, `train` |

`TrainConfig::threads = 0` means `std::thread::hardware_concurrency()`, not
"no threads".

```cpp
void Validate(const WordsConfig& config, std::size_t vocabularySize);
```

`buckets` defaults to **0**, so a run is plain SGNS unless it is asked for
otherwise; `minN`/`maxN` are only checked when it is non-zero.

Throws `ConfigError` when a setting cannot produce a usable run. It takes the
vocabulary size so it can also catch a negative-sampling table smaller than the
vocabulary — a condition that otherwise surfaces from deep inside
`NegativeSampler`'s constructor, far from the setting that caused it.

### `config_json.h`

`NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE` for all four config structs, so a
`WordsConfig` round-trips through JSON (the GUI stores it in its settings).

**Include this only where you actually serialise.** It pulls in
`nlohmann/json.hpp`, which is why it is a separate header from `config.h` rather
than the bottom of it.

### `error.h`

```
std::runtime_error
└── Words::Error
    ├── Words::IoError           missing, unreadable, truncated or wrong-format file
    ├── Words::VocabularyError   the vocabulary is unusable (empty, or too large for TWordId)
    └── Words::ConfigError       a setting that cannot produce a usable run
```

Exceptions are for things that are nobody's fault at the keyboard: a broken
file, an impossible configuration. Bad *user input* — an unknown word, an
unparseable expression — is not exceptional and is reported through
`QueryStatus` instead; see [`query/README.md`](query/README.md). The CLI catches
`Words::Error` once, in `main`, and exits non-zero.

`Error` stays here rather than moving to `core/lib`: neither the classifier nor
the generator has an exception taxonomy of its own, so hoisting it would mean
inventing a home namespace for the benefit of a single consumer.
