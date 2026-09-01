# Word embeddings

Skip-gram with negative sampling (SGNS), trained from a plain text dump.

## Pipeline

```
raw text ──inspect──> statistics
    │
    ├──buildvoc──> .voc ─┐
    │                    ├──buildcor──> .cor ─┐
    └────────────────────┘                    ├──train──> embeddings
                                              │
                          .voc ────────────────┘
                                              │
   embeddings + .voc ──> neighbours | expression | oddone | axis | evaluate
```

```bash
MatrixGui_words inspect  --input-file corpus.txt
MatrixGui_words buildvoc --input-file corpus.txt --output-file built.voc --min-count 5
MatrixGui_words buildcor --input-file corpus.txt --vocabulary built.voc --output-file built.cor
MatrixGui_words train    --vocabulary built.voc --corpus built.cor --output-file emb.bin \
                         --dim 100 --epochs 5
MatrixGui_words neighbours --vocabulary built.voc --embeddings emb.bin --word king
```

## Module layout

| Directory | Contains |
|---|---|
| `core/words/` | The pipeline: vocabulary, corpus, samplers, model, trainer, queries, evaluation |
| `core/words/report/` | **All** console formatting |
| `core/words_cli/` | Subcommand option structs and bodies (CLI-only, not in `CORE_SOURCES`) |

### The one rule

> No `#include <iostream>` anywhere under `core/words/` except in `core/words/report/`.

Checked with a single grep:

```bash
grep -rln 'include <iostream>' core/words/ | grep -v '^core/words/report/'
```

Everything that produces output takes an explicit `std::ostream&`. That is what
lets the same code serve the CLI and a GUI: a Qt tab can point the stream at
`ThreadSafeTerminalOStream` (`gui_common/advanced_terminal.h`), or ignore the
`report/` layer entirely and render the report structs into widgets.

`core/words_cli/` is deliberately outside `CORE_SOURCES`, because that list is
compiled into the Qt binary too and CLI11 does not belong there.

## Compute, then report

Every operation splits in two: a function returning a struct, and a printer for
that struct.

```cpp
const auto report = Words::QueryNeighbours(vocabulary, index, "king", 10);
Words::PrintNeighbourReport(std::cout, vocabulary, report);   // optional
```

This holds for queries (`queries.h`), evaluation (`evaluate.h`) and training
(`TrainProgress` / `TrainSummary`).

## Errors

Two mechanisms, chosen by whose fault it is:

| Situation | Mechanism |
|---|---|
| Missing/truncated/wrong-format file, impossible config | throw `Words::Error` (`IoError`, `VocabularyError`, `ConfigError`) |
| Word not in the vocabulary, unparseable expression, too few words | `QueryStatus` on the result struct |

Bad user input is not exceptional: an expression box would otherwise throw on
every half-typed word. `QueryStatus` renders as an inline message; an exception
renders as a dialog. The CLI catches `Words::Error` once, in `main`, and exits
non-zero.

Exit codes: `0` success, `1` a failed check or reported error, `2` an internal
error.

## Training

`Words::Trainer` mirrors `Neural::Classifier::Trainer`, so a GUI controller
drives it the same way:

```cpp
Words::Trainer trainer;
trainer.setVocabulary(vocabulary);          // shared_ptr<const Vocabulary>
trainer.setCorpus(corpus);                  // shared_ptr<const TCorpus>
trainer.setModelConfig(config.model);
trainer.setSamplingConfig(config.sampling);
trainer.setProgressCallback([](const Words::TrainProgress& p) { /* chart */ });
trainer.setOutputStream(&terminal);         // or setVerbose(false)

const auto summary = trainer.train(config.train);
```

The trainer builds its own `Subsampler`, `WindowSampler` and `NegativeSampler`
from the vocabulary and `SamplingConfig`.

Two things to know:

- **The callback runs on the thread that called `train()`**, not the UI thread.
  A Qt controller must marshal it (`QMetaObject::invokeMethod`).
- **`setVerbose(false)` means silent.** Note this differs from
  `Neural::Classifier::Trainer`, which falls back to `std::cerr` when not
  verbose and therefore still prints.

`requestStop()` is honoured at chunk granularity (~1000 corpus tokens) and sets
`TrainSummary::stopped`.

### Known race

The monitor thread calls `MeanProbeLoss` while workers are mutating the same
rows (Hogwild-style lock-free updates through a `mutable` model). This is
formally a data race and ThreadSanitizer will report it. It is left as-is
deliberately: the reported loss is a progress indicator, and the contention-free
updates are the point of the design.

## Configuration

`core/words/config.h` holds `ModelConfig` (shape), `SamplingConfig` (which pairs
exist), `TrainConfig` (schedule) and the `WordsConfig` aggregate.
`Words::Validate` rejects impossible combinations up front, rather than letting
them surface from a sampler constructor after a worker thread has started.

JSON bindings are in `config_json.h`, kept separate so that including
`config.h` does not pull nlohmann into every translation unit.

## GUI

The `MatrixGui` application has a Words mode (the fourth button on the mode
picker) with three sub-tabs sharing one terminal:

- **Data & Training** — inspect a raw dump, build/load the vocabulary and
  corpus, train with live probe-loss and speed charts, save/load embeddings.
- **Explore** — neighbours, the default battery, vector expressions,
  odd-one-out and axis projection. Results print into the terminal through the
  same `report/` functions the CLI uses, so the output is identical. A model
  trained elsewhere is loaded right on this tab: Load vocabulary…, then Load
  embeddings… (embeddings are indexed by the vocabulary's word ids, so the
  vocabulary comes first).
- **Evaluate** — analogy and similarity datasets. Analogy evaluation runs on a
  worker thread and cannot be cancelled once started.

After an in-session training run the query index is built in memory
(`EmbeddingIndex` from the trainer's embeddings) — no save/load round-trip is
needed before exploring. Reloading or rebuilding the vocabulary resets the
loaded corpus, since its encoded ids belong to the old vocabulary.

## Tests

```bash
cmake -B build -DBUILD_TESTS=ON && cmake --build build -j
ctest --test-dir build --output-on-failure     # unit tests (doctest)
tests/golden/compare.sh                        # CLI snapshot
```

Two layers:

- **Unit tests** (`tests/words/`) over synthetic fixtures — deterministic on
  fixed seeds, whole suite under three seconds. `tests/support/fixtures.h`
  builds toy vocabularies and embeddings whose neighbours are analytically
  known.
- **A CLI snapshot** (`tests/golden/`) running every subcommand against a
  generated corpus and diffing stdout and exit codes. `capture.sh` records,
  `compare.sh` checks. Regenerate the corpus with `make_corpus.py`; it is seeded,
  and the expectations are pinned to its exact bytes.

The `validate-*` subcommands that once self-checked the samplers and the model
are gone: they predate the test suite, and every invariant they verified
(gradient check, chi-squared draw distribution, initialisation statistics,
window invariants) now lives in the unit tests, which assert instead of
printing "passed".

The snapshot normalises the training log: progress ticks are sampled on a
wall-clock timer, so both their values and their number vary between runs. The
banner and the final summary are reproducible and are compared.

CI (`.github/workflows/ci.yml`) builds the core and runs the unit tests on
every push to master and on every pull request into it, plus compiles the Qt
GUI in a second job. The golden snapshot stays local: its expectations pin
float output of a trained model, which is not bit-portable across CPUs.
