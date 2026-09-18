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

| Directory | Contains | Reference |
|---|---|---|
| `core/words/` | Configuration and the exception taxonomy | [README](../core/words/README.md) |
| `core/words/data/` | Raw-dump survey; the vocabulary, corpus and embedding matrix, and their file formats | [README](../core/words/data/README.md) |
| `core/words/train/` | Samplers, the SGNS model, the trainer | [README](../core/words/train/README.md) |
| `core/words/query/` | Nearest neighbours, expressions, the evaluation benchmarks | [README](../core/words/query/README.md) |
| `core/words/report/` | **All** console formatting | [README](../core/words/report/README.md) |
| `cli/words/` | Subcommand option structs and bodies (the `matrixgui_cli_words` library) | [README](../cli/words/README.md) |
| `gui/words/` | The Qt Words mode: controller, three tabs, config editor | [README](../gui/words/README.md) |

Each README is the file-level reference for its folder — what every unit
exposes, what it guarantees, and what will bite. This page stays the narrative:
the pipeline, the CLI, and the behaviour that spans folders. Everything the
source files used to say in comments is in the READMEs.

### The one rule

> No `#include <iostream>` anywhere under `core/words/`.

Checked with a single grep:

```bash
grep -rln 'include <iostream>' core/words/    # must print nothing
```

Everything that produces output takes an explicit `std::ostream&`. That is what
lets the same code serve the CLI and a GUI: a Qt tab can point the stream at
`ThreadSafeTerminalOStream` (`gui_common/advanced_terminal.h`), or ignore the
`report/` layer entirely and render the report structs into widgets.

The rule used to carve out `core/words/report/`, which needed `std::cerr` for a
default log stream. That plumbing moved to `core/lib/stream_format.h`
(`NullStream`, `DefaultLogStream`, `StreamFormatGuard`), so the carve-out is
gone and the grep above is exact.

`cli/words/` is a separate library (`matrixgui_cli_words`) rather than part of
`matrixgui_words`, so CLI11 never reaches the Qt binary.

## Compute, then report

Every operation splits in two: a function returning a struct, and a printer for
that struct.

```cpp
const auto report = Words::QueryNeighbours(vocabulary, index, "king", 10);
Words::PrintNeighbourReport(std::cout, vocabulary, report);   // optional
```

This holds for queries (`query/queries.h`), evaluation (`query/evaluate.h`) and
training (`TrainProgress` / `TrainSummary` in `train/trainer.h`).

## Errors

Two mechanisms, chosen by whose fault it is:

| Situation | Mechanism |
|---|---|
| Truncated/wrong-format file, impossible config | throw `Words::Error` (`IoError`, `VocabularyError`, `ConfigError`) |
| A file that cannot be opened | throw `Io::Error` (`core/lib/file_stream.h`) |
| Word not in the vocabulary, unparseable expression, too few words | `QueryStatus` on the result struct |

Nothing under `core/words/` opens a file. Serialization takes `std::istream&` /
`std::ostream&`; the front ends open the stream through `Io::ReadFile` /
`Io::WriteFile`, which append the file name to whatever the core throws — so the
messages users see still name the file.

Loading an artifact that is missing, unreadable or truncated throws. It does not
quietly hand back an empty vocabulary or a zero-filled embedding matrix, which
would otherwise surface much later as a model that trains but learns nothing.

Bad user input is not exceptional: an expression box would otherwise throw on
every half-typed word. `QueryStatus` renders as an inline message; an exception
renders as a dialog. The CLI catches `Words::Error` once, in the command wrappers of
`cli/words/commands.cpp`, and exits non-zero.

Exit codes: `0` success, `1` a failed check or reported error, `2` an internal
error. Parse failures carry CLI11's own codes: `106` for no subcommand or a
missing required option, `105` for a file check that failed, `109` for an
unknown flag.

## Training

`Words::Trainer` mirrors `Neural::Classifier::Trainer`, so a GUI controller
drives it the same way:

```cpp
Words::Trainer trainer;
trainer.setVocabulary(vocabulary);          // shared_ptr<const Vocabulary>
trainer.setCorpus(corpus);                  // shared_ptr<const Words::Corpus>
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
`TrainSummary::stopped`. The cancellation check deliberately sits at the chunk
boundary rather than inside the pair loop: generating pairs is the innermost
loop of the whole trainer, and a test on every pair would cost a branch there for
no user-visible benefit. A chunk is small enough that cancellation still feels
immediate.

**`--sample 0` disables subsampling**, and that path needs care. With subsampling
off there are no keep-probabilities to sum, so a naive token estimate would come
back as zero — and since `EstimateTotalPairs` feeds the learning-rate schedule, a
total of zero would pin progress at 0 and the rate would never decay. `Subsampler`
special-cases it and reports the whole corpus instead.

### Known race

The monitor thread calls `MeanProbeLoss` while workers are mutating the same
rows (Hogwild-style lock-free updates through a `mutable` model). This is
formally a data race and ThreadSanitizer will report it. It is left as-is
deliberately: the reported loss is a progress indicator, and the contention-free
updates are the point of the design.

Subwords make the same trade louder: each pair now writes `1 + k` input rows,
and the handful of buckets that every other word contains are written by every
thread. The updates that collide are lost, which is the accepted cost of
lock-free training here as it is in fastText.

## CLI reference

`MatrixGui_words` takes a subcommand. Each one binds to its own options struct in
`cli/words/options.h`, and its body lives in `cli/words/commands.cpp` —
CLI11 calls the right handler through `->callback()`, so there is no dispatch
chain to keep in step with the registrations.

Follow the pipeline order: `inspect` to see what you have, `buildvoc` and
`buildcor` to prepare it, `train`, then any of the query commands.

<details>
<summary><code>inspect</code> — summarise a raw text dump</summary>

Counts tokens and types and lists the most frequent words. Use it to pick a
`--min-count` for `buildvoc` before committing to a build.

| Flag | Default | Description |
|------|---------|-------------|
| `--input-file <path>` | *required* | Raw text file |
| `--top <n>` | `15` | How many frequent words to list |

</details>

<details>
<summary><code>buildvoc</code> / <code>loadvoc</code> — the vocabulary</summary>

`buildvoc` scans a raw dump and writes a `.voc` file; `loadvoc` reads one back
and prints its statistics.

| Flag | Default | Description |
|------|---------|-------------|
| `--input-file <path>` | *required* | Raw text file |
| `--output-file <path>` | *required* | Where to write the vocabulary |
| `--min-count <n>` | `5` | Drop words occurring fewer times than this |
| `--prune-threshold <n>` | `20000000` | Prune rare words whenever distinct words exceed this; `0` disables. Approximate, as in the original word2vec: the output reports how many prune runs fired and the final reduce counter |

`loadvoc` takes only `--input-file`, pointing at a built `.voc`.

</details>

<details>
<summary><code>buildcor</code> / <code>loadcor</code> — the corpus</summary>

`buildcor` encodes a raw dump into word ids using an existing vocabulary, writing
a `.cor` file. Because the ids belong to that vocabulary, a corpus and the
vocabulary it was built against must always be used together.

| Flag | Default | Description |
|------|---------|-------------|
| `--input-file <path>` | *required* | Raw text file |
| `--vocabulary <path>` | *required* | Built vocabulary |
| `--output-file <path>` | *required* | Where to write the corpus |

`loadcor` takes only `--input-file`, pointing at a built `.cor`.

</details>

<details>
<summary><code>train</code> — train the embeddings</summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--vocabulary <path>` | *required* | Built vocabulary |
| `--corpus <path>` | *required* | Built corpus |
| `--output-file <path>` | *required* | Where to write the embeddings |
| `--dim <n>` | `100` | Embedding dimension |
| `--negatives <n>` | `5` | Negative samples drawn per pair |
| `--window <n>` | `5` | Maximum window radius |
| `--epochs <n>` | `5` | Passes over the corpus |
| `--sample <x>` | `1e-4` | Subsampling threshold; `0` disables subsampling |
| `--lr <x>` | `0.025` | Initial learning rate |
| `--threads <n>` | `0` | Worker threads; `0` uses the hardware concurrency |
| `--buckets <n>` | `0` | Hash buckets for character n-grams. `0` means no subwords at all — the run is then bit-for-bit the plain SGNS it always was |
| `--min-n <n>` | `3` | Shortest character n-gram; ignored unless `--buckets` is set |
| `--max-n <n>` | `6` | Longest character n-gram; ignored unless `--buckets` is set |
| `--corpus-storage <mode>` | `auto` | How the corpus is held: `load` reads it into memory, `mmap` maps the file read-only, `auto` loads only when the file is small (< 4 GB and < ¼ of `MemAvailable`). The chosen mode is printed at startup |
| `--subwords-file <path>` | `<output-file>.sub` | Where the n-gram vectors go; written only when `--buckets` is non-zero |

</details>

<details>
<summary>Subword (fastText-style) training</summary>

With `--buckets N` a word's vector during training is the mean of its own row
and the rows of its character n-grams, so word forms that share a stem share
most of their vectors. This is what Russian needs: `кот`, `кота` and `котом`
are three vocabulary entries splitting the statistics of one word.

```bash
MatrixGui_words train --vocabulary ru.voc --corpus ru.cor --output-file ru.emb \
                      --dim 300 --buckets 2000000 --min-n 3 --max-n 6
MatrixGui_words neighbours --vocabulary ru.voc --embeddings ru.emb \
                           --subwords-file ru.emb.sub --word котёнком
```

Two files come out of such a run:

| File | Holds | Used by |
|---|---|---|
| `--output-file` (`.emb`) | the composed `V × dim` matrix, one row per word, in the usual format | every query command and the GUI, unchanged |
| `--subwords-file` (`.sub`) | the `buckets × dim` n-gram matrix plus `min-n`/`max-n`/`buckets` | `neighbours --subwords-file`, for words the vocabulary does not have |

The composition happens once, at the end of training, so a query still costs one
pass over `V` rows rather than recomposing every word. The n-gram rows for
**words in the vocabulary** are already folded into the `.emb`; the `.sub` file
exists for the words that are not.

**n-grams are counted in characters, not bytes.** `кот` wrapped as `<кот>` is
five characters, so `--min-n 3 --max-n 6` gives six n-grams. A byte-wise split
would give fifteen, each covering one and a half Cyrillic letters.

Cost, measured on a 1.7M-word Russian vocabulary (`n = 3..6`): 30.6 n-grams per
vocabulary word, 18.5 per corpus token, 4.2M distinct n-grams. Since every one
of those rows is read and written per pair, training slows down by roughly the
same factor as the row count grows, and `--buckets 2000000` adds
`2000000 × dim × 4` bytes — 2.4 GB at `--dim 300`. `--min-n 5 --max-n 5` is the
cheap end: 4.1 rows per token and 1.3M distinct n-grams.

`--buckets 0` is the default precisely so that an English run stays comparable
with every run recorded before this existed.

</details>

<details>
<summary><code>neighbours</code> — nearest words</summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--vocabulary <path>` | *required* | Built vocabulary |
| `--embeddings <path>` | *required* | Trained embeddings |
| `--word <w>` | *(empty)* | Query word. Leaving it empty runs a default battery of words |
| `--subwords-file <path>` | — | `.sub` file from a subword run. Only consulted when `--word` is **not** in the vocabulary, in which case its vector is assembled from n-grams alone |
| `--count <n>` | `10` | How many neighbours to show |

</details>

<details>
<summary><code>expression</code> — free-form vector arithmetic</summary>

Takes the expression as a positional argument.

```bash
MatrixGui_words expression "king - man + woman" \
  --vocabulary built.voc --embeddings emb.bin
```

| Argument / flag | Default | Description |
|------|---------|-------------|
| `<expression>` | *required* | Positional, e.g. `"king - man + woman"` |
| `--vocabulary <path>` | *required* | Built vocabulary |
| `--embeddings <path>` | *required* | Trained embeddings |
| `--count <n>` | `10` | How many results |

When the terms form a three-word analogy, the 3CosMul reranking is reported
alongside the plain vector result.

</details>

<details>
<summary><code>oddone</code> — the odd word out</summary>

Takes the word list as one positional argument.

```bash
MatrixGui_words oddone "breakfast cereal lunch dinner" \
  --vocabulary built.voc --embeddings emb.bin
```

| Argument / flag | Default | Description |
|------|---------|-------------|
| `<words>` | *required* | Positional, a quoted space-separated list |
| `--vocabulary <path>` | *required* | Built vocabulary |
| `--embeddings <path>` | *required* | Trained embeddings |

</details>

<details>
<summary><code>axis</code> — project onto a semantic axis</summary>

Builds a direction from two words and ranks words along it.

```bash
MatrixGui_words axis "good - bad" \
  --vocabulary built.voc --embeddings emb.bin
```

| Argument / flag | Default | Description |
|------|---------|-------------|
| `<axis>` | *required* | Positional, e.g. `"good - bad"` |
| `--vocabulary <path>` | *required* | Built vocabulary |
| `--embeddings <path>` | *required* | Trained embeddings |
| `--words <list>` | *(empty)* | Words to project; empty scans the vocabulary |
| `--restrict-to <n>` | `30000` | Scan only the top-N most frequent words; `0` = all |
| `--count <n>` | `10` | How many to show at each end |

</details>

<details>
<summary><code>evaluate</code> — analogy and similarity benchmarks</summary>

Give it at least one of `--analogies` or `--similarity`.

| Flag | Default | Description |
|------|---------|-------------|
| `--vocabulary <path>` | *required* | Built vocabulary |
| `--embeddings <path>` | *required* | Trained embeddings |
| `--analogies <path>` | — | A `questions-words.txt`-style file |
| `--similarity <path>` | — | A WordSim/SimLex-style file |
| `--score-column <n>` | `2` | 0-based column holding the human score in the similarity file |
| `--restrict-to <n>` | `30000` | Search only the top-N most frequent words; `0` = all |
| `--threads <n>` | `0` | Worker threads; `0` uses the hardware concurrency |

</details>

<details>
<summary>Memory notes</summary>

Subword training adds a third matrix of `--buckets × --dim` floats on top of the
two `V × --dim` ones, plus the flat n-gram table (about 4 bytes per n-gram
reference: 209 MB for a 1.7M-word Russian vocabulary at `n = 3..6`). At
`--dim 300` and `--buckets 2000000` that is 2.4 GB of matrix on top of whatever
the vocabulary already costs, so check `--buckets` against available memory
before a large run.

Building a vocabulary streams the whole raw dump into an `unordered_map`, which
is pre-sized for roughly a million distinct words so it never rehashes
mid-stream. That is the right trade for a wiki-scale corpus, but it means
`buildvoc` allocates on the order of 10 MB of buckets even for a tiny test
corpus. It is a fixed floor, not a leak.

</details>

## Configuration

`core/words/config.h` holds `ModelConfig` (shape), `SamplingConfig` (which pairs
exist), `TrainConfig` (schedule) and the `WordsConfig` aggregate.
`Words::Validate` rejects impossible combinations up front, rather than letting
them surface from a sampler constructor after a worker thread has started.

JSON bindings are in `config_json.h`, kept separate so that including
`config.h` does not pull nlohmann into every translation unit.

## GUI

The `MatrixGui` application has a Words mode — the **Word embeddings (SGNS)**
button on the mode picker — with three sub-tabs sharing one terminal. For the
shell around it (threading rules, the terminal, themes) see [gui.md](gui.md);
what follows is specific to this mode:

- **Data & Training** — inspect a raw dump, build/load the vocabulary and
  corpus, train with live probe-loss and speed charts, save/load embeddings.
- **Explore** — neighbours, the default battery, vector expressions,
  odd-one-out and axis projection. Results print into the terminal through the
  same `report/` functions the CLI uses, so the output is identical. A model
  trained elsewhere is loaded right on this tab: Load vocabulary…, then Load
  embeddings… (embeddings are indexed by the vocabulary's word ids, so the
  vocabulary comes first). When a subword model is loaded, a word the vocabulary
  does not have is answered from its n-grams instead of rejected.
- **Evaluate** — analogy and similarity datasets. Analogy evaluation runs on a
  worker thread and cannot be cancelled once started.

After an in-session training run the query index is built in memory
(`EmbeddingIndex` from the trainer's embeddings) — no save/load round-trip is
needed before exploring. Reloading or rebuilding the vocabulary resets the
loaded corpus, since its encoded ids belong to the old vocabulary.

### Subwords in the GUI

The training panel has `Subword buckets` (shown as *off* at zero, the default),
`min n` and `max n`, matching `--buckets`, `--min-n` and `--max-n`. A run with
buckets keeps the n-gram matrix alongside the embeddings, and **Save
embeddings… writes it as `<name>.sub` beside the file you picked** — the same
name the CLI defaults to. There is no separate dialog: the two files are useless
apart, so the sidecar's name is derived rather than asked for.

Load embeddings… picks that sidecar back up when it is there and the dimensions
agree, and says so in the terminal. A sidecar that does not match, or is not a
`.sub` at all, is named and ignored — the embeddings still load. Loading plain
embeddings with no sidecar drops any n-grams that were held, so a stale matrix
can never be written next to vectors it does not belong to.

## Tests

```bash
cmake -B build -DBUILD_TESTS=ON && cmake --build build -j
ctest --test-dir build --output-on-failure     # unit tests (doctest)
tests/golden/compare.sh words                  # CLI snapshot
```

Two layers:

- **Unit tests** (`tests/words/`) over synthetic fixtures — deterministic on
  fixed seeds, whole suite under three seconds. `tests/support/fixtures.h`
  builds toy vocabularies and embeddings whose neighbours are analytically
  known.
- **A CLI snapshot** (`tests/golden/words/`) running every subcommand against a
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
