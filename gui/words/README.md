# `gui/words` — the Words mode

The Qt front end over [`core/words`](../../core/words/README.md). One controller
plus a mode widget holding three tabs and a terminal. No embedding logic lives
here: every button ends in a call into `core/words`, and every result comes back
as one of its report structs.

The other front end is [`cli/words`](../../cli/words/README.md). The
two share the whole pipeline; they differ only in where the `std::ostream` goes.

| File | Contains |
|---|---|
| `words_controller.h/.cpp` | `WordsController` — owns the model state and runs every long operation |
| `words_mode_widget.h/.cpp` | `WordsModeWidget` — the tab container and the terminal |
| `words_data_tab_widget.h/.cpp` | Data & Training tab: build/load artifacts, run training, charts |
| `words_explore_tab_widget.h/.cpp` | Explore tab: neighbours, expression, odd-one-out, axis |
| `words_evaluate_tab_widget.h/.cpp` | Evaluate tab: analogy and similarity benchmarks |
| `words_train_config_widget.h/.cpp` | `WordsTrainConfigWidget` — editor for the training knobs |

Widgets never talk to each other. Each holds a `WordsController*`, calls slots on
it, and repaints from `WordsController::Info` when `infoUpdated()` fires. General
GUI conventions — the `valueChanged` loop, mode registration — are in
[`docs/gui.md`](../../docs/gui.md).

## `WordsController`

`ModeController` subclass, registered as a mode through `gui/lib/mode_factory.cpp`.

`getInfo()` returns an `Info` snapshot: what is loaded (`hasVocabulary`,
`hasCorpus`, `hasEmbeddings`, `hasIndex`), the sizes behind those, the six
configured paths, and the current `Words::WordsConfig`. Widgets enable and
disable themselves from it rather than tracking state of their own.

### Threading contract

**Every member is written on the GUI thread only.** Worker lambdas capture
immutable snapshots — `shared_ptr`s, copies of paths — up front, and publish
results back with `QMetaObject::invokeMethod`. One worker thread at a time
serves all long operations, guarded by `runTask(label, isTraining, task)`.

`runTask` sets `busyFlag` on the GUI thread *before* starting the thread, so a
double click cannot slip through the gap between the check and the start. A
request arriving while busy is logged and dropped, not queued.

Which operations go where:

| Tab | Work | Thread |
|---|---|---|
| Data & Training | inspect, build/load vocabulary and corpus, train, save/load embeddings | worker |
| Explore | neighbours, battery, expression, odd-one-out, axis | GUI thread — these are milliseconds |
| Evaluate | analogies, similarity | worker |

**`Words::Trainer` invokes its progress callback on the thread that called
`train()`.** The controller marshals it into the `trainProgressed` signal;
`WordsDataTabWidget::handleTrainProgress` then drives the progress bar, the ETA
label and the two `TimeChart`s. Never touch a widget from that callback
directly.

`requestStop()` is honoured at chunk granularity (~1000 corpus tokens) and sets
`TrainSummary::stopped`. `Trainer::train()` resets its own stop flag, so the
member `trainer` survives repeated runs and does not need recreating.

### Composed versus normalised embeddings

The controller holds both, and the distinction matters:

```cpp
std::shared_ptr<const Words::Embeddings>     embeddings;  // composed: the only thing worth saving
std::shared_ptr<const Words::EmbeddingIndex> index;       // normalised: queries only
std::shared_ptr<const Words::SubwordVectors> subwords;    // n-grams: unknown words only
```

`EmbeddingIndex` normalises the rows it is given, so saving from it would write
unit vectors and silently lose magnitude. Save from `embeddings`; query through
`index`. `hasEmbeddings` and `hasIndex` are separate `Info` flags for the same
reason.

`embeddings` comes from `Trainer::getWordEmbeddings()`, not
`getInputEmbeddings()`: with subwords on, that is the `V × dim` matrix composed
from each word row and its n-gram rows. Without subwords the two are the same
object, so nothing is copied that was not copied before.

### Subwords

Training with `Subword buckets` greater than zero also leaves the n-gram matrix
in `subwords`. Three rules keep it honest:

- **It is saved beside the embeddings**, as `<embeddings path>.sub`, by the same
  Save embeddings… button. The name is derived rather than asked for, because
  the two files are useless apart — the same reasoning as a `.voc` and its
  `.cor`. The CLI's `--subwords-file` defaults to the same name.
- **It is loaded from beside them too.** Load embeddings… looks for that sidecar
  and picks it up when the dimensions agree, reporting what it found; a `.sub`
  whose width does not match is named and ignored rather than half-used.
- **It is dropped when the embeddings change without one.** Loading a plain
  `.emb` clears `subwords`, so a stale n-gram matrix can never be written next
  to vectors it does not belong to, nor answer a query about them.

With it loaded, the Explore tab's neighbours query answers a word the vocabulary
does not have: the query vector is the mean of the word's n-gram rows and the
result prints through `PrintSubwordNeighbourReport` (`out of vocabulary, N
n-grams` instead of an id and a count). A word the vocabulary *does* have never
takes that path — its composed row is already in the index.

**Cost to know about:** `subwords` is a copy of the bucket matrix, so it holds
`buckets × dim × 4` bytes on top of the model — 2.4 GB at 2M buckets and dim
300. The copy is what makes the saved file match the embeddings it sits beside
even after another run starts.

Vocabulary and corpus are `shared_ptr<const>` because `Words::Trainer` shares
rather than takes ownership — the tabs keep showing statistics after a run ends,
and the Explore tab needs the same `Vocabulary` to turn ids back into words.

## Output

`WordsModeWidget` owns an `AdvancedTerminal` and hands the controller a
`std::ostream*` over it via `setLogger` — a `ThreadSafeTerminalOStream` from
`gui_common/advanced_terminal.h`, which is what makes writes from a worker
thread safe.

That stream is passed straight to `Words::Trainer::setOutputStream` and to the
printers in `core/words/report/`, so the GUI terminal shows the same banner,
ticks, summary and query tables the CLI prints. Every controller line is
prefixed `[controller] `; when no logger is set it falls back to `std::cerr`,
which is what makes the mode usable from a console build.

## `WordsTrainConfigWidget`

Editor for the knobs the CLI exposes: `dim`, `negatives`, `window`, `epochs`,
`sample`, learning rate, `threads`, and the three subword settings — `buckets`
(shown as *off* at zero, which is the default and means plain SGNS), `min n` and
`max n`.

It keeps a full `Words::WordsConfig` in `stored` and overwrites only the fields
it shows. Everything else — `chunkSize`, `syncEvery`, `reportEveryMs`,
`probePairs`, `seed`, `negativeTableSize`, `negativePower`,
`minLearningRateFactor` — **round-trips through `getConfig()` untouched**, so it
stays editable by hand in the settings file. Do not rebuild the config from
scratch in `getConfig()`; that would silently reset the hidden fields on the
first click.

The whole `WordsConfig` persists through `ModeSettings` as JSON, using the
serialisers in `core/words/config_json.h`.

## Traps

- Read widget values at click time, never on `valueChanged` — that loops through
  `setValue` → `valueChanged` → `infoUpdated`. See [`docs/gui.md`](../../docs/gui.md).
- `busyLabel` is GUI-thread-only; `busyFlag` and `trainingFlag` are atomics
  because the worker clears them.
- Explore queries run synchronously and will freeze the UI if pointed at a
  vocabulary far larger than the ones this mode is used with. `restrictTo` on
  the axis query exists to bound that scan.
- There is no test framework for the GUI. Behaviour changes here are verified by
  hand, or by an out-of-tree harness driving the controller with a
  `std::ostringstream` logger under `QT_QPA_PLATFORM=offscreen`.
