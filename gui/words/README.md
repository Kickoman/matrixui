# `gui/words` — the Words mode

The Qt front end over [`core/words`](../../core/words/README.md). One controller
plus a mode widget holding three tabs and a terminal. No embedding logic lives
here: every button ends in a call into `core/words`, and every result comes back
as one of its report structs.

The other front end is [`core/words_cli`](../../core/words_cli/README.md). The
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

### Raw versus normalised embeddings

The controller holds both, and the distinction matters:

```cpp
std::shared_ptr<const Words::Embeddings>    embeddings;   // raw: the only thing worth saving
std::shared_ptr<const Words::EmbeddingIndex> index;       // normalised: queries only
```

`EmbeddingIndex` normalises the rows it is given, so saving from it would write
unit vectors and silently lose magnitude. Save from `embeddings`; query through
`index`. `hasEmbeddings` and `hasIndex` are separate `Info` flags for the same
reason.

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
`sample`, learning rate, `threads`.

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
