# `core/words/report` — console rendering

Every printer in the words module, and nothing else. These files turn the report
structs the rest of `core/words` returns into text on a `std::ostream`. They
compute nothing, hold no state, and open no files.

| File | Renders |
|---|---|
| `inspect_report.h/.cpp` | `CorpusStatistics` from `data/inspect.h` |
| `train_report.h/.cpp` | the training banner, progress ticks and summary |
| `query_report.h/.cpp` | the six report structs from `query/queries.h` |
| `evaluate_report.h/.cpp` | `AnalogyReport` and `SimilarityReport` from `query/evaluate.h` |

The stream plumbing they share — `NullStream()`, `DefaultLogStream()` and
`StreamFormatGuard` — is not here. It lives in `core/lib/stream_format.h`,
because none of it has anything to do with embeddings.

## Compute, then report

Every operation in the module splits in two: a function that returns a struct,
and a printer for that struct. The printer is always optional.

```cpp
const auto report = Words::QueryNeighbours(vocabulary, index, "king", 10);
Words::PrintNeighbourReport(std::cout, vocabulary, report);   // optional
```

That is what lets one implementation serve both front ends. The CLI calls the
printer; a Qt tab can point the stream at `ThreadSafeTerminalOStream`
(`gui_common/advanced_terminal.h`), or skip this folder entirely and render the
same struct into widgets.

The dependency arrow runs one way — `report/` includes the computing headers,
never the reverse. There is exactly one exception: `train/trainer.cpp` (the
`.cpp`, not the header) calls `PrintTrainBanner`, `PrintTrainProgress` and
`PrintTrainSummary` so a long run prints as it goes.

## Formatting discipline

Every printer that touches `std::setprecision`, `std::scientific`, `std::setw`
or `std::setfill` opens with

```cpp
const StreamFormatGuard guard(out);
```

The guard restores `flags`, `precision` and `fill` on the way out, including on
an exception. Without it those settings leak into whatever else writes to the
stream — which, on a stream shared with a GUI terminal, means every later line
in the app.

## The printers

### `inspect_report.h`

```cpp
void PrintCorpusStatistics(std::ostream& out, const CorpusStatistics& statistics);
```

Symbol, token and type counts, the min-count survivor table, and the top-N words
with their share of the corpus.

### `train_report.h`

```cpp
void PrintTrainBanner(std::ostream& out,
                      const ModelConfig&, const SamplingConfig&, const TrainConfig&,
                      std::size_t threadCount, std::size_t estimatedPairs,
                      std::size_t probeCount, double initialLoss);
void PrintTrainProgress(std::ostream& out, const TrainProgress& progress);
void PrintTrainSummary(std::ostream& out, const TrainSummary& summary);
```

Three points in a run: the banner before the first pair, one tick every
`TrainConfig::reportEveryMs`, and the closing figures.

The banner takes the resolved values, not just the config — `threadCount` after
`0` has become `hardware_concurrency`, `estimatedPairs` after
`EstimateTotalPairs`, and the loss measured on the probe set before any update.
That is what makes the banner and the summary directly comparable.

### `query_report.h`

```cpp
void PrintNeighbourReport(std::ostream&, const Vocabulary&, const NeighbourReport&);
void PrintAnalogyQueryReport(std::ostream&, const Vocabulary&, const AnalogyQueryReport&);
void PrintExpressionReport(std::ostream&, const Vocabulary&, const ExpressionReport&);
void PrintOddOneOutReport(std::ostream&, const Vocabulary&, const OddOneOutReport&);
void PrintAxisReport(std::ostream&, const Vocabulary&, const AxisReport&);
void PrintBatteryReport(std::ostream&, const Vocabulary&, const BatteryReport&);
```

Each takes the `Vocabulary` as well, because the report structs carry `TWordId`s
and only the vocabulary can turn one back into a word.

A report whose `QueryStatus` is not `ok` prints its message instead of a table;
the caller does not need to check first.

Two row layouts are shared internally: an indented `word  score` used by the
neighbour, analogy and expression reports, and a signed `score  word` used by
the axis report, where the sign is the point.

### `evaluate_report.h`

```cpp
void PrintAnalogyReport(std::ostream&, const AnalogyReport&);
void PrintSimilarityReport(std::ostream&, const std::string& name, const SimilarityReport&);
```

The analogy report prints one row per category with the 3CosAdd and 3CosMul
columns side by side, then the semantic/syntactic/overall roll-ups.
`PrintSimilarityReport` takes a `name` because a run typically evaluates several
similarity datasets and the report struct itself does not know which one it is.

## Adding a printer

1. Put the computation in `data/`, `train/` or `query/`, returning a struct.
2. Add the printer here, taking `std::ostream&` first.
3. Open with a `StreamFormatGuard` if you touch stream formatting.
4. Add the `.cpp` to the `# Words: console rendering` block in `CMakeLists.txt`.
5. If the CLI prints it, cover it in `tests/words/golden_print_test.cpp` — that
   suite renders reports into a string and compares, which is what keeps the
   golden CLI snapshots honest.

Never add `#include <iostream>` here. The module invariant is that it appears
nowhere under `core/words/`; take the stream as a parameter.
