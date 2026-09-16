# `cli/functions` — the `MatrixGui_functions` subcommands

The command-line front end over [`core/functions`](../../core/functions/): a
genetic algorithm that evolves an arithmetic expression fitting a table of
expected values. One subcommand so far, `run`.

The layout follows [`cli/words`](../words/README.md) — same four files, same
`Guarded` dispatch, same exit codes:

| File | Contains |
|---|---|
| `options.h` | `RunOptions`: the three paths plus the whole `FunctionsConfig` |
| `commands.h` | `int Run(std::ostream&, std::ostream&, const RunOptions&)` and the exit-code constants |
| `commands.cpp` | The body, plus config validation and the config dump |
| — | The data format itself lives in [`core/functions/expected_csv.{h,cpp}`](../../core/functions/expected_csv.h) |
| `main.cpp` | The CLI11 app: flag definitions and dispatch |

## The data file

`--data` takes a CSV whose header names the variables and whose last column must
be called `expected`:

```
x,y,expected
1,1,2
1,2,3
2,1,4
```

One row is one data point. The variable names are whatever the header says —
they are the same names the evolved expressions use, and `addExpected` is what
teaches the mutation config that they exist. Parse failures name the line and
the column: `line 3, column 'x': not a number: 'abc'`.

`ParseExpectedCsv` started here as a CLI input format, but the GUI's point
import and export made it an interchange format both front ends read and write,
so it lives in `core/functions/expected_csv.{h,cpp}` next to a writer,
`WriteExpectedCsv`. The same is true of `Genetizer::Validate`
([`core/functions/validate.h`](../../core/functions/validate.h)): the GUI has to
run the same checks before a run, so they are no longer private to this file.

## Config

Every knob is a field of `Genetizer::FunctionsConfig`
([`core/functions/config.h`](../../core/functions/config.h)), so the same
settings can come from a JSON file or from flags:

```bash
MatrixGui_functions run --data points.csv --config cfg.json --epochs 200
```

`--config` is read out of `argv` before the options are registered — the same
load-bearing ordering as [`cli/classifier`](../classifier/README.md). A config
file supplies the defaults, a flag still overrides its one field, and
`capture_default_str()` snapshots the loaded values so `--help` shows what will
actually be used.

`--save-config` writes the effective config back out. That file is a valid
`--config` input, which is how a run is reproduced: it carries the seed, the
starting population and every GA parameter.

The starting population comes from `initialExpressions` (or repeated
`--expression`) plus `randomCount` random trees. An expression that does not
parse fails the run and is named in the error.

**Seeded expressions are the only source of exact constants.** Random constants
come from a `uniform_real_distribution` and mutation nudges them by a normal
step, so a constant that is exactly `2` has probability zero of ever appearing
on its own. `(x*x+x)/2` is therefore reachable only if something in the gene
pool already carries an integer — which is what `--expression "x/2"`,
`--expression "x+1"` and friends are for. Without them the search finds the
right shape with drifted constants, e.g. `(1.1827+x)*x/2.0253`.

## Fitness

Three terms, blended by weights that are normalized by their sum, so only their
ratio matters:

| Term | Meaning |
|---|---|
| accuracy | `1 / (1 + mean error)`, the error of each point divided by `max(1, mean(abs(expected)))` |
| complexity | `1 / (1 + rpn units / 50)` |
| length | `1 / (1 + printed characters / 50)` |

**The accuracy term is normalized by both the point count and the magnitude of
the data**, so the same problem measured in units and in thousands ranks
identically, and adding points does not shrink the accuracy term. Without that
normalization the summed error on a dataset reaching 190 was ~1140 for a
trivial expression, the accuracy term collapsed to 0.0009 against complexity's
0.41, and the fitness degenerated into "shortest expression wins" — the
population converged onto `x` and no partial solution could ever get credit.

The weights are tuned around that normalization, which is why
`complexityWeight` is 0.2 rather than something closer to the accuracy weight;
see the comment on `FitnessConfig` in
[`core/functions/applier.h`](../../core/functions/applier.h). A point that
evaluates to inf or nan costs 100 in the same normalized units.

## World output

Identical expressions collapse into one row with a `Copies` count, and a
`unique K / N` line reports how many distinct expressions the whole world holds
— a converged population is mostly clones, so the uncollapsed top rows were N
copies of one string. `--print-top` counts **distinct** expressions, and `Birth`
is the earliest birth in the clone group (clones tie on rank and the sort is not
stable, so the first one encountered is not reproducible).

## Early stopping

`--patience` stops the run when the best rank has not improved for that many
consecutive epochs; `0` disables it and every epoch runs. The naming follows
`MatrixGui_headless train`, with one deliberate difference: the classifier
requires an improvement of at least 0.001, while ranks here improve by much
less than that late in a run, so any improvement at all resets the counter.

## Order inside the body

`RunRun` is short but its order is load-bearing, and the pieces are not
interchangeable:

1. **The applier is declared before the genetizer.** `getRankFunction()` and
   friends return lambdas capturing `this`, and the genetizer keeps them for its
   whole life.
2. **`addExpected` runs before `seedRandom`.** The variable list a random tree
   draws from is built by `addExpected`; seeding first throws out of
   `std::vector::at`.
3. **`setConfig` runs before `runEpoch`.** It sizes the tournament buffer.
4. **`setMutationOptions`, `setFitnessOptions` and every `addExpected` run
   before the first `addOrganism`,** because adding an organism ranks it
   immediately — and because `addExpected` and `setFitnessOptions` clear the
   rank cache, every rank taken before them being measured against different
   data or different weights.

## Seeds

`seed` 0 (the default) leaves both engines on `std::random_device`, as the
library does on its own. A non-zero seed pins both: `genetizer.setSeed()` for the
tournament draw and `FunctionGenetizerApplier::SeedThreadRng()` for mutation,
crossover and random tree growth. They get different values from the one seed so
the two streams do not march in step.

Two runs with the same seed and config produce byte-identical output — that is
what makes [`tests/golden/functions`](../../tests/golden/functions/) possible,
and `tests/cli/functions_commands_test.cpp` asserts it directly.

## Errors and exit codes

The body does not catch; it throws `std::runtime_error` — from its own
validation, from `ParseExpectedCsv`, or from `Matematyka::Expression` when an
expression will not parse. `Guarded` catches once, prints the `error: ` line to
`err` and returns the code.

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | a failed check or reported error |
| `2` | an internal error |
| `105`/`106`/`109` | CLI11's own parse-failure codes: failed file check / missing required option or subcommand / unknown flag |

This is one catch-site fewer than `cli/words`, which also catches `Words::Error`
and `Io::Error`: `core/functions` has no error hierarchy of its own, and reads
its input through the CLI layer.

## Adding a subcommand

1. Add an options struct to `options.h`.
2. Define `RunName(out, options)` and wrap it via `Guarded`.
3. Register the flags and a `->callback()` that assigns `exitCode` in `main.cpp`.
4. Extend `tests/golden/functions/capture.sh` and re-record `expected/`.

Keep computation out of the body: anything worth testing belongs in
`core/functions`, or in `csv.cpp` if it is about the CLI's input format.
