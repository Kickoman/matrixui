# Golden CLI snapshots

One subdirectory per tool — `words/`, `classifier/`, `generator/`, `functions/` —
each with its own `capture.sh` (records a snapshot), `compare.sh` (re-captures and
diffs against `expected/`) and `expected/` (the committed snapshot). The top-level
[`compare.sh`](compare.sh) runs all four, or just the ones named:

```bash
tests/golden/compare.sh                 # everything
tests/golden/compare.sh classifier      # one tool
tests/golden/classifier/compare.sh      # same, directly; [binary] overrides ./build/<name>
```

Each captured case produces `<name>.out`, `<name>.err`, `<name>.code`; the
normalized forms (`*.out.normalized`, `*.err.normalized`) and the exit codes
are what `compare.sh` diffs. Raw `.out`/`.err` stay untracked, for eyeballing.

## Local only

Snapshots run on the machine that recorded them, never in CI: release builds
use `-march=native`, and several expectations pin float-derived output (words
training losses, the classifier's `predict` digit), which is not bit-portable
across CPUs. See `docs/building.md`.

## Determinism and fixtures

**words** regenerates everything at capture time. Its corpus is committed
(`words/corpus.txt`, produced by `words/make_corpus.py`, seed 20260831) and the
pipeline is deterministic by construction: a hardcoded training seed, one
thread, fixed corpus.

**classifier / generator cannot do that**: weight initialization, dataset
shuffling and GAN noise all come from `std::random_device`, so training is
unreproducible. Hence:

- The 8×8 PNG dataset is regenerated deterministically at capture time by
  `classifier/fixtures/make_dataset.py` (stdlib-only, seed 20260831) and is
  not committed.
- `classifier/fixtures/network.wgt` and `generator/fixtures/generator.wgt`
  are **frozen artifacts**, trained once by
  `classifier/fixtures/make_fixtures.sh` and committed. The goldens pin
  behavior given these exact bytes. Re-running `make_fixtures.sh` produces
  different weights — if you do, re-record both tools' `expected/` in the same
  commit.
- What cannot be pinned is normalized away by each tool's `capture.sh`:
  the classifier's final accuracy and its epoch chatter on stderr, the GAN's
  per-epoch lines, the classifier's opinion of a freshly generated image.

**functions** is deterministic whenever `--seed` is passed: it seeds both
sources of randomness — the genetizer's tournament draw and the thread-local
engine behind mutation, crossover and random tree growth — so every captured
case pins one. Its data file (`functions/data.csv`) is committed, the run loop
prints no timings, and its output turned out identical between a Debug and a
`-march=native` Release binary, so only the work directory and the binary path
need normalizing.

## Re-recording

After an intentional output change, re-record the affected tool and review the
diff before committing:

```bash
tests/golden/words/capture.sh      tests/golden/words/expected
tests/golden/classifier/capture.sh tests/golden/classifier/expected
tests/golden/generator/capture.sh  tests/golden/generator/expected
tests/golden/functions/capture.sh  tests/golden/functions/expected
```
