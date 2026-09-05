# Golden CLI snapshots

One subdirectory per tool — `words/`, `classifier/`, `generator/` — each with
its own `capture.sh` (records a snapshot), `compare.sh` (re-captures and diffs
against `expected/`) and `expected/` (the committed snapshot). The top-level
[`compare.sh`](compare.sh) runs all three, or just the ones named:

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

For the same reason these are **expected to differ on macOS** — a different CPU
and a different standard library reorder float accumulation. The scripts
themselves are kept portable (macOS ships bash 3.2 and a BSD userland, so
`mktemp` gets an explicit template, the tool list uses positional parameters
rather than an array, and path normalisation accounts for `/var` being a symlink
to `/private/var`), but a clean diff on a Mac is not something to expect. The
unit tests are the portable check. See `docs/macos.md`.

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

## Re-recording

After an intentional output change, re-record the affected tool and review the
diff before committing:

```bash
tests/golden/words/capture.sh      tests/golden/words/expected
tests/golden/classifier/capture.sh tests/golden/classifier/expected
tests/golden/generator/capture.sh  tests/golden/generator/expected
```
