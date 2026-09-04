# Classifier

The classifier trains a fully-connected neural network to sort images into
labelled classes, and can run inference on a single image. The shipped defaults
target MNIST-style handwritten digits — 28×28 pixels, ten classes — but both the
image size and the class count are configurable.

The CLI binary is `MatrixGui_headless`, which has two subcommands: `train` and
`predict`. The GUI offers the same things interactively through its **Classifier**
and **Recognizer** modes.

## Dataset layout

Training and testing expect a root directory whose immediate subdirectories are
named `0` through `N-1`, where `N` is the network's output size. Each holds the
PNG images of that class.

```
data/mnist_split/
  0/  *.png
  1/  *.png
  ...
  9/  *.png
```

Images are read at 28×28 pixels by default and flattened into a single input
vector. Use `--dataset-img-width` and `--dataset-img-height` to change that — but
`width × height` must equal the first entry of `--layers`, or the run stops with
exit code 5.

---

## Training mode

```bash
./build/MatrixGui_headless train --network models/digits.wgt --dataset /path/to/data
```

**Required**

- `--network <path.wgt>` — Where to load weights from, and where to save after each epoch. Created fresh if the file does not exist.
- A resolved training path and testing path:
  - `--dataset <dir>` — Use the same root for both training and testing, **or**
  - `--train-dataset <dir>` and/or `--test-dataset <dir>` — Each side falls back to `--dataset` when unset; you must end up with both paths covered.

**Dataset**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-img-width <n>` | `28` | Width images are read at |
| `--dataset-img-height <n>` | `28` | Height images are read at |
| `--test-file-limit <n>` | `0` (all) | Cap test images per class for the final accuracy report |
| `--working-directory <dir>` | `training-data` | Where per-run output is written (see below) |

<details>
<summary>Network topology — used only when creating a new <code>.wgt</code></summary>

Ignored if `--network` points at a file that loads successfully.

| Flag | Default | Description |
|------|---------|-------------|
| `--layers <n,n,...>` | `784,50,20,10` | Comma-separated layer sizes. The first must equal `width × height`, the last is the class count |
| `--hidden-activation <name>` | `relu` | `sigmoid`, `relu`, `tanh`, or `softmax` |
| `--output-activation <name>` | `softmax` | Same options |

</details>

<details>
<summary>Learning schedule</summary>

Defaults come from `Neural::Classifier::LearningConfig`. For guidance on what to
change and when, see [hyperparameters.md](hyperparameters.md#classifier).

| Flag | Default | Description |
|------|---------|-------------|
| `--initial-lr <x>` | `0.2` | Starting learning rate |
| `--min-lr <x>` | `0.0001` | Floor; decay stops here |
| `--lr-decay <x>` | `0.7` | Multiplicative decay applied when validation stops improving |
| `--max-epochs <n>` | `500` | Hard epoch limit |
| `--patience <n>` | `10` | Epochs without improvement before LR is decayed |
| `--inner-epochs <n>` | `1` | Backprop passes per sample per epoch |
| `--dropout <x>` | `0.0` | Dropout rate applied during training |
| `--dataset-limit-per-label <n>` | `100` | Cap training images per class; `0` = no cap |
| `--dataset-file-limit <n>` | — | Alias for `--dataset-limit-per-label` |

</details>

### What a training run leaves behind

Each run creates its own directory under `--working-directory`, named after the
network file plus the start time — for example
`training-data/digits.wgt_1757000000/`. Inside it:

| File | Contents |
|------|----------|
| `learning-config.json` | The hyperparameters this run actually used |
| `network-config.json` | The topology this run actually used |
| `log.jsonl` | One JSON line per epoch: loss, learning rate, timing |
| `testing-log.jsonl` | One JSON line per evaluation pass |
| `backup-<N>` | Network weights snapshotted after epoch `N` |

The two JSON files are written in exactly the format `--learning-config` and
`--network-config` read back, so any run can be replayed or resumed. The sweep
scripts in `experiments/` consume `log.jsonl` and `testing-log.jsonl`.

### Config files

| Flag | Description |
|------|-------------|
| `--learning-config <file.json>` | Load hyperparameters from a JSON file |
| `--network-config <file.json>` | Load topology from a JSON file; only used when creating a new network |

Precedence is **file first, then flags**: the file supplies a new set of
defaults, and any flag you also pass on the command line overrides just that
field. So this reuses a previous run's settings but pushes the epoch limit up:

```bash
./build/MatrixGui_headless train \
  --network models/digits.wgt \
  --dataset /path/to/data \
  --learning-config training-data/digits.wgt_1757000000/learning-config.json \
  --max-epochs 1000
```

Malformed JSON in either file aborts with exit code 4.

### Examples

Minimal run — one directory for both train and test:

```bash
./build/MatrixGui_headless train \
  --network models/digits.wgt \
  --dataset /path/to/data/mnist_split
```

Separate train and test roots:

```bash
./build/MatrixGui_headless train \
  --network models/digits.wgt \
  --train-dataset /path/to/train \
  --test-dataset /path/to/test
```

Quick experiment — 50 samples per class, 100 epochs, evaluated on 200 test images:

```bash
./build/MatrixGui_headless train \
  --network models/quick.wgt \
  --dataset /path/to/data \
  --dataset-limit-per-label 50 \
  --max-epochs 100 \
  --test-file-limit 200
```

<details>
<summary>Custom architecture</summary>

Only takes effect if `models/custom.wgt` does not already exist:

```bash
./build/MatrixGui_headless train \
  --network models/custom.wgt \
  --dataset /path/to/data \
  --layers 784,128,64,10 \
  --hidden-activation relu \
  --output-activation softmax
```

For non-MNIST geometry, keep the first layer and the image size in step:

```bash
./build/MatrixGui_headless train \
  --network models/big.wgt \
  --dataset /path/to/data \
  --dataset-img-width 32 --dataset-img-height 32 \
  --layers 1024,256,64,10
```

</details>

---

## Single-image prediction mode

**Required**

- `--network <path.wgt>` — Must exist and load successfully.
- `--image <path.png>` — One PNG file.

Optionally `--dataset-img-width` / `--dataset-img-height`, which must match what
the network was trained at.

**Output**

Prints the predicted class index followed by a newline to **stdout**. Errors go
to **stderr** with a non-zero exit code.

### Examples

```bash
./build/MatrixGui_headless predict \
  --network models/digits.wgt \
  --image samples/seven.png
```

Capture the result in a shell variable:

```bash
digit=$(./build/MatrixGui_headless predict --network models/digits.wgt --image photo.png)
echo "Predicted: $digit"
```

---

## Exit codes

| Code | Situation |
|------|-----------|
| `0` | Success |
| `2` | Training: neither `--dataset` nor both of `--train-dataset`/`--test-dataset` resolved |
| `3` | Dataset root does not contain subdirectories `0`..`N-1` |
| `4` | `--layers` has fewer than two entries; malformed config JSON; or an exception during prediction |
| `5` | Predict: the `.wgt` failed to load — **or** training: `width × height` does not equal the network's input size |
| `106` | Command line did not parse: no subcommand, a missing required option, an unknown flag, or a file that does not exist |

Code `5` covers two different problems; the stderr message distinguishes them.
`106` comes from the argument parser and is what a typo produces.

---

## See also

- [Hyperparameters guide](hyperparameters.md) — learning rate schedules, architecture choices, dropout
- [GUI guide](gui.md) — the Classifier and Recognizer modes
- `core/classifier/learning_config.h` — default values
- [`cli/classifier/README.md`](../cli/classifier/README.md) — how the subcommands are wired
- `cli/classifier/main.cpp` — argument parsing
