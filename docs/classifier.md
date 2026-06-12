# Classifier

The classifier trains a fully-connected neural network to recognise handwritten digits (0–9) and can run inference on individual images.

The CLI binary is `MatrixGui_headless`. The GUI provides the same functionality interactively through the **Classifier** and **Recognizer** tabs.

## Dataset layout

Both training and testing expect a root directory whose immediate subdirectories are named `0` through `9`, each containing PNG images of that digit class.

```
data/mnist_split/
  0/  *.png
  1/  *.png
  ...
  9/  *.png
```

Images are loaded at 28×28 pixels and flattened to a 784-element input vector.

---

## Training mode

**Required**

- `--network <path.wgt>` — Where to load weights from, and where to save after each epoch. Created fresh if the file does not exist.
- A resolved training path and testing path:
  - `--dataset <dir>` — Use the same root for both training and testing, **or**
  - `--train-dataset <dir>` and/or `--test-dataset <dir>` — Each side falls back to `--dataset` when set; you must end up with both paths covered.

**Network topology** (used only when creating a new `.wgt`; ignored if the file loads successfully)

| Flag | Default | Description |
|------|---------|-------------|
| `--layers <n,n,...>` | `784,50,20,10` | Comma-separated layer sizes |
| `--hidden-activation <name>` | `relu` | `sigmoid`, `relu`, `tanh`, or `softmax` |
| `--output-activation <name>` | `softmax` | Same options |

**Learning schedule** (defaults from `Neural::Classifier::LearningConfig`)

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
| `--dataset-file-limit <n>` | — | Deprecated alias for `--dataset-limit-per-label` |

**Post-training evaluation**

| Flag | Default | Description |
|------|---------|-------------|
| `--test-file-limit <n>` | `0` (all) | Cap test images per class for the final accuracy report |

### Examples

Minimal run — one directory for both train and test:

```bash
./build/MatrixGui_headless \
  --network models/digits.wgt \
  --dataset /path/to/data/mnist_split
```

Separate train and test roots:

```bash
./build/MatrixGui_headless \
  --network models/digits.wgt \
  --train-dataset /path/to/train \
  --test-dataset /path/to/test
```

Quick experiment — 50 samples per class, 100 epochs, evaluated on 200 test images:

```bash
./build/MatrixGui_headless \
  --network models/quick.wgt \
  --dataset /path/to/data \
  --dataset-limit-per-label 50 \
  --max-epochs 100 \
  --test-file-limit 200
```

Custom architecture (only takes effect if `models/custom.wgt` does not exist):

```bash
./build/MatrixGui_headless \
  --network models/custom.wgt \
  --dataset /path/to/data \
  --layers 784,128,64,10 \
  --hidden-activation relu \
  --output-activation softmax
```

---

## Single-image prediction mode

**Required**

- `--network <path.wgt>` — Must exist and load successfully.
- `--predict-image <path.png>` — One PNG file.

**Output**

Prints a single digit `0`–`9` followed by a newline to **stdout**. Errors go to **stderr** with a non-zero exit code.

### Examples

```bash
./build/MatrixGui_headless \
  --network models/digits.wgt \
  --predict-image samples/seven.png
```

Capture the result in a shell variable:

```bash
digit=$(./build/MatrixGui_headless --network models/digits.wgt --predict-image photo.png)
echo "Predicted: $digit"
```

---

## Exit codes

| Code | Situation |
|------|-----------|
| `0` | Success |
| `1` | Missing or invalid `--network` / `--predict-image` arguments |
| `2` | Missing dataset paths in training mode |
| `3` | Dataset path does not match the expected `0`..`9` layout |
| `4` | Invalid numeric or enum arguments, invalid `--layers`, or exception during prediction |
| `5` | Predict mode: failed to load `.wgt` |
| `6` | Predict mode: image path not found |

---

## See also

- [Hyperparameters guide](hyperparameters.md) — guidance on learning rate schedules, architecture choices, and dropout
- `core/classifier/learning_config.h` — default values
- `core/main_headless.cpp` — argument parsing
