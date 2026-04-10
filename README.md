# MatrixGui

Neural network tooling for digit recognition. This repository includes a Qt-based GUI and a **headless** command-line program for training and inference without a display.

## Building the headless CLI

The headless target is enabled when `BUILD_CLI` is on (default). From the project root:

```bash
cmake -B build -DBUILD_GUI=OFF -DBUILD_READER=OFF
cmake --build build
```

The executable is named `MatrixGui_headless` (see `CMakeLists.txt` and the `PROJECT_NAME` value). With the default configuration that also builds the GUI, the same binary is produced alongside the main application:

```bash
cmake -B build
cmake --build build --target MatrixGui_headless
```

## Dataset layout

Training and testing expect a **directory** whose **immediate subdirectories** are named `0` through `9`. Each subdirectory holds PNG images for that digit class (same convention as the GUI).

Example:

```text
data/mnist_split/
  0/   *.png
  1/   *.png
  ...
  9/   *.png
```

Images are loaded at 28x28, converted to a single row of 784 values, and fed into the network.

## Overview of modes

| Mode | Purpose | Requires dataset |
|------|---------|------------------|
| Training | Fit or continue training; optional post-training evaluation | Yes |
| Single image | Load a saved network and classify one PNG | No |

Run `--help` for the full flag list as implemented in `core/main_headless.cpp`.

```bash
./build/MatrixGui_headless --help
```

---

## Training mode

**Required**

- `--network <path.wgt>` — Where to load weights from, and where to save after each epoch. If the file does not exist, a new network is created (see network options below).
- A resolved **training** path and **testing** path for data:
  - `--dataset <dir>` — Use the same tree for both train and test, or
  - `--train-dataset <dir>` and/or `--test-dataset <dir>` — Each side falls back to `--dataset` when that flag is set; otherwise both train and test paths must be covered (for example only `--train-dataset` and `--test-dataset`, or `--dataset` plus one override).

**Network (only when creating a new `.wgt`; ignored if the file loads successfully)**

- `--layers <n,n,...>` — Comma-separated layer sizes. Default: `784,50,20,10`.
- `--hidden-activation <name>` — `sigmoid`, `relu`, `tanh`, or `softmax`. Default: `relu`.
- `--output-activation <name>` — Same set. Default: `softmax`.

**Learning (defaults match `Neural::LearningConfig` in `core/learning_config.h`)**

- `--initial-lr`, `--min-lr`, `--lr-decay`
- `--max-epochs`, `--patience`, `--inner-epochs`
- `--dropout`
- `--dataset-limit-per-label <n>` — Cap training files per class; `0` means no cap.
- `--dataset-file-limit <n>` — Deprecated alias for `--dataset-limit-per-label`.

**After training**

- `--test-file-limit <n>` — Cap evaluation files per class on the **test** dataset (`0` = all). Runs once after training and prints accuracy.

### Training examples

Minimal run: one dataset for train and test, new or existing weights at `models/digits.wgt`:

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

Train on a default “full” tree but evaluate on a smaller held-out copy:

```bash
./build/MatrixGui_headless \
  --network models/digits.wgt \
  --dataset /path/to/full \
  --test-dataset /path/to/small_eval
```

Limit training samples per class and shorten the schedule (illustrative):

```bash
./build/MatrixGui_headless \
  --network models/quick.wgt \
  --dataset /path/to/data \
  --dataset-limit-per-label 50 \
  --max-epochs 100 \
  --test-file-limit 200
```

Define a new architecture only when the weight file is missing:

```bash
./build/MatrixGui_headless \
  --network models/custom.wgt \
  --dataset /path/to/data \
  --layers 784,128,64,10 \
  --hidden-activation relu \
  --output-activation softmax
```

---

## Single-image classification

**Required**

- `--network <path.wgt>` — Must exist and load successfully (no training, no dataset).
- `--predict-image <path.png>` — One image file.

**Output**

- Prints a single digit `0`–`9` and a newline to **stdout** (suitable for scripts).

**Example**

```bash
./build/MatrixGui_headless \
  --network models/digits.wgt \
  --predict-image samples/seven.png
```

Capture the digit in a shell variable:

```bash
digit=$(./build/MatrixGui_headless --network models/digits.wgt --predict-image photo.png)
echo "Predicted class: $digit"
```

Errors (missing file, failed load, invalid image or network output) are reported on **stderr** with a non-zero exit status.

---

## Exit codes (typical)

Exact codes are defined in `core/main_headless.cpp`. In practice:

| Code | Situation |
|------|-----------|
| 0 | Success |
| 1 | Missing or invalid `--network` / predict arguments |
| 2 | Missing dataset paths in training mode |
| 3 | Dataset path does not match the expected `0`..`9` layout |
| 4 | Invalid numeric or enum arguments, invalid `--layers`, or exception during predict |
| 5 | Predict mode: failed to load `.wgt` |
| 6 | Predict mode: image path not found |

---

## See also

- `core/main_headless.cpp` — Argument parsing and behavior.
- `core/learning_config.h` — Default training hyperparameters.
- `gui/learning_config_widget.cpp` — GUI defaults aligned with the same `LearningConfig` structure.
