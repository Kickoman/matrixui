# MatrixGui

Neural network tooling for handwritten digit recognition and generation. The project ships a Qt-based GUI and headless CLI tools for training and inference without a display.

## Modes

| Mode | GUI tab | CLI binary | Description |
|------|---------|------------|-------------|
| Classifier | Classifier | `MatrixGui_headless` | Train a network to classify digits 0–9 |
| Recognizer | Recognizer | `MatrixGui_headless --predict-image` | Draw or load a digit and classify it |
| Generator (GAN) | Generator | `MatrixGui_gan` | Train a conditional GAN and generate synthetic digit images |

## Quick start

**Build everything:**

```bash
cmake -B build
cmake --build build
```

**Train a classifier:**

```bash
./build/MatrixGui_headless \
  --network models/digits.wgt \
  --dataset /path/to/mnist_split
```

**Classify a single image:**

```bash
./build/MatrixGui_headless \
  --network models/digits.wgt \
  --predict-image samples/seven.png
```

**Train a GAN** (requires a trained classifier):

```bash
./build/MatrixGui_gan \
  --classifier models/digits.wgt \
  --dataset /path/to/mnist_split
```

**Generate digit images:**

```bash
./build/MatrixGui_gan \
  --generate --label 7 \
  --generator models/generator.wgt \
  --output samples/seven.png
```

## Dataset layout

Training and testing expect a root directory whose immediate subdirectories are named `0` through `9`, each containing PNG images of that digit class:

```
data/mnist_split/
  0/  *.png
  1/  *.png
  ...
  9/  *.png
```

Images are loaded at 28×28 pixels and flattened to a 784-element input vector.

## Documentation

- [Building](docs/building.md) — CMake options, build targets, dependencies
- [Classifier](docs/classifier.md) — Training and inference CLI reference
- [GAN](docs/gan.md) — GAN training and image generation CLI reference
- [Hyperparameters](docs/hyperparameters.md) — Learning rate schedules, architecture guidance, GAN stability tricks
