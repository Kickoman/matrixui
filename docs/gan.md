# GAN (Generator)

The GAN mode trains a conditional generative adversarial network that learns to produce synthetic digit images. The generator is conditioned on a digit class label, so it can be directed to generate a specific digit.

The CLI binary is `MatrixGui_gan`. The GUI provides the same functionality through the **Generator** tab.

## How it works

Training requires a pre-trained and frozen classifier alongside real digit images. The discriminator learns to distinguish real images from generator output; the generator is rewarded both for fooling the discriminator and for producing images the classifier recognises as the requested digit class.

**Prerequisite:** train a classifier first (see [classifier.md](classifier.md)) and keep its `.wgt` file.

---

## Training mode

**Required**

- `--classifier <path.wgt>` — Frozen classifier used to compute the generator's class loss.
- `--dataset <dir>` — Real training images root (`0/`–`9/` subdirectories of PNGs, same layout as the classifier).

**Network paths** (created fresh if the file does not exist)

| Flag | Default | Description |
|------|---------|-------------|
| `--generator <path.wgt>` | `generator.wgt` | Generator save path |
| `--discriminator <path.wgt>` | `discriminator.wgt` | Discriminator save path |

**Network topology** (used only when creating new networks; ignored when loading existing)

| Flag | Default | Description |
|------|---------|-------------|
| `--latent-dim <n>` | `100` | Size of the random noise vector fed to the generator |
| `--gen-layers <n,n,...>` | `256,512` | Generator hidden layer sizes (input and output are inferred) |
| `--disc-layers <n,n,...>` | `512,256` | Discriminator hidden layer sizes (input is 784, output is 1) |

**GAN training**

| Flag | Default | Description |
|------|---------|-------------|
| `--gen-lr <x>` | `0.0002` | Generator learning rate |
| `--disc-lr <x>` | `0.0001` | Discriminator learning rate |
| `--epochs <n>` | `300` | Total training epochs |
| `--batch-size <n>` | `64` | Samples per batch |
| `--disc-steps <n>` | `3` | Discriminator updates per generator update |
| `--dropout <x>` | `0.3` | Discriminator dropout rate |
| `--classifier-weight <x>` | `1.0` | Weight of the classifier loss in the generator update |
| `--num-classes <n>` | `10` | Number of digit classes |
| `--dataset-limit <n>` | `0` (all) | Max real images per label to load |

**Adaptive learning rate** (disabled by default)

Enable with `--adaptive-lr`. Each epoch the discriminator scores `D(real)` and `D(G(z))` are tracked with an EMA; learning rates are adjusted when the discriminator or generator dominates.

| Flag | Default | Description |
|------|---------|-------------|
| `--adaptive-lr` | off | Enable adaptive LR adjustment |
| `--lr-ema-alpha <x>` | `0.9` | EMA smoothing factor; higher = slower reaction |
| `--lr-adjust-factor <x>` | `1.05` | Multiplicative LR step per epoch (~5%) |
| `--lr-min <x>` | `1e-6` | Lower LR clamp |
| `--lr-max <x>` | `1e-2` | Upper LR clamp |
| `--lr-warmup-epochs <n>` | `5` | Epochs to skip before adjustments begin |

**Flatness detection** (disabled by default)

Enable with `--flatness-detection`. When both EMA signals plateau for several consecutive epochs a "kick" is applied: the discriminator dropout is boosted and the generator LR is spiked temporarily to escape the plateau.

| Flag | Default | Description |
|------|---------|-------------|
| `--flatness-detection` | off | Enable plateau detection |
| `--flatness-threshold <x>` | `0.005` | Max EMA change per epoch to count as flat |
| `--flatness-window <n>` | `10` | Consecutive flat epochs before kick fires |
| `--flatness-kick-duration <n>` | `5` | Epochs to hold the kick |
| `--flatness-dropout-boost <x>` | `2.0` | Multiply discriminator dropout by this during kick |
| `--flatness-gen-lr-boost <x>` | `3.0` | Multiply generator LR by this during kick |

### Training examples

Basic run with defaults:

```bash
./build/MatrixGui_gan \
  --classifier models/digits.wgt \
  --dataset /path/to/data/mnist_split
```

Longer run with adaptive LR and flatness detection:

```bash
./build/MatrixGui_gan \
  --classifier models/digits.wgt \
  --dataset /path/to/data/mnist_split \
  --generator models/generator.wgt \
  --discriminator models/discriminator.wgt \
  --epochs 1000 \
  --adaptive-lr \
  --flatness-detection
```

Custom architecture:

```bash
./build/MatrixGui_gan \
  --classifier models/digits.wgt \
  --dataset /path/to/data/mnist_split \
  --latent-dim 64 \
  --gen-layers 128,256,512 \
  --disc-layers 512,256,128
```

---

## Generation mode

Load a trained generator and produce sample images. No dataset or discriminator is needed.

**Required**

- `--generate` — Activate generation mode.
- `--label <n>` — Digit class to generate (0–9).
- `--generator <path.wgt>` — Generator weights to load.

**Optional**

| Flag | Default | Description |
|------|---------|-------------|
| `--num-samples <n>` | `1` | Number of images to generate |
| `--output <path.png>` | `generated.png` | Output file path; a counter is inserted before the extension when generating multiple samples (e.g. `digit_0.png`, `digit_1.png`) |
| `--classifier <path.wgt>` | — | If provided, the classifier's predicted label is printed next to each saved file |
| `--num-classes <n>` | `10` | Must match the value used during training |

### Generation examples

Generate one image of the digit 7:

```bash
./build/MatrixGui_gan \
  --generate \
  --label 7 \
  --generator models/generator.wgt \
  --output samples/seven.png
```

Generate 10 images of the digit 3, with classifier verification:

```bash
./build/MatrixGui_gan \
  --generate \
  --label 3 \
  --generator models/generator.wgt \
  --classifier models/digits.wgt \
  --num-samples 10 \
  --output samples/three.png
```

---

## End-to-end workflow

```bash
# 1. Train the classifier
./build/MatrixGui_headless \
  --network models/digits.wgt \
  --dataset /path/to/mnist_split

# 2. Train the GAN
./build/MatrixGui_gan \
  --classifier models/digits.wgt \
  --dataset /path/to/mnist_split \
  --generator models/generator.wgt \
  --discriminator models/discriminator.wgt

# 3. Generate samples
./build/MatrixGui_gan \
  --generate \
  --label 5 \
  --generator models/generator.wgt \
  --classifier models/digits.wgt \
  --num-samples 5 \
  --output output/five.png
```

---

## See also

- [Hyperparameters guide](hyperparameters.md) — GAN stability, adaptive LR, flatness kicks
- `core/generator/gan_config.h` — default values with inline explanations
- `core/main_headless_gan.cpp` — argument parsing
