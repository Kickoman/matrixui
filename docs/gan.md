# GAN (Generator)

This mode trains a conditional generative adversarial network that produces
synthetic digit images. "Conditional" means the generator takes a class label
alongside its random noise, so you can ask it for a specific digit rather than
whatever it feels like drawing.

The CLI binary is `MatrixGui_gan`, with two subcommands: `train` and `generate`.
The GUI offers the same through its **Generator** mode.

## How it works

Three networks are involved. The **discriminator** learns to tell real images
from generated ones. The **generator** tries to fool it. A third network — a
**frozen, pre-trained classifier** — grades whether the generated image actually
looks like the digit that was requested, and that grade is folded into the
generator's loss. Without it the generator would happily produce convincing
digits of the wrong class.

**Prerequisite:** train a classifier first (see [classifier.md](classifier.md))
and keep its `.wgt` file.

---

## Training mode

```bash
./build/MatrixGui_gan train --classifier models/digits.wgt --dataset /path/to/data
```

**Required**

- `--classifier <path.wgt>` — Frozen classifier used to compute the generator's class loss. Its output size also sets the number of classes.
- `--dataset <dir>` — Real training images root (`0/`–`9/` subdirectories of PNGs, same layout as the classifier).

**Network paths** (created fresh if the file does not exist)

| Flag | Default | Description |
|------|---------|-------------|
| `--generator <path.wgt>` | `generator.wgt` | Generator save path |
| `--discriminator <path.wgt>` | `discriminator.wgt` | Discriminator save path |

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
| `--dataset-limit <n>` | `0` (all) | Max real images per label to load |

<details>
<summary>Network topology — used only when creating new networks</summary>

Ignored when the `.wgt` files load successfully.

| Flag | Default | Description |
|------|---------|-------------|
| `--latent-dim <n>` | `100` | Size of the random noise vector fed to the generator |
| `--gen-layers <n,n,...>` | `256,512` | Generator hidden layer sizes (input and output are inferred) |
| `--disc-layers <n,n,...>` | `512,256` | Discriminator hidden layer sizes (input is the image size, output is 1) |
| `--dataset-img-width <n>` | `28` | Width real images are read at, and the width the generator emits |
| `--dataset-img-height <n>` | `28` | Height real images are read at, and the height the generator emits |

The generator's input layer is `--latent-dim` plus the class count (the label
arrives one-hot encoded), and its output layer is `width × height`.

</details>

<details>
<summary>Adaptive learning rate — off by default</summary>

Enable with `--adaptive-lr`. Each epoch, the discriminator's scores on real
images `D(real)` and on generated ones `D(G(z))` are smoothed with an EMA. When
either network starts running away with the game, the learning rates are nudged
in opposite directions to bring it back.

The four target bands decide what "running away" means. They select the three
branches in `core/generator/trainer.cpp:194-206`:

| Flag | Default | Description |
|------|---------|-------------|
| `--adaptive-lr` | off | Enable adaptive LR adjustment |
| `--lr-ema-alpha <x>` | `0.9` | EMA smoothing factor; higher = slower reaction |
| `--d-real-target-low <x>` | `0.30` | `D(real)` below this (with `D(G(z))` also low) ⇒ discriminator collapsed |
| `--d-real-target-high <x>` | `0.80` | `D(real)` above this (with `D(G(z))` low) ⇒ discriminator dominating |
| `--gen-fool-target-low <x>` | `0.30` | `D(G(z))` below this ⇒ the generator is not fooling anyone |
| `--gen-fool-target-high <x>` | `0.60` | `D(G(z))` above this ⇒ generator dominating |
| `--lr-adjust-factor <x>` | `1.05` | Multiplicative LR step per epoch (~5%) |
| `--lr-min <x>` | `1e-6` | Lower LR clamp |
| `--lr-max <x>` | `1e-2` | Upper LR clamp |
| `--lr-warmup-epochs <n>` | `5` | Epochs to skip before adjustments begin |

The bands are deliberately asymmetric: the healthy range for `D(real)` is wider
than for `D(G(z))`. Widening them further makes the scheduler intervene less
often.

</details>

<details>
<summary>Flatness detection — off by default</summary>

Enable with `--flatness-detection`. When both EMA signals barely move for
`--flatness-window` consecutive epochs, training has stalled, and a "kick" is
applied: the discriminator's dropout is boosted and the generator's learning rate
is spiked for a few epochs, then everything is restored.

Adaptive LR is **suspended while a kick is active**, so the two mechanisms do not
fight each other.

| Flag | Default | Description |
|------|---------|-------------|
| `--flatness-detection` | off | Enable plateau detection |
| `--flatness-threshold <x>` | `0.005` | Max EMA change per epoch to count as flat |
| `--flatness-window <n>` | `10` | Consecutive flat epochs before kick fires |
| `--flatness-kick-duration <n>` | `5` | Epochs to hold the kick |
| `--flatness-dropout-boost <x>` | `2.0` | Multiply discriminator dropout by this during kick |
| `--flatness-gen-lr-boost <x>` | `3.0` | Multiply generator LR by this during kick |

</details>

### Training examples

Basic run with defaults:

```bash
./build/MatrixGui_gan train \
  --classifier models/digits.wgt \
  --dataset /path/to/data/mnist_split
```

Longer run with adaptive LR and flatness detection:

```bash
./build/MatrixGui_gan train \
  --classifier models/digits.wgt \
  --dataset /path/to/data/mnist_split \
  --generator models/generator.wgt \
  --discriminator models/discriminator.wgt \
  --epochs 1000 \
  --adaptive-lr \
  --flatness-detection
```

<details>
<summary>Custom architecture</summary>

```bash
./build/MatrixGui_gan train \
  --classifier models/digits.wgt \
  --dataset /path/to/data/mnist_split \
  --latent-dim 64 \
  --gen-layers 128,256,512 \
  --disc-layers 512,256,128
```

</details>

---

## Generation mode

Load a trained generator and produce sample images. No dataset or discriminator
is needed.

**Required**

- `--label <n>` — Class to generate.
- `--generator <path.wgt>` — Generator weights to load. Defaults to `generator.wgt` in the current directory.

**Optional**

| Flag | Default | Description |
|------|---------|-------------|
| `--num-samples <n>` | `1` | Number of images to generate |
| `--output <path.png>` | `generated.png` | Output file path; a counter is inserted before the extension when generating multiple samples (e.g. `digit_0.png`, `digit_1.png`) |
| `--classifier <path.wgt>` | — | If provided, the classifier's predicted label is printed next to each saved file |
| `--num-classes <n>` | `10` | Class count the generator was trained with. **Ignored when `--classifier` is given** — the classifier's output size wins |
| `--dataset-img-width <n>` | `28` | Width of the emitted image; must match training |
| `--dataset-img-height <n>` | `28` | Height of the emitted image; must match training |

### Generation examples

Generate one image of the digit 7:

```bash
./build/MatrixGui_gan generate \
  --label 7 \
  --generator models/generator.wgt \
  --output samples/seven.png
```

Generate 10 images of the digit 3, with classifier verification:

```bash
./build/MatrixGui_gan generate \
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
./build/MatrixGui_headless train \
  --network models/digits.wgt \
  --dataset /path/to/mnist_split

# 2. Train the GAN
./build/MatrixGui_gan train \
  --classifier models/digits.wgt \
  --dataset /path/to/mnist_split \
  --generator models/generator.wgt \
  --discriminator models/discriminator.wgt

# 3. Generate samples
./build/MatrixGui_gan generate \
  --label 5 \
  --generator models/generator.wgt \
  --classifier models/digits.wgt \
  --num-samples 5 \
  --output output/five.png
```

---

## Exit codes

| Code | Situation |
|------|-----------|
| `0` | Success |
| `2` | The generator or the classifier failed to load |
| `4` | An exception during generation |
| `106` | Command line did not parse: no subcommand, a missing required option, or an unknown flag |

Note that `--generator` has a default value, and the "file must exist" check does
not apply to defaults — so running `generate` in a directory without a
`generator.wgt` gives you `2`, not `106`.

---

## See also

- [Hyperparameters guide](hyperparameters.md) — GAN stability, adaptive LR, flatness kicks
- [GUI guide](gui.md) — the Generator mode
- `core/generator/learning_config.h` — default values
- `core/main_headless_gan.cpp` — argument parsing
