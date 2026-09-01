# Hyperparameters

This page is advice, not reference: what each knob does to training, and which
way to turn it when something goes wrong. The full flag tables live in
[classifier.md](classifier.md) and [gan.md](gan.md).

Defaults are defined in `core/classifier/learning_config.h` and
`core/generator/learning_config.h`.

---

## Classifier

### Learning rate schedule

The classifier uses a simple decay-on-plateau schedule. The LR starts at `--initial-lr`, decays by `--lr-decay` whenever validation accuracy does not improve for `--patience` epochs, and stops decaying once it hits `--min-lr`.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--initial-lr` | `0.2` | Intentionally high — the large first step moves weights quickly out of the random initialisation zone |
| `--lr-decay` | `0.7` | Applied multiplicatively: `lr *= 0.7` each time patience is exhausted |
| `--min-lr` | `0.0001` | Once the LR reaches this floor training continues but the LR no longer changes |
| `--patience` | `10` | Increase this (e.g. `20`) if the loss is noisy and you want to give each LR level more time |

**When to tune:**
- The network oscillates or diverges early → lower `--initial-lr` (try `0.05`–`0.1`).
- Accuracy plateaus quickly without improving → lower `--lr-decay` (try `0.5`) so each step is more aggressive, or lower `--patience` to decay sooner.
- Training stops too early (LR collapses before the network converges) → raise `--patience` or lower `--lr-decay`.

### Architecture

The default `784,50,20,10` is deliberately small for fast iteration. The first
and last layer sizes are not free: the input must equal
`--dataset-img-width × --dataset-img-height` (784 for the default 28×28), and the
output must equal the number of class subdirectories in your dataset.

<details>
<summary>Reasonable starting points</summary>

| Goal | `--layers` suggestion |
|------|-----------------------|
| Fast baseline | `784,50,20,10` (default) |
| More capacity | `784,128,64,10` |
| Deeper | `784,256,128,64,10` |

</details>

`relu` hidden + `softmax` output is a solid default for multi-class classification. Use `sigmoid` output only if you have a specific reason; `softmax` produces a proper probability distribution over classes.

### Dropout

`--dropout 0` (default) is fine for small datasets. Add dropout (e.g. `0.2`–`0.4`) if the training accuracy is significantly higher than the test accuracy — a sign of overfitting.

### Dataset limit

`--dataset-limit-per-label 100` (default) caps training at 100 images per class. Set it to `0` to use all available images, or raise it for larger runs. Reducing it speeds up epochs at the cost of accuracy.

---

## GAN

GAN training is inherently less stable than supervised learning. The key challenge is keeping the generator and discriminator balanced — if one dominates the other, training stalls.

### Learning rates

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--gen-lr` | `0.0002` | |
| `--disc-lr` | `0.0001` | Lower than the generator LR because the discriminator typically converges faster |

The asymmetric defaults exist because if the discriminator gets too strong it provides no useful gradient to the generator. Keep `disc-lr` ≤ `gen-lr` as a starting heuristic. If the generator loss collapses, try reducing `disc-lr` further.

### Discriminator steps

`--disc-steps 3` runs three discriminator updates for each generator update. This also helps prevent the discriminator from lagging behind, but more than 5 steps typically causes the discriminator to dominate. If you observe the generator loss stuck near 0 (discriminator wins immediately), reduce `--disc-steps` to `1`.

### Dropout on the discriminator

`--dropout 0.3` regularises the discriminator and prevents it from memorising the real training set. If the discriminator is too strong (generator can't fool it), raise dropout (try `0.4`–`0.5`). If generated images are blurry or incoherent, the discriminator may be too weak — reduce dropout.

### Classifier weight

`--classifier-weight 1.0` balances the generator's adversarial loss with its class-conditioning loss. Increasing it (e.g. `2.0`) pushes the generator harder to produce images that the frozen classifier labels correctly, at the risk of mode collapse onto a narrow set of patterns. Decreasing it (e.g. `0.5`) relaxes class conditioning.

### Batch size

`--batch-size 64` is a safe default. Smaller batches (32) add gradient noise which can help escape local minima; larger batches (128) give more stable gradients but require more memory and may over-smooth early training.

---

## Adaptive learning rate

Enable with `--adaptive-lr`. The scheduler tracks `D(real)` (how well the discriminator recognises real images) and `D(G(z))` (how often it is fooled by the generator) using an exponential moving average and adjusts both LRs each epoch.

**When to enable:** runs of 500+ epochs where you want the balance to self-correct. Not needed for short experiments.

<details>
<summary>Every adaptive-LR knob</summary>

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--lr-ema-alpha` | `0.9` | Higher = slower EMA reaction. Lower (e.g. `0.7`) makes the scheduler react faster but noisier |
| `--lr-adjust-factor` | `1.05` | Multiplicative step (~5% per epoch). Reduce to `1.02` for a gentler schedule |
| `--lr-warmup-epochs` | `5` | Hold LRs fixed for this many epochs while the EMA stabilises |
| `--lr-min` / `--lr-max` | `1e-6` / `1e-2` | Hard clamps; keep `lr-max` well below `0.1` to avoid divergence |
| `--d-real-target-low` / `--d-real-target-high` | `0.30` / `0.80` | The healthy band for `D(real)`. Widen it to make the scheduler intervene less often |
| `--gen-fool-target-low` / `--gen-fool-target-high` | `0.30` / `0.60` | The healthy band for `D(G(z))`. Deliberately narrower than the `D(real)` band |

The bands decide what counts as "out of balance". Three situations are
recognised: both signals low (the discriminator has collapsed), `D(real)` high
with `D(G(z))` low (the discriminator is dominating), and `D(G(z))` high (the
generator is dominating). Each nudges the two learning rates in opposite
directions.

</details>

---

## Flatness detection

Enable with `--flatness-detection`. When both EMA signals change by less than `--flatness-threshold` for `--flatness-window` consecutive epochs, the training is considered stuck. A kick is applied:

1. Discriminator dropout is multiplied by `--flatness-dropout-boost` (weakens D).
2. Generator LR is multiplied by `--flatness-gen-lr-boost` (encourages G to explore).
3. After `--flatness-kick-duration` epochs both are restored.

Adaptive LR is suspended for as long as a kick is active, so the two mechanisms
do not pull against each other.

**When to enable:** long runs (1000+ epochs) on challenging datasets where training visually flatlines. For shorter runs the overhead is rarely worth it.

<details>
<summary>Every flatness knob</summary>

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--flatness-threshold` | `0.005` | Reduce to `0.002` to trigger kicks less aggressively |
| `--flatness-window` | `10` | Increase to `20` to require a longer plateau before kicking |
| `--flatness-kick-duration` | `5` | Raise to `10` if a single kick isn't enough to restart progress |
| `--flatness-dropout-boost` | `2.0` | Multiplies the current dropout rate — don't set this so high it pushes dropout above `0.8` |
| `--flatness-gen-lr-boost` | `3.0` | Multiplies the current generator LR — ensure `gen-lr * boost` stays below `lr-max` |

</details>

---

## Sweeping automatically

Rather than guessing one setting at a time, `experiments/` holds a small harness
that runs the classifier across a grid of hyperparameters and analyses the
results.

```bash
pip install -r experiments/requirements.txt
python3 experiments/run_experiments.py          # or run_experiments_parallel.py
python3 experiments/analyze.py
```

<details>
<summary>How the harness works</summary>

`run_experiments.py` holds a `PARAM_GRID` near the top of the file. For each
combination it writes a temporary learning-config JSON, invokes
`MatrixGui_headless train --learning-config …` with a unique network file so
every run starts from scratch, and collects that run's `log.jsonl` and
`testing-log.jsonl`. `run_experiments_parallel.py` does the same across several
processes at once.

`analyze.py` then reads those logs and produces `enriched_results.csv`,
a set of plots under `analysis_plots/`, and a written `analysis_report.md`.

**One thing to watch:** the runner's summary CSV records only *train* accuracy,
which is a poor criterion for choosing hyperparameters — bigger networks, more
data and less dropout nearly always raise train accuracy while hurting
generalisation. `analyze.py` exists partly to correct for this: it makes test
accuracy the primary metric and reports the train/test gap so overfitting is
visible. Select on its output, not on the raw CSV.

The scripts assume a dataset at `~/Documents/projects/mnist-pngs/{train,test}`;
edit `BASE_CMD` if yours lives elsewhere. Everything under `experiments/` except
the scripts themselves is gitignored.

</details>
