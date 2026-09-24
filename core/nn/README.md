# `core/nn` — the network itself

Layers, the network they compose into, the thing that runs it, and the `.wgt`
format it persists to. Everything here is written against
[`Matrix`](../matrix/README.md) and knows nothing about images, datasets on disk
beyond a reader callback, or serving.

| File | Contains |
|---|---|
| `layers.h/.cpp` | `DenseLayer`, `ActivationLayer`, `DropoutLayer`, `SoftmaxLayer`, the `LayerData` variant, `ActivationType`, `ForwardContext` |
| `neural_network.h` | `NeuralNetworkConfiguration`, `NeuralNetwork`, the layer-list text helpers |
| `neural_network_applier.h/.cpp` | `Neural::Predict` and `class NeuralNetworkApplier` |
| `neural_network_loader.h/.cpp` | `CreateNetwork`, `SaveNetwork`, `LoadNetwork`, `LoadConfig` |
| `dataset.h/.cpp`, `directory_dataset.h/.cpp` | `Sample`, the `Dataset` interface, one implementation over a directory tree |
| `loss_functions.h` | `mse_gradient`, `bce_gradient` |

## Two entry points, and only one of them is for inference

```cpp
Matrix Neural::Predict(const NeuralNetwork&, const Matrix&);   // inference
Matrix NeuralNetworkApplier::forward(const Matrix&, double);   // training
```

`Predict` is a free function taking the network by `const&`. It builds
`ForwardContext{false, 0.0}` — **hard-coded, no overload takes a context** — and
keeps every intermediate in a function-local `Matrix`.

`NeuralNetworkApplier` is the training side. It holds a `NeuralNetwork` **by
value**, so `initializeNetwork` deep-copies every weight, and its `forward`
hard-codes `ctx{true, dropoutRate}` and caches activations in a member for
`backward` to read. Nothing on an inference path should touch it; its `predict`
is a thin forward to `Neural::Predict` and exists only for callers that already
own an applier.

## Why `Predict` is safe on a network shared between threads

Several threads may call `Predict` on one `shared_ptr<const NeuralNetwork>`
without a lock. That rests on exactly three things, and losing any one of them is
silent:

1. The parameter is `const NeuralNetwork&`.
2. `ForwardContext::training` is `false`, hard-coded at the top of `Predict`.
3. Every intermediate is function-local.

The state that point 2 is protecting is **`DropoutLayer::mask`, which is a
`mutable` member** — so `const` does not stop it being written — plus a
function-local `static std::mt19937` inside `DropoutLayer::forward` that is
shared by every dropout layer of every network in the process. Both are touched
only on the `ctx.training` branch. Constructing a `ForwardContext` with
`training` set and running it on a shared network compiles with **zero
diagnostics** and is a data race; the most likely way in is reaching for
`NeuralNetworkApplier::forward` because it is the function that takes a dropout
rate.

Every other layer on the forward path is genuinely row-independent:
`ApplySoftMax` recomputes its max and sum inside the per-row loop,
`ActivationLayer` is elementwise, and `DropoutLayer` returns its input unchanged
at inference.

## Preconditions are checked here, not by Eigen

`Predict` refuses an input whose column count is not `network.inputSize()`, and
`Matrix::multiplyAdd` refuses a multiplier the left side cannot meet. Neither is
belt-and-braces: Eigen's own dimension guard is an `eigen_assert` that `NDEBUG`
removes, so in a Release build a wrong-width row is undefined behaviour rather
than an abort — one value short returns a plausible, silently wrong answer, one
value long reads past the weights. [`core/matrix/README.md`](../matrix/README.md)
has the measurements. The two checks are at different levels on purpose:
`multiplyAdd` also covers `backward` and the training path, which do not go
through `Predict`.

## Activation derivatives take the *output*

`applyActivationDerivative(x, type)` expects `x` to be the **post-activation**
value, not the pre-activation one. For Sigmoid and Tanh that is the only form
that is cheap; for ReLU and LeakyReLU input and output have the same sign, so the
derivative is recoverable from either and `ActivationLayer::backward` passes the
input for those two and the output for the rest. Reading the call site without
this in mind suggests a bug where there is none.

## What `LoadNetwork` builds

`CreateNetwork` and `LoadNetwork` both assemble the stack the same way: per
transition a `DenseLayer`, then for a hidden layer an `ActivationLayer` followed
by a `DropoutLayer`, and for the last layer either a `SoftmaxLayer` or an
`ActivationLayer`. So **every loaded network carries dropout layers**, one per
hidden layer, inert at inference but each holding that `mutable mask`.

`LoadNetwork` also allocates `gradientWeights` and `gradientBiases` for every
dense layer even though inference never reads them, which is why a model is about
**twice its declared weight bytes** resident.

## The `.wgt` format

```
version   u32   exactly 1
hidden    u8    ActivationType
output    u8    ActivationType
numLayers u32
sizes     u64 x numLayers
weights   f64   row-major per dense layer, then its biases
```

Little-endian throughout, via [`core/lib/write.h`](../lib/write.h).

**There is no magic number** — unlike `.cor` and `.sub`
([`core/words/data/`](../words/data/README.md)), a `.wgt` file identifies itself
only by its version field, so the first four bytes of any file are a plausible
version and a wrong file is diagnosed by whatever the header implies rather than
by refusing up front. The two `u8` activations at offsets 4 and 5 also leave
every following `u64` at offset ≡ 2 (mod 8).

Ceilings, all refused before anything is allocated:

| Constant | Value | Bounds |
|---|---|---|
| `kMaxLayers` | 4096 | declared layer count |
| `kMaxLayerSize` | 16777216 | one layer's width |
| `kMaxWeightBytes` | 4 GiB | the whole network |

A declared size the file cannot back is refused by comparing against the bytes
actually remaining, which is why `LoadNetwork` needs a seekable stream.

`LoadConfig` reads only the header and returns the configuration, for callers
that want the topology without paying for the weights. It is the non-throwing
sibling: every refusal is `std::nullopt`, so a caller probing an unknown file
does not have to catch anything.
