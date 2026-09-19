# `core/serving` — loading a model artifact

Turns a directory on disk into a validated, immutable model in memory, or
refuses saying which number did not match. This is the first piece of an
inference service; nothing here serves anything yet.

| File | Contains |
|---|---|
| `error.h` | `Error` and its three children — `ManifestError`, `IntegrityError`, `ContractError` |
| `manifest.h/.cpp` | `ModelManifest` and the parts it is made of, plus `ParseManifest` |
| `loaded_model.h/.cpp` | `class LoadedModel`, `enum class IntegrityCheck` |
| `model_loader.h/.cpp` | `LoadModel` — the only entry point |

This library links `matrixgui_nn` for `NeuralNetwork` and `LoadNetwork`, and
`matrixgui_core_lib` privately for `Io::ReadFile` and `Hash::Sha256OfStream`.
It deliberately does **not** link Qt, `matrixgui_classifier` or
`matrixgui_png`: the inference path knows nothing about training or about image
formats.

## What is on disk

```
any-directory-name/
    manifest.json
    weights.wgt
```

**The directory name is decorative.** `name` and `version` live in the manifest
and are not checked against it: `models/mnist-v3/` holding a manifest that says
`"name": "digits"` loads without complaint. The manifest is the only truth. A
consequence that lands on the future registry rather than here: two directories
can declare the same `name` + `version`, and catching that collision is the
registry's job when it walks the tree.

`weights.wgt` is the format `Neural::SaveNetwork` writes — see
[`core/nn/`](../nn/). Nothing about it changed for serving; the manifest is the
envelope that carries what the blob cannot say about itself.

## `manifest.json`

```json
{
  "manifestVersion": 1,
  "name": "mnist",
  "version": "v3",
  "weights": {
    "path": "weights.wgt",
    "sha256": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
  },
  "input":  { "size": 784, "dtype": "f64", "layout": "hwc", "shape": [28, 28, 1],
              "range": { "min": 0.0, "max": 1.0 } },
  "output": { "size": 10, "kind": "classification",
              "labels": ["0","1","2","3","4","5","6","7","8","9"] },
  "annotations": { "trainedOn": "mnist-8x8" }
}
```

| Field | Required | Meaning |
|---|---|---|
| `manifestVersion` | yes | Format version of this file. Integer, matched exactly against the versions the loader knows (currently `1`) |
| `name` | yes | `[a-z0-9][a-z0-9_-]{0,63}` |
| `version` | yes | Same shape as `name` |
| `weights.path` | yes | Relative to this file's directory. Must stay inside it |
| `weights.sha256` | no | 64 lowercase hex digits. Checked when present |
| `input.size` | yes | Values the network takes, as one row |
| `input.dtype` | no | `f64` only, which is what `Matrix` holds. Defaults to `f64` |
| `input.layout` | no | `flat` or `hwc`. Defaults to `flat` |
| `input.shape` | no | How a client's tensor flattens into `input.size`. The product must equal `input.size` |
| `input.range` | no | `{min, max}` the model was trained for |
| `output.size` | yes | Values the network produces |
| `output.kind` | no | `raw`, `classification`, `regression`, `embedding`. Defaults to `raw` |
| `output.labels` | no | One per output. The count must equal `output.size` |
| `annotations` | no | Opaque object, not validated, carried through as-is |

**Unknown keys are refused**, at every level except inside `annotations`. A
manifest saying `"lables"` fails rather than loading with anonymous classes —
finding that kind of typo is what this loader is for. The cost is that adding a
field is a breaking change, and `manifestVersion` is the channel for it;
`annotations` is the escape hatch for anything that does not need the loader's
understanding.

### Why `input` describes rather than prescribes

A network takes one row of `input.size` doubles and nothing else, so `size` is
the only field with a mechanical meaning. `layout`, `shape` and `range`
describe how a caller's data is meant to become that row — enough to hand to
clients later and to validate requests against — but **the loader does not
enforce any of it**. It never decodes a PNG, never rescales, never clamps.
A model taking embeddings writes `layout: "flat"` and omits `shape`; an image
classifier writes `hwc` and `[28, 28, 1]`. Both load the same way.

`output.kind` is not cross-checked against the network's output activation. A
multi-label classifier legitimately ends in Sigmoid, so the claim would be
wrong as often as it was right.

### Versions have no order

`version` is a string, and **the format does not define an ordering over
versions** — there is no `latest`. Real model versions are not monotone
(`v3`, `2026-09-19`, `rc1`), so an integer would either lie or force artifacts
to be renamed. Choosing a default version is the registry's job and must be
explicit; deriving one from the manifests would mean a model silently changing
under a running client.

## What `LoadModel` checks, in order

1. The directory exists and holds a `manifest.json`.
2. The JSON parses and satisfies everything in the table above — including that
   `weights.path` is relative and, after `weakly_canonical` (so a symlink
   pointing out is caught too), still inside the model directory.
3. The weights file exists.
4. `weights.sha256`, when declared, matches the blob. Checked **before** the
   network is parsed, so a mismatched file is reported as a mismatch instead of
   as whatever the parser trips over first. This reads the file a second time
   rather than holding it next to the matrices it is about to become.
5. `Neural::LoadNetwork` parses it, which is where a truncated or oversized
   header is refused.
6. `input.size` against `network.inputSize()`, `output.size` against
   `network.outputSize()`. Label count is already tied to `output.size` by
   step 2, so it reaches the network transitively.

Failures throw, never return a sentinel:

| Exception | Raised for |
|---|---|
| `ManifestError` | The manifest: missing file, bad JSON, unknown key, a path leaving the directory |
| `IntegrityError` | The blob: truncated, unparseable, or not the one the digest names |
| `ContractError` | The two disagreeing about what the model takes or returns |

The three are separate because the HTTP layer will map them to different
outcomes, and a CLI to different exit codes.

## `LoadedModel`

```cpp
const ModelManifest& manifest() const;
const std::shared_ptr<const Neural::NeuralNetwork>& network() const;
const std::filesystem::path& directory() const;
IntegrityCheck integrity() const;   // NotDeclared | Verified
```

Immutable through having no mutators, not through `const` members — those would
kill move, and the registry will want these in a container.

The network is a `shared_ptr<const>` rather than a value for two reasons: several
executors can run it at once without copying the weights, and a request that
started on one version keeps it alive while the registry swaps in another.
Running it is `Neural::Predict(*model.network(), input)`
([`core/nn/neural_network_applier.h`](../nn/neural_network_applier.h)) —
`NeuralNetworkApplier` holds a `NeuralNetwork` by value and would copy every
weight per executor.

That sharing is safe because the only mutable state on the forward path,
`DropoutLayer::mask`, is written only when `ForwardContext::training` is set,
which `Predict` never sets. There is a comment on the member saying so; it is
the one thing holding this up.

`LoadModel` does not cache. Loading the same directory twice gives two
independent models with two networks — deciding what to keep is the registry's
job, not this one's.

## Not here

HTTP, request serialisation, the model registry, lookup by name and version,
snapshots, hot reload, worker pools — and inference itself beyond the single
`Predict` call above. Also no PNG decoding and no preprocessing of any kind.
