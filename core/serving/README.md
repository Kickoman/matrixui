# `core/serving` — loading and holding model artifacts

Turns a directory on disk into a validated, immutable model in memory, or
refuses saying which number did not match. This is the first piece of an
inference service; nothing here serves anything yet.

| File | Contains |
|---|---|
| `error.h` | `Error` and its three children — `ManifestError`, `IntegrityError`, `ContractError` |
| `manifest.h/.cpp` | `ModelManifest` and the parts it is made of, plus `ParseManifest` |
| `loaded_model.h/.cpp` | `class LoadedModel`, `enum class IntegrityCheck` |
| `model_loader.h/.cpp` | `LoadModel` — one directory into one model |
| `snapshot.h/.cpp` | `RegistrySnapshot`, `ModelKey`, `ModelFailure`, `Find`, `FindDefault` |
| `build.h/.cpp` | `Build` — a directory of directories into one snapshot |
| `registry.h/.cpp` | `RegistryConfig`, `class ModelRegistry` — holds the snapshot and swaps it |

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

`manifest.json` is refused above `kMaxManifestBytes = 1MiB`, checked with
`file_size` before the parse. The largest legitimate manifest is dominated by
`output.labels`, and a thousand labels of thirty characters is about 30KB, so
that is thirty times the headroom. It is bounded at all because a registry turns
one unbounded parse into one per directory, and nlohmann materialises roughly
sixteen bytes a node on top of the text — 1MiB of JSON can be 10–20MB resident,
which is the reason not to raise it casually.
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
std::shared_ptr<const Neural::NeuralNetwork> network() const;
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

`network()` returns the pointer **by value**, not by const reference: every call
is a reference-count increment, so a request calls it once and keeps the result
for the whole forward pass rather than reaching through `model` per layer.

That sharing is safe because the only mutable state on the forward path,
`DropoutLayer::mask`, is written only when `ForwardContext::training` is set,
which `Predict` never sets. That invariant is not this module's to keep:
[`core/nn/README.md`](../nn/README.md) states it, names the `static std::mt19937`
next to it, and spells out the edit that loses it without any compiler
diagnostic. This module only depends on it.

`LoadModel` does not cache. Loading the same directory twice gives two
independent models with two networks — deciding what to keep is the registry's
job, not this one's.

## The registry

`ModelRegistry` holds one published `RegistrySnapshot` and replaces it whole.
Building a composition and installing it are separate operations:

```cpp
RegistrySnapshot Build(const RegistryConfig& config);   // never publishes
void ModelRegistry::publish(RegistrySnapshot&& next);   // stamps, installs
std::shared_ptr<const RegistrySnapshot> ModelRegistry::rebuild();   // both, under one lock
```

`publish` takes the composition by rvalue and turns it into a
`shared_ptr<const>` itself. That is deliberate: if it took a
`shared_ptr<RegistrySnapshot>`, the caller would keep a mutable handle to a
snapshot that is already serving readers, and could empty it from the outside
with an ordinary `clear()`.

### What the swap guarantees

The slot is a `std::atomic<std::shared_ptr<const RegistrySnapshot>>`.
`snapshot()` is one `load()` returning a pointer that *owns* the composition, so
a later `publish()` cannot destroy what a reader is holding. A request that
started on one composition keeps it — models, networks, strings, diagnostics —
until it drops the pointer, and whichever thread drops the last reference pays
for freeing the old tree.

Three things about it are easy to get wrong:

- **`snapshot()` is not free and readers do not run in parallel.** libstdc++
  implements `atomic<shared_ptr>` as `_Sp_atomic`, a spinlock in the low bit of
  the control-block pointer, and `_Atomic_count::lock()` is *exclusive* — readers
  take it too. Two concurrent `snapshot()` calls serialise. The spin is
  `__builtin_ia32_pause()` with no `sched_yield`. Measured here, nanoseconds per
  acquisition: 21.5 at one reader, 53.2 at two, 132 at four, 292 at eight, while
  a `shared_mutex` flattens at ~108. Those are tight-loop numbers — threads doing
  nothing but acquiring — so they are an upper bound on contention, not what a
  worker that touches the slot once and then spends microseconds in `Predict`
  will see; a rerun on different hardware moves them and should not be pasted
  over these. It is still the right primitive, for that once-per-request reason
  — but the margin is far thinner than the first draft of this paragraph
  claimed, and "readers never block" must not be written anywhere.

  That draft said one `Predict` costs 33µs–3.7ms. Those are `-O0` numbers. `build/`
  is configured `Debug` — it is the tree CLAUDE.md tells you to create for the
  tests, and it is the tree the benchmark was run in. Rebuilt with the flags the
  project actually ships, `-O3 -march=native`, the same three topologies cost
  0.8µs (64-16-10), 34µs (784-128-10) and 87µs (784-256-128-10): forty to fifty
  times less. An acquisition under contention is therefore comparable to a whole
  small prediction, not to a thousandth of one. Measure the forward pass against
  the tree the service ships, never against `build/`.
- **Atomicity is per acquisition.** One `snapshot()` per request and every lookup
  through that pointer; two loads can land on different compositions. This is why
  there is deliberately no `registry.find(name, version)` shortcut hiding the
  load inside itself.
- **Default `seq_cst`, and not because ordering is moot.** `_Sp_atomic` clamps
  anything weaker — `load` up to acquire, `store` up to release — so a `relaxed`
  annotation in the source would be a lie rather than an optimisation.

`Build` on one thread and `publish` on another is safe only when the hand-off is
itself synchronised (a return value, a join, a queue under a mutex). Stashing a
raw pointer is not.

`slot` carries `alignas(64)`, and so does the member after it. `alignas` sets a
member's alignment, not the padding after it, so aligning only `slot` leaves the
next member on the same line: measured here, `configuration` lands at offset 16
of the line readers are CAS-ing. With `alignas` on both it starts at 64.

### The snapshot

```cpp
struct RegistrySnapshot {
    std::map<ModelKey, std::shared_ptr<const LoadedModel>, ModelKeyLess> models;
    std::map<std::string, std::string, std::less<>> defaults;
    std::vector<ModelFailure> failures;
    std::uint64_t generation = 0;
};
```

`std::map` rather than `unordered_map`: a composition is tens of models, so the
`log N` is irrelevant, and a deterministic order is what makes diagnostics,
collision messages and the CLI snapshot reproducible. The same scar is recorded
in [`core/words/data/README.md`](../words/data/README.md).

`ModelKeyLess` is transparent, so resolving a request does not have to build a
`ModelKey` out of two `std::string`s — two allocations on the path a request
takes. There is no `versionsByName` index: the map is keyed by `(name, version)`,
so every version of one name is already contiguous and `VersionsOf` walks that
range. A second index would only be a second thing to keep in step.

`generation` is stamped by `publish`, not by whoever built the composition: the
number is about publication order, which only the registry knows.

### Lookup

A miss is data, not an exception. Not finding a model is an ordinary outcome of a
request; `Serving::Error` stays for a build that went wrong.

| `LookupStatus` | Means | The HTTP layer will want |
|---|---|---|
| `Found` | | 200 |
| `UnknownModel` | no such name at all | 404 |
| `UnknownVersion` | the name is served, that version is not | 404, different text |
| `NoDefaultVersion` | the name is served, none asked for, none is default | 400 |

The two "which one?" misses carry `availableVersions`, so the HTTP layer never
re-asks the snapshot just to write the error text.

`FindDefault` resolves the default *first* and only builds a version list on a
miss. The route without a version is the common one, and the list is needed only
for the message.

### One bad directory does not sink the rest

`Build` loads what loads and records the rest in `failures`. It throws only when
the root itself cannot be walked.

The unit of failure is a directory; the unit of value is a model; they are
independent, so coupling them would be a choice rather than a consequence. The
usual objection to degrading — that the failure is displaced in time and space,
a log line at 03:14 becoming a customer's 404 at 11:40 — holds only when the
reason burns in a log. Here it does not: `failures` is a field of the snapshot,
`MatrixGui_models list` prints it, and a future `/readyz` reads the same field.

Strictness stays available to the caller, one line before publishing:

```cpp
auto next = Build(config);
if (!next.failures.empty()) { throw ...; }
registry.publish(std::move(next));
```

The reverse is not available: a library that throws cannot hand back a partial
composition. And because publication is the last statement, a failed rebuild is
non-destructive — the previous composition keeps serving.

`FailureKind` is an enum rather than a string so the HTTP layer never matches on
message text; `reason` is the exception's own `what()`, because the loader
already put the numbers in it.

### Walking the root

Depth is exactly one: `<root>/<directory>/manifest.json`.

| Found in the root | Result |
|---|---|
| a directory with a `manifest.json` | a candidate |
| a directory without one | recorded as `Skipped` |
| a plain file | ignored silently |
| a symlink to a directory | recorded as `Skipped`, not followed |
| a symlink to anything else | ignored silently, like a plain file |
| an entry whose status cannot be read | recorded as `Skipped` |

Entries are sorted before they are loaded. `directory_iterator` hands them back
in filesystem order — creation order here — and letting that decide anything
would make the result depend on how the tree was built.

Symlinks are not followed. Step 1 already refuses a symlink that leaves a model
directory, and following them in the root invites loops and registering one model
twice under two directory names. `models/current -> models/mnist-v3` is a common
deployment habit, so it is recorded rather than ignored: the operator has to be
able to see why it did nothing.

Every status query uses the `error_code` overload and the walk increments with
`increment(ec)`. `directory_entry::symlink_status()` is the throwing form and
does not consult the cached `d_type`, so on a root that is readable but not
searchable — or when an entry vanishes between `readdir` and `lstat`, which is an
ordinary `rsync` race — the throwing form would emit a `filesystem_error`, and
that is not a `Serving::Error`.

### Collisions

Two directories declaring the same `name` + `version` are **both** rejected.
Picking a winner would make the answer depend on the traversal order, and that
order is the filesystem's. The message names both directories; a third claimant
names only itself, because the first two already named each other.

### Defaults

Which version answers a request that does not name one comes from
`RegistryConfig::defaults`, supplied by the caller — a repeated `--default
mnist=v3` on the CLI today, a service configuration later. It is never derived
from disk: the manifest format defines no ordering over versions, and a
`defaults.json` in the root would be a second file format with its own version,
parser and messages for a map the caller already has.

The resolved map is copied *into* the snapshot, so composition and defaults swap
as one unit; otherwise a rebuild could publish new models with stale defaults.

A default naming a version that is not loaded is recorded in `failures` and the
model stays: the default is broken, the model is not, so explicit versions keep
working and a request without one gets `NoDefaultVersion`.

### Rebuilding

`rebuild()` is called by the owner — a CLI invocation, a SIGHUP handler, a future
admin endpoint. There is no watcher, no timer and no background thread: `inotify`
fires in the middle of an `rsync` and would publish a composition built from a
half-written tree, and nothing in `core/` owns a daemon thread.

**Everything is re-read. Nothing is reused.** Two independent reasons:

- Filesystem metadata lies, and not exotically. `tar --mtime='@0' --clamp-mtime`
  and `SOURCE_DATE_EPOCH` pin mtime to a constant as a matter of reproducible-build
  practice, `cp -p` and `git checkout` do weaker versions of the same, and
  `SaveNetwork` writes a fixed layout, so retraining the same topology yields a
  file of exactly the same length. `(size, mtime)` would call that the same model.
- Hashing to decide costs more than reloading. Measured with the Release flags on
  a 5.1MB blob: the digest is 23.8ms against 11.8ms for the whole load — 2.0x
  worse, because SHA-256 here runs at about 240MB/s. The 173ms against 62.9ms
  this bullet used to quote came from the same `-O0` tree as the `Predict`
  numbers above; there the two are 205ms against 70ms. The ratio survives the
  rebuild, the magnitudes do not.

The price is measured and is not hidden: with the Release flags, 1.3ms for a
795KB model and 11.8ms for 5.2MB, roughly 3.5x that when a digest is declared.
Twenty MNIST-sized models rebuild in about 0.03s. In `build/` all of these are
eight times slower, which is where the 10ms and 63ms of earlier drafts came from.

### Ceilings

```cpp
std::size_t maxRootEntries = 4096;
std::uint64_t maxDeclaredWeightBytes = 2ull << 30;
```

`maxRootEntries` is counted *while* iterating and bails before pushing, so a root
with millions of entries does not cost a vector of paths before being refused.

The weight budget is in *declared* bytes — `(rows*cols + cols) * 8` summed across
the tree — not in bytes on disk. 2GiB rather than 4: `LoadNetwork` already refuses
a single network above `kMaxWeightBytes = 4GiB`, so a registry-wide 4GiB ceiling
would be weaker than the existing per-model cap as soon as there are two models.

**What the budget does and does not bound.** It bounds the total a composition
*retains*. It does not bound the peak during the walk: the declared size is known
only after `LoadModel` has already built the matrices, so one model up to the
per-model 4GiB cap can materialise before the budget refuses it. Nor is resident
size the same as declared: `LoadNetwork` allocates `gradientWeights` and
`gradientBiases` for every dense layer even though inference never reads them, so
a model is **2x its declared bytes** resident, and a rebuild holds two
compositions at once, so the peak is about **4x**. For scale: twenty
784/256/128/10 models are 36MB declared, 72MB resident, 144MB at the swap. The
default exists for a corrupt or hostile tree, not for capacity planning.

## Looking at a tree from the shell

`MatrixGui_models` ([`cli/serving/`](../../cli/serving/)) is the operator-facing
side of all of the above, and the thing the golden snapshot in
[`tests/golden/serving/`](../../tests/golden/serving/) pins.

```
MatrixGui_models list --root models [--default mnist=v3 ...]
```

It builds a composition, prints what loaded as a table on stdout, and every
reason the rest did not on stderr. Exit codes: `0` everything loaded, `1` the
composition is incomplete, `2` the root could not be walked, `3` a malformed
`--default`. The split matters — a caller can take stdout as the inventory and
still see the problems, and `1` is what a deployment check would gate on.

## Not here

HTTP, request serialisation, JSON for any of these structures, worker pools,
batching, hot-reload triggers — and inference itself beyond the single `Predict`
call above. Also no PNG decoding and no preprocessing of any kind.
