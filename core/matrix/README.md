# `core/matrix` — the one matrix type

A single class wrapping `Eigen::MatrixXd`, so that nothing above this folder
includes Eigen. Every layer, trainer and applier in [`core/nn/`](../nn/) is
written against `Matrix` and against nothing else.

| File | Contains |
|---|---|
| `matrix.h` | `class Matrix`, `operator<<` |
| `matrix.cpp` | all of it |

The Eigen member is **private and there is no accessor**: the two
`Eigen::MatrixXd`-taking constructors are private too, so an expression can only
enter or leave through the named operations below. That is what keeps Eigen out
of the rest of the tree, and it is also why filling a `Matrix` from a flat buffer
is an element loop rather than a `memcpy`.

## What it exposes

Construction is `(rows, cols)`, `(rows, cols, value)`, `(rows, cols, generator)`,
`std::vector<std::vector<double>>`, and the `identity` / `zeros` / `ones`
factories. Access is `operator()(row, col)`.

Arithmetic is `+ - *` with a `Matrix`, `*` and `/` with a scalar, `+=`, `-=`,
`hadamard`, `transpose`, plus four fused forms that exist to avoid materialising
an intermediate on the training path:

| Operation | Computes |
|---|---|
| `transposeMultiply(b)` | `this->transpose() * b` |
| `multiplyTranspose(b)` | `this * b.transpose()` |
| `addTransposeMultiply(a, b)` | `*this += a.transpose() * b` |
| `multiplyAdd(w, bias)` | `this * w`, then `bias` added to **every row** |
| `substractScaled(alpha, b)` | `*this -= alpha * b` |

## What will bite you

**`eigen_assert` is not a check.** Eigen guards its own dimension preconditions
with `eigen_assert`, which `NDEBUG` compiles away — and `NDEBUG` is in
`CMAKE_CXX_FLAGS_RELEASE`. So in the build that ships, a dimension mismatch is
not an abort but undefined behaviour: measured on a 784-input network, a row one
value **short** returns a plausible, silently wrong answer (the dot product is
truncated, which is bit-identical to zero-padding), and a row one value **long**
is a heap-buffer-overflow read inside `gemm_pack_rhs`. Anything reachable from
outside the process must therefore be checked *here*, in `Matrix`, and not left
to Eigen. `multiplyAdd`, `transform`, `operator/` and the
`vector<vector<double>>` constructor throw `std::invalid_argument`; the rest of
the operations still rely on Eigen and are only as safe as their callers.

**Whether a sanitizer notices is not monotone in any dimension.** A one-column
overread escapes the weights allocation — and so gets reported — at `16x5`, but
stays inside the block Eigen already over-allocated at `8x10`, the same forty
values wider. A memory-safety test written on a toy topology can therefore pass
while proving nothing; `tests/nn/neural_network_applier_test.cpp` pins its
canary to `64-16-10` and says so.

**`multiplyAdd` broadcasts, and Eigen's `+=` does not.** The bias a dense layer
carries is `1 x cols` while the product is `rows x cols`. Written as
`result += bias` that is a size mismatch — correct for a single row, undefined
for a batch, and silent in Release. It is `result.rowwise() += bias.row(0)`, and
`multiplyAdd` refuses a bias that is not exactly `1 x cols` so the broadcast
cannot be mistaken for a shape-matching add. Before this was fixed, four
identical input rows produced four different answers, each summing to one and
each looking like a reasonable distribution.

**`Eigen::MatrixXd` is column-major.** `transform(rows, cols)` is a *row-major*
reshape and therefore a strided copy, not a view: it walks the source in
row-major order and writes in row-major order, which is what callers flattening
an image expect. It only ever reshapes — the cell count must match.

**`hadamard` still asserts.** It uses plain `assert` on both dimensions, so it
has exactly the Release-build blind spot described above. Nothing outside
training reaches it today.
