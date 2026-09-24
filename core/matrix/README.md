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

**The vendored Eigen needs `EIGEN_MAX_ALIGN_BYTES` pinned, or a big enough product
kills the process.** `eigen/` is 3.3.90, a snapshot between 3.3 and 3.4. On a
`-march` that has AVX-512 it enables the AVX-512 kernels while still computing an
alignment of 32, and the kernel then issues a `_mm512_store_pd`, which needs 64,
into a buffer aligned to 32. The result is a SIGSEGV inside `gemm_pack_lhs` on the
first product large enough for Eigen to take its blocked path — reachable over the
network in a service, and reachable from training, where the matrices are bigger.

Verified by compiling an assertion against the vendored headers, which needs no
AVX-512 hardware to show:

| `-march` | `EIGEN_MAX_ALIGN_BYTES` | AVX-512 kernels | |
|---|---|---|---|
| `skylake` | 32 | off | consistent |
| `sapphirerapids` | 32 | **on** | **inconsistent** |
| `skylake-avx512` | 32 | **on** | **inconsistent** |
| `sapphirerapids` + `EIGEN_MAX_ALIGN_BYTES=64` | 64 | on | consistent |
| `x86-64-v3` | 32 | off | consistent |

The root `CMakeLists.txt` therefore sets `EIGEN_MAX_ALIGN_BYTES=64` on the
`matrixgui_eigen` interface target rather than in the compiler flags. That target
is what carries the include path, and `core/matrix/matrix.h` is the only file in
the tree that includes Eigen, so every translation unit able to see Eigen also sees
the definition — 108 of them, with none missed. Putting it in
`CMAKE_CXX_FLAGS_RELEASE` next to `-march=native` would have left the Debug tree
and the hand-written sanitizer build on a different value, which is the same class
of mismatch one step removed.

The threshold where it starts crashing is a property of Eigen's blocking
heuristics, not of any number here: for 784x128 it sits between 64 and 96 rows on
one machine, and it moves with the shape of the matrices and with the host.
`tests/matrix/matrix_test.cpp` runs a 256x784 product for no reason other than to
be above it, because every other matrix in the suite is below — which is exactly
how this survived undetected until a batch request went looking for it.

Updating to Eigen 3.4, where the alignment and the kernels agree, is the real fix
and a separate piece of work: it re-measures everything that rests on the current
numbers.

**`Eigen::MatrixXd` is column-major.** `transform(rows, cols)` is a *row-major*
reshape and therefore a strided copy, not a view: it walks the source in
row-major order and writes in row-major order, which is what callers flattening
an image expect. It only ever reshapes — the cell count must match.

**`hadamard` still asserts.** It uses plain `assert` on both dimensions, so it
has exactly the Release-build blind spot described above. Nothing outside
training reaches it today.
