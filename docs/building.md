# Building MatrixGui

One CMake project builds everything: a Qt desktop application, four command-line
tools and a test binary. They share a set of small libraries under `core/`, each
built once and linked by whichever front end needs it.

The GUI is the only part that needs Qt. If you only want the command-line tools,
pass `-DBUILD_GUI=OFF` and you can skip installing Qt entirely.

**Before your first build**, populate the Eigen submodule — the build will not
configure without it:

```bash
git submodule update --init
```

Then:

```bash
cmake -B build
cmake --build build
```

## Requirements

| Requirement | Notes |
|-------------|-------|
| C++20 compiler | GCC or Clang |
| CMake 3.16+ | |
| Qt 6 (or Qt 5) | GUI only. Modules: Gui, Widgets, Qml, QuickWidgets, Concurrent, Charts |
| Eigen | Bundled as a git submodule in `eigen/` |

On Debian/Ubuntu, the Qt package set CI uses is:

```bash
sudo apt-get install -y --no-install-recommends \
  qt6-base-dev qt6-declarative-dev qt6-charts-dev
```

## CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_GUI` | `ON` | Qt-based graphical application (`MatrixGui`) |
| `BUILD_CLI` | `ON` | Command-line tools (`MatrixGui_headless`, `MatrixGui_gan`, `MatrixGui_words`, `MatrixGui_functions`) |
| `BUILD_TESTS` | `ON` | Unit-test binary (`MatrixGui_tests`) and the `unit` CTest entry |
| `BUILD_SHARED_LIBS` | `OFF` | Build the `matrixgui_*` libraries as `.so` instead of `.a` |

## Build targets

Libraries — none of them use Qt. Their kind follows `BUILD_SHARED_LIBS`, static
by default, and they land in `build/lib/`:

| Target | Source | Depends on | Description |
|--------|--------|-----------|-------------|
| `matrixgui_matrix` | `core/matrix/` | Eigen | Matrix type used by everything numeric |
| `matrixgui_png` | `core/png/` | `matrix` | Image loading/writing over vendored stb |
| `matrixgui_core_lib` | `core/lib/` | — | Framework-free utilities: text, stats, stream formatting, file IO, RNG, caches. Knows nothing about matrices or networks |
| `matrixgui_nn` | `core/nn/` | `matrix`, `core_lib` | Layers, network, applier, loader, datasets |
| `matrixgui_classifier` | `core/classifier/` | `nn` | Classifier training loop and its config |
| `matrixgui_generator` | `core/generator/` | `nn`, `matrix` | Conditional-GAN training loop and its config |
| `matrixgui_words` | `core/words/` | `core_lib` | The whole SGNS pipeline. Notably does **not** link `nn`, `matrix` or Eigen |
| `matrixgui_functions` | `core/functions/` | `core_lib` | Genetic symbolic regression: the genetizer, the fitness applier, and the expected-values CSV |
| `matrixgui_cli_lib` | `cli/lib/` | `matrix` | Helpers shared by the command-line tools: cached PNG reader, JSON config loading, argv pre-scan |
| `matrixgui_cli_words` | `cli/words/` | `words` | `MatrixGui_words` subcommand bodies, minus `main()` |
| `matrixgui_cli_classifier` | `cli/classifier/` | `classifier`, `nn` | `MatrixGui_headless` subcommand bodies, minus `main()` |
| `matrixgui_cli_generator` | `cli/generator/` | `generator` | `MatrixGui_gan` subcommand bodies, minus `main()` |
| `matrixgui_cli_functions` | `cli/functions/` | `functions` | `MatrixGui_functions` subcommand bodies, minus `main()` |
| `matrixgui_gui_common` | `gui_common/` | Qt | Reusable widgets; built only with `BUILD_GUI=ON` |

Executables — these land in `build/` itself:

| Target | Kind | Description |
|--------|------|-------------|
| `MatrixGui` | GUI application | Full Qt interface for all five modes |
| `MatrixGui_headless` | CLI | Classifier training and single-image prediction |
| `MatrixGui_gan` | CLI | GAN training and image generation |
| `MatrixGui_words` | CLI | Word-embedding (SGNS) pipeline and queries |
| `MatrixGui_functions` | CLI | Genetic symbolic regression over a CSV of expected values |
| `MatrixGui_tests` | test binary | doctest suite, built when `BUILD_TESTS=ON` |

## Common build configurations

**Everything (default):**

```bash
cmake -B build
cmake --build build
```

**CLI tools only (no Qt required):**

```bash
cmake -B build -DBUILD_GUI=OFF
cmake --build build
```

**Single target:**

```bash
cmake -B build
cmake --build build --target MatrixGui_words
```

**Shared libraries:**

```bash
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build
```

The `.so` files go to `build/lib/`; the executables in `build/` find them through
an embedded RPATH, so no `LD_LIBRARY_PATH` is needed. Symbol visibility is left
at the compiler default on purpose — see the comment in `CMakeLists.txt`.

**Debug build:**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

`CMAKE_BUILD_TYPE` defaults to `Release` when you do not set it.

Everything compiles with `-Wall -Wextra`. Release adds `-O3 -DNDEBUG
-march=native`, Debug adds `-g`.

`-march=native` means release binaries are tuned for the machine that built them
and are not portable to a different CPU. That is also why the golden CLI
snapshot only runs locally: it pins the float output of a trained model, and a
different instruction set accumulates floats in a different order.

## Tests

```bash
cmake -B build -DBUILD_TESTS=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

The suite is doctest-based and registered with CTest under the name `unit`; you
can also run `./build/MatrixGui_tests` directly. It covers the words and
functions modules plus the CLI command bodies. See [words.md](words.md#tests) for what it asserts and for the
separate golden CLI snapshots (`tests/golden/README.md`).

<details>
<summary>What CI builds</summary>

`.github/workflows/ci.yml` runs on every push to `master` and every pull request
into it, as two jobs:

- **Core build & unit tests** — configures with `-DBUILD_GUI=OFF -DBUILD_TESTS=ON`,
  builds, and runs `ctest`. No Qt, so it stays fast.
- **GUI build (Qt6)** — installs the Qt packages listed above and configures with
  `-DBUILD_TESTS=OFF`. This job only compiles; nothing is executed.

Both check out with `submodules: true`, since Eigen is required to configure.

The golden CLI snapshots (`tests/golden/compare.sh`, covering all four
command-line tools) deliberately do not run in CI, for the `-march=native`
reason given above. They stay a local pre-commit tool; the unit tests carry the
same invariants with tolerances. See `tests/golden/README.md`.

</details>

<details>
<summary>Source layout</summary>

Every directory that produces a target owns its own `CMakeLists.txt`. The root
file holds only the options, three interface targets and the `add_subdirectory`
calls:

```
core/       the libraries: matrix, png, lib, nn, classifier, generator, words, functions
cli/        one folder per command-line tool, mirroring gui/
gui/        the Qt application
gui_common/ reusable Qt widgets
tests/      the doctest suite
```

Three `INTERFACE` targets carry the settings, and every first-party target links
`matrixgui_base`:

| Target | Carries |
|---|---|
| `matrixgui_base` | The repo-root include path (which is what makes `#include "core/nn/layers.h"` resolve), C++20, `-Wall -Wextra` |
| `matrixgui_contrib` | `contrib/` as a system include — nlohmann, CLI11, doctest, magic_enum |
| `matrixgui_eigen` | `eigen/` as a system include, so Eigen's own warnings stay quiet without silencing ours |

Dependencies are declared `PUBLIC` only when a type actually appears in the
target's headers, `PRIVATE` otherwise. That is what keeps `matrixgui_words` free
of Eigen and CLI11 out of the Qt binary — the split that used to be maintained
by hand as a carve-out from a flat `CORE_SOURCES` list.

AUTOMOC is set per target in `gui/` and `gui_common/` rather than globally, so it
never runs over the CLI or core targets. AUTOUIC is not enabled anywhere: there
are no `.ui` files.

</details>

<details>
<summary>Why the build files look the way they do</summary>

Non-obvious constraints. Each of these will look like a pointless detail until
you change it.

**Symbol visibility is left at the compiler default, on purpose.** Setting
`CMAKE_CXX_VISIBILITY_PRESET hidden` breaks more than linking: the `Words::Error`
hierarchy in `core/words/error.h` is header-only, thrown inside
`matrixgui_words` and caught in `cli/words/main.cpp`. With hidden visibility each
module gets its own `typeinfo`, and `catch (const Words::Error&)` silently misses
into `std::terminate`. Making hidden visibility safe needs export macros on the
exception types, not a flag flip.

**`CMAKE_RUNTIME_OUTPUT_DIRECTORY` is load-bearing.** Executables are declared in
`cli/*/` and `gui/`, so without it they would land in `build/cli/words/` and
friends. `README.md`, this file, and the `tests/golden/*/` scripts all default
to `./build/MatrixGui_words` and friends.

**Do not add `include_directories(eigen)` back.** GCC ignores an `-isystem P`
if a plain `-I P` for the same path appeared earlier on the command line, which
would make `matrixgui_eigen` a no-op and bring Eigen's warnings back. For the
same reason `eigen/` is never `add_subdirectory`'d — its own `project()` drags in
tests, BLAS probing and install rules.

**`MATRIXGUI_LIB_TYPE` is passed to `add_library()` explicitly** instead of
relying on `BUILD_SHARED_LIBS` implicitly, so the kind of a library cannot be
changed out from under a leaf directory by a shadowed variable.
`contrib/qt-dark-theme` hardcodes `STATIC` and should stay that way: its `.qrc`
is registered by a static initializer.

**`enable_testing()` must stay in the root file.** Called only from
`tests/CMakeLists.txt`, it generates no root `build/CTestTestfile.cmake`, and
`ctest --test-dir build` — what CI runs — finds nothing.

**Test sources go straight into `MatrixGui_tests`, never into a library.**
doctest registers cases through file-scope global constructors; in a static
archive the linker drops the objects and the suite silently shrinks to zero
cases.

**Every `cli/*/` directory builds its library unconditionally**, and only the
executables are gated on `BUILD_CLI`. `-DBUILD_CLI=OFF -DBUILD_TESTS=ON` is the
shape of CI's first job, and it must stay configurable if a test ever links one
of them.

**`core/png/pngreader.cpp` wraps its stb includes in a `#pragma GCC diagnostic`
block.** The `*_IMPLEMENTATION` defines pull the whole vendored implementation
into that translation unit; the pragma keeps its warnings out of the build
without silencing the rest of the file.

</details>

## Eigen submodule

The `eigen/` directory must be populated before building:

```bash
git submodule update --init
```

Warnings from Eigen's own headers are suppressed with `-w` so they do not drown
the project's `-Wall -Wextra` output.
