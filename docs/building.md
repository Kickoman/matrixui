# Building MatrixGui

One CMake project builds everything: a Qt desktop application, three command-line
tools and a test binary. They all link the same static core library, so the code
is compiled once and shared.

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
| CMake 3.14+ | |
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
| `BUILD_CLI` | `ON` | Command-line tools (`MatrixGui_headless`, `MatrixGui_gan`, `MatrixGui_words`) |
| `BUILD_TESTS` | `ON` | Unit-test binary (`MatrixGui_tests`) and the `unit` CTest entry |

## Build targets

| Target | Kind | Description |
|--------|------|-------------|
| `matrixgui_core` | static library | The framework-free core. Linked by every target below; nothing in it uses Qt |
| `MatrixGui` | GUI application | Full Qt interface for all four modes |
| `MatrixGui_headless` | CLI | Classifier training and single-image prediction |
| `MatrixGui_gan` | CLI | GAN training and image generation |
| `MatrixGui_words` | CLI | Word-embedding (SGNS) pipeline and queries |
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
can also run `./build/MatrixGui_tests` directly. It currently covers the words
module only. See [words.md](words.md#tests) for what it asserts and for the
separate golden CLI snapshot.

<details>
<summary>What CI builds</summary>

`.github/workflows/ci.yml` runs on every push to `master` and every pull request
into it, as two jobs:

- **Core build & unit tests** — configures with `-DBUILD_GUI=OFF -DBUILD_TESTS=ON`,
  builds, and runs `ctest`. No Qt, so it stays fast.
- **GUI build (Qt6)** — installs the Qt packages listed above and configures with
  `-DBUILD_TESTS=OFF`. This job only compiles; nothing is executed.

Both check out with `submodules: true`, since Eigen is required to configure.

The golden CLI snapshot (`tests/golden/compare.sh`) deliberately does not run in
CI, for the `-march=native` reason given above. It stays a local pre-commit tool;
the unit tests carry the same invariants with tolerances.

</details>

<details>
<summary>Source layout</summary>

`CMakeLists.txt` groups sources into three lists:

- **`CORE_SOURCES`** — `core/lib/`, `core/classifier/`, `core/generator/`,
  `core/words/` (with its `data/`, `train/`, `query/` and `report/` subfolders,
  each a labelled group in the list), plus `matrix/` and `png/`. Compiled once
  into `matrixgui_core`. No Qt, so AUTOMOC/AUTOUIC/AUTORCC are turned off for
  it.
- **`GUI_SOURCES`** and **`GUI_COMMON_SOURCES`** — `gui/` and `gui_common/`,
  linked only into the `MatrixGui` executable.

`core/words_cli/` is deliberately **not** in `CORE_SOURCES`: that list is
compiled into the Qt GUI too, and CLI11 has no business being there. Its two
files are listed directly on the `MatrixGui_words` target instead.

</details>

## Eigen submodule

The `eigen/` directory must be populated before building:

```bash
git submodule update --init
```

Warnings from Eigen's own headers are suppressed with `-w` so they do not drown
the project's `-Wall -Wextra` output.
