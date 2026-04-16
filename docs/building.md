# Building MatrixGui

## Requirements

| Requirement | Notes |
|-------------|-------|
| C++20 compiler | GCC or Clang |
| CMake 3.14+ | |
| Qt 6 (or Qt 5) | Only for the GUI target; Gui, Widgets, Concurrent, Charts modules |
| Eigen | Bundled as a git submodule in `eigen/` |

## CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_GUI` | `ON` | Qt-based graphical application (`MatrixGui`) |
| `BUILD_CLI` | `ON` | Headless CLI tools (`MatrixGui_headless`, `MatrixGui_gan`) |
| `USE_EIGEN` | `ON` | Use the bundled Eigen library for matrix operations |

## Build targets

| Target | Binary | Description |
|--------|--------|-------------|
| `MatrixGui` | GUI application | Full Qt interface for all three modes |
| `MatrixGui_headless` | CLI | Classifier training and inference |
| `MatrixGui_gan` | CLI | GAN training and image generation |

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
cmake --build build --target MatrixGui_headless
cmake --build build --target MatrixGui_gan
```

**Debug build:**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Release builds compile with `-O3 -march=native`, debug builds with `-g`.

## Eigen submodule

The `eigen/` directory must be populated before building:

```bash
git submodule update --init
```
