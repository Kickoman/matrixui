# Building and running on macOS

The project is developed on Ubuntu. This page is what you need on a Mac, and
what to do if you are checking the build *for* someone who does not own one.

## Requirements

| Requirement | Notes |
|-------------|-------|
| Xcode Command Line Tools | `xcode-select --install`. Provides Apple Clang, `git` and `python3`. |
| Apple Clang 15+ (Xcode 15+) | `core/nn/neural_network.cpp` uses the C++20 range constructor of `std::string_view` (P1391), which libc++ only implements from LLVM 16. Older toolchains fail on that one file. |
| CMake 3.16+ | `brew install cmake` |
| Qt 6 | `brew install qt` — GUI only. The Homebrew `qt` formula already contains Qt Charts. |

```bash
brew install cmake qt
```

Both Apple Silicon and Intel are supported.

## Building

CMake does not find Homebrew's Qt on its own, so point it there:

```bash
git submodule update --init
cmake -B build -DCMAKE_PREFIX_PATH="$(brew --prefix qt)"
cmake --build build -j"$(sysctl -n hw.ncpu)"
```

Use `sysctl -n hw.ncpu`; macOS has no `nproc`.

For the command-line tools only, skip Qt entirely with `-DBUILD_GUI=OFF`.

## Where the binaries land

The three CLI tools go where they do everywhere else — `build/MatrixGui_headless`,
`build/MatrixGui_gan`, `build/MatrixGui_words`.

**The GUI is the exception.** On macOS it is built as an application bundle, so
it is not `build/MatrixGui`:

```bash
open build/MatrixGui.app                          # normal launch
./build/MatrixGui.app/Contents/MacOS/MatrixGui    # launch with console output
```

The bundle is not cosmetic: without one, a Qt application runs as a faceless
background process with no Dock icon that cannot reliably be brought to the
front, and `QSettings` writes under an empty bundle identifier.

## Release builds and `-march=native`

On Linux, Release adds `-march=native`. Apple Clang on arm64 rejects that flag
outright, so `CMakeLists.txt` probes for it (and for `-mcpu=native`) and quietly
drops it when neither is accepted — which is the normal outcome on Apple
Silicon. Nothing to configure; `-DENABLE_NATIVE_ARCH=OFF` turns the probe off
altogether.

Because the flag changes floating-point instruction selection, the golden CLI
snapshots in `tests/golden/` are expected to differ on a Mac. That is not a
failure — see `tests/golden/README.md`. The unit tests (`ctest`) are the
portable check and *are* expected to pass.

## Checking the build for someone without a Mac

There is a single script that does everything and produces one archive to send
back:

```bash
git clone --recurse-submodules <repo>
cd matrixui
bash scripts/macos-check.sh
```

It records the toolchain versions, configures and builds, runs `ctest`, exercises
the CLI on a generated fixture dataset, launches the GUI twice (once through the
real Cocoa plugin, once offscreen) and saves screenshots. It deliberately does
**not** stop at the first failure, so one run reports every problem.

When it finishes, send back `macos-check-report.tar.gz`.

If `brew` or Qt is missing it will say so and still check as much as it can.

## Notes for the maintainer

Two macOS behaviours have bitten this project and are worth remembering, because
both are reproducible on Linux:

- Finder writes `.DS_Store` into every folder it displays, and volume roots carry
  `.Spotlight-V100` and `.fseventsd`. `DirectoryLister` therefore skips dotted
  entries; without that, `.DS_Store` reached the PNG reader and `std::stoull`
  threw on the Spotlight directory. Reproduce with `touch dataset/0/.DS_Store`.
- `std::filesystem::directory_iterator` returns an unspecified order that differs
  between ext4 and APFS, so listings are sorted before any `limit` is applied.
  Otherwise the same command selects a different subset of images per platform.

`$TMPDIR` on macOS lives under `/var`, a symlink to `/private/var`, so paths that
round-trip through the C++ side come back with a `/private` prefix. The golden
capture scripts normalise both spellings.
