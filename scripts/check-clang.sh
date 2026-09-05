#!/usr/bin/env bash
# Approximate a macOS build without a Mac.
#
# macOS compiles with Clang against libc++; Linux defaults to GCC against
# libstdc++. Most "it only breaks on the Mac" failures are not really about the
# OS -- they are about those two swaps, and both can be reproduced here:
#
#   libc++     catches headers that lean on libstdc++'s transitive includes,
#              which is the single largest class of macOS build breakage.
#   clang      catches the places where GCC only warns and Clang errors.
#
# What this canNOT catch: the Apple SDK, arm64 codegen, .app bundle behaviour,
# Cocoa at runtime, and bash 3.2. Those still need a real Mac -- see
# scripts/macos-check.sh, which is what you hand to someone who has one.
#
# Usage: scripts/check-clang.sh [build-root]
set -u

cd "$(dirname "$0")/.."

BUILD_ROOT="${1:-build-portability}"
CXX_BIN="${CLANGXX:-}"
if [ -z "$CXX_BIN" ]; then
    for candidate in clang++-18 clang++-17 clang++-16 clang++-15 clang++; do
        if command -v "$candidate" > /dev/null 2>&1; then
            CXX_BIN="$candidate"
            break
        fi
    done
fi

if [ -z "$CXX_BIN" ]; then
    cat >&2 <<'MSG'
check-clang: no clang++ found.

  sudo apt-get install -y clang libc++-dev libc++abi-dev

Note that core/nn/neural_network.cpp uses the C++20 range constructor of
std::string_view (P1391), which needs libc++ 16+ (or libstdc++ 12+). Ubuntu
22.04 only ships libc++ 15, so for the libc++ leg you want a newer LLVM from
https://apt.llvm.org -- otherwise that one file will fail here for a reason
that has nothing to do with macOS.
MSG
    exit 1
fi

echo "check-clang: using $CXX_BIN"
"$CXX_BIN" --version | head -1

status=0

step() {
    local name="$1"; shift
    echo
    echo "=== $name ==="
    if "$@"; then
        echo "--- $name: ok"
    else
        echo "--- $name: FAILED"
        status=1
    fi
}

# Leg 1 -- core, CLI and the unit tests under libc++. This is the valuable one:
# libc++ is literally the standard library macOS ships.
libcxx_build() {
    cmake -B "$BUILD_ROOT/libcxx" \
        -DBUILD_GUI=OFF -DBUILD_TESTS=ON \
        -DCMAKE_CXX_COMPILER="$CXX_BIN" \
        -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
        -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libc++" > /dev/null &&
    cmake --build "$BUILD_ROOT/libcxx" -j"$(nproc)" &&
    ctest --test-dir "$BUILD_ROOT/libcxx" --output-on-failure
}

# Leg 2 -- the GUI under Clang but with the default libstdc++. Qt on Debian and
# Ubuntu is built against libstdc++, so linking the GUI to libc++ would be an ABI
# mismatch; this leg is here for Clang's stricter front end, not for the library.
clang_gui_build() {
    cmake -B "$BUILD_ROOT/clang-gui" \
        -DBUILD_TESTS=OFF \
        -DCMAKE_CXX_COMPILER="$CXX_BIN" > /dev/null &&
    cmake --build "$BUILD_ROOT/clang-gui" -j"$(nproc)"
}

# Leg 3 -- configure the way Apple Silicon does, where no -march/-mcpu=native is
# accepted, so the flag probe in CMakeLists.txt has to degrade to plain -O3.
no_native_build() {
    cmake -B "$BUILD_ROOT/no-native" \
        -DBUILD_GUI=OFF -DBUILD_TESTS=ON \
        -DENABLE_NATIVE_ARCH=OFF > /dev/null &&
    cmake --build "$BUILD_ROOT/no-native" -j"$(nproc)"
}

step "core + cli + tests, clang/libc++" libcxx_build
step "gui, clang/libstdc++"             clang_gui_build
step "no native arch (Apple Silicon)"   no_native_build

echo
if [ "$status" -eq 0 ]; then
    echo "check-clang: all legs passed"
else
    echo "check-clang: something failed (see above)"
fi
exit "$status"
