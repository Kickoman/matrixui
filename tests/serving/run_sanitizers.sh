#!/usr/bin/env bash
# Build and run the serving, matrix, nn and handler tests under a sanitizer, out of tree.
#
# Usage: tests/serving/run_sanitizers.sh [asan|tsan]
#
# A standalone binary rather than a sanitized MatrixGui_tests: core/lib/rpn.h
# has a static_assert that stops being a constant expression under -fsanitize,
# so the whole-suite build does not compile. Nothing here depends on that code.
set -eu

cd "$(dirname "$0")/../.."
KIND="${1:-asan}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

case "$KIND" in
    asan) FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"; WRAP="" ;;
    tsan) FLAGS="-fsanitize=thread"; WRAP="setarch -R" ;;
    *)    echo "usage: $0 [asan|tsan]" >&2; exit 2 ;;
esac

SOURCES=(
    tests/main.cpp
    tests/matrix/matrix_test.cpp
    tests/nn/neural_network_applier_test.cpp
    tests/cli/serving_handlers_test.cpp
    cli/serving/handlers.cpp
    tests/serving/registry_test.cpp
    tests/serving/build_test.cpp
    tests/serving/manifest_test.cpp
    tests/serving/model_loader_test.cpp
    core/serving/manifest.cpp
    core/serving/snapshot.cpp
    core/serving/registry.cpp
    core/serving/build.cpp
    core/serving/loaded_model.cpp
    core/serving/model_loader.cpp
    core/nn/layers.cpp
    core/nn/neural_network.cpp
    core/nn/neural_network_loader.cpp
    core/nn/neural_network_applier.cpp
    core/matrix/matrix.cpp
    core/lib/sha256.cpp
    core/lib/text.cpp
)

echo "building with $FLAGS"
g++ -std=c++20 -O1 -g $FLAGS \
    -I. -Ieigen -Icontrib \
    "${SOURCES[@]}" -o "$OUT/serving_tests" -pthread

echo "running"
# TSan does not start on Ubuntu 24.04 kernels without this: the default ASLR
# entropy conflicts with its shadow mapping.
$WRAP "$OUT/serving_tests" "${@:2}"
