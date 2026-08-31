#!/usr/bin/env bash
# Re-capture the CLI snapshot and diff it against tests/golden/expected.
#
# Usage: tests/golden/compare.sh [binary]
#
# Compares the *.normalized files and the exit codes. Raw *.out are kept for
# eyeballing but are not diffed, since the training log contains timings.
set -u

BIN="${1:-./build/MatrixGui_words}"
ACTUAL="$(mktemp -d)"
trap 'rm -rf "$ACTUAL"' EXIT

tests/golden/capture.sh "$ACTUAL" "$BIN" > /dev/null 2>&1

status=0
for expected in tests/golden/expected/*.normalized tests/golden/expected/*.code; do
    name="$(basename "$expected")"
    if ! diff -u "$expected" "$ACTUAL/$name" > /dev/null 2>&1; then
        echo "=== CHANGED: $name"
        diff -u "$expected" "$ACTUAL/$name" | sed -n '3,40p'
        status=1
    fi
done

if [ "$status" -eq 0 ]; then
    echo "golden: all snapshots match"
else
    echo "golden: differences above -- review, then re-record with:"
    echo "  tests/golden/capture.sh tests/golden/expected"
fi
exit "$status"
