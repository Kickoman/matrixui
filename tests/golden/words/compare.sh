#!/usr/bin/env bash
# Re-capture the CLI snapshot and diff it against tests/golden/words/expected.
#
# Usage: tests/golden/words/compare.sh [binary]
#
# Compares the *.normalized files and the exit codes. Raw *.out are kept for
# eyeballing but are not diffed, since the training log contains timings.
set -u

BIN="${1:-./build/MatrixGui_words}"
ACTUAL="$(mktemp -d "${TMPDIR:-/tmp}/matrixgui-golden.XXXXXXXX")"
trap 'rm -rf "$ACTUAL"' EXIT

tests/golden/words/capture.sh "$ACTUAL" "$BIN" > /dev/null 2>&1

status=0
for expected in tests/golden/words/expected/*.normalized tests/golden/words/expected/*.code; do
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
    echo "  tests/golden/words/capture.sh tests/golden/words/expected"
fi
exit "$status"
