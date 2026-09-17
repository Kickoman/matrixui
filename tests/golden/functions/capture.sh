#!/usr/bin/env bash
# Capture stdout+stderr and exit code of every MatrixGui_functions subcommand.
#
# Usage: tests/golden/functions/capture.sh <output-dir> [binary]
set -u

OUT="${1:?usage: capture.sh <output-dir> [binary]}"
BIN="${2:-./build/MatrixGui_functions}"
WORK="$OUT/_work"

rm -rf "$OUT"
mkdir -p "$WORK"

DATA=tests/golden/functions/data.csv

run() {
    local name="$1"; shift
    "$BIN" "$@" > "$OUT/$name.out" 2> "$OUT/$name.err"
    echo "$?" > "$OUT/$name.code"
    echo "  $name -> $(cat "$OUT/$name.code")"
}

echo "capturing to $OUT using $BIN"

# --- help ---
run help     --help
run run-help run --help

# --- runs (fixed seed keeps the evolution reproducible) ---
run run-fixed run --data "$DATA" --seed 42 --epochs 5 \
    --max-population 128 --random-count 64 --print-top 5 --print-every 5

run run-expressions run --data "$DATA" --seed 42 --epochs 2 \
    --max-population 64 --random-count 0 --expression "x" --expression "x+1" \
    --print-top 3 --print-every 0

run run-patience run --data "$DATA" --seed 42 --epochs 50 --patience 2 \
    --max-population 64 --random-count 16 --print-top 3 --print-every 0

run run-weights run --data "$DATA" --seed 42 --epochs 3 \
    --accuracy-weight 1 --complexity-weight 0 --length-weight 0 \
    --max-population 64 --random-count 16 --print-top 3 --print-every 0

# --- the mutation function set ---
run run-functions-subset run --data "$DATA" --seed 42 --epochs 5 \
    --functions sin,cos --max-population 64 --random-count 16 \
    --print-top 5 --print-every 0

run run-functions-none run --data "$DATA" --seed 42 --epochs 5 \
    --functions none --max-population 64 --random-count 16 \
    --print-top 5 --print-every 0

run save-config run --data "$DATA" --seed 42 --epochs 1 \
    --max-population 64 --random-count 16 --print-top 2 \
    --save-config "$WORK/cfg.json"
cat "$WORK/cfg.json" > "$OUT/saved-config.out"
: > "$OUT/saved-config.err"
echo 0 > "$OUT/saved-config.code"

# --- error paths (must not terminate) ---
run err-missing-data run --data /no/such.csv
printf 'x,y\n1,2\n' > "$WORK/bad.csv"
run err-bad-header run --data "$WORK/bad.csv" --seed 42
printf 'x,expected\n1,abc\n' > "$WORK/nan.csv"
run err-bad-cell run --data "$WORK/nan.csv" --seed 42
run err-unknown-function run --data "$DATA" --seed 42 --functions sqrt
run err-none-combined run --data "$DATA" --seed 42 --functions none,sin

# The work directory and the binary path (echoed in help usage lines) are
# absolute; nothing else in the output is volatile, since --seed pins both
# engines and the run loop prints no timings.
normalize() {
    sed -E \
        -e "s#$WORK#<WORK>#g" \
        -e "s#$BIN#<BIN>#g" \
        "$1" | cat -s
}

for f in "$OUT"/*.out "$OUT"/*.err; do
    normalize "$f" > "$f.normalized"
done

echo "done"
