#!/usr/bin/env bash
# Capture stdout+stderr and exit code of the MatrixGui_headless subcommands.
#
# Usage: tests/golden/classifier/capture.sh <output-dir> [binary]
set -u

OUT="${1:?usage: capture.sh <output-dir> [binary]}"
BIN="${2:-./build/MatrixGui_headless}"
WORK="$OUT/_work"

rm -rf "$OUT"
mkdir -p "$WORK"

NET=tests/golden/classifier/fixtures/network.wgt
python3 tests/golden/classifier/fixtures/make_dataset.py "$WORK/dataset" > /dev/null
IMG="$WORK/dataset/3/img_0.png"
printf 'garbage' > "$WORK/corrupt.wgt"
printf '{nope' > "$WORK/bad-learning.json"
mkdir -p "$WORK/empty"

run() {
    local name="$1"; shift
    "$BIN" "$@" > "$OUT/$name.out" 2> "$OUT/$name.err"
    echo "$?" > "$OUT/$name.code"
    echo "  $name -> $(cat "$OUT/$name.code")"
}

echo "capturing to $OUT using $BIN"

# --- help ---
run help         --help
run help-predict predict --help
run help-train   train --help

# --- predict on the committed fixture (deterministic for these bytes) ---
run predict predict --network "$NET" --image "$IMG" \
    --dataset-img-width 8 --dataset-img-height 8

# --- parse failures (CLI11 owns these codes) ---
run err-no-subcommand
run err-missing-required predict --image "$IMG"
run err-unknown-flag     predict --network "$NET" --image "$IMG" --bogus
run err-nonexistent-network predict --network "$WORK/nope.wgt" --image "$IMG"

# --- reported failures (codes owned by the commands) ---
run err-corrupt-network  predict --network "$WORK/corrupt.wgt" --image "$IMG"
run err-train-no-dataset train --network "$WORK/out.wgt"
run err-train-bad-layers train --network "$WORK/out.wgt" --dataset "$WORK/dataset" --layers 64
run err-train-size-mismatch train --network "$WORK/fresh.wgt" --dataset "$WORK/dataset" --layers 64,16,10
run err-train-invalid-dataset train --network "$WORK/out2.wgt" --dataset "$WORK/empty" \
    --layers 64,16,10 --dataset-img-width 8 --dataset-img-height 8
run err-bad-learning-config train --learning-config "$WORK/bad-learning.json" \
    --network "$WORK/out.wgt" --dataset "$WORK/dataset"

# --- a bounded real run ---
run train train --network "$WORK/trained.wgt" --dataset "$WORK/dataset" \
    --layers 64,16,10 --dataset-img-width 8 --dataset-img-height 8 \
    --max-epochs 2 --patience 1 --test-file-limit 1 --working-directory "$WORK/td"

# Normalise the volatile parts.
#
# The work directory and the binary path (echoed in help usage lines) are
# absolute. Training accuracy depends on random weight init, so the final
# figure is masked; everything else on train's stdout is a verbatim echo of
# the options. Train's stderr is epoch-by-epoch chatter whose every number is
# nondeterministic, so it is blanked wholesale below -- same reasoning as the
# words snapshot dropping its progress ticks.
normalize() {
    sed -E \
        -e "s#$WORK#<WORK>#g" \
        -e "s#$BIN#<BIN>#g" \
        -e 's#Final test accuracy: .*#Final test accuracy: <ACC>#' \
        "$1" | cat -s
}

for f in "$OUT"/*.out "$OUT"/*.err; do
    normalize "$f" > "$f.normalized"
done
: > "$OUT/train.err.normalized"

echo "done"
