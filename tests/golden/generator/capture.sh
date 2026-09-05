#!/usr/bin/env bash
# Capture stdout+stderr and exit code of the MatrixGui_gan subcommands.
#
# Usage: tests/golden/generator/capture.sh <output-dir> [binary]
set -u

OUT="${1:?usage: capture.sh <output-dir> [binary]}"
BIN="${2:-./build/MatrixGui_gan}"
WORK="$OUT/_work"

rm -rf "$OUT"
mkdir -p "$WORK"

GEN=tests/golden/generator/fixtures/generator.wgt
CLS=tests/golden/classifier/fixtures/network.wgt
python3 tests/golden/classifier/fixtures/make_dataset.py "$WORK/dataset" > /dev/null
printf 'garbage' > "$WORK/corrupt.wgt"

run() {
    local name="$1"; shift
    "$BIN" "$@" > "$OUT/$name.out" 2> "$OUT/$name.err"
    echo "$?" > "$OUT/$name.code"
    echo "  $name -> $(cat "$OUT/$name.code")"
}

echo "capturing to $OUT using $BIN"

# --- help ---
run help          --help
run help-generate generate --help
run help-train    train --help

# --- generation from the committed fixture ---
run generate generate --label 3 --generator "$GEN" --num-classes 10 \
    --output "$WORK/digit.png" --dataset-img-width 8 --dataset-img-height 8
run generate-multi generate --label 3 --generator "$GEN" --num-classes 10 \
    --output "$WORK/many.png" --num-samples 3 --dataset-img-width 8 --dataset-img-height 8
run generate-classified generate --label 3 --generator "$GEN" --classifier "$CLS" \
    --output "$WORK/cls.png" --dataset-img-width 8 --dataset-img-height 8

# --- parse failures (CLI11 owns these codes) ---
run err-no-subcommand
run err-missing-label generate
run err-unknown-flag  generate --label 0 --bogus
run err-nonexistent-generator generate --label 0 --generator "$WORK/nope.wgt"
run err-train-nonexistent-dataset train --classifier "$CLS" --dataset "$WORK/absent"

# --- reported failures ---
run err-corrupt-generator generate --label 0 --generator "$WORK/corrupt.wgt"
run err-train-missing-classifier train --classifier "$WORK/absent.wgt" --dataset "$WORK/dataset"

# --- a bounded real run ---
run train train --classifier "$CLS" --dataset "$WORK/dataset" \
    --generator "$WORK/g.wgt" --discriminator "$WORK/d.wgt" \
    --gen-layers 16 --disc-layers 16 --latent-dim 8 \
    --epochs 1 --batch-size 4 --dataset-img-width 8 --dataset-img-height 8

# Normalise the volatile parts.
#
# The work directory and the binary path (echoed in help usage lines) are
# absolute. Generation noise comes from std::random_device, so the classifier's
# opinion of a generated image varies per run, as does every number in the GAN
# epoch lines.
normalize() {
    sed -E \
        -e "s#$WORK#<WORK>#g" \
        -e "s#$BIN#<BIN>#g" \
        -e 's#\(classifier: [0-9]+\)#(classifier: <N>)#' \
        -e '/^Epoch [0-9]+\//d' \
        "$1" | cat -s
}

for f in "$OUT"/*.out "$OUT"/*.err; do
    normalize "$f" > "$f.normalized"
done

echo "done"
