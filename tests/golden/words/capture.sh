#!/usr/bin/env bash
# Capture stdout+stderr and exit code of every MatrixGui_words subcommand.
#
# Usage: tests/golden/words/capture.sh <output-dir> [binary]
set -u

OUT="${1:?usage: capture.sh <output-dir> [binary]}"
BIN="${2:-./build/MatrixGui_words}"
WORK="$OUT/_work"

rm -rf "$OUT"
mkdir -p "$WORK"

TEXT=tests/golden/words/corpus.txt
VOC="$WORK/built.voc"
COR="$WORK/built.cor"
EMB="$WORK/emb.bin"
SUBEMB="$WORK/sub.emb"

run() {
    local name="$1"; shift
    "$BIN" "$@" > "$OUT/$name.out" 2> "$OUT/$name.err"
    echo "$?" > "$OUT/$name.code"
    echo "  $name -> $(cat "$OUT/$name.code")"
}

echo "capturing to $OUT using $BIN"

# --- artifact construction (order matters: later commands need these) ---
run inspect   inspect  --input-file "$TEXT"
run buildvoc  buildvoc --input-file "$TEXT" --output-file "$VOC" --min-count 5
run loadvoc   loadvoc  --input-file "$VOC"
run buildcor  buildcor --input-file "$TEXT" --vocabulary "$VOC" --output-file "$COR"
run loadcor   loadcor  --input-file "$COR"

# --- training (1 epoch, 1 thread so pair scheduling is deterministic;
#     explicit storage so the snapshot does not depend on available memory) ---
run train train --vocabulary "$VOC" --corpus "$COR" --output-file "$EMB" \
    --dim 32 --epochs 1 --threads 1 --corpus-storage load

# --- training with character n-grams (same seed and thread count, so the
#     composed vectors are reproducible too) ---
run train-subwords train --vocabulary "$VOC" --corpus "$COR" --output-file "$SUBEMB" \
    --dim 32 --epochs 1 --threads 1 --corpus-storage load \
    --buckets 1000 --min-n 3 --max-n 6

# --- query commands ---
run neighbours-word neighbours --vocabulary "$VOC" --embeddings "$EMB" --word king --count 5
run neighbours-battery neighbours --vocabulary "$VOC" --embeddings "$EMB"
run expression  expression "king - man + woman" --vocabulary "$VOC" --embeddings "$EMB" --count 5
run oddone      oddone "king queen boy computer" --vocabulary "$VOC" --embeddings "$EMB"
run axis        axis "good - bad" --vocabulary "$VOC" --embeddings "$EMB" --count 5
run axis-words  axis "good - bad" --words "king queen war music" --vocabulary "$VOC" --embeddings "$EMB"

# --- out-of-vocabulary queries, which are the point of the n-gram vectors ---
run neighbours-oov neighbours --vocabulary "$VOC" --embeddings "$SUBEMB" \
    --subwords-file "$SUBEMB.sub" --word zzzznotaword --count 5
run neighbours-oov-short neighbours --vocabulary "$VOC" --embeddings "$SUBEMB" \
    --subwords-file "$SUBEMB.sub" --word zz --count 5

# --- error paths (must not terminate) ---
run err-missing-voc  loadvoc --input-file /dev/null
run err-unknown-word neighbours --vocabulary "$VOC" --embeddings "$EMB" --word zzzznotaword
run err-subwords-as-embeddings neighbours --vocabulary "$VOC" --embeddings "$SUBEMB.sub" --word king

# Normalise the volatile parts of the training log.
#
# The per-tick progress lines are sampled by the monitor thread at wall-clock
# intervals, so which pair-counts get reported, the loss measured at that
# instant, and even HOW MANY ticks fire all vary between runs -- a fast machine
# may finish before the first report window elapses. Only the header and the
# final summary are reproducible, so the progress lines are dropped entirely.
# The work directory is also absolute.
normalize() {
    sed -E \
        -e '/^ *[0-9]+(\.[0-9]+)?%  pairs /d' \
        -e 's#in [0-9]+(\.[0-9]+)?s \([0-9]+(\.[0-9]+)?k pairs/s\)#in <SECS> (<RATE>)#' \
        -e "s#$WORK#<WORK>#g" \
        "$1" | cat -s
}

for f in "$OUT"/*.out "$OUT"/*.err; do
    normalize "$f" > "$f.normalized"
done

echo "done"
