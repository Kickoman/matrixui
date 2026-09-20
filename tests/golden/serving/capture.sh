#!/usr/bin/env bash
# Capture stdout+stderr and exit code of the MatrixGui_models subcommands.
#
# Usage: tests/golden/serving/capture.sh <output-dir> [binary]
set -u

OUT="${1:?usage: capture.sh <output-dir> [binary]}"
BIN="${2:-./build/MatrixGui_models}"
WORK="$OUT/_work"

rm -rf "$OUT"
mkdir -p "$WORK"

# The two frozen classifier/generator fixtures are used as weight blobs: their
# bytes are committed, so the table below does not depend on a random init.
SMALL=tests/golden/classifier/fixtures/network.wgt     # 64 -> 16 -> 10
WIDE=tests/golden/generator/fixtures/generator.wgt     # 18 -> 16 -> 64

manifest() {   # <dir> <name> <version> <in> <out> [extra-json]
    mkdir -p "$1"
    python3 - "$@" <<'PY'
import json, sys
directory, name, version, inputs, outputs = sys.argv[1:6]
extra = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
document = {
    "manifestVersion": 1, "name": name, "version": version,
    "weights": {"path": "weights.wgt"},
    "input": {"size": int(inputs)}, "output": {"size": int(outputs)},
}
for section, fields in extra.items():
    document.setdefault(section, {}).update(fields)
open(directory + "/manifest.json", "w").write(json.dumps(document, indent=2) + "\n")
PY
}

ROOT="$WORK/models"
mkdir -p "$ROOT"

manifest "$ROOT/mnist-v1" mnist v1 64 10
cp "$SMALL" "$ROOT/mnist-v1/weights.wgt"

manifest "$ROOT/mnist-v3" mnist v3 64 10 \
    "{\"input\": {\"layout\": \"hwc\", \"shape\": [8, 8, 1]},
      \"output\": {\"kind\": \"classification\",
                   \"labels\": [\"0\",\"1\",\"2\",\"3\",\"4\",\"5\",\"6\",\"7\",\"8\",\"9\"]},
      \"weights\": {\"sha256\": \"$(sha256sum "$SMALL" | cut -d' ' -f1)\"}}"
cp "$SMALL" "$ROOT/mnist-v3/weights.wgt"

manifest "$ROOT/gen-v1" gen v1 18 64
cp "$WIDE" "$ROOT/gen-v1/weights.wgt"

# Every shape of refusal the walk can produce.
manifest "$ROOT/truncated" broken v1 64 10
head -c 60 "$SMALL" > "$ROOT/truncated/weights.wgt"
mkdir -p "$ROOT/bad-json" && printf '{nope' > "$ROOT/bad-json/manifest.json"
mkdir -p "$ROOT/no-manifest"
manifest "$ROOT/dup-a" dup v1 64 10 && cp "$SMALL" "$ROOT/dup-a/weights.wgt"
manifest "$ROOT/dup-b" dup v1 64 10 && cp "$SMALL" "$ROOT/dup-b/weights.wgt"
manifest "$ROOT/contract" other v1 100 10 && cp "$SMALL" "$ROOT/contract/weights.wgt"
ln -s mnist-v3 "$ROOT/current"
printf 'models live here\n' > "$ROOT/README.md"

# A root with nothing in it, and one with only good models.
mkdir -p "$WORK/nothing"
CLEAN="$WORK/clean"
mkdir -p "$CLEAN"
manifest "$CLEAN/mnist-v1" mnist v1 64 10 && cp "$SMALL" "$CLEAN/mnist-v1/weights.wgt"
manifest "$CLEAN/mnist-v3" mnist v3 64 10 && cp "$SMALL" "$CLEAN/mnist-v3/weights.wgt"

run() {
    local name="$1"; shift
    "$BIN" "$@" > "$OUT/$name.out" 2> "$OUT/$name.err"
    echo "$?" > "$OUT/$name.code"
    echo "  $name -> $(cat "$OUT/$name.code")"
}

echo "capturing to $OUT using $BIN"

run help      --help
run help-list list --help

run list-clean        list --root "$CLEAN"
run list-default      list --root "$CLEAN" --default mnist=v3
run list-empty-root   list --root "$WORK/nothing"
run list-with-failures list --root "$ROOT"
run list-dangling-default list --root "$CLEAN" --default mnist=v9

run err-no-subcommand
run err-no-root          list
run err-missing-root     list --root "$WORK/absent"
run err-bad-default      list --root "$CLEAN" --default mnist
run err-unknown-flag     list --root "$CLEAN" --bogus

# Paths in refusals come back from weakly_canonical, so they are absolute and
# carry the checkout prefix; that has to go too or the snapshot is machine-bound.
HERE="$(pwd)"
normalize() {
    sed -E \
        -e "s#$HERE/$WORK#<WORK>#g" \
        -e "s#$WORK#<WORK>#g" \
        -e "s#$HERE#<REPO>#g" \
        -e "s#$BIN#<BIN>#g" \
        "$1" | cat -s
}

for f in "$OUT"/*.out "$OUT"/*.err; do
    normalize "$f" > "$f.normalized"
done

echo "done"
