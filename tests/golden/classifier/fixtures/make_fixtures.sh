#!/usr/bin/env bash
# One-off: (re)train the committed .wgt fixtures for the classifier and
# generator goldens.
#
# Training is NOT reproducible (weight init, shuffling and GAN noise come from
# std::random_device), so the committed .wgt files are frozen artifacts: the
# goldens pin behavior GIVEN THESE BYTES. Re-running this script produces
# different weights -- if you do, re-record both tools' expected/ in the same
# commit.
#
# Usage (from the repo root): tests/golden/classifier/fixtures/make_fixtures.sh [build-dir]
set -eu

BUILD="${1:-./build}"
HERE=tests/golden/classifier/fixtures
GEN=tests/golden/generator/fixtures
TMP="$(mktemp -d "${TMPDIR:-/tmp}/matrixgui-fixtures.XXXXXXXX")"
trap 'rm -rf "$TMP"' EXIT

python3 "$HERE/make_dataset.py" "$TMP/dataset"

"$BUILD/MatrixGui_headless" train \
    --network "$HERE/network.wgt" \
    --dataset "$TMP/dataset" \
    --layers 64,16,10 \
    --dataset-img-width 8 --dataset-img-height 8 \
    --max-epochs 30 --patience 10 \
    --working-directory "$TMP/wd"

"$BUILD/MatrixGui_gan" train \
    --classifier "$HERE/network.wgt" \
    --dataset "$TMP/dataset" \
    --generator "$GEN/generator.wgt" \
    --discriminator "$TMP/discriminator.wgt" \
    --gen-layers 16 --disc-layers 16 --latent-dim 8 \
    --epochs 2 --batch-size 4 \
    --dataset-img-width 8 --dataset-img-height 8

ls -la "$HERE/network.wgt" "$GEN/generator.wgt"
