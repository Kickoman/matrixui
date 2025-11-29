#!/bin/bash

set -e

REV_A=$1
REV_B=$2

ORIGINAL=$PWD
CACHE_DIR=$PWD/.compare_cache
WORKDIR=$CACHE_DIR/compare_${REV_A}_${REV_B}
BUILDIR_PREFIX=$CACHE_DIR/build_
BUILDIR_A=${BUILDIR_PREFIX}_$REV_A
BUILDIR_B=${BUILDIR_PREFIX}_$REV_B
PREFIX=/usr/local/Qt-6.9.3
PERFTEST_EXECUTABLE="./perftest/perfbench_main"

mkdir -p $WORKDIR
mkdir -p $BUILDIR_A
mkdir -p $BUILDIR_B

echo "Build first revision"
git checkout $REV_A
git submodule update --init --recursive
cd $BUILDIR_A
cmake $ORIGINAL -DCMAKE_PREFIX_PATH=$PREFIX -DUSE_EIGEN=ON -DBUILD_MATRIX_BENCHMARK=ON -DBUILD_GUI=OFF -DBUILD_CLI=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
cd ..

echo "Build second revision"
git checkout $REV_B
git submodule update --init --recursive
cd $BUILDIR_B
cmake $ORIGINAL -DCMAKE_PREFIX_PATH=$PREFIX -DUSE_EIGEN=ON -DBUILD_MATRIX_BENCHMARK=ON -DBUILD_GUI=OFF -DBUILD_CLI=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel

cd $BUILDIR_A
echo "Run benchmark..."
$PERFTEST_EXECUTABLE --output $WORKDIR/rev_a.json --warmup 100 --iterations 10000 --randomize

cd $BUILDIR_B
echo "Run benchmark..."
$PERFTEST_EXECUTABLE --output $WORKDIR/rev_b.json --warmup 100 --iterations 10000 --randomize

echo "Compare..."
$PERFTEST_EXECUTABLE --compare $WORKDIR/rev_a.json $WORKDIR/rev_b.json --output $WORKDIR/compare_result.json
