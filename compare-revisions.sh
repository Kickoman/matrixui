#!/bin/bash

set -e

REV_A=$1
REV_B=$2

ORIGINAL=$PWD
WORKDIR=$PWD/compare
BUILDIR=$PWD/build_compare
PREFIX=/usr/local/Qt-6.9.3
PERFTEST_EXECUTABLE="./perftest/perfbench_main"

mkdir -p $WORKDIR
mkdir -p $BUILDIR

echo "Build first revision"
git checkout $REV_A
git submodule update --init --recursive

cd $BUILDIR
rm -rf *
cmake ../ -DCMAKE_PREFIX_PATH=$PREFIX -DUSE_EIGEN=ON -DBUILD_MATRIX_BENCHMARK=ON
cmake --build . --parallel

echo "Run benchmark..."
$PERFTEST_EXECUTABLE --output $WORKDIR/rev_a.json
cd ..

echo "Build second revision"
git checkout $REV_B
git submodule update --init --recursive

cd $BUILDIR
rm -rf *
cmake ../ -DCMAKE_PREFIX_PATH=$PREFIX -DUSE_EIGEN=ON -DBUILD_MATRIX_BENCHMARK=ON
cmake --build . --parallel

echo "Run benchmark..."
$PERFTEST_EXECUTABLE --output $WORKDIR/rev_b.json


echo "Compare..."
$PERFTEST_EXECUTABLE --compare $WORKDIR/rev_a.json $WORKDIR/rev_b.json --output $WORKDIR/compare_result.json
