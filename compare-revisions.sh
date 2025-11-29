#!/bin/bash

set -e

REV_A=$1
REV_B=$2

REPOSITORY="git@github.com:Kickoman/matrixui.git"
PREFIX=/usr/local/Qt-6.9.3
BUILD_PARAMS="-DCMAKE_PREFIX_PATH=${PREFIX} -DUSE_EIGEN=ON -DBUILD_MATRIX_BENCHMARK=ON -DBUILD_GUI=OFF -DBUILD_CLI=OFF -DCMAKE_BUILD_TYPE=Release"

CACHE_DIR=/tmp/.compare_cache
WORKDIR=$CACHE_DIR/compare_${REV_A}_${REV_B}
SOURCE_A=$WORKDIR/source_${REV_A}
SOURCE_B=$WORKDIR/source_${REV_B}
BUILDDIR_A=$WORKDIR/build_${REV_A}
BUILDDIR_B=$WORKDIR/build_${REV_B}
PERFTEST_EXECUTABLE="./perftest/perfbench_main"

mkdir -p $WORKDIR
mkdir -p $BUILDDIR_A
mkdir -p $BUILDDIR_B

echo "Retrieve source for ${REV_A}"

if [ ! -d "$SOURCE_A" ]; then
    git clone $REPOSITORY $SOURCE_A --recurse-submodules
fi
cd $SOURCE_A
git checkout $REV_A
git submodule update --init --recursive

echo "Retrieve source for ${REV_B}"
if [ ! -d "$SOURCE_B" ]; then
    git clone $REPOSITORY $SOURCE_B --recurse-submodules
fi
cd $SOURCE_B
git checkout $REV_B
git submodule update --init --recursive

echo "Build first revision"
cd $BUILDDIR_A
cmake $SOURCE_A $BUILD_PARAMS
cmake --build . --parallel

echo "Build second revision"
cd $BUILDDIR_B
cmake $SOURCE_B $BUILD_PARAMS
cmake --build . --parallel


echo "Retrieving benchmark list"
cd $BUILDDIR_A
mapfile -t benchmarks_list < <($PERFTEST_EXECUTABLE --list)

for iteration in {1..10}; do
    echo "Running benchmark for iteration $iteration"
    for benchmark in "${benchmarks_list[@]}"; do
        echo "Run benchmark $benchmark for revision $REV_A"
        cd $BUILDDIR_A
        $PERFTEST_EXECUTABLE --output $WORKDIR/rev_a.json --warmup 100 --iterations 10 --names $benchmark --append-json-report
        echo ""
        echo "Run benchmark $benchmark for revision $REV_B"
        cd $BUILDDIR_B
        $PERFTEST_EXECUTABLE --output $WORKDIR/rev_b.json --warmup 100 --iterations 10 --names $benchmark --append-json-report
        echo ""
    done
done

# cd $BUILDDIR_A
# echo "Run benchmark..."
# $PERFTEST_EXECUTABLE --output $WORKDIR/rev_a.json --warmup 100 --iterations 10000 --randomize

# cd $BUILDDIR_B
# echo "Run benchmark..."
# $PERFTEST_EXECUTABLE --output $WORKDIR/rev_b.json --warmup 100 --iterations 10000 --randomize

echo "Compare..."
$PERFTEST_EXECUTABLE --compare $WORKDIR/rev_a.json $WORKDIR/rev_b.json --output $WORKDIR/compare_result.json
