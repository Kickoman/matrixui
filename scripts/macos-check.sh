#!/usr/bin/env bash
# One-shot macOS build-and-run check, meant to be handed to someone who owns a
# Mac when the author does not. It never stops at the first failure: it runs
# every stage, records everything, and packs a single archive to send back, so
# one round trip reports every problem rather than just the first.
#
#   bash scripts/macos-check.sh
#
# Written for the bash 3.2 that macOS still ships: no associative arrays, no
# mapfile, no ${var,,}, and no expansion of a possibly-empty array under set -u.
set -u

cd "$(dirname "$0")/.."

REPORT_DIR="macos-check-report"
LOG="$REPORT_DIR/report.txt"
rm -rf "$REPORT_DIR"
mkdir -p "$REPORT_DIR"

failures=0

say() { echo "$@" | tee -a "$LOG"; }
rule() { say ""; say "=== $* ==="; }

# Run a stage, tee its output into its own log, and keep going either way.
stage() {
    stage_name="$1"; shift
    rule "$stage_name"
    if "$@" >> "$LOG" 2>&1; then
        say "[ ok ] $stage_name"
        return 0
    else
        say "[FAIL] $stage_name"
        failures=$((failures + 1))
        return 1
    fi
}

# ---------------------------------------------------------------- environment
rule "Environment"
{
    echo "--- sw_vers";        sw_vers 2>&1
    echo "--- uname";          uname -a 2>&1
    echo "--- arch";           uname -m 2>&1
    echo "--- xcode-select";   xcode-select -p 2>&1
    echo "--- clang";          clang --version 2>&1
    echo "--- cmake";          cmake --version 2>&1
    echo "--- bash";           echo "$BASH_VERSION"
    echo "--- python3";        python3 --version 2>&1
    echo "--- brew qt prefix"; brew --prefix qt 2>&1
    echo "--- qmake";          "$(brew --prefix qt 2>/dev/null)/bin/qmake" --version 2>&1
    echo "--- git describe";   git describe --always --dirty 2>&1
} >> "$LOG" 2>&1
say "recorded (see $LOG)"

# --------------------------------------------------------------- preconditions
rule "Prerequisites"
missing=""
if ! xcode-select -p > /dev/null 2>&1; then
    missing="$missing xcode-command-line-tools"
fi
if ! command -v cmake > /dev/null 2>&1; then
    missing="$missing cmake"
fi
QT_PREFIX=""
if command -v brew > /dev/null 2>&1; then
    QT_PREFIX="$(brew --prefix qt 2>/dev/null || true)"
fi
if [ -z "$QT_PREFIX" ] || [ ! -d "$QT_PREFIX" ]; then
    missing="$missing qt"
fi

if [ -n "$missing" ]; then
    say "Missing:$missing"
    say ""
    say "Install with:"
    say "    xcode-select --install        # if xcode-command-line-tools is listed"
    say "    brew install cmake qt         # if cmake or qt is listed"
    say ""
    say "Then re-run this script."
    # Qt is only needed for the GUI, so carry on if that is the only gap.
    case "$missing" in
        *cmake*|*xcode*) say "Cannot continue without a compiler and cmake."; exit 1 ;;
    esac
else
    say "all present (Qt at $QT_PREFIX)"
fi

# ------------------------------------------------------------------ submodule
stage "Eigen submodule" git submodule update --init

# ------------------------------------------------------------------ configure
BUILD=build-mac
CONFIGURE_ARGS="-DBUILD_TESTS=ON"
if [ -n "$QT_PREFIX" ] && [ -d "$QT_PREFIX" ]; then
    CONFIGURE_ARGS="$CONFIGURE_ARGS -DCMAKE_PREFIX_PATH=$QT_PREFIX"
else
    say "note: building without the GUI, Qt was not found"
    CONFIGURE_ARGS="$CONFIGURE_ARGS -DBUILD_GUI=OFF"
fi

rm -rf "$BUILD"
# Unquoted on purpose: CONFIGURE_ARGS is a list of separate cmake arguments.
# shellcheck disable=SC2086
stage "Configure" cmake -B "$BUILD" $CONFIGURE_ARGS

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 2)"
stage "Build" cmake --build "$BUILD" -j"$JOBS"

# ---------------------------------------------------------------------- tests
stage "Unit tests (ctest)" ctest --test-dir "$BUILD" --output-on-failure

# ------------------------------------------------------------------- cli smoke
rule "CLI smoke test"
SMOKE="$BUILD/_smoke"
rm -rf "$SMOKE"
mkdir -p "$SMOKE"

if python3 tests/golden/classifier/fixtures/make_dataset.py "$SMOKE/dataset" >> "$LOG" 2>&1; then
    say "[ ok ] generated the fixture dataset"

    # A .DS_Store in a class folder and a Spotlight directory at the root are what
    # a real Mac puts there; both used to be fed to the PNG reader or to stoull.
    : > "$SMOKE/dataset/0/.DS_Store"
    mkdir -p "$SMOKE/dataset/.Spotlight-V100"
    say "planted .DS_Store and .Spotlight-V100 to exercise the macOS filesystem path"

    # The fixture images are 8x8, so the geometry has to be stated explicitly.
    stage "headless train" "./$BUILD/MatrixGui_headless" train \
        --dataset "$SMOKE/dataset" \
        --network "$SMOKE/smoke.wgt" \
        --working-directory "$SMOKE/work" \
        --layers 64,16,10 \
        --dataset-img-width 8 --dataset-img-height 8 \
        --max-epochs 1
    stage "headless predict" "./$BUILD/MatrixGui_headless" predict \
        --network tests/golden/classifier/fixtures/network.wgt \
        --image "$SMOKE/dataset/3/img_0.png" \
        --dataset-img-width 8 --dataset-img-height 8
    stage "gan --help" "./$BUILD/MatrixGui_gan" --help
    stage "words --help" "./$BUILD/MatrixGui_words" --help
else
    say "[FAIL] could not generate the fixture dataset (python3 missing?)"
    failures=$((failures + 1))
fi

# ------------------------------------------------------------------------ gui
rule "GUI"
GUI_BIN=""
if [ -x "$BUILD/MatrixGui.app/Contents/MacOS/MatrixGui" ]; then
    GUI_BIN="$BUILD/MatrixGui.app/Contents/MacOS/MatrixGui"
    say "bundle built at $BUILD/MatrixGui.app"
elif [ -x "$BUILD/MatrixGui" ]; then
    GUI_BIN="$BUILD/MatrixGui"
    say "plain executable at $GUI_BIN (expected a bundle -- worth reporting)"
fi

if [ -z "$GUI_BIN" ]; then
    say "[skip] no GUI binary was built"
else
    # Native platform plugin first: this is the one that proves Cocoa actually
    # loads. Then offscreen, which works with no window server (e.g. over ssh).
    stage "GUI, native (cocoa)" "./$GUI_BIN" --screenshot "$REPORT_DIR/gui-cocoa.png"

    export QT_QPA_PLATFORM=offscreen
    stage "GUI, offscreen" "./$GUI_BIN" --screenshot "$REPORT_DIR/gui-offscreen.png"
    unset QT_QPA_PLATFORM
fi

# --------------------------------------------------------------------- goldens
# Informational only. The snapshots pin float output from a -march=native build,
# so a different CPU is expected to differ; we record the result, not judge it.
rule "Golden snapshots (informational, differences are expected on a new CPU)"
bash tests/golden/compare.sh >> "$LOG" 2>&1
say "golden exit status: $? (non-zero here is not necessarily a bug)"

# ---------------------------------------------------------------------- pack up
rule "Summary"
cp "$BUILD/CMakeCache.txt" "$REPORT_DIR/CMakeCache.txt" 2>/dev/null || true
say "failed stages: $failures"

ARCHIVE="macos-check-report.tar.gz"
tar -czf "$ARCHIVE" "$REPORT_DIR" 2>/dev/null

echo
echo "==============================================================="
echo " Done. Please send back: $ARCHIVE"
echo " Failed stages: $failures"
echo "==============================================================="
exit 0
