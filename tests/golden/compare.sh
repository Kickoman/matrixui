#!/usr/bin/env bash
# Run the golden CLI snapshots. With no arguments, all three tools; otherwise
# only the ones named.
#
# Usage: tests/golden/compare.sh [words|classifier|generator ...]
set -u

cd "$(dirname "$0")/../.."

# Deliberately the positional list rather than an array: macOS ships bash 3.2,
# where expanding an empty array under `set -u` aborts with "unbound variable" --
# which is exactly the no-argument case documented above.
if [ "$#" -eq 0 ]; then
    set -- words classifier generator
fi

status=0
for tool in "$@"; do
    if tests/golden/"$tool"/compare.sh; then
        echo "golden[$tool]: ok"
    else
        echo "golden[$tool]: FAILED"
        status=1
    fi
done
exit "$status"
