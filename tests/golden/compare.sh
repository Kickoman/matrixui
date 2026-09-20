#!/usr/bin/env bash
# Run the golden CLI snapshots. With no arguments, all tools; otherwise
# only the ones named.
#
# Usage: tests/golden/compare.sh [words|classifier|generator|functions ...]
set -u

cd "$(dirname "$0")/../.."

TOOLS=("$@")
if [ "${#TOOLS[@]}" -eq 0 ]; then
    TOOLS=(words classifier generator functions serving)
fi

status=0
for tool in "${TOOLS[@]}"; do
    if tests/golden/"$tool"/compare.sh; then
        echo "golden[$tool]: ok"
    else
        echo "golden[$tool]: FAILED"
        status=1
    fi
done
exit "$status"
