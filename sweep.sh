#!/usr/bin/env bash
# sweep.sh - rebuild + run the test program for a range of SIZE / blockSize values
# and print one CSV row per test. Run from the repo root.
#
#   bash sweep.sh size  12 16 20 24 26      # SIZE = 2^12 ... 2^26, blockSize as in source
#   bash sweep.sh block 64 128 256 512 1024 # blockSize sweep, SIZE = 2^$SIZE_EXP (default 22)
#   REPEAT=5 bash sweep.sh size 20 24       # run the exe 5 times per build, report the min
#
# Temporarily edits src/main.cpp and the two blockSize constants, then restores them.
set -euo pipefail

MAIN=src/main.cpp
NAIVE=stream_compaction/naive.cu
EFF=stream_compaction/efficient.cu
EXE=build/bin/Release/cis5650_stream_compaction_test.exe
SIZE_EXP="${SIZE_EXP:-22}"
REPEAT="${REPEAT:-1}"

[ -f "$MAIN" ] || { echo "run from the repo root" >&2; exit 1; }
mode="${1:-}"; shift || true
[ "$mode" = size ] || [ "$mode" = block ] || { sed -n '2,7p' "$0"; exit 1; }
[ $# -gt 0 ] || { echo "give at least one value" >&2; exit 1; }

# backup + always restore
cp "$MAIN" "$MAIN.bak"; cp "$NAIVE" "$NAIVE.bak"; cp "$EFF" "$EFF.bak"
trap 'mv -f "$MAIN.bak" "$MAIN"; mv -f "$NAIVE.bak" "$NAIVE"; mv -f "$EFF.bak" "$EFF"' EXIT

set_size()  { sed -i -E "s/^const int SIZE = 1 << [0-9]+;/const int SIZE = 1 << $1;/" "$MAIN"; }
set_block() { sed -i -E "s/static const int blockSize = [0-9]+;/static const int blockSize = $1;/" "$NAIVE" "$EFF"; }

run_once() {  # $1 = label prefix for the CSV row; builds once, runs REPEAT times, keeps the min
    cmake --build build --config Release >/dev/null 2>&1 || { echo "build failed" >&2; exit 1; }
    for ((r = 0; r < REPEAT; r++)); do echo | "$EXE"; done | tr -d '\r' | awk -v pre="$1" '
        /^==== /        { name=$0; gsub(/^==== | ====$/, "", name) }
        /elapsed time:/ { t=$3 + 0; sub(/ms$/, "", t)
                          if (!(name in best) || t < best[name]) best[name] = t
                          if (!(name in seen)) { seen[name] = 1; order[++n] = name } }
        END { for (i = 1; i <= n; i++) printf "%s,%s,%s\n", pre, order[i], best[order[i]] }'
}

echo "param,value,test,ms"
if [ "$mode" = size ]; then
    for e in "$@"; do set_size "$e"; run_once "size,2^$e"; done
else
    set_size "$SIZE_EXP"
    for b in "$@"; do set_block "$b"; run_once "block,$b"; done
fi
