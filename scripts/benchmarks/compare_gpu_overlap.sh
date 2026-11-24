#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/Projects/qwen3.8-27B"
MODEL="$HOME/models/Qwen3.5-0.8B-MLX-8bit"
PROMPT="What is the capital of France? Answer briefly."

cd "$REPO"

cp Makefile Makefile.overlap-backup
trap 'mv -f Makefile.overlap-backup Makefile' EXIT

build_and_run() {
    local mode="$1"

    rm -f qwen-fortran-cuda
    rm -rf build/fortran

    LFORTRAN_LINKER=gcc make fortran-gpu \
        > "build-${mode}.log" 2>&1

    ./qwen-fortran-cuda \
        -m "$MODEL" \
        --no-think \
        --ids \
        -n 1 \
        -p "$PROMPT" \
        > "run-${mode}.txt" 2>&1

    grep -E '^(HEAD|LAYER|final_normalized|output ids)' \
        "run-${mode}.txt" \
        > "trace-${mode}.txt"
}

echo "Building overlap version..."
build_and_run overlap

echo "Building no-overlap version..."
sed -i 's/[[:space:]]*--gpu-overlap//g' Makefile

echo "Confirming --gpu-overlap is absent..."
if grep -q -- '--gpu-overlap' Makefile; then
    echo "ERROR: --gpu-overlap still present"
    exit 1
fi

build_and_run no-overlap

echo
echo "Output tokens:"
grep 'output ids' run-overlap.txt || true
grep 'output ids' run-no-overlap.txt || true

echo
echo "First differing trace lines:"
diff -u trace-overlap.txt trace-no-overlap.txt \
    > overlap-diff.txt || true

sed -n '1,160p' overlap-diff.txt

echo
echo "Files created:"
printf '%s\n' \
    trace-overlap.txt \
    trace-no-overlap.txt \
    overlap-diff.txt \
    run-overlap.txt \
    run-no-overlap.txt
