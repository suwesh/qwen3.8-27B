#!/usr/bin/env bash
set -euo pipefail

MODEL="${QWEN_MODEL_DIR:-$HOME/models/Qwen3.5-0.8B-MLX-8bit}"
EXE="${EXE:-./qwen-fortran-cuda}"
OUT="${OUT:-artifacts/qwen35-08b/benchmarks/lfortran-cuda}"
WARMUPS="${WARMUPS:-2}"
RUNS="${RUNS:-7}"

PROMPT="What is the capital of France? Answer in a poem"
MAX_NEW=160

mkdir -p "$OUT"

if [[ ! -x "$EXE" ]]; then
    echo "Executable not found: $EXE" >&2
    exit 1
fi

if [[ ! -d "$MODEL" ]]; then
    echo "Model directory not found: $MODEL" >&2
    exit 1
fi

run_once() {
    local kind="$1"
    local index="$2"
    local prefix="$OUT/${kind}-${index}"

    "$EXE" \
      -m "$MODEL" \
      --no-think \
      -n "$MAX_NEW" \
      -p "$PROMPT" \
      >"${prefix}.stdout.txt" \
      2>"${prefix}.stderr.txt"
}

printf 'Running %d warm-ups...\n' "$WARMUPS"

for i in $(seq 1 "$WARMUPS"); do
    run_once warmup "$i"
done

printf 'Running %d measured iterations...\n' "$RUNS"

for i in $(seq 1 "$RUNS"); do
    printf 'Measured run %d/%d\n' "$i" "$RUNS"
    run_once measured "$i"
done

grep -h 'prefill:' "$OUT"/measured-*.stderr.txt \
    > "$OUT/results.txt"

printf '\nMeasured results:\n'
cat "$OUT/results.txt"
