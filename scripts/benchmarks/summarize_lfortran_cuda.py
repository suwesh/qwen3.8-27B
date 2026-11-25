import re
import statistics
from pathlib import Path

result_dir = Path(
    "artifacts/qwen35-08b/benchmarks/lfortran-cuda"
)

pattern = re.compile(
    r"prefill:\s+"
    r"(?P<prompt_tokens>\d+)\s+tokens in\s+"
    r"(?P<prefill_seconds>[0-9.]+)\s+s\s+"
    r"\((?P<prefill_rate>[0-9.]+)\s+tok/s\);\s+"
    r"decode:\s+"
    r"(?P<decode_tokens>\d+)\s+tokens in\s+"
    r"(?P<decode_seconds>[0-9.]+)\s+s\s+"
    r"\((?P<decode_rate>[0-9.]+)\s+tok/s\)"
)

rows = []

for path in sorted(result_dir.glob("measured-*.stderr.txt")):
    text = path.read_text(errors="replace")
    match = pattern.search(text)

    if match is None:
        raise RuntimeError(f"Could not parse timing from {path}")

    row = {
        "run": path.stem,
        "prompt_tokens": int(match.group("prompt_tokens")),
        "prefill_seconds": float(match.group("prefill_seconds")),
        "prefill_rate": float(match.group("prefill_rate")),
        "decode_tokens": int(match.group("decode_tokens")),
        "decode_seconds": float(match.group("decode_seconds")),
        "decode_rate": float(match.group("decode_rate")),
    }

    rows.append(row)

if not rows:
    raise RuntimeError("No measured results found")

prefill_rates = [row["prefill_rate"] for row in rows]
decode_rates = [row["decode_rate"] for row in rows]
decode_latencies_ms = [
    1000.0 * row["decode_seconds"] / row["decode_tokens"]
    for row in rows
    if row["decode_tokens"] > 0
]

def summarize(name, values, unit):
    print(
        f"{name}: "
        f"median={statistics.median(values):.3f} {unit}, "
        f"min={min(values):.3f}, "
        f"max={max(values):.3f}, "
        f"mean={statistics.mean(values):.3f}, "
        f"stdev={statistics.stdev(values):.3f}"
        if len(values) > 1
        else
        f"{name}: {values:.3f} {unit}"
    )

print(f"Measured runs: {len(rows)}")
print()

for row in rows:
    latency = (
        1000.0 * row["decode_seconds"] / row["decode_tokens"]
        if row["decode_tokens"] > 0
        else float("nan")
    )

    print(
        f"{row['run']}: "
        f"prefill={row['prefill_rate']:.3f} tok/s, "
        f"decode={row['decode_rate']:.3f} tok/s, "
        f"decode_latency={latency:.3f} ms/token, "
        f"decode_tokens={row['decode_tokens']}"
    )

print()
summarize("Prefill throughput", prefill_rates, "tok/s")
summarize("Decode throughput", decode_rates, "tok/s")
summarize("Decode latency", decode_latencies_ms, "ms/token")
