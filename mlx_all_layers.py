from pathlib import Path

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template

MODEL_DIR = Path.home() / "models" / "Qwen3.5-0.8B-MLX-8bit"
PROMPT = "What is the capital of France? Answer briefly."
NUM_LAYERS = 24
VALUES = 8

model, processor = load(str(MODEL_DIR))

formatted_prompt = apply_chat_template(
    processor,
    model.config,
    PROMPT,
    num_images=0,
)

token_ids = processor.tokenizer.encode(
    formatted_prompt,
    add_special_tokens=False,
)

inputs = mx.array([token_ids])
cache = model.language_model.make_cache()

captured = []

final_hidden = model.language_model.model(
    inputs,
    cache=cache,
    capture_layer_ids=list(range(NUM_LAYERS)),
    hidden_sink=captured,
)

mx.eval(final_hidden)

if len(captured) != NUM_LAYERS:
    raise RuntimeError(
        f"Expected {NUM_LAYERS} captured layers, got {len(captured)}"
    )

print(f"prompt_tokens={len(token_ids)}")
print(f"captured_layers={len(captured)}")
print()

for layer_id, hidden in enumerate(captured):
    mx.eval(hidden)

    if hidden.shape != (1, len(token_ids), 1024):
        print(
            f"layer={layer_id:02d} "
            f"unexpected_shape={hidden.shape}"
        )
        continue

    residual = hidden[0, -1, :]
    mx.eval(residual)

    values = [float(x) for x in residual[:VALUES].tolist()]
    formatted = " ".join(f"{x:.9e}" for x in values)

    print(f"layer={layer_id:02d} {formatted}")

final_last = final_hidden[0, -1, :]
mx.eval(final_last)

values = [float(x) for x in final_last[:VALUES].tolist()]
formatted = " ".join(f"{x:.9e}" for x in values)

print()
print(f"final_normalized {formatted}")
