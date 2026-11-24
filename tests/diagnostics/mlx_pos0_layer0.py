from pathlib import Path

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template

model_dir = Path.home() / "models" / "Qwen3.5-0.8B-MLX-8bit"
prompt = "What is the capital of France? Answer briefly."

model, processor = load(str(model_dir))

formatted_prompt = apply_chat_template(
    processor,
    model.config,
    prompt,
    num_images=0,
)

token_ids = processor.tokenizer.encode(
    formatted_prompt,
    add_special_tokens=False,
)

first_token = mx.array([[token_ids[0]]])
cache = model.language_model.make_cache()
captured = []

base_output = model.language_model.model(
    first_token,
    cache=cache,
    capture_layer_ids=[0],
    hidden_sink=captured,
)

mx.eval(base_output)

print("first_prompt_token_id:", token_ids[0])
print("captured_entries:", len(captured))

if len(captured) != 1:
    raise RuntimeError(
        f"Expected one captured layer, got {len(captured)}"
    )

layer0 = captured[0]
mx.eval(layer0)

values = layer0[0, -1, :8]
mx.eval(values)

print(
    "mlx_pos0_layer0",
    " ".join(
        f"{float(value):.8e}"
        for value in values.tolist()
    ),
)
