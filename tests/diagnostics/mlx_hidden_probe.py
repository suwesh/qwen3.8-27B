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

input_ids = processor.tokenizer.encode(
    formatted_prompt,
    add_special_tokens=False,
)

inputs = mx.array([input_ids])
captured = []
cache = model.language_model.make_cache()
base_output = model.language_model.model(
    inputs,
    cache=cache,
    capture_layer_ids=[23],
    hidden_sink=captured,
)

print("Input shape:", inputs.shape)
print("Base output type:", type(base_output))
print("Base output is None:", base_output is None)
print("Captured entries:", len(captured))

for index, entry in enumerate(captured):
    print(f"\nCaptured entry {index}")
    print("Type:", type(entry))

    if entry is None:
        print("Value: None")
        continue

    if isinstance(entry, (tuple, list)):
        print("Elements:", len(entry))

        for subindex, value in enumerate(entry):
            print(
                f"  Element {subindex}:",
                "type =", type(value),
                "shape =", getattr(value, "shape", None),
            )

            if value is None or not hasattr(value, "shape"):
                continue

            mx.eval(value)

            if len(value.shape) == 3:
                last = value[0, -1, :]
                mx.eval(last)

                print(
                    "  Last-position first 8:",
                    [float(x) for x in last[:8].tolist()],
                )

        continue

    print("Shape:", getattr(entry, "shape", None))

    if hasattr(entry, "shape"):
        mx.eval(entry)

        if len(entry.shape) == 3:
            last = entry[0, -1, :]
            mx.eval(last)

            print(
                "Last-position first 8:",
                [float(x) for x in last[:8].tolist()],
            )

print("\nBase model attributes containing norm:")
print("\nBase output shape:", base_output.shape)

final_hidden = base_output[0, -1, :]
mx.eval(final_hidden)

print(
    "Final normalized hidden first 8:",
    [float(x) for x in final_hidden[:8].tolist()],
)
for name in dir(model.language_model.model):
    if "norm" in name.lower():
        print(name)
