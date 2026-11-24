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

input_ids = mx.array([input_ids])

with mx.stream(mx.default_stream(mx.default_device())):
    output = model.language_model(input_ids)

print("Output type:", type(output))
logits = output.logits
last_logits = logits[0, -1, :]

mx.eval(last_logits)

top_token = int(mx.argmax(last_logits).item())

print("Logits shape:", logits.shape)
print("Top token ID:", top_token)
print("Top token:", processor.tokenizer.convert_ids_to_tokens([top_token]))
print("Decoded top token:", processor.tokenizer.decode([top_token]))

print("Logit token 43:", float(last_logits[43].item()))
print("Logit token 90562:", float(last_logits[90562].item()))

print("First 8 logits:")
print([float(x) for x in last_logits[:8].tolist()])