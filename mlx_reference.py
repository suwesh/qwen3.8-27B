from pathlib import Path

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model_dir = Path.home() / "models" / "Qwen3.5-0.8B-MLX-8bit"

model, processor = load(str(model_dir))

prompt = "What is the capital of France? Answer briefly."

formatted_prompt = apply_chat_template(
    processor,
    model.config,
    prompt,
    num_images=0,
)

result = generate(
    model,
    processor,
    formatted_prompt,
    max_tokens=16,
    temperature=0.0,
    verbose=False,
)

if isinstance(result, str):
    generated_text = result
elif hasattr(result, "text"):
    generated_text = result.text
else:
    generated_text = str(result)

output_ids = processor.tokenizer.encode(
    generated_text,
    add_special_tokens=False,
)

print("Generated text:")
print(generated_text)

print("\nGenerated token IDs:")
print(*output_ids)

print("\nGenerated tokens:")
print(processor.tokenizer.convert_ids_to_tokens(output_ids))
