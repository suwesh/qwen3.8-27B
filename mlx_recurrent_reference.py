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
last_id = token_ids[-1]

embedding = model.language_model.model.embed_tokens(
    mx.array([[last_id]])
)

normalized = model.language_model.model.layers[0].input_layernorm(
    embedding
)

mx.eval(embedding, normalized)

print("\nMLX embedding residual:")
print([float(x) for x in embedding[0, 0, :8].tolist()])

print("\nMLX embedding normalized:")
print([float(x) for x in normalized[0, 0, :8].tolist()])

cache = model.language_model.make_cache()
output = None

for position, token_id in enumerate(token_ids):
    token = mx.array([[token_id]])

    output = model.language_model(
        token,
        cache=cache,
    )

    mx.eval(output.logits)

last_logits = output.logits[0, -1, :]
mx.eval(last_logits)

next_token = int(mx.argmax(last_logits).item())

print("Prompt IDs:")
print(*token_ids)

print("\nRecurrent first-token ID:")
print(next_token)

print("\nRecurrent first token:")
print(processor.tokenizer.convert_ids_to_tokens([next_token]))

print("\nRecurrent decoded token:")
print(processor.tokenizer.decode([next_token]))

print("\nLogit token 43:")
print(float(last_logits[43].item()))

print("\nLogit token 57590:")
print(float(last_logits[57590].item()))

print("\nLogit token 90562:")
print(float(last_logits[90562].item()))

print("\nFirst 8 logits:")
print([float(x) for x in last_logits[:8].tolist()])
