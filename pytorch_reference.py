from pathlib import Path

import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    MetalConfig,
)

model_dir = Path.home() / "models" / "Qwen3.5-0.8B-MLX-8bit"
prompt = "What is the capital of France? Answer briefly."

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    model_dir,
    trust_remote_code=True,
)

print("Loading and dequantizing MLX checkpoint on CPU...")

quantization_config = MetalConfig(
    dequantize=True,
)

model = AutoModelForMultimodalLM.from_pretrained(
    model_dir,
    device_map="cpu",
    torch_dtype=torch.float32,
    quantization_config=quantization_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

model.eval()

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt,
            }
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
)

inputs = {
    name: value.to("cpu") if isinstance(value, torch.Tensor) else value
    for name, value in inputs.items()
}

prompt_ids = inputs["input_ids"][0].tolist()
prompt_length = inputs["input_ids"].shape[1]

print("\nPrompt token count:")
print(len(prompt_ids))

print("\nPrompt token IDs:")
print(*prompt_ids)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        use_cache=True,
    )

generated_ids = outputs[0, prompt_length:].tolist()

print("\nGenerated token IDs:")
print(*generated_ids)

print("\nFirst generated token ID:")
print(generated_ids[0] if generated_ids else "No token generated")

print("\nGenerated tokens:")
print(
    processor.tokenizer.convert_ids_to_tokens(
        generated_ids
    )
)

print("\nGenerated text:")
print(
    processor.decode(
        generated_ids,
        skip_special_tokens=False,
    )
)
