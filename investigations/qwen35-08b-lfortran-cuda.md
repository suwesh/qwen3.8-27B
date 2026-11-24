# Porting Qwen3.5-0.8B MLX 8-bit to the LFortran CUDA Path

## Engineering investigation and lessons from the debugging battlefield

**Checkpoint:** `lmstudio-community/Qwen3.5-0.8B-MLX-8bit`  
**Repository:** `suwesh/qwen3.8-27B`  
**GPU:** NVIDIA GeForce MX450, 2 GB, compute capability 7.5  
**Compiler:** LFortran 0.69.0-714-gf43701e5d2  
**CUDA toolkit:** NVIDIA HPC SDK CUDA 13.3  
**Status:** Correct coherent generation established on the LFortran CUDA path

## Objective

The objective was to reproduce the existing Fortran GPU inference path using a smaller Qwen3.5-0.8B MLX 8-bit checkpoint that fits on the local MX450.

The initial expectation was that the 0.8B checkpoint would require only minor handling of multimodal tensors. In practice, the checkpoint exposed several assumptions that were specific to the original 27B configuration.

The final result was a working end-to-end LFortran CUDA inference path that generated coherent text and correctly answered the validation prompt.

## Environment setup

The installed `nvcc` came from NVIDIA HPC SDK:

```text
/opt/nvidia/hpc_sdk/Linux_x86_64/26.9/compilers/bin/nvcc


The CUDA runtime headers were under:

/opt/nvidia/hpc_sdk/Linux_x86_64/26.9/cuda/13.3/targets/x86_64-linux/include


The first LFortran build failed with:

fatal error: cuda_runtime.h: No such file or directory


The problem was not a missing CUDA compiler. LFortran invoked the host C compiler to build its CUDA runtime, but the host compiler could not see the NVIDIA HPC SDK headers.

The environment was fixed by exporting:

export HPCSDK=/opt/nvidia/hpc_sdk/Linux_x86_64/26.9
export CUDA_HOME="$HPCSDK/cuda/13.3"

export PATH="$HPCSDK/compilers/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export LFORTRAN_LINKER=gcc


This environment is preserved in:

scripts/environment/activate-qwen-env.sh

Command-line interface discovery

The Fortran executable uses short options:

-m DIR
-p PROMPT
-n MAX_NEW
-c MAX_CTX


The correct launch form is:

./qwen-fortran-cuda \
  -m "$HOME/models/Qwen3.5-0.8B-MLX-8bit" \
  --no-think \
  -n 16 \
  -p "What is the capital of France? Answer briefly."


The longer options used by other repository implementations are not accepted by this executable.

Checkpoint incompatibilities discovered
Missing output_gate_type

The original configuration parser required:

"output_gate_type": "swish"


The 0.8B checkpoint does not contain this field.

The model still uses compatible swish or SiLU gating semantics. The field therefore needs to be optional, while an explicitly present unsupported value should continue to be rejected.

Simply commenting out the check allowed investigation to proceed, but the proper implementation should parse the field optionally and retain swish as the supported behavior.

F32 A_log tensors

The original SafeTensors loader accepted only:

BF16
U32


The 0.8B checkpoint stores the Gated DeltaNet A_log tensors as:

F32


For example:

language_model.model.layers.0.linear_attn.A_log
dtype: F32
shape: (16,)


Support was added for:

F32 metadata;
four-byte F32 payload sizes;
read_f32;
read_f32_flat;
a dedicated vec_f32 path;
loading A_log through the F32 reader.

The loader was validated against Python SafeTensors. The first eight values matched exactly:

0.2578125
-2.03125
-2.125
-0.56640625
-2.609375
-2.25
-1.8046875
-2.984375


This ruled out byte-order, offset, conversion, and F32 payload errors.

Tied input and output embeddings

The checkpoint does not contain:

language_model.lm_head.weight


The 0.8B configuration uses tied embeddings, so the output projection reuses the input embedding matrix.

The compatibility path falls back to:

language_model.model.embed_tokens


when a separate LM head is absent.

The current implementation loads the embedding-backed quantized matrix twice. This is correct enough for functional validation but wastes memory. A future implementation should share or alias the storage.

Correctness investigation
Prompt tokenization matched

The Fortran prompt IDs matched the Transformers reference exactly:

248045 846 198 3710 369 279 6511 314 9338 30
21134 25899 13 248046 198 248045 74455 198
248068 271 248069 271


This ruled out:

the Fortran tokenizer;
the chat template;
prompt formatting;
input-token alignment.
The detokenizer was not responsible

The incorrect Fortran output IDs were decoded using the reference tokenizer.

The reference tokenizer produced the same gibberish, proving that the Fortran model was generating incorrect token IDs. The problem was upstream of detokenization.

MLX reference execution

Both MLX execution styles were tested:

whole-prompt or chunked execution;
recurrent execution one token at a time with a hybrid cache.

Both selected:

57590 = Paris


The incorrect Fortran path selected:

90562 = importe


This ruled out chunked-versus-recurrent prefill as the explanation.

Argmax was correct

Before the final fix, the Fortran logits contained:

logit[43]    = -1.397
logit[90562] = 12.102


The argmax implementation correctly selected token 90562. The error was already present in the logits.

Embedding loading was correct

The last prompt token’s embedding residual was compared between MLX and Fortran.

MLX began with:

0.0048828125
0.01220703125
0.00390625
-0.03857421875


Fortran began with:

0.00469970703
0.0120887756
0.00413131714
-0.0384979248


The values were sufficiently close to rule out a gross embedding layout or quantization-loading error.

Layer-by-layer comparison

All 24 post-layer residuals were captured in both MLX and Fortran.

The first Gated DeltaNet layers were relatively close, while divergence grew sharply around the full-attention layers.

This shifted attention from:

tokenization;
embeddings;
F32 loading;
output-head handling;
argmax;

toward the full-attention configuration.

GPU overlap was ruled out

The model was built both with and without:

--gpu-overlap


The following were identical:

per-head values;
per-layer residuals;
final normalized hidden state;
output token IDs.

Therefore, asynchronous GPU overlap was not the cause.

Root cause

The full-attention implementation contained:

integer, parameter :: AG = 6


AG is the number of query heads mapped to each KV head.

The original 27B configuration uses six query heads per KV head.

The 0.8B checkpoint uses:

8 query heads
2 KV heads
8 / 2 = 4 query heads per KV head


Therefore, the 0.8B model requires:

integer, parameter :: AG = 4


Using AG=6 corrupted every full-attention layer even though the model loaded successfully and all kernels executed.

After changing the attention grouping to four, the program generated coherent text immediately.

Correct execution result

Validation prompt:

What is the capital of France? Answer in a poem


Generated output:

The golden sun rises high,
Across the rolling hills of France.
The Seine flows through the heart,
Where ancient kings and queens meet.
The capital, the jewel of the land,
Is Paris, where the light is found.
From the Seine to the Seine,
The city stands as a beacon.


Preliminary valid single-run result:

prompt: 23 tokens
prefill: 23 tokens in 12.889 s, 1.8 tok/s
decode: 66 tokens in 2.656 s, 24.8 tok/s


This result is a valid correctness smoke test and preliminary performance observation. It is not yet the controlled median benchmark.

Important current limitation

The current AG=4 change is checkpoint-specific.

It fixes Qwen3.5-0.8B but does not preserve the original Qwen3.5-27B configuration, which requires AG=6.

The proper implementation should derive:

AG = num_attention_heads / num_key_value_heads


from config.json.

The derived grouping must then be passed into the attention implementation instead of remaining a compile-time constant.

This limitation must be communicated clearly before presenting the change as suitable for upstream integration.

Lessons from the battlefield
Loading successfully does not prove compatibility

The model loaded all 24 layers and completed inference while still producing incorrect logits.

A successful build and complete execution path only prove that the program ran. They do not prove that it executed the intended model mathematics.

Model family does not eliminate configuration differences

Both checkpoints belong to Qwen3.5, but they differ in:

attention head grouping;
tied versus separate output embeddings;
auxiliary tensor dtypes;
explicit versus omitted configuration fields.

A model-family implementation must read these differences from configuration rather than encode one checkpoint’s values as global constants.

An 8-bit checkpoint still contains higher-precision tensors

The large projection weights are quantized, but small or numerically sensitive tensors can remain BF16 or F32.

The term “8-bit checkpoint” does not mean every tensor uses an 8-bit dtype.

Correctness should be localized through boundaries

The useful debugging order was:

prompt IDs
→ output-ID decoding
→ reference first token
→ reference logits
→ F32 payload values
→ embeddings
→ layer residuals
→ attention configuration


This progressively eliminated entire classes of causes.

Same-shape tensors do not guarantee correct semantics

The embedding and LM-head fallback had matching dimensions, but shape agreement alone was insufficient evidence.

Configuration and execution semantics still needed validation.

Performance numbers require correct computation

The incorrect model initially produced higher decode rates near 31–34 steps/s.

Those numbers were invalid as inference benchmarks because the model selected the wrong first token.

The first meaningful local result is the coherent run at approximately 24.8 decode tok/s. Controlled warm-up and repeated measurements are still required.

Generalization should follow understanding

The investigation revealed which parts need to become configuration-driven:

attention grouping;
tied embeddings;
optional configuration fields;
tensor dtypes.

Abstracting the runtime before finding these differences would only have hidden incorrect assumptions behind new interfaces.

Reusable investigation files

Reference scripts:

tests/reference/mlx_reference.py
tests/reference/mlx_recurrent_reference.py
tests/reference/mlx_logits_probe.py
tests/reference/mlx_all_layers.py


Diagnostic scripts:

tests/diagnostics/mlx_hidden_probe.py
tests/diagnostics/mlx_pos0_layer0.py


Environment:

scripts/environment/activate-qwen-env.sh


Benchmark and compiler experiments:

scripts/benchmarks/compare_gpu_overlap.sh


Generated logs are stored locally under:

artifacts/qwen35-08b/


The artifacts directory is excluded from Git.

Next steps
Remove all temporary diagnostic prints from the production Fortran source.
Make query-to-KV attention grouping configuration-driven.
Preserve both the original 27B behavior and the new 0.8B behavior.
Run at least three greedy correctness prompts.
Establish the LFortran CUDA median baseline using two warm-ups and five measured runs.
Adapt and validate the C+CUDA path for the same checkpoint.
Build CPU baselines with GFortran, ifx, LFortran, and NVFORTRAN.
Test NVFORTRAN GPU offloading only after output correctness is established.
Communicate the result and checkpoint-specific limitation to Certík before proposing upstream integration. 

