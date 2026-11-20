# Fortran implementation

Qwen3.8-27B inference (MLX 8-bit checkpoint) in plain Fortran: no C, no vendor libraries. It is the
CUDA implementation ([`src-cuda`](../src-cuda)) written in Fortran. The GPU work is `do concurrent`
loops, which LFortran offloads to the GPU with `--gpu=cuda`.

```console
$ LFORTRAN_LINKER=gcc make fortran-gpu     # from the repository root; builds ./qwen-fortran-cuda
$ ./qwen-fortran-cuda -m "$MODEL_DIR" --no-think -n 128 \
    -p "Write a Fortran function that returns the n-th Fibonacci number."
```

Building needs a recent LFortran `main` (`2e12da0696` or later, with the GPU work of
[#18109](https://github.com/lfortran/lfortran/pull/18109), [#18113](https://github.com/lfortran/lfortran/pull/18113), [#18121](https://github.com/lfortran/lfortran/pull/18121), [#18122](https://github.com/lfortran/lfortran/pull/18122), [#18124](https://github.com/lfortran/lfortran/pull/18124),
[#18240](https://github.com/lfortran/lfortran/pull/18240)).
The linker defaults to clang; `LFORTRAN_LINKER=gcc` uses gcc instead.

On one H200 NVL this decodes at about 119 tok/s (the CUDA implementation: about 120.5 tok/s), and the
greedy token ids are identical to those of `src-cuda`.

## Files

| File | Lines | What |
|---|---|---|
| [`kernels.f90`](kernels.f90) | ~450 | the GPU kernels, as `do concurrent` loops |
| [`qwen.f90`](qwen.f90) | ~410 | weight loading, the per-token forward pass, greedy decoding, the CLI |
| [`qwen_tokenizer.f90`](qwen_tokenizer.f90) | ~2970 | byte-level BPE, the Qwen pre-tokenizer, NFC, the chat template, streaming detokenization (about 1300 lines are generated Unicode tables) |
| [`safetensors_io.f90`](safetensors_io.f90) | ~480 | sharded safetensors reader |
| [`qwen_config.f90`](qwen_config.f90) | ~250 | `config.json` reader and checks |
| [`json_tokens.f90`](json_tokens.f90) | ~550 | JSON parser (config, safetensors index and headers) |
| [`model_io_support.f90`](model_io_support.f90) | ~100 | error type and checked file I/O |
| [`tests/`](tests) | | tokenizer test |

## The kernels

`kernels.f90` follows `src-cuda/kernels.cu`. Each CUDA kernel becomes one `do concurrent` loop, except
that the attention and the argmax are two each:

| `src-cuda/kernels.cu` | `kernels.f90` |
|---|---|
| `embed_kernel` | `embed` |
| `gemv_kernel` (q8 GEMV with the RMSNorm, residual and SwiGLU fused in) | `gemv_norm`, `gemv_resid` |
| `gdn_kernel` | `gdn`: one value head per iteration |
| `attention_kernel` | `attn_part`: one kv head and chunk of 8 positions per iteration; `attn_combine` |
| `argmax_kernel` | `argmax_part`, `argmax_final` |

The weights stay in the checkpoint's format (U32 bytes with bf16 scales and biases per group of 64),
and matrices that multiply the same input are stacked into one, as in `src-cuda`. Activations, the
recurrent state and the KV cache are fp32.

As in `src-cuda`, the sum of `x**2` for the RMSNorm is added up by the GEMV that produces `x`, here
with a `reduce` into a 64-bit fixed-point sum, which does not depend on the order of the additions. In
`gdn` and `attn_part` an iteration works on a whole head: its steps are inner loops that the threads of
the iteration share, over arrays local to the iteration, which LFortran keeps in shared memory.

Every token, the prompt included, goes through the same single-token forward pass (`step`). The next
token is picked on the GPU and passed to the next step in device memory, so the host queues the steps
ahead of the tokens it prints and the GPU never waits for the host.

## Usage

```
./qwen-fortran-cuda [-m DIR] [-p PROMPT] [-n MAX_NEW] [-c MAX_CTX] [--no-think] [--raw] [--ids]
```

- `-m`: model directory (default `/tmp/certik/model`).
- `-p`: the prompt (default `Hello, who are you?`). It is wrapped in the model's chat template as one
  user message; the model thinks first unless `--no-think` is given. `--raw` tokenizes the prompt as is.
- `-n`: at most this many new tokens (default 512). Generation also stops at the stop tokens of
  `generation_config.json` (or `<|im_end|>` / `<|endoftext|>`).
- `-c`: context capacity in tokens (default 8192).
- `--ids`: print the prompt and output token ids to stderr.

The answer is streamed to stdout; the load, prefill and decode times go to stderr.

## Tokenizer test

`test_tokenizer.py` runs the tokenizer (through `tests/test_tokenizer.f90`) on recorded HuggingFace
`tokenizers` output (`tokenizer_cases.json`): token ids, decoding, invalid UTF-8, the chat template and
NFC. If the `tokenizers` package is installed, it also compares random texts and every code point
against it. It needs only the model's tokenizer files:

```console
$ make fortran-tokenizer-test MODEL_DIR="$MODEL_DIR"                   # LFortran
$ pixi run -e fortran-precision tokenizer-test "$MODEL_DIR"            # GFortran and Flang
```


## TODOs
- **Batched prefill (`GEMM`):** Implement a pure ISO Fortran (`do concurrent`) quantized `q8` GEMM kernel with zero external dependencies (no BLAS/OpenMP), structured for SIMD/AVX vectorization (`-march=native` / `--fast`) on `lfortran` and `gfortran` (plus optional vendor BLAS crossover benchmarks).
- **`nvfortran` compatibility:** Fix compilation/offload issues on `nvfortran`.
