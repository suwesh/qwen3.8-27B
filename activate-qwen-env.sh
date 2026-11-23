#!/usr/bin/env bash

export HPCSDK=/opt/nvidia/hpc_sdk/Linux_x86_64/26.9
export CUDA_HOME="$HPCSDK/cuda/13.3"

export PATH="$HPCSDK/compilers/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export LFORTRAN_LINKER=gcc
export QWEN_MODEL_DIR="$HOME/models/Qwen3.5-0.8B-MLX-8bit"

echo "Qwen Fortran GPU environment activated"
echo "CUDA_HOME=$CUDA_HOME"
echo "MODEL=$QWEN_MODEL_DIR"

command -v lfortran
command -v nvcc
