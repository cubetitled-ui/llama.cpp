#!/bin/bash
# OpenMythos falsifiable bench: same prompt/seed, T=1 vs T=2/4/8 + sham
# Usage: ./scripts/openmythos_bench.sh <llama-cli> <model>
CLI=${1:-build-cuda-vnni/bin/llama-cli}
MODEL=${2:-/home/cune/models/Falcon-H1R-7B-IQ4_XS.gguf}
export LD_LIBRARY_PATH=$(dirname $(readlink -f $CLI)):$LD_LIBRARY_PATH
PROMPT="Solve step by step: A train travels 120 km at 60 km/h, then 120 km at 40 km/h. What is the average speed? Answer with a number."
for T in 1 2 4 8; do
  echo "=== T=$T ==="
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -n1
  time $CLI -m $MODEL -n 64 -p "$PROMPT" --seed 7 -t 6 --n-gpu-layers 28 \
    --recurrent-t $T --recurrent-layer 12 -fa on 2>&1 | tail -n 12
  echo "VRAM after:"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -n1
done
