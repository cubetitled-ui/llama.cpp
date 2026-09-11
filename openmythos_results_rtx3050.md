# OpenMythos on RTX 3050 6GB: measured results (Falcon-H1R-7B-IQ4_XS)

Hardware: NVIDIA GeForce RTX 3050 6GB Laptop GPU (5803 MiB usable), 6 CPU threads, CUDA sm_86.
Model: Falcon-H1R-7B-IQ4_XS.gguf (3.88 GiB, hybrid Mamba+attention, reasoning model).
Build: llama.cpp b10262 + recurrent-core patch (`build_recurrent_core`, single + A->B->A + gate).
Flags: `-ngl 28 -t 6`, seed fixed, `--reasoning-budget 0` for generation tests.

## 1. Throughput (llama-bench, p16/tg32)

| config | pp16 (t/s) | tg32 (t/s) |
|---|---:|---:|
| T=1 (vanilla) | 34.29 | 12.21 |
| T=3 legacy | 30.49 (-11%) | 11.39 (-7%) |

Extra passes cost ~7-11%: the graph executes, no crash.

## 2. Stability (perplexity, same text, n_ctx=512)

| config | PPL | note |
|---|---:|---|
| T=1 | 10.61 +/- 0.53 | baseline |
| T=2 legacy (raw h+e sum) | 1292 | DIVERGED |
| T=3 legacy | NaN (247..600 per-chunk) | DIVERGED, residual explosion |
| T=2 + input-RMSNorm, gate=1.0 | 13.13 | stable, +24% |
| T=3 + input-RMSNorm, gate=1.0 | 15.13 | stable, +43% |
| T=3 + input-RMSNorm, gate=0.2 | 10.64 +/- 0.54 | stable, == baseline |

Findings:
- Naive `h = A*h + B*e + F(h+e)` diverges on a pretrained hybrid checkpoint from the FIRST
  extra pass (T=2 already garbage). An output RMSNorm does NOT fix it (tested PPL 1390):
  re-scaling the whole residual stream breaks every coda layer.
- Fix that works: `combined = RMSNorm(h + e)`, `h = A*h + B*e + gate*F(combined)`.
  Input normalization keeps the block on-distribution; no output norm preserves stream scale.
- gate=0.2 keeps PPL at baseline while still changing the trajectory (see reasoning tests).

## 3. Reasoning spot-checks (same prompt, seed=7, generation)

Q1: train 120km@60 + 120km@40, average speed? (truth: 48)
- T=1: "1.5" (wrong)
- T=3 gate=0.2: full 4-step solution -> "48 km/h" (right)

Q2: book costs 10 + half its price? (truth: 20)
- T=1: rambles toward 15 (wrong)
- T=3 gate=0.2: sets up P = 10 + P/2 -> "20" (right)

Status: 2/2 anecdotes favor recurrence. NOT a benchmark: no statistics, no blind judging,
no GSM8K/MBPP harness yet. Next: scripted harness (>=100 tasks), sham-controller control,
bootstrap CIs.

## 4. Memory

Peak VRAM during T=3 generation: 4724 MiB / 6144 MiB. Weights 3.88 GiB + context/scratch fit
with ~1.4 GB headroom. Temporary latent state adds no persistent weights; KV isolation for
intermediate passes is still open (overwrite semantics observed, dedicated scratch pending).

## 5. What failed (kept, not hidden)

- Toy fact-recovery: recurrence cannot restore information lost to aggressive compression
  (16-dim bottleneck of 256-dim code: baseline 0.825 -> recurrence 0.49 collapse; anchored
  0.81; model-based iteration flat). Extra compute != new facts.
- Branching/rollback/verifier: designed, not yet implemented or measured.
- A->B->A alternating core: implemented behind flags, NOT yet benchmarked for quality.

## Reproduce

```bash
cmake --build build-cuda-vnni --target llama-bench llama-perplexity llama-cli -j1
export LD_LIBRARY_PATH=$(pwd)/build-cuda-vnni/bin
./build-cuda-vnni/bin/llama-bench -m MODEL -ngl 28 -t 6 -p 16 -n 32 --recurrent-t 3
./build-cuda-vnni/bin/llama-perplexity -m MODEL -f TEXT -ngl 28 -t 6 \
  --recurrent-t 3 --recurrent-gate 0.2
printf '/exit\n' | ./build-cuda-vnni/bin/llama-cli -m MODEL -n 300 -p "PROMPT" \
  --seed 7 -t 6 --n-gpu-layers 28 --recurrent-t 3 --recurrent-gate 0.2 \
  --reasoning-budget 0 --no-conversation
```
