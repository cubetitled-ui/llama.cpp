# Fact recovery under aggressive compression: toy result

## Setup
- fact: 32 bits distributed via random codebook into dim=256
- observation: x = facts @ code + noise
- compression: z = x @ proj, bottleneck B = 8/16/32/64
- decoder: linear, trained only on compressed z
- upper bound: linear decoder on uncompressed x = 1.0000

## Results
- B=8 (ratio 0.031): baseline 0.6775 -> recurrence collapses to 0.4976 by step 4
- B=16 (ratio 0.062): baseline 0.8255 -> step1 0.7773 -> step8 0.4976
- B=32 (ratio 0.125): baseline 0.9985 -> step2 0.8885 -> step4 0.4976
- B=64 (ratio 0.250): baseline 1.0000 -> step3 collapse to 0.4976

Anchored recurrence (h = A*h + B*e + correction):
- A=0.9 B=0.1 -> final 0.6606 (worse than baseline 0.8255)
- A=0.5 B=0.5 -> final 0.8065 (still worse)
- A=0.0 B=1.0 -> final 0.8192 (anchor only, correction hurts)

Model-based iteration (forward model code+proj + gradient step):
- 8 steps: 0.8255 -> 0.8255 (no gain, stuck)

## Technical conclusion
1. Naive recurrence `state = (1-g)*state + g*recon` destroys information: latent_norm shrinks
   45.8 -> 6.1, accuracy -> chance level. This is attractor collapse, not denoising.
2. Anchor injection (OpenMythos `h=A*h+B*e`) prevents total collapse but does NOT recover
   lost bits: best anchored result still below baseline.
3. Even with perfect forward model, iterative refinement without extra prior/signal cannot
   exceed the compressed linear baseline. Information-theoretic loss is not undone by compute.
4. Implication for LLM on 6GB: quantization/KV-compression loss cannot be recovered by extra
   loops alone. Recursion helps only for *composition/search/planning*, not for *missing facts*.
   To compensate compression, loops need an error signal (verifier, uncompressed anchor,
   external evidence), not just more passes.

## Where I was wrong / right
- Wrong: "recurrence can reconstruct scattered fact" — false in this toy without extra signal.
- Right: single-stream recurrence converges to wrong attractor; anchor is necessary but not sufficient.
- Right: equal-FLOP branching costs more memory; verifier quality bounds rollback gain.

## Next code step in llama.cpp
Do NOT add A->B->A. Implement:
1. `--recurrent-no-kv-write` scratch isolation (attention + SSM temp state)
2. best-state rollback behind flag, score = task verifier first (math/code), hidden-cosine only as tie-break
3. instrumentation: branch separation, verifier score vs correctness, KV bytes, peak VRAM
4. Benchmark: T=1 vs T=2/4/8 on GSM8K/MBPP subsets, same seed/prompt, plus sham-controller control.
