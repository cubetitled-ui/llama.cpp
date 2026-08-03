# HOW IT FUCKING WORKS

The complete, no-bullshit technical guide to what `llamar.cpp` actually does under the hood.

This fork of `llama.cpp` takes ordinary transformer LLMs (LLaMA, Qwen, Gemma, Mistral, ...) and adds **inference-time recurrence** without any fine-tuning. It also carries a set of CPU/GPU speed optimizations for MoE models. Everything below is driven purely by environment variables - the model weights are never touched.

---

## 1. The Big Idea: KV-Decoupled Recurrent Layers

Normal transformers scale quadratically with context length (attention is O(n^2)). Recurrent / linear-attention models (Mamba, Gated Delta Net, RWKV) scale linearly because they carry a fixed-size **state** instead of a growing key-value cache.

**KV-Decoupled** is the hybrid trick: inside one layer you have two parallel streams

- a **full-attention** (KV cache) stream for precise, long-range lookup, and
- a **recurrent** stream (Gated Delta Net state) that compresses history into fixed-size state tensors.

The two streams are combined, then fed through the FFN. Architectures that ship this by default: Qwen3-Next (qwen35), Delta-Net variants, Gemma4, etc.

### The Gated Delta Net recurrence (in one block)

For a token `t`, the recurrent state update is:

```
S[t] = S[t-1] * g[t] + k[t] * v[t]^T        (gated state carry)
out  = S[t] * q[t] + b[t]                    (query readout)
```

where `g` is a learned gate in (0, 1), `S` is the `[S_v, S_v, H_v, n_seqs]` state tensor, `k`/`v` are the key/value projections, `q` the query, `b` a bias. This is exactly what `ggml_gated_delta_net` computes, and what `src/models/delta-net-base.cpp` wraps:

- `build_delta_net_chunking` - chunked parallel scan (prefill, `K` tokens at once)
- `build_delta_net_autoregressive` - single-token step (generation)
- `build_delta_net_fused` - fused CUDA op via `ggml_gated_delta_net`
- `build_conv_state` - the short convolutional state (`conv_kernel-1` history) that also feeds the delta net

A **hybrid** model interleaves these: some layers are delta-net (natively recurrent, linear in context), some are plain full attention. `hparams.is_recr(il)` tells you which.

---

## 2. The Core Feature: Inference-Time Recurrence

This is the real point of the fork. During generation, instead of passing each layer's output forward once, we run the layer **`iters` times** on the same input, feeding the previous iteration's output back in. It is "reasoning in place" - the hidden state is refined over multiple passes before moving to the next layer.

### Why it works (loosely)

Running a layer repeatedly on its own output is like doing several Jacobi / fixed-point iterations toward a better representation. Because every iteration reads the **same** input embedding (for the first iteration) but progressively more informed hidden states, deep reasoning emerges from a small model without training. It is the same family of idea as "test-time compute" / chain-of-thought, but applied **inside** the network instead of on the token stream.

### Euler Step Scaling (the stabilizer)

Naively iterating a layer blows up or drifts. To prevent that, after each layer iteration we blend the new hidden state with the original input using an Euler-style convex combination:

```
h_new = alpha * h_layer_out + beta * h_input       alpha + beta = 1
```

- Default `alpha = 1.0 / iters` (the more iterations, the smaller the per-step contribution)
- `RECURRENT_STEP_MODE=harmonic` overrides it with `alpha = 1.0 / (iter + 1)` - the first step moves a lot, later steps fine-tune. Theoretically guarantees fixed-point convergence.
- `RECURRENT_ALPHA` / `RECURRENT_BETA` override both constants directly.

### KV cache writes: from the LAST iteration

By default the KV cache is now written on the **final** iteration only, so the cache reflects the *refined* hidden state rather than the first noisy pass. Controlled by `RECURRENT_KV`:

| value   | behaviour                                        |
|---------|--------------------------------------------------|
| `last`  | write KV only on `iter == iters-1` (default)     |
| `first` | write KV only on `iter == 0` (old behaviour)     |
| `all`   | write KV on every iteration                      |

Implementations: `get_store_kv(iter, iters)` in `src/models/models.h`, threaded through every `build_attn(...)` call.

---

## 3. How Recurrence Is Scheduled (the math)

This is `get_recurrent_iters()` in `src/models/models.h`. It decides, per layer, how many iterations to run. Every architecture file copies the same block:

```cpp
int S = 50; int D = 12;                       // env-overridable
int k  = n_layer / 4;                          // 4 zones
int r  = n_layer % 4;
// zones: [0..k), [k..2k), [2k..3k), [3k..n_layer)
int offset = (S * (size-1)) / 100;             // S = "stability" -> percent shift
int L2 = k     + offset2;                      // anchor layers inside zones 2,3,4
int L3 = 2*k   + offset3;
int L4 = 3*k   + offset4;
int c2 = (D+3)/6;   int c3 = (D+1)/2;   int c4 = D - c2 - c3;
recurrent_iters[L2] = c2;                      // shallow
recurrent_iters[L3] = c3;                      // medium
recurrent_iters[L4] = c4;                      // deep
```

So recurrence is applied at **three anchor layers** positioned one-third, half-way and deep in the network, with increasing depth `c2 < c3 < c4`. All other layers run once (`iters = 1`).

For `D = 12`: `c2 = 2`, `c3 = 6`, `c4 = 4` (2+6+4 = 12). `S` shifts the anchors; at `S = 0` the anchors sit at the zone starts, at `S = 100` at the zone ends.

### Choosing how many layers recur (`RECURRENT_LAYERS_COUNT`)

By default recurrence hits exactly **3** anchor layers. You can change that with `RECURRENT_LAYERS_COUNT`:

```
RECURRENT_LAYERS_COUNT=8 RECURRENT_D=12   # 8 recurrent layers, depth spread across them
```

When `N != 3`, the network is split into `N` evenly-spaced zones (one recurrent layer per zone, placed by `S` inside its zone) and the total depth `D` is distributed across the `N` layers (`D / N` each, first `D % N` get one extra). All other layers still run once.

### Hybrid models (qwen35 / qwen35moe)

Recurrence only applies to **full-attention** layers. Native delta-net layers (`hparams.is_recr(il)`) already have recurrence built in, so they execute exactly once, even if the scheduler would have given them `iters > 1`. The Euler blend is likewise skipped for them.

### Full manual override

`RECURRENT_LAYERS` + `RECURRENT_DEPTHS` bypass the whole scheduler:

```
RECURRENT_LAYERS="10,20,30" RECURRENT_DEPTHS="3,6,3"
```

---

## 4. Environment Variables Reference

| variable           | default | meaning |
|--------------------|---------|---------|
| `RECURRENT_D`      | `12`    | total recurrence depth; `0` disables recurrence entirely |
| `RECURRENT_S`      | `50`    | anchor-layer placement (percent) |
| `RECURRENT_LAYERS_COUNT` | `3` | how many layers get recurrence. `3` = classic anchors; any N spreads recurrence over N evenly-spaced layers (depth `D` split across them) |
| `RECURRENT_C2/C3/C4` | auto  | per-anchor iteration counts (override the D split) |
| `RECURRENT_ALPHA`  | `1/iters` | Euler blend coefficient for the layer output |
| `RECURRENT_BETA`   | `1-alpha` | Euler blend coefficient for the input |
| `RECURRENT_STEP_MODE` | -    | `harmonic` -> adaptive alpha |
| `RECURRENT_LAYERS` | -      | comma-separated layer IDs (full override) |
| `RECURRENT_DEPTHS` | -      | comma-separated iters per layer (with LAYERS) |
| `RECURRENT_KV`     | `last` | when to write KV cache: `last` / `first` / `all` |

---

## 5. Supported Architectures

Recurrence is injected for (all in `src/models/`):

| family        | file(s)                                  |
|---------------|------------------------------------------|
| LLaMA 2/3/3.1/3.2 | `llama.cpp`                          |
| Qwen / Qwen2 / Qwen2.5 / Qwen3 | `qwen2.cpp`, `qwen3.cpp` |
| Qwen2 / Qwen3 MoE | `qwen2moe.cpp`, `qwen3moe.cpp`      |
| Qwen3.5 Next (dense) | `qwen35.cpp`                       |
| Qwen3.5 MoE    | `qwen35moe.cpp`                          |
| Gemma 2        | `gemma2.cpp`                              |
| Mistral v0.3 / Mixtral 8x22B | `mistral3.cpp`               |

`Bonsai-27B` maps to the `qwen35` (Qwen3-Next) architecture: 64 layers, embed 5120, SSM state 128, conv_kernel 4 - fully supported.

---

## 6. The Speed Optimizations (why it's fast)

1. **MoE Fused Gate-Up (`-fgu` / `--fuse-gu`)** - concatenates MoE `gate_exps` and `up_exps` into one `gate_up` matmul per layer, cutting the two serial expert groups + two barrier syncs into one. Halves token memory reads on CPU.

2. **Kahan-Compensated Recurrence** - the delta-net state update `S += delta * k` loses low-order bits over long contexts. Kahan summation tracks the lost bits in a compensation term to keep FP32 accuracy on >4K-token runs.

3. **Expert Batching (GEMM dispatch)** - `mul_mat_id` reordered weight-major: each `Q4_K_M` block is dequantized **once** and reused for all tokens routed to that expert. Guarded by `GGML_EXPERIMENTAL_BUILD`.

4. **Token Prefetching** - `__builtin_prefetch` inside the sparse expert-routing loop hides memory latency.

5. **SIMD Recurrence Loop Control (`--simdv`)** - runtime flag for experimental SIMD-vectorized recurrence inner loops.

6. **PCIe offload tuning (MoE / prefill)** - `GGML_CUDA_REGISTER_HOST=1` pins host memory for DMA; `GGML_SCHED_PREFETCH_EXPERTS=1` double-buffers next-layer experts on a second CUDA queue (requires `GGML_CUDA_DISABLE_GRAPHS=1`). ~+64% prefill on RTX 3060.

---

## 7. Where The Code Lives

| concern                | location |
|------------------------|----------|
| recurrence helpers     | `src/models/models.h` (`get_recurrent_iters`, `get_recurrent_alpha`, `get_store_kv`) |
| delta-net kernels      | `src/models/delta-net-base.cpp` |
| KV/attention backend   | `src/llama-graph.h` (`build_attn`, `store_kv`) |
| per-arch injection     | `src/models/*.cpp` (the `for iter` loop + Euler blend) |
| MoE math kernels       | `ggml/src/ggml-cpu/ops.cpp` (`mul_mat_id`, `add_q_f32`) |
| meta/split tensor cache| `ggml/src/ggml-backend-meta.cpp` |

---

## 7.5 Empirical Comparison (Bonsai-27B-Q1_0)

Trick-question benchmark, fixed seed, one prompt per config so differences come only from the recurrence layout.

| # | Task (correct answer) | D=0 | 3/D=12 | 8/D=12 | 3/D=24 | 8/D=24 |
|---|-----------------------|-----|--------|--------|--------|--------|
| 1 | 17 sheep, all but 9 run away (9) | Yes | Yes | Yes | Yes | Yes |
| 2 | 6 matchsticks -> 4 triangles (tetrahedron) | Yes | **No** | Yes | Yes | Yes |
| 3 | Bat & ball $1.10, bat $1.00 more ($0.05) | Yes | Yes | Yes | Yes | Yes |
| 4 | 5 machines -> 5 widgets in 5 min; 100->100 (5 min) | Yes | Yes | Yes | Yes | Yes |
| 5 | Sibling puzzle (7 children) | Yes | Yes | Yes | **Loop** | Yes |
| 6 | Passing trains 150+120 m (27/7 s) | Yes | **Loop** | **Loop** | Yes | Yes |
| 7 | Digit 9 in 1..100 (20) | Yes | Yes | Yes | Yes | Yes |
| 8 | Shirt $80 -25% -15% +10% tax ($56.10) | Yes | Yes | Yes | Yes | Yes |
| 9 | Doubling lily, half on which day (day 29) | Yes | Yes | Yes | Yes | Yes |

Speed: `D` is the main cost (~21.5 t/s at D=0, ~19 at D=12, ~17 at D=24). Spreading the same `D` across 8 layers vs 3 is nearly free. "Loop" = the model never produced a final answer within the token budget.

Key takeaway: recurrence is not uniformly "more = better". The default 3/D=12 is actually the weakest config on this set (misses the tetrahedron, loops on the trains). 8/D=12 fixes both of those at zero speed cost. Higher `D` (24) adds reasoning length but costs ~2.5 t/s and can itself introduce loops (siblings at 3/D=24).

---

## 8. Quickstart

```bash
# default = recurrence ON, D=12, for every supported model
./build/bin/llama-cli -m model.gguf -c 8192

# explicit deep reasoning
RECURRENT_D=12 RECURRENT_KV=last ./build/bin/llama-cli -m model.gguf -c 32468

# baseline (no recurrence at all)
RECURRENT_D=0 ./build/bin/llama-cli -m model.gguf

# hybrid delta-net model (e.g. Bonsai-27B): recurrence hits full-attn layers only
RECURRENT_D=12 ./build/bin/llama-cli -m Bonsai-27B-Q1_0.gguf -c 32468

# MoE + GPU offload
RECURRENT_D=12 ./build/bin/llama-cli -m qwen35-moe.gguf -ngl 28 --n-cpu-moe 36 -fa on -fgu -t 12
```
