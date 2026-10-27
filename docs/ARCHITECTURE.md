# How It Works: Weight-Tied Recurrent Transformer Layers in llama.cpp

This document describes the inference-time recurrence implemented in the `llamar.cpp` fork: its rationale, the exact equations executed, the graph construction in the model builders, and the stability properties of the scheme. It is the authoritative reference for the recurrence mechanism; the README is the quick overview.

## 1. Problem statement

A standard transformer applies every layer exactly once per token. The effective depth — and therefore the reasoning capacity — of a single forward pass is fixed by the architecture, independent of token difficulty. The usual workaround is to spend reasoning effort in *context space*: the model emits many intermediate tokens, which grow the KV cache quadratically in the number of such tokens and increase latency linearly.

The alternative explored here is to spend effort in *latent space*: apply a subset of the network more than once per token, without allocating any new parameters and without feeding additional tokens to the attention cache. Weight-tied recurrence reuses one existing layer as the "reasoning core" and iterates it, so the effective depth of that stage becomes an input-dependent hyperparameter.

## 2. Design overview

The network of `N` layers is split into three sequential stages for each token:

```
Layers:   [ 0 .. lo-1 ]    [ lo .. hi ] recurred T times     [ bridge ]     [ hi+1 .. N-1 ]
Stage:    prelude (1 pass) recurrent core (T passes)         (sandwich only) coda (1 pass)
```

- **Prelude** decodes the token through the lower layers once. Its output `e` is the **frozen anchor**: computed once per token and constant for the whole recurrence.
- **Recurrent core** is the layer interval `[lo .. hi]`, applied `T` times. `T = 1` takes the plain single-pass path and builds the same graph upstream builds. `T > 1` runs pass 0 with `store_kv = true` and passes `1 .. T-1` with `store_kv = false`, feeding each pass the previous pass's own output.
- **Bridge** (sandwich topology only, see §3.3) is a single layer strictly inside `(lo, hi)` that runs once between the loop and the coda.
- **Coda** decodes the final recurrent state through the upper layers once, then a normalisation and the LM head produce logits.

`lo`/`hi` come from `recurrent_layer` / `recurrent_layer_b`. A negative `recurrent_layer` with `T > 1` resolves to `0.38 * n_layer` at context creation; `0.38` is a heuristic constant, not a fitted value.

## 3. Mathematics

Let `F_l(h)` be the forward function of layer `l`. For the architectures involved `F_l` is a residual block; for example, an attention-SSM-FFN hybrid layer computes

```
F_l(h) = h + FFN(RMSNorm( Attn(RMSNorm(h)) + SSM(RMSNorm(h)) ))
```

where the SSM term is present only for hybrid architectures (Falcon-H1, Qwen3.5 DeltaNet). The core is resolved from `llama_context_params` (CLI `--recurrent-layer`, `--recurrent-layer-b`, `--recurrent-bridge`, `--recurrent-t`, `--recurrent-gate`, `--recurrent-mode`), validated in the `llama_context` constructor, and consumed by the single builder entry point `build_recurrent_core()` in `src/llama-graph.cpp`. Builders never read the environment.

### 3.1 Prelude

```
h_0  = embed(x)
e    = F_{lo-1}( ... F_1(F_0(h_0)) ... )           // anchor, frozen
```

### 3.2 Recurrent core

Pass 0 is the plain forward pass through the core:

```
Delta_0 = F_hi( ... F_{lo}(e) ... ) - e
h       = e + Delta_0
```

For `t = 1 .. T-1` the core is re-applied to its own output:

```
Delta_t        = F_hi( ... F_{lo}(h_t) ... ) - h_t      // store_kv = false
Delta_used     = modify_mode(Delta_t, Delta_0, h_t, e)   // identity for mode 0
h_{t+1}        = h_t + gamma * Delta_used
```

`gamma` is `recurrent_gate`, constrained to `[0, 1]`. The same tensors `F_lo .. F_hi` (attention weights, SSM, FFN) are reused for every pass -- hence *weight-tied*.

There is no `A*h_t + B*e` term. The earlier LTI formulation this document used to describe has been removed from the code together with the `recurrent_a`/`recurrent_b` parameters, because nothing in the graph used the anchor after pass 0 and no measurement in this repository justified a decay/anchor-injection pair.

`modify_mode` is the only place the modes differ:

| mode | `Delta_used` |
|---|---|
| 0 `vanilla` | `Delta_t` |
| 1 `orsd` | `RMSNorm(Delta_t - sum_{j<t} (<Delta_t,u_j>/max(<u_j,u_j>,1e-7)) u_j) * RMS(Delta_0)`, `u_0 = Delta_0` |
| 2 `snc` | `RMSNorm(0.6*m_{t-1} + 0.4*Delta_t)`, `m_0 = Delta_0` |
| 3 `cav` | `Delta_t * clamp(cos(h_t, e), 0, 1)` |
| 4 `dscc` | `Delta_t + 0.2*(RMSNorm(0.65*v + 0.35*Delta_t) - h_t)`, `v_0 = h_0` |

`RMSNorm` in this table is `ggml_rms_norm`, i.e. plain RMS normalization with **no learned weight**: it maps `x` to `x / sqrt(mean(x^2) + eps)`. Modes 1, 2 and 4 therefore replace the true magnitude of the block difference with a magnitude chosen by the formula. Modes 5 (`hybrid`) and 6 (`akr`) do not exist: there is no branch for them in `build_recurrent_core()`, so `parse_recurrent_mode()` in `common/arg.cpp` and the context constructor both refuse them instead of letting them run as mode 0.

### 3.3 Sandwich topology

`recurrent_layer_b` widens the core to `[min(A,B) .. max(A,B)]` and every layer in that range runs each pass. When `recurrent_bridge` is set, or when `|A - B| == 2`, the core instead loops only over `{lo, hi}` and the layer in `(lo, hi)` runs exactly once between the loop and the coda. The context constructor rejects a bridge that is not strictly inside `(lo, hi)`, so no layer is ever executed twice for one token.

### 3.4 Coda and output

```
c   = F_{N-1}( ... F_{hi+2}(F_{hi+1}(h_T)) ... )
out = LMHead(RMSNorm(c))
```

## 4. What the scheme does and does not guarantee

- **KV cache isolation is exact, not approximate.** Pass 0 is the only pass with `store_kv = true`, and the bridge and coda also run with `store_kv = true`, once each. A token therefore writes its K/V exactly once per layer, so a downstream token reads the history a single-pass forward pass would have left.
- **Boundedness is not a design property of mode 0.** The update is `h + gamma * (block(h) - h)` with `gamma <= 1`, and nothing in the code bounds the growth of `block(h) - h` over repeated application. The only stability trace stored in this repository (`docs/openmythos/openmythos_results_rtx3050.md`) measures the **removed** LTI variant (`T=2 legacy (raw h+e sum)`, perplexity `10.61 -> 1292`; `T=3 legacy`, `NaN`) and an input-RMSNorm variant of it, not the update rule that is in the graph now. So it establishes that naive multi-pass re-execution of a pretrained block can diverge; it says nothing about the current code, which has never been measured for perplexity. Modes 1, 2 and 4 bound the magnitude of the injected update by construction (their RMS normalization caps the gain at `1/sqrt(eps)`), but bounding the update is not the same as bounding the state, because `block(h)` itself is re-evaluated at the new `h`.
- **No distributional argument survives.** The old RMSNorm scale-invariance proof in this document assumed the core input was `h + e`. It is not: passes `1 .. T-1` feed the core `h_t` alone. The identity has been removed rather than repaired.

## 5. Graph construction (exact code path)

`build_recurrent_core()` lives in `src/llama-graph.cpp` and is invoked from the model builders; `src/llama-context.cpp` holds the validation.

### 5.1 Configuration (context level)

The recurrence parameters live in `llama_context_params` (`recurrent_t`, `recurrent_layer`, `recurrent_layer_b`, `recurrent_bridge`, `recurrent_gate`, `recurrent_mode`); the CLI tools expose them as `--recurrent-*` flags with `LLAMA_ARG_RECURRENT_*` environment fallback, and `--recurrent-config` reads them from a JSON or `.rlang` file. They are resolved and validated once in the `llama_context` constructor (`src/llama-context.cpp`): a negative `recurrent_layer` with `recurrent_t > 1` resolves to `0.38 * n_layer`; an out-of-range layer A, layer B or bridge, a `recurrent_gate` outside `[0, 1]`, and a `recurrent_mode` outside `0..4` all abort with a diagnostic; and one `llamar.cpp: recurrent core enabled` line reports the resolved configuration. Builders never touch the environment.

### 5.2 Decoder closure

A single decoder closure takes a layer index, an input tensor and a `store_kv` flag, and returns the layer output (SSM, attention, aggregation, FFN). For the recurrent core this closure is invoked `T` times with the same layer indices, which is what makes the core weight-tied in the compute graph.

### 5.3 Shared control flow

```cpp
// src/llama-graph.cpp
if (recurrent_t <= 1 || recurrent_layer < 0) {
    for (int il = 0; il < n_layer; ++il) inpL = decoder(il, inpL, true);
    return inpL;
}
// prelude
for (int il = 0; il < lo; ++il) inpL = decoder(il, inpL, true);
ggml_tensor * anchor_e = inpL;
// pass 0: the only store_kv = true pass through the core
for (int il : core_layers) cur = decoder(il, cur, true);
ggml_tensor * delta_0 = ggml_sub(ctx0, h, anchor_e);
for (int t = 1; t < recurrent_t; ++t) {
    for (int il : core_layers) cur = decoder(il, cur, false);   // store_kv = false
    ggml_tensor * delta_t = ggml_sub(ctx0, block_out, step_in);
    delta_t = modify_mode(delta_t);                            // mode 0: unchanged
    h = ggml_add(ctx0, h, ggml_scale(ctx0, delta_t, recurrent_gate));
}
if (bridge_layer >= 0) inpL = decoder(bridge_layer, inpL, true);
for (int il = hi + 1; il < n_layer; ++il) inpL = decoder(il, inpL, true);
```

`core_layers` is `[lo .. hi]`, or `{lo, hi}` in the sandwich topology. The function also takes an optional `on_entry(il, inpL)` hook fired before each layer decode; `qwen35.cpp` uses it to record `res->t_layer_inp[il]` for `embeddings_nextn` extraction, which the vanilla loop also performed. The vanilla branch is taken whenever `recurrent_t = 1` (the default) or the layer did not resolve, so default inference is graph-identical to upstream.

## 6. Empirical status

**There is no quality measurement of the recurrent core in this repository.** The throughput table this section used to carry (Falcon-H1R-7B-IQ4_XS, `149.5`/`18.5` t/s baseline against `111.4`/`15.6` t/s at `--recurrent-t 3 --recurrent-layer 16`) was removed: no log of that run is stored, and the two rows are single measurements with no repeats and no variance. What is stored is the perplexity trace in `docs/openmythos/openmythos_results_rtx3050.md`, and it points the wrong way for `T > 1`.

The cost of the feature is known structurally rather than measured: each additional pass re-executes the core, so the per-token compute of `T` passes is the single-pass cost plus `(T-1)` core executions. There is no tuned `gamma`, no tuned `T`, and no tuned mode in this repository; the defaults are what the code was written against.

**Effective T.** An operational definition of where the loop stops paying is `dppl(T+1) < eps` for a fixed input distribution: the smallest `T` beyond which the output distribution no longer moves measurably. Because factual recall and serial reasoning are expected to exhibit *different* convergence profiles (monotone degradation vs. dip-then-saturate), the eval run must measure two curves -- perplexity on held-out text and task accuracy -- on the same sweep of `T`. That run has not been done.

## 7. Known limitations

- **Latency vs. reasoning trade-off.** Each pass adds a full execution of the core; total cost scales linearly with `T`. Since `T = 1` is always the fastest configuration, any product claim is conditional on measured quality at a constrained latency/KV budget -- and that measurement does not exist here.
- **Core selection is a heuristic.** `0.38 * n_layer` is a constant, not a fitted or validated layer choice. No layer sweep result is stored in this repository.
- **Modes 1-4 are unmeasured.** They alter the injected update and none of them has a quality measurement behind it. Mode 4 in particular is scale-degenerate: `v` is resealed to unit RMS while `h` is the full residual stream, so the update collapses to approximately `Delta_t - 0.2*h_t`. See `README.md` for the per-mode table.
- **SSM hybrid layers.** For architectures with native SSM state, the native SSM state is re-evaluated per pass; only `h` and the mode-specific accumulators carry memory across passes.
- **Fixed `T` per graph.** Per-token adaptive depth is not possible on a static scheduled graph; per-*chunk* adaptive `T` (graph rebuilt at chunk boundaries, `T` chosen from a rolling convergence estimate) is the follow-on design.
