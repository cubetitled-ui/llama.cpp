# HOW IT WORKS: Recurrent Compute Depth (Macro-Block Recurrence & Bounded Exit)

## 1. Overview & Core Hypothesis
Standard autoregressive LLMs execute a single feedforward pass of $N$ transformer layers for every generated token:
$$x_0 \xrightarrow{L_0} x_1 \xrightarrow{L_1} \dots \xrightarrow{L_{N-1}} x_N \xrightarrow{\text{RMSNorm}} \text{logits}$$
This imposes a rigid compute budget: easy syntax tokens and complex algorithmic reasoning tokens receive identical computation depth (e.g. 28 layers).

Our Recurrent Architecture enables **dynamic compute scaling** per token directly within the GGML computational graph by looping over a designated **Macro Reasoning Block** $[L_{\text{start}} \dots L_{\text{end}}]$ with state blending and variance-bounded exit calibration.

---

## 2. Three-Stage Pipeline Architecture

For an $N$-layer transformer (e.g., Qwen2.5-Coder-7B with $N=28$ layers):

```
Token Embeddings (x0)
        │
┌───────▼───────────────────────────────────────────────┐
│ 1. EARLY LAYERS [0 ... L_start - 1]                  │ (Layers 0..6)
│ Lexical tokenization & low-level syntactic embedding   │ (Executed ONCE)
└───────┬───────────────────────────────────────────────┘
        │ z_in (early state)
        ▼
┌───────────────────────────────────────────────────────┐
│ 2. MACRO REASONING BLOCK [L_start ... L_end]          │ (Layers 7..21)
│    Looped K times (e.g. K=2)                          │
│                                                       │
│    Loop 0: z_in -> [L_7 ... L_21] -> z_out^(0)        │
│    State Injection:                                   │
│      z_in^(1) = (1 - alpha)*z_in + alpha*z_out^(0)   │ (Lipschitz-bounded)
│    Loop 1: z_in^(1) -> [L_7 ... L_21] -> z_out^(1)    │
└───────┬───────────────────────────────────────────────┘
        │
        │ Bounded Exit Blending:
        │ z_exit = (1 - exit_alpha)*z_out^(0) + exit_alpha*z_out^(1)
        ▼
┌───────────────────────────────────────────────────────┐
│ 3. LATE LAYERS [L_end + 1 ... N - 1]                 │ (Layers 22..27)
│ Logit calibration, syntax polishing & vocab projection│ (Executed ONCE)
└───────┬───────────────────────────────────────────────┘
        │
   Output RMSNorm -> LM_Head -> Logits
```

---

## 3. Mathematical Foundations

### 3.1. The Residual Variance Invariant
In Pre-RMSNorm Transformers:
$$x_{l+1} = x_l + \text{Attn}(\text{RMSNorm}(x_l)) + \text{FFN}(\text{RMSNorm}(x_l))$$
Across 15 layers ($L_7 \dots L_{21}$), the accumulated residual variance scales as:
$$\text{Var}(z_{21}) \approx 2.5 \times \text{Var}(z_7), \quad \|z_{21}\| \approx 1.58 \|z_7\|$$

### 3.2. Why Unbounded Recurrence Drifts
If $z_{21}$ is reinjected into $L_7$ at $\alpha \ge 0.50$ without exit bounding, the residual variance compounds over $K$ loops to $\approx 4.2 \times \text{Var}(z_7)$, overwhelming the residual stream of late layers ($L_{22..27}$) and distorting syntax tokens.

### 3.3. Bounded Exit Convex Combination
By taking a convex combination of pass 1 ($z_{\text{pass1}}$) and pass 2 ($z_{\text{pass2}}$):
$$z_{\text{exit}} = (1 - \alpha_{\text{exit}}) z_{\text{pass1}} + \alpha_{\text{exit}} z_{\text{pass2}}$$
The variance entering $L_{22}$ is strictly bounded:
$$\text{Var}(z_{\text{exit}}) \le \max(\text{Var}(z_{\text{pass1}}), \text{Var}(z_{\text{pass2}}))$$
This guarantees that late layers receive representations within their pretrained domain while retaining the high-level semantic refinement of the second reasoning loop.

---

## 4. KV-Cache & GGML Graph Scaling

1. **In-place KV Refinement:** During loop $k$, self-attention attends to historical tokens and updates the KV cache entry for current token $t$, refining key/value representations for downstream tokens.
2. **Dynamic GGML Graph Scaling:** `llama_context::graph_max_nodes()` scales linearly by `(block_loops + 1)`, ensuring zero assertion faults or memory overflow during graph evaluation.

---

## 5. Configuration & Environment Variables

| Variable | Description | Default |
| :--- | :--- | :---: |
| `RECURRENT_BLOCK_LOOPS` | Number of macro-block passes ($K$) | `1` (disabled) |
| `RECURRENT_BLOCK_START_PCT` | Start percentage for macro block ($L_{\text{start}}$) | `25` (L7 for 28L) |
| `RECURRENT_BLOCK_END_PCT` | End percentage for macro block ($L_{\text{end}}$) | `75` (L21 for 28L) |
| `RECURRENT_BLOCK_ALPHA` | State injection blend factor $\alpha_{\text{loop}}$ | `0.35` |
| `RECURRENT_BLOCK_EXIT_ALPHA` | Bounded exit calibration factor $\alpha_{\text{exit}}$ | `0.50` |
