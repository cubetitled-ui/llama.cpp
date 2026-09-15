# KV Cache Architecture in Recurrent GGML Execution

## 1. Physical Role of the KV Cache in `llama.cpp`

In standard autoregressive decoding, generating token $p$ requires computing attention over all preceding tokens $0 \dots p-1$. Recomputing Keys ($K$) and Values ($V$) for prior tokens at every step would scale inference cost quadratically: $\mathcal{O}(N^2)$ FLOPs per sequence.

The **KV Cache** resolves this by persisting previously computed Key and Value matrices in contiguous GPU/system memory:
- Structure: For each layer $il \in [0, N-1]$, dedicated memory buffers `kv_self.k_l[il]` and `kv_self.v_l[il]` store the projections:
  $$K_i = W_k \cdot x_i, \quad V_i = W_v \cdot x_i \quad \forall i \in [0, p-1]$$
- Decoding Step: When token $p$ is processed, only $Q_p = W_q x_p$, $K_p = W_k x_p$, and $V_p = W_v x_p$ are computed. $K_p$ and $V_p$ are appended to the cache at offset $p$, and attention is evaluated as:
  $$\text{Attention}(Q_p, K_{0:p}, V_{0:p}) = \text{Softmax}\left(\frac{Q_p K_{0:p}^T}{\sqrt{d_k}}\right) V_{0:p}$$

---

## 2. The Recurrent Loop Challenge ($T > 1$)

In a recurrent graph (e.g., layers $13 \dots 14$ executed for $T$ iterations per token):
- The layer block is called repeatedly **within the exact same token timestep $p$**.
- At iteration $t = 0$: layer 13 receives prelude output, computing draft vectors $K_p^{(0)}, V_p^{(0)}$.
- At iteration $t = 1$: layer 13 receives refined state $h^{(1)}$, computing updated vectors $K_p^{(1)}, V_p^{(1)}$.

Without explicit cache management, naive multi-pass execution creates conflicts: does iteration $t=1$ overwrite the cache at position $p$, append to the sequence, or ignore caching?

---

## 3. Supported KV Cache Modes in Topology Specs

Our engine defines three explicit operational modes configurable in `topology.json`:

### A. `step_overwrite` (Default & Recommended for Latent Refinement)
* **Mechanics**:
  During iterations $t = 0 \dots T-1$, layer operations read historical context $K_{0:p-1}, V_{0:p-1}$ from the cache. The slot at index $p$ is overwritten on each iteration.
* **Benefit**:
  When generation moves to token $p+1$, it attends to the **converged, fully refined representation** $K_p^{(T-1)}, V_p^{(T-1)}$ rather than the initial unrefined draft.
* **Memory Cost**: **0% overhead**. Cache size remains identical to a standard feedforward model.

### B. `frozen_kv` (Strict Intra-Token Thinking)
* **Mechanics**:
  Only the initial pass ($t = 0$) writes $K_p^{(0)}, V_p^{(0)}$ to the KV cache. On subsequent iterations $t \ge 1$, the attention operation queries the existing cache without writing updates.
* **Benefit**:
  Zero distributional drift for downstream tokens if the base model's key/value projection distribution must remain completely unmodified.
* **Memory Cost**: **0% overhead**.

### C. `virtual_depth` (Time-Unrolled Trajectory Memory)
* **Mechanics**:
  Each recurrence step $t$ is mapped to a distinct virtual layer in an expanded cache buffer:
  $$il_{\text{virtual}} = \text{lo} + t \cdot (\text{hi} - \text{lo} + 1) + (il - \text{lo})$$
* **Benefit**:
  Allows subsequent tokens to attend across both spatial depth and intermediate temporal reasoning states.
* **Memory Cost**: Scales linearly with $T \times (\text{layers in loop})$.

---

## 4. Mathematical Interaction: `pure_delta` vs. KV Cache Stability

A critical discovery in our engineering audits is that the **residual formulation directly dictates KV cache numeric stability**:

$$\text{block\_out} = \text{combined} + G(\text{combined})$$

1. **The Double-Residual Failure Mode (`raw_unsubtracted`)**:
   Adding `block_out` directly into the recurrent update:
   $$h_{t+1} = a \cdot h_t + b \cdot e + \gamma \cdot \text{block\_out}$$
   injects `combined` twice on every loop pass. The hidden state magnitude explodes exponentially:
   $$\|h_t\| \approx 2^t \cdot \|h_0\|$$
   Because $K = W_k h$ and $V = W_v h$, this exponential explosion overflows FP16 dynamic ranges ($> 65,504$) and drives attention logits $Q K^T / \sqrt{d}$ into extreme saturation ($\pm \infty$), causing immediate NaN collapse in the KV cache.

2. **The Pure Delta Formulation (`pure_delta`)**:
   By explicitly subtracting the input:
   $$\Delta_{\text{thought}} = \text{block\_out} - \text{combined} = G(\text{combined})$$
   $$h_{t+1} = a \cdot h_t + b \cdot e + \gamma \cdot \Delta_{\text{thought}}$$
   The linear time-invariant system remains strictly contractive when $|a| < 1$. Hidden state vectors stay bounded within $[-3.0, +3.0]$, ensuring that all values committed to the KV cache match the base model's calibrated dynamic range.

---

## 5. Quick-Reference Configuration Schema

```json
{
  "type": "recurrent_loop",
  "name": "center_core",
  "layers": [13, 14],
  "iterations": 2,
  "recurrent_a": 0.90,
  "recurrent_b": 0.10,
  "recurrent_gate": 1.00,
  "delta_mode": "pure_delta",
  "kv_mode": "step_overwrite",
  "norm_eps": 1e-06,
  "save_state": true
}
```
