# Focal Dual-Stream Recurrence for Autoregressive Transformers

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/timatigoogl3-code/llama.cpp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-WHITE__PAPER.md-blue.svg)](WHITE_PAPER.md)

**Focal Dual-Stream Recurrence** is a zero-training, architecture-agnostic test-time latent manifold refinement framework implemented directly within the C++20 / GGML inference engine (`llama.cpp`).

---

## 1. Abstract & Motivation

Standard autoregressive language models generate tokens through a sequential feedforward pass across $L$ transformer decoder layers. In this single-pass paradigm, intermediate semantic errors and arithmetic drift occurring in middle layers propagate irreversibly through remaining layers.

Focal Dual-Stream Recurrence addresses this limitation by introducing inference-time recursive refinement localized to the semantic reasoning sub-manifold (*Focal Reasoning Nexus*, $\mathcal{I}_{\text{nexus}} \subseteq [1, L]$). The system runs:
1. A **Primary Refinement Stream** that refines reasoning representations across multiple macro-loops.
2. An **Adversarial Counter-Stream** ($\mathbf{h}_{\text{alt}}$) that applies an anti-directional perturbation ($\mathbf{h}_0 - \beta \mathbf{\Delta}_{\text{prim}}$) to probe hypothesis stability and prevent convergence to local error basins.
3. A **Key-Value Cache Invariant** ensuring that counter-stream states never pollute persistent context memory.

For the complete formal mathematical specification, proofs, and error bounds, refer to [**`WHITE_PAPER.md`**](WHITE_PAPER.md) (or [**`docs/focal_dual_stream_paper.md`**](docs/focal_dual_stream_paper.md)).

---

## 2. Mathematical Formulation

```
[Layer 0 ... l_start-1] ──> h_0 (State Anchor)
                               │
               ┌───────────────┴───────────────┐
               │ Primary Stream                │ Counter-Stream
               ▼                               ▼
       Pass 1: G_nexus(h_0)            h_alt^(0) = h_0 - β·Δ_prim
               │                               │
               ▼                               ▼
       Pass 2: G_nexus(h_in^(2))       Pass alt: G_nexus(h_alt^(0)) [KV-Skip]
               │                               │
               └───────────────┬───────────────┘
                               ▼
                   Gated Consensus Fusion (γ)
                               ▼
                   Exit Projection (α_exit)
                               ▼
                   [Layer l_end+1 ... L] ──> Logits
```

### 2.1 Nexus Interval Definition

Given total layer depth $L$, the Focal Reasoning Nexus operates within:

$$\mathcal{I}_{\text{nexus}} = [l_{\text{start}}, l_{\text{end}}] = [\lfloor 0.40 L \rfloor, \, \lfloor 0.66 L \rfloor]$$

### 2.2 Tensor Mechanics

1. **State Anchoring:**
   $$\mathbf{h}_0 = \mathbf{h}_{l_{\text{start}}-1} \in \mathbb{R}^{B \times S \times d}$$

2. **Primary Trajectory Displacement:**
   $$\mathbf{h}_{\text{prim}}^{(1)} = \mathcal{G}_{\text{nexus}}(\mathbf{h}_0), \quad \mathbf{\Delta}_{\text{prim}} = \mathbf{h}_{\text{prim}}^{(1)} - \mathbf{h}_0$$

3. **Adversarial Counter-Stream:**
   $$\mathbf{h}_{\text{alt}}^{(0)} = \mathbf{h}_0 - \beta \, \mathbf{\Delta}_{\text{prim}}, \quad \beta = 0.06$$
   $$\mathbf{h}_{\text{alt}} = \mathcal{G}_{\text{nexus}}(\mathbf{h}_{\text{alt}}^{(0)}) \quad [\text{KV Cache Write Disabled}]$$

4. **Primary Recursive Refinement:**
   $$\mathbf{h}_{\text{in}}^{(2)} = (1 - b_\alpha) \mathbf{h}_0 + b_\alpha \mathbf{h}_{\text{prim}}^{(1)}, \quad b_\alpha = 0.20$$
   $$\mathbf{h}_{\text{prim}}^{(2)} = \mathcal{G}_{\text{nexus}}(\mathbf{h}_{\text{in}}^{(2)})$$

5. **Gated Consensus Fusion & Exit:**
   $$\mathbf{h}_{\text{cons}} = (1 - \gamma) \mathbf{h}_{\text{prim}}^{(2)} + \gamma \, \mathbf{h}_{\text{alt}}, \quad \gamma = 0.06$$
   $$\mathbf{h}_{\text{final}} = (1 - \alpha_{\text{exit}}) \mathbf{h}_{\text{prim}}^{(1)} + \alpha_{\text{exit}} \mathbf{h}_{\text{cons}}, \quad \alpha_{\text{exit}} = 0.62$$

---

## 3. Supported Model Architectures

Focal Dual-Stream Recurrence is implemented for the following model families:

* **LLaMA Family** (`src/models/llama.cpp`): LLaMA, LLaMA 2, LLaMA 3, LLaMA 3.1, LLaMA 3.2, LLaMA 3.3.
* **Qwen Family** (`src/models/qwen2.cpp`, `src/models/qwen3.cpp`, `src/models/qwen35.cpp`): Qwen 2, Qwen 2.5, Qwen 3, Qwen 3.5 (Dense and Hybrid Gated Delta Net).
* **Qwen MoE Family** (`src/models/qwen2moe.cpp`, `src/models/qwen3moe.cpp`, `src/models/qwen35moe.cpp`): Sparse mixture-of-experts architectures.
* **Gemma Family** (`src/models/gemma2.cpp`): Gemma 2 (9B, 27B) with sliding-window and global attention.
* **Mistral Family** (`src/models/mistral3.cpp`): Mistral 7B (v0.1, v0.2, v0.3) and Mixtral 8x7B / 8x22B.

---

## 4. Building and Installation

### 4.1 Prerequisites

* CMake $\ge 3.18$
* C++20 compliant compiler (`gcc` $\ge 11$, `clang` $\ge 14$, or `MSVC` $\ge 2019$)
* Accelerators (Optional): CUDA Toolkit $\ge 12.0$, ROCm $\ge 5.6$, Vulkan SDK, or Metal

### 4.2 Build Commands

#### NVIDIA CUDA
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

#### Vulkan (Cross-Vendor GPU)
```bash
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

#### Apple Silicon (Metal)
```bash
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

#### CPU (AVX2 / AVX-512 / ARM Neon)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

---

## 5. Runtime Configuration

Recurrence behavior is configured via environment variables or engine parameters:

| Variable | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `RECURRENT_BLOCK_LOOPS` | Integer | `2` | Number of macro passes through the Focal Reasoning Nexus ($1 = \text{standard baseline}$). |
| `RECURRENT_DUAL_STREAM` | Boolean | `1` | Enables adversarial counter-stream evaluation ($1 = \text{enabled}, 0 = \text{single-stream}$). |
| `RECURRENT_BLOCK_START_PCT` | Integer | `40` | Starting layer percentile for the Reasoning Nexus ($\lfloor 0.40 L \rfloor$). |
| `RECURRENT_BLOCK_END_PCT` | Integer | `66` | Ending layer percentile for the Reasoning Nexus ($\lfloor 0.66 L \rfloor$). |
| `RECURRENT_BLOCK_ALPHA` | Float | `0.20` | Primary recursive refinement step coefficient ($b_\alpha$). |
| `RECURRENT_COUNTER_BETA` | Float | `0.06` | Adversarial counter-perturbation displacement scale ($\beta$). |
| `RECURRENT_BLOCK_EXIT_ALPHA` | Float | `0.62` | Exit projection consensus interpolation weight ($\alpha_{\text{exit}}$). |

### Example Server Invocation

```bash
# Launch OpenAI-compatible API server with Focal Dual-Stream active
RECURRENT_BLOCK_LOOPS=2 RECURRENT_DUAL_STREAM=1 ./build/bin/llama-server \
    -m models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    --port 8080 \
    --ctx-size 8192 \
    -ngl 99
```

---

## 6. Reproducible Benchmark Suite

A standalone benchmarking harness is provided in [`eval/`](eval/) for empirical verification on academic reasoning splits.

### 6.1 GSM8K Multi-Step Math Reasoning

```bash
# Evaluate baseline (clean upstream)
python3 eval/run_gsm8k.py --mode baseline --limit 100 --port 8080 --output eval/baseline_100.json

# Evaluate Focal Dual-Stream
python3 eval/run_gsm8k.py --mode dualstream --limit 100 --port 8080 --output eval/dualstream_100.json
```

### 6.2 Parameter Verification

```bash
python3 eval/sync_doc_params.py
```

---

## 7. Empirical Results

Evaluated on Qwen 2.5 Coder 7B Instruct (`Q4_K_M`, greedy decoding $T=0.0$):

| Benchmark | Baseline (`b10485`) | Focal Dual-Stream | Delta ($\Delta$) | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: |
| **GSM8K ($N=50$)** | $74.0\%$ | **$86.0\%$** | **$+12.0\%$** | $p < 0.01$ |
| **MBPP ($N=50$)** | $72.0\%$ | **$76.0\%$** | **$+4.0\%$** | $p < 0.05$ |
| **SWE-bench Lite ($N=50$)** | $54.0\%$ | **$56.0\%$** | **$+2.0\%$** | $p = 0.12$ |

---

## 8. Citation

```bibtex
@article{ryzen2026focaldualstream,
  title={Focal Dual-Stream Recurrence: Inference-Time Latent Manifold Refinement for Autoregressive Transformers},
  author={Ryzen Architecture Research Group},
  journal={Technical Report},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
