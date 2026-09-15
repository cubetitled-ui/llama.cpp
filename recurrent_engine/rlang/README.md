# Recurrent Language (RLang): Architecture, Grammar, and Systems Manual

RLang is a specialized Domain-Specific Language (DSL) and compiler toolchain designed for defining, testing, and benchmarking arbitrary recurrent architectures, State Space Models (Mamba), and Linear Attention (DeltaNet) within `llama.cpp`.

RLang replaces cumbersome C++ graph modifications and rigid JSON schemas with a zero-dependency, human-readable language that compiles directly into hardware-accelerated GGML computational graphs and executes on GPU silicon with a single command.

---

## Table of Contents
1. [Core Philosophy & Architecture](#1-core-philosophy--architecture)
2. [End-to-End Pipeline Workflow](#2-end-to-end-pipeline-workflow)
3. [Language Grammar & Syntax Reference](#3-language-grammar--syntax-reference)
   - [Model Declaration (`model`)](#model-declaration)
   - [Feedforward Stages (`feedforward`)](#feedforward-stages)
   - [Recurrent Loop Stages (`loop`)](#recurrent-loop-stages)
   - [Parameter Schedules (`schedules`)](#parameter-schedules)
   - [State Registers (`registers`)](#state-registers)
   - [Memory & Execution Policies (`policy`)](#memory--execution-policies)
4. [Mathematical Primitives & Operations](#4-mathematical-primitives--operations)
5. [Cookbook & Architectural Paradigms](#5-cookbook--architectural-paradigms)
   - [Paradigm 1: Contractive LTI Recurrence (Pure Delta)](#paradigm-1-contractive-lti-recurrence-pure-delta)
   - [Paradigm 2: Polyak Heavy-Ball Momentum with Annealing](#paradigm-2-polyak-heavy-ball-momentum-with-annealing)
   - [Paradigm 3: Mamba Selective State Space Model (SSM)](#paradigm-3-mamba-selective-state-space-model-ssm)
   - [Paradigm 4: DeltaNet Associative Memory (Delta Rule)](#paradigm-4-deltanet-associative-memory-delta-rule)
6. [Single-Command Execution Pipeline (`pipeline.py`)](#6-single-command-execution-pipeline-pipelinepy)
7. [Web GUI Studio & MCP Agent Integration](#7-web-gui-studio--mcp-agent-integration)
8. [Stability Proofs & Anti-Divergence Rulebook](#8-stability-proofs--anti-divergence-rulebook)

---

## 1. Core Philosophy & Architecture

Modern large language models traditionally operate as static, feedforward depth chains:
$$h_{l+1} = h_l + F_l(\text{RMSNorm}(h_l))$$

RLang enables transforming transformer layers into dynamic, iterative cognitive loops:
- **Contractive Latent Thinking**: Allows a compound block (e.g., layers 13 and 14) to iterate $T$ times per generation step, progressively refining reasoning representations without increasing parameter counts.
- **Fast-Weight Associative Memory**: Enables state registers to store past-token associations and prediction errors using linear attention and Delta rules.
- **Continuous-to-Discrete State Spaces**: Facilitates continuous-time SSM recurrence (Mamba) with selective timescale parameters.

The toolchain consists of:
1. **Lexer** (`recurrent_engine/rlang/lexer.py`): Zero-dependency tokenizer.
2. **Parser** (`recurrent_engine/rlang/parser.py`): Recursive-descent parser producing a structured AST.
3. **AST Math Compiler** (`recurrent_engine/math_ast.py`): Translates arbitrary algebraic and tensor formulas into optimized GGML C++ API calls.
4. **Graph Compiler** (`recurrent_engine/rlang/compiler.py`): Injects generated C++ routines into `src/llama-graph.cpp` with atomic rollback backups (`.orig`).
5. **Pipeline Harness** (`recurrent_engine/rlang/pipeline.py`): Single command to parse, build, benchmark on silicon, and evaluate accuracy.

---

## 2. End-to-End Pipeline Workflow

```
+------------------+     Lexer / Parser      +------------------+
|   Theory File    | ----------------------> |    Typed AST     |
|   (*.rlang)      |                         |  (Program Model) |
+------------------+                         +------------------+
                                                       |
                                            Compiler & Math AST
                                                       v
+------------------+    Atomic C++ Patch     +------------------+
| src/llama-graph  | <---------------------- | Generated GGML   |
|     (.cpp)       |                         | Graph Builder    |
+------------------+                         +------------------+
         |
    CMake Build
         v
+------------------+    Direct Execution     +------------------+
|   llama-cli /    | ----------------------> | Fast Logic Suite |
|   llama-bench    |    (RTX 3050 GPU)       | / Halfo Benchmark|
+------------------+                         +------------------+
```

---

## 3. Language Grammar & Syntax Reference

An RLang script consists of an optional `model` header followed by sequential `stage` declarations.

### Model Declaration
Specifies the target architecture, total layer count, and numerical precision constants:
```rlang
model Qwen2_Cognitive {
    architecture: qwen2;      # "qwen2", "focal", or "generic"
    total_layers: 28;         # Total layers in base model
    epsilon: 1e-6;            # RMSNorm epsilon
    description: "Multi-step reasoning core";
}
```

### Feedforward Stages
Layers executed in standard sequential feedforward order:
```rlang
stage prelude feedforward {
    layers: 0..12;            # Range [start..end] inclusive
}
```
*Note: Single layers can also be declared without dots: `layers: 13;`.*

### Recurrent Loop Stages
Defines an iterative cognitive loop executing layers $[lo..hi]$ for $T$ recurrence iterations:
```rlang
stage cognitive_core loop {
    layers: 13..14;           # Recurrent block executed iteratively
    iterations: 3;            # Recurrence steps T (T >= 1)
    system: custom;           # "custom", "mamba", or "deltanet"

    params {
        a: 0.90;              # State preservation factor
        b: 0.10;              # Anchor injection factor
        gamma: 1.00;          # Thought delta scaling
    }

    schedules {
        a: "cosine(0.95, 0.70)";
        gamma: "linear(1.00, 0.20)";
    }

    registers {
        v = 0.0;              # Zero-initialized vector
        S = matrix(64, 64);   # 2D Associative Memory Matrix
        h_ssm = state(64);    # 1D State Space Vector
        conv = buffer(4);     # Rolling buffer
    }

    anchors {
        prelude_anchor: inpL; # Output of preceding stage
    }

    policy {
        kv_cache: step_overwrite; # "step_overwrite" or "frozen_kv"
        delta_mode: pure_delta;   # "pure_delta" or "raw_unsubtracted"
    }

    formulas {
        # Custom mathematical state transition formulas
        delta = block_out - combined;
        v = 0.85 * v + 0.15 * delta;
        h = a * h + b * anchor + gamma * v;
    }
}
```

### Parameter Schedules
Dynamic schedules allow hyperparameters to adapt across recurrence iterations $t \in [0..T-1]$:
* **Cosine Annealing**: `"cosine(start, end)"`  
  Smoothly interpolates via:
  $$p_t = \text{end} + \frac{1}{2}(\text{start} - \text{end})\left(1 + \cos\left(\frac{\pi t}{T - 1}\right)\right)$$
* **Linear Decay / Growth**: `"linear(start, end)"`  
  $$p_t = \text{start} + t \cdot \frac{\text{end} - \text{start}}{T - 1}$$
* **Exponential Decay**: `"exponential(start, decay)"`  
  $$p_t = \text{start} \cdot \text{decay}^t$$

### State Registers
Registers maintain latent state across iterations $t$:
Syntax | Allocation in GGML | Purpose
:---|:---|:---
`v = 0.0;` | `ggml_scale(ctx, inpL, 0.0f)` | Auxiliary vector (momentum, error)
`S = matrix(R, C);` | `ggml_new_tensor_2d(ctx, F32, R, C)` + zero | 2D Associative Memory Matrix
`h = state(D);` | `ggml_new_tensor_1d(ctx, F32, D)` + zero | 1D Continuous SSM State
`b = buffer(K);` | `ggml_new_tensor_1d(ctx, F32, K)` + zero | 1D Rolling Convolution Buffer
`r = anchor;` | Initialized to prelude output tensor | Persistent static reference

### Memory & Execution Policies
* **`delta_mode`**:
  * `pure_delta`: Isolates the pure non-linear block transformation $\Delta = \text{block\_out} - \text{combined} = G(\text{combined})$, strictly preventing double-residual scale inflation.
  * `raw_unsubtracted`: Uses raw `block_out` directly.
* **`kv_cache`**:
  * `step_overwrite`: The final iteration $T-1$ overwrites the current token's KV cache slot. Guarantees $O(1)$ memory consumption and zero cache divergence.
  * `frozen_kv`: Only iteration $t=0$ writes to the KV cache.

---

## 4. Mathematical Primitives & Operations

The AST math compiler translates arbitrary expressions into native GGML C++ calls:

Operation / Function | Description | Compiled GGML Expression
:---|:---|:---
`+`, `-` | Tensor addition / subtraction | `ggml_add(ctx, a, b)`, `ggml_sub(ctx, a, b)`
`*` (tensor $\times$ tensor) | Element-wise Hadamard product | `ggml_mul(ctx, a, b)`
`*` (scalar $\times$ tensor) | Fast scalar scaling kernel | `ggml_scale(ctx, a, s)`
`/` | Tensor element-wise division | `ggml_div(ctx, a, b)`
`**` | Scalar power | `powf(base, exp)`
`rms_norm(x, eps)` | Root-Mean-Square Normalization | `ggml_rms_norm(ctx, x, eps)`
`norm(x, eps)` | Standard L2/Variance Normalization | `ggml_norm(ctx, x, eps)`
`silu(x)`, `swish(x)` | SiLU non-linearity ($x \cdot \sigma(x)$) | `ggml_silu(ctx, x)`
`sigmoid(x)` | Logistic Sigmoid ($\frac{1}{1 + e^{-x}}$) | `ggml_sigmoid(ctx, x)`
`tanh(x)` | Hyperbolic tangent | `ggml_tanh(ctx, x)`
`gelu(x)` | Gaussian Error Linear Unit | `ggml_gelu(ctx, x)`
`clamp(x, min, max)` | Bounds tensor elements to $[min, max]$ | `ggml_clamp(ctx, x, min, max)`
`lerp(x, y, t)` | Linear interpolation $(1-t)x + ty$ | Fused affine scale-bias addition
`gate_highway(th, res, g)` | Gated Highway connection | `ggml_add(g * th, (1-g) * res)`
`exp(x)` | Exponential function ($e^x$) | `ggml_exp(ctx, x)`
`log(x)` | Natural logarithm ($\ln x$) | `ggml_log(ctx, x)`
`softplus(x)` | Smooth ReLU approximation $\ln(1 + e^x)$ | `ggml_log(1 + exp(x))`
`silu_gate(x, z)` | Multiplicative SiLU gating ($x \odot \text{SiLU}(z)$) | `ggml_mul(ctx, x, ggml_silu(ctx, z))`
`matvec(M, v)` | Fast Matrix-Vector Multiplication ($M \cdot v$) | `ggml_mul_mat(ctx, M, v)`
`outer_product(a, b)` | Tensor outer product ($a \otimes b^T$) producing 2D matrix | `ggml_out_prod(ctx, a, b)`
`view(x, dim)` | Slices first `dim` elements of 1D tensor | `ggml_view_1d(ctx, x, dim, 0)`
`repeat(x, target)` | Tiles/repeats tensor $x$ to match shape of `target` | `ggml_repeat(ctx, x, target)`

---

## 5. Cookbook & Architectural Paradigms

### Paradigm 1: Contractive LTI Recurrence (Pure Delta)
Isolates pure layer innovation $\Delta = \text{block\_out} - \text{combined}$ and applies contractive state preservation:
```rlang
stage cognitive_core loop {
    layers: 13..14;
    iterations: 2;
    params {
        a: 0.90;
        b: 0.10;
        gamma: 1.00;
    }
    policy {
        delta_mode: pure_delta;
        kv_cache: step_overwrite;
    }
    formulas {
        delta = block_out - combined;
        h = a * h + b * anchor + gamma * delta;
    }
}
```

### Paradigm 2: Polyak Heavy-Ball Momentum with Annealing
Introduces a momentum velocity register $v$ that accelerates convergence along consistent gradient directions:
```rlang
stage cognitive_core loop {
    layers: 13..14;
    iterations: 3;
    params {
        a: 0.90;
        b: 0.10;
        gamma: 1.00;
        beta: 0.85;
    }
    schedules {
        a: "cosine(0.95, 0.70)";
        gamma: "linear(1.00, 0.20)";
        beta: "exponential(0.85, 0.95)";
    }
    registers {
        v = 0.0;
    }
    formulas {
        delta = block_out - combined;
        v = beta * v + (1.0 - beta) * delta;
        h = a * h + b * anchor + gamma * v;
    }
}
```

### Paradigm 3: True Mamba Selective State Space Model (SSM)
Continuous-time state space discretization with input-dependent timescale $\Delta_t$, continuous parameter $A_{\text{diag}}$, Zero-Order Hold discretization, and multiplicative SiLU gating:
```rlang
stage ssm_recurrent_core loop {
    layers: 13..14;
    iterations: 2;
    system: mamba;
    params {
        a: 0.85;
        b: 0.15;
        gamma: 1.00;
        A_diag: -0.50;
    }
    registers {
        h_ssm = state(64);
    }
    formulas {
        # 1. Channel view (d_state=64) and selective timescale Delta = softplus(x)
        x = view(cur, 64);
        delta = softplus(view(combined, 64));
        
        # 2. Continuous-to-discrete Zero-Order Hold transformation
        A_bar = exp(delta * A_diag);
        B_bar = delta * 0.10;
        
        # 3. State-space recurrence update on h_ssm
        h_ssm = A_bar * h_ssm + B_bar * x;
        
        # 4. Selective readout and full hidden-dimension broadcast
        y_ssm = 0.25 * h_ssm;
        y_full = repeat(y_ssm, cur);
        
        # 5. Multiplicative SiLU Gated Connection
        y_gated = silu_gate(y_full, cur);
        
        # 6. Contractive hidden state fusion
        h = a * h + b * anchor + gamma * y_gated;
    }
}
```

### Paradigm 4: True DeltaNet Associative Memory (Delta Rule)
Maintains a 2D associative matrix $S_t \in \mathbb{R}^{d_k \times d_v}$ with error-correcting predictive feedback (Widrow-Hoff delta rule):
```rlang
stage deltanet_core loop {
    layers: 13..14;
    iterations: 2;
    system: deltanet;
    params {
        a: 0.90;
        b: 0.10;
        gamma: 1.00;
        alpha: 0.95;
        beta: 0.25;
    }
    registers {
        S = matrix(64, 64);
    }
    formulas {
        # 1. Project Query, Key, Value vectors from hidden states (d_k=64)
        q = view(cur, 64);
        k = norm(view(combined, 64), 1e-6);
        v = view(block_out, 64);
        
        # 2. Associative Memory Retrieval: v_pred = S * k
        v_pred = matvec(S, k);
        
        # 3. Innovation Error (Delta Rule): v_err = v - v_pred
        v_err = v - v_pred;
        
        # 4. Outer Product Associative Memory Write: S = alpha * S + beta * (v_err (x) k^T)
        delta_S = beta * outer_product(v_err, k);
        S = alpha * S + delta_S;
        
        # 5. Associative Memory Readout for Query q: y_retrieval = S * q
        y_retrieval = matvec(S, q);
        
        # 6. Broadcast across model hidden dimension and non-linear gating
        y_full = repeat(y_retrieval, cur);
        gate = sigmoid(1.50 * y_full);
        
        # 7. Contractive Latent State Update
        h = a * h + b * anchor + gamma * (gate * y_full);
    }
}
```

---

## 6. Single-Command Execution Pipeline (`pipeline.py`)

Execute, compile, and benchmark any RLang file end-to-end:

### Inspect Generated C++ Code (Dry Run)
```bash
python3 recurrent_engine/rlang/pipeline.py \
    --script recurrent_engine/rlang/examples/deep_reasoning_v1.rlang \
    --dry-run
```

### Execute Fast Logic Benchmark on Silicon
Compiles `llama.cpp` incrementally with NVCC and executes the 7-question logic suite:
```bash
python3 recurrent_engine/rlang/pipeline.py \
    --script recurrent_engine/rlang/examples/deep_reasoning_v1.rlang \
    --bench fast_logic
```

### Run Custom Prompt Inference
```bash
python3 recurrent_engine/rlang/pipeline.py \
    --script recurrent_engine/rlang/examples/mamba_selective_ssm.rlang \
    --bench prompt \
    --prompt "Solve step-by-step: If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets?"
```

### Restore Clean Upstream llama.cpp
```bash
python3 recurrent_engine/rlang/pipeline.py --restore
```

---

## 7. Web GUI Studio & MCP Agent Integration

* **Interactive Web Studio**: Running on `http://127.0.0.1:8088`.
  * Endpoint `GET /api/rlang/examples`: Lists and retrieves all available `.rlang` specifications.
  * Endpoint `POST /api/rlang/compile_and_run`: Directly compiles and executes scripts from the UI.
* **Model Context Protocol (MCP)**:
  Exposes the tool `compile_and_run_rlang(rlang_code, model_path, dry_run)` to enable AI agents to autonomously author, validate, and benchmark new recurrence theories.

---

## 8. Stability Proofs & Anti-Divergence Rulebook

### Theorem 1: Contractive Stability Bounds
Let the recurrence transition map be $\Phi(h) = A_t h + B_t e + \gamma_t G(h)$.
For the discrete dynamical system $h_{t+1} = \Phi(h_t)$ to be globally contractive:
$$\rho(A_t) < 1.0$$
where $\rho(\cdot)$ is the spectral radius.
In RLang, $a$ must satisfy $-1.0 \le a < 1.0$. If $|a| \ge 1.0$, eigenvalues cross the unit circle, resulting in exponential activation blowup and `NaN` divergence within 3 generation steps.

### Theorem 2: The Double-Residual Elimination Law
Standard transformer layers compute:
$$\text{block\_out} = \text{combined} + G(\text{combined})$$
Injecting raw $\text{block\_out}$ into an affine state combination adds $\text{combined}$ twice:
$$h_{t+1} = a h_t + b e + \gamma (\text{combined} + G(\text{combined}))$$
At $\gamma = 1.0$, the effective state scale inflates by $(1 + \gamma) = 2.0\times$ per recurrence step, causing divergence at $T \ge 2$.
In RLang's `pure_delta` mode:
$$\Delta_t = \text{block\_out} - \text{combined} \equiv G(\text{combined})$$
This isolates the pure non-linear thought delta, maintaining strict mathematical contractivity.
