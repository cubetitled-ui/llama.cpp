---
name: model-logic-bench
description: >-
  Fast logic benchmarking, recurrent graph validation, and model verification tools for AI agents.
  Use to evaluate GGUF models on standardized logic questions, tune recurrent execution parameters,
  and inspect hardware telemetry via MCP and Web Studio.
---

# Model Logic Benchmark & Recurrent Verification Skill

This skill provides an automated, high-throughput verification suite to evaluate language models on reasoning, mathematical deduction, and recurrent latent refinement on NVIDIA RTX 3050 Laptop GPU (GA107, 6GB VRAM).

---

## 1. Quick Capabilities Overview

Component        | Location                                           | Purpose
:--------------- | :------------------------------------------------- | :----------------------------------------------------------------------
**MCP Server**   | `/home/cune/llama.cpp/tools/mcp_server/server.py`  | Model Context Protocol tools for AI agents to benchmark and run models.
**Web Dashboard**| `http://127.0.0.1:8088`                            | Real-time Web GUI for interactive testing, scoring, and telemetry.
**CLI Harness**  | `/home/cune/llama.cpp/benchmarks/halfo/run_halfo.py`| Automated test runner across logic, tool calling, and translation.
**Spec Compiler**| `/home/cune/llama.cpp/recurrent_engine/engine_cli.py`| Validates and emits GGML C++ code for custom recurrent theories.

---

## 2. Standard Logic Verification Suite (7 Questions)

The fast verification suite tests resistance to cognitive biases and multi-step tracking:

1. **Bat & Ball**: "$1.10 total, bat costs $1.00 more than ball. How much is the ball?" -> **5 cents**
2. **Train Speed**: "120 km in 2.5 hours average speed?" -> **48 km/h**
3. **Snail in Well**: "10m well, +3m day, -2m night. Reaches top on which day?" -> **Day 8**
4. **Lily Pads**: "Doubles daily, covers lake in 48 days. How many days for half?" -> **47 days**
5. **Widget Machines**: "5 machines take 5 min for 5 widgets. 100 machines for 100 widgets?" -> **5 min**
6. **Sheep Puzzle**: "17 sheep, all but 9 run away. Left?" -> **9 sheep**
7. **Sister Age**: "When brother was 4, sister was half. Brother is 100. Sister age?" -> **98 years**

---

## 3. MCP Server Configuration & Available Tools

To register the MCP server with an agent or IDE, add to `mcp_config.json`:

```json
{
  "mcpServers": {
    "model-logic-bench": {
      "command": "/usr/bin/python3",
      "args": ["/home/cune/llama.cpp/tools/mcp_server/server.py"]
    }
  }
}
```

### Available MCP Tools:
* `benchmark_logic_fast(model_path, recurrent_t, recurrent_layer, recurrent_layer_b, recurrent_a, recurrent_b, recurrent_gate, recurrent_config, lora_path)`:
  Runs the 7 logic tests sequentially and returns a JSON summary with pass rate %, latencies, and error snippets. Supports passing a config file or RLang script directly via `recurrent_config`.
* `run_model_inference(model_path, prompt, recurrent_t, recurrent_layer, ..., recurrent_config, lora_path)`:
  Executes single prompt inference with custom recurrent graph parameters or loaded directly from a `.json` / `.conf` / `.rlang` file.
* `get_hardware_status()`:
  Returns current GPU VRAM utilization, used/free MiB, and GPU core temperature.
* `list_models()`:
  Discovers and returns all valid GGUF models available in `models/`.
* `compile_and_run_rlang(rlang_code, model_path, dry_run)`:
  Parses, mathematically validates, compiles custom RLang scripts (.rlang) into C++ GGML computational graphs, rebuilds llama.cpp, and evaluates logic performance on silicon.

### Direct Native llama-cli Flags for Recurrent Configs & RScripts:
```bash
# Load recurrent parameters and LoRA directly from JSON config:
llama-cli -m model.gguf -rc /path/to/recurrent.json -p "Prompt"

# Load recurrent parameters directly from RLang script:
llama-cli -m model.gguf -rs /path/to/theory.rlang -p "Prompt"
```

---

## 4. Single-Command RLang Pipeline

Execute any custom recurrent theory end-to-end with a single command:
```bash
python3 /home/cune/llama.cpp/recurrent_engine/rlang/pipeline.py \
    --script /home/cune/llama.cpp/recurrent_engine/rlang/examples/deep_reasoning_v1.rlang \
    --bench fast_logic
```

Options:
- `--script <path.rlang>`: Custom theory written in RLang.
- `--model <path.gguf>`: Target GGUF model (auto-detected if omitted).
- `--bench <fast_logic|halfo|throughput|prompt|none>`: Benchmark suite to run.
- `--dry-run`: Emit and inspect generated C++ GGML computational graph without compiling.
- `--restore`: Revert `src/llama-graph.cpp` to clean upstream baseline and rebuild.

---

## 5. Web Dashboard Usage

The interactive studio runs locally at:
```bash
http://127.0.0.1:8088
```

Features:
* Dropdown selection across all local GGUF models.
* Parameter adjustment sliders for $T$, layers $[lo..hi]$, and $(a, b, \gamma)$.
* Single-click **"EXECUTE LOGIC BENCHMARK"** with visual pass/fail scoring cards.
* Live VRAM and GPU core telemetry polling every 3 seconds.

---

## 6. Verification Runbook for AI Agents

1. **Step 1: Check System Health**:
   Query `get_hardware_status()` to verify VRAM headroom ($\ge 2.0$ GB free).
2. **Step 2: Establish Baseline ($T=1$)**:
   Run `benchmark_logic_fast(model_path, recurrent_t=1)`. Record baseline accuracy (e.g. 57.1%).
3. **Step 3: Evaluate Recurrent Configuration ($T=2$, layers 13-14)**:
   Run `benchmark_logic_fast(model_path, recurrent_t=2, recurrent_layer=13, recurrent_layer_b=14, recurrent_a=0.90, recurrent_b=0.10, recurrent_gate=1.00)`.
4. **Step 4: Verify Delta Contractivity**:
   Ensure accuracy matches or exceeds baseline without state divergence or runaway latency.

