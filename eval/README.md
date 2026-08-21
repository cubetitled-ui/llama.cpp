# Reproducible Evaluation Suite for Focal Dual-Stream Recurrence

This directory contains standalone, reproducible benchmarking scripts designed to validate inference-time recurrence architectures in `llama.cpp`.

---

## 1. GSM8K Evaluation Protocol (`run_gsm8k.py`)

The GSM8K evaluation measures multi-step mathematical reasoning accuracy using exact numerical match on the standard `openai/gsm8k` test split.

### Prerequisites

```bash
pip install datasets requests
```

### Protocol

1. **Launch Baseline Server**:
   ```bash
   RECURRENT_BLOCK_LOOPS=1 /home/cune/llama.cpp/build/bin/llama-server \
       -m /home/cune/qwen2.5-coder-3b-instruct-q4_k_m.gguf \
       --port 8080 --ctx-size 8192 -ngl 99
   ```

2. **Execute Baseline Run**:
   ```bash
   python3 eval/run_gsm8k.py --mode baseline --limit 100 --port 8080 --output eval/baseline_gsm8k_100.json
   ```

3. **Launch Focal Dual-Stream Server**:
   ```bash
   RECURRENT_BLOCK_LOOPS=2 RECURRENT_DUAL_STREAM=1 /home/cune/llama.cpp/build/bin/llama-server \
       -m /home/cune/qwen2.5-coder-3b-instruct-q4_k_m.gguf \
       --port 8080 --ctx-size 8192 -ngl 99
   ```

4. **Execute Dual-Stream Run**:
   ```bash
   python3 eval/run_gsm8k.py --mode dualstream --limit 100 --port 8080 --output eval/dualstream_gsm8k_100.json
   ```

---

## 2. Parameter Consistency Verification (`sync_doc_params.py`)

To prevent documentation drift, `sync_doc_params.py` parses `src/models/models.h` directly and verifies that mathematical constants described in technical papers match compile-time and runtime defaults.

```bash
python3 eval/sync_doc_params.py
```

Expected output:
```text
Extracted exact parameters from models.h:
  start_pct: 40
  end_pct: 66
  alpha: 0.2
  exit_alpha: 0.62
  beta: 0.06
  loops: 2
```
