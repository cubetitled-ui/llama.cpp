#!/usr/bin/env python3
"""
compare_recurrent_strategies_no_lora.py
Evaluates multiple unquantized/pure-weight recurrent strategies WITHOUT LoRA
against the 13 Brutal Systems Coding Tasks.
Compares directly to the established Baseline (T=1: 7/13, 53.8%).
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, Any, List

sys.path.insert(0, "/home/cune/llama.cpp")
from benchmarks.comprehensive.run_benchmarks import run_inference, evaluate_python_code

DATASET_PATH = "/home/cune/llama.cpp/benchmarks/comprehensive/dataset.json"
MODEL_PATH = "/home/cune/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
CLI_BIN = "/home/cune/llama.cpp/build-cuda-vnni/bin/llama-cli"

STRATEGIES = [
    {
        "name": "Strategy A (Contractive Mid-Core T=2, L13-14, a=0.9, b=0.1, g=0.2)",
        "flags": ["--recurrent-t", "2", "--recurrent-layer", "13", "--recurrent-layer-b", "14", "--recurrent-a", "0.90", "--recurrent-b", "0.10", "--recurrent-gate", "0.20"]
    },
    {
        "name": "Strategy B (High-Anchor Conservative T=2, L13-14, a=0.7, b=0.3, g=0.15)",
        "flags": ["--recurrent-t", "2", "--recurrent-layer", "13", "--recurrent-layer-b", "14", "--recurrent-a", "0.70", "--recurrent-b", "0.30", "--recurrent-gate", "0.15"]
    },
    {
        "name": "Strategy C (Late-Stage Synthesis T=2, L20-21, a=0.9, b=0.1, g=0.2)",
        "flags": ["--recurrent-t", "2", "--recurrent-layer", "20", "--recurrent-layer-b", "21", "--recurrent-a", "0.90", "--recurrent-b", "0.10", "--recurrent-gate", "0.20"]
    },
    {
        "name": "Strategy D (Tri-Pass Deep Reflection T=3, L13-14, a=0.8, b=0.2, g=0.10)",
        "flags": ["--recurrent-t", "3", "--recurrent-layer", "13", "--recurrent-layer-b", "14", "--recurrent-a", "0.80", "--recurrent-b", "0.20", "--recurrent-gate", "0.10"]
    }
]

def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    coding_tasks = data["tasks"]["coding"]

    print("=" * 80)
    print("RECURRENT STRATEGIES BENCHMARK (WITHOUT LORA) ON 13 BRUTAL CODING TASKS")
    print(f"Model: {MODEL_PATH}")
    print(f"Baseline Score (T=1, Vanilla): 7/13 (53.8%)")
    print("=" * 80)

    summary_results = {
        "baseline": {
            "score": "7/13",
            "accuracy_pct": 53.8,
            "passed": ["code_02_bytecode_vm", "code_04_lisp_interpreter", "code_05_dinic_max_flow", "code_06_diff_patch_engine", "code_08_transactional_key_value", "code_11_topological_lexical_kahn", "code_13_knapsack_with_reconstruction"],
            "failed": ["code_01_regex_nfa", "code_03_lazy_segment_tree", "code_07_avl_tree_invariants", "code_09_expression_calculator_shunting_yard", "code_10_interval_tree_overlap", "code_12_tarjan_scc"]
        },
        "strategies": {}
    }

    for strat in STRATEGIES:
        s_name = strat["name"]
        flags = strat["flags"]
        print(f"\nEvaluating: {s_name}")
        print(f"Flags: {' '.join(flags)}")
        
        passed_count = 0
        passed_ids = []
        failed_ids = []
        task_details = []

        t_start = time.time()
        for idx, task in enumerate(coding_tasks):
            tid = task["id"]
            prompt = task["prompt"]
            t0 = time.time()
            out = run_inference(CLI_BIN, MODEL_PATH, flags, prompt, max_tokens=512)
            latency = time.time() - t0
            
            is_pass = evaluate_python_code(task, out)
            status = "PASS" if is_pass else "FAIL"
            if is_pass:
                passed_count += 1
                passed_ids.append(tid)
            else:
                failed_ids.append(tid)

            task_details.append({"id": tid, "status": status, "latency": round(latency, 2)})
            print(f"  [{status}] {tid} ({latency:.2f}s)")

        total_time = round(time.time() - t_start, 2)
        acc_pct = round((passed_count / len(coding_tasks)) * 100, 1)
        delta_pct = round(acc_pct - 53.8, 1)
        delta_str = f"+{delta_pct}%" if delta_pct > 0 else f"{delta_pct}%"

        print(f"Score for {s_name}: {passed_count}/{len(coding_tasks)} ({acc_pct}%) [Delta vs Baseline: {delta_str}] in {total_time}s")

        summary_results["strategies"][s_name] = {
            "score": f"{passed_count}/{len(coding_tasks)}",
            "accuracy_pct": acc_pct,
            "delta_vs_baseline": delta_pct,
            "total_time_s": total_time,
            "passed": passed_ids,
            "failed": failed_ids,
            "details": task_details
        }

        # Dump progressive results
        with open("/home/cune/llama.cpp/benchmarks/comprehensive/recurrent_no_lora_results.json", "w") as f:
            json.dump(summary_results, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL STRATEGY COMPARISON COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
