#!/usr/bin/env python3
"""
compare_all.py
Runs the Comprehensive Multi-Domain Benchmark Suite across:
1. Baseline (T=1)
2. T=2 (Vanilla No LoRA)
3. T=2 (LoRA Step 100)
4. T=2 (LoRA Step 200)
5. T=2 (LoRA Step 300)
6. T=3 (LoRA Step 300)

Produces a structured markdown comparison matrix across Tool Calling, Logic, Coding, and Web Development.
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, Any, List

sys.path.insert(0, "/home/cune/llama.cpp/benchmarks/comprehensive")
from run_benchmarks import run_inference, evaluate_json_tool, evaluate_contains, evaluate_keywords, evaluate_python_code

DATASET_PATH = "/home/cune/llama.cpp/benchmarks/comprehensive/dataset.json"
MODEL_PATH = "/home/cune/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
CLI_PATH = "/home/cune/llama.cpp/build-cuda-vnni/bin/llama-cli"

CONFIGS = [
    ("Baseline (T=1)", []),
    ("T=2 No LoRA", ["--recurrent-t", "2", "--recurrent-layer", "13", "--recurrent-layer-b", "14"]),
    ("T=2 Step 100", ["--recurrent-t", "2", "--recurrent-layer", "13", "--recurrent-layer-b", "14", "--lora", "/home/cune/llama.cpp/models/recurrent_core_lora/checkpoint_step_100.gguf"]),
    ("T=2 Step 200", ["--recurrent-t", "2", "--recurrent-layer", "13", "--recurrent-layer-b", "14", "--lora", "/home/cune/llama.cpp/models/recurrent_core_lora/checkpoint_step_200.gguf"]),
    ("T=2 Step 300", ["--recurrent-t", "2", "--recurrent-layer", "13", "--recurrent-layer-b", "14", "--lora", "/home/cune/llama.cpp/models/recurrent_core_lora/checkpoint_step_300.gguf"]),
    ("T=3 Step 300", ["--recurrent-t", "3", "--recurrent-layer", "13", "--recurrent-layer-b", "14", "--lora", "/home/cune/llama.cpp/models/recurrent_core_lora/checkpoint_step_300.gguf"]),
]

def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks_dict = data["tasks"]
    categories = list(tasks_dict.keys())

    summary_matrix = {}

    for cfg_name, flags in CONFIGS:
        print(f"\n==================================================")
        print(f"EVALUATING: {cfg_name}")
        print(f"==================================================")
        cfg_results = {}
        total_p = 0
        total_n = 0

        for cat in categories:
            cat_tasks = tasks_dict[cat]
            passed = 0
            for item in cat_tasks:
                prompt = item["prompt"]
                vtype = item.get("validation_type", "contains")
                max_tok = 384 if cat in ["coding", "web_dev"] else 128

                t0 = time.time()
                out = run_inference(CLI_PATH, MODEL_PATH, flags, prompt, max_tokens=max_tok)
                dt = time.time() - t0

                ok = False
                if vtype == "json_tool_call":
                    ok = evaluate_json_tool(item, out)
                elif vtype == "contains":
                    ok = evaluate_contains(item["expected"], out)
                elif vtype == "keywords":
                    ok = evaluate_keywords(item["keywords"], out)
                elif vtype == "python_unit_test":
                    ok = evaluate_python_code(item, out)

                if ok:
                    passed += 1
                    total_p += 1
                total_n += 1

                status_sym = "PASS" if ok else "FAIL"
                snip = out.replace("\n", " ")[:50]
                print(f"  [{status_sym}] {item['id']} ({dt:.1f}s) | {snip}")

            acc = (passed / len(cat_tasks) * 100) if cat_tasks else 0
            cfg_results[cat] = f"{passed}/{len(cat_tasks)} ({acc:.1f}%)"
            print(f"Category {cat}: {passed}/{len(cat_tasks)} ({acc:.1f}%)")

        tot_acc = (total_p / total_n * 100) if total_n else 0
        cfg_results["TOTAL"] = f"{total_p}/{total_n} ({tot_acc:.1f}%)"
        summary_matrix[cfg_name] = cfg_results

    # Print Final Markdown Table
    print("\n" + "=" * 78)
    print("FINAL COMPARISON MATRIX:")
    print("=" * 78)
    header = ["Configuration"] + [c.upper() for c in categories] + ["TOTAL"]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for cfg_name, res in summary_matrix.items():
        row = [cfg_name] + [res.get(c, "-") for c in categories] + [res.get("TOTAL", "-")]
        print("| " + " | ".join(row) + " |")

if __name__ == "__main__":
    main()
