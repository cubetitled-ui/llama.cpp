#!/usr/bin/env python3
"""
run_10_architectures.py
Benchmarking 10 Distinct Recurrent / Cognitive Architectures on NVIDIA RTX 3050 (GA107, 6GB).
Evaluates all 10 architectures + Baseline (T=1) on the standardized 7-question Deductive Logic suite.
Uses real silicon execution via llama-cli, LoRA Step 200 checkpoint, and strict evaluation metrics.
"""

import os
import sys
import json
import time
import re
from typing import Dict, Any, List

sys.path.insert(0, "/home/cune/llama.cpp")
from tools.mcp_server.server import LOGIC_QUESTIONS, execute_llama_cli

MODEL_PATH = "/home/cune/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
LORA_PATH = "/home/cune/llama.cpp/models/recurrent_core_lora/checkpoint_step_200.gguf"

ARCHITECTURES = [
    {
        "id": "baseline",
        "name": "Baseline Vanilla (T=1)",
        "flags": []
    },
    {
        "id": "arch_1_ccmc",
        "name": "Arch 1: Canonical Contractive Mid-Core (CCMC-2)",
        "flags": ["--recurrent-t", "2", "--recurrent-layer", "13", "--recurrent-layer-b", "14", "--recurrent-a", "0.90", "--recurrent-b", "0.10", "--recurrent-gate", "1.00", "--lora", LORA_PATH]
    },
    {
        "id": "arch_2_mwqc",
        "name": "Arch 2: Meso-Window Quad-Core (MWQC-4)",
        "flags": ["--recurrent-t", "2", "--recurrent-layer", "11", "--recurrent-layer-b", "14", "--recurrent-a", "0.90", "--recurrent-b", "0.08", "--recurrent-gate", "0.85", "--lora", LORA_PATH]
    },
    {
        "id": "arch_3_hocl",
        "name": "Arch 3: High-Order Cognitive Loop (HOCL-3)",
        "flags": ["--recurrent-t", "3", "--recurrent-layer", "13", "--recurrent-layer-b", "14", "--recurrent-a", "0.90", "--recurrent-b", "0.10", "--recurrent-gate", "1.00", "--lora", LORA_PATH]
    },
    {
        "id": "arch_4_cadc",
        "name": "Arch 4: Conservative Attractor Damped Core (CADC-2)",
        "flags": ["--recurrent-t", "2", "--recurrent-layer", "13", "--recurrent-layer-b", "14", "--recurrent-a", "0.95", "--recurrent-b", "0.05", "--recurrent-gate", "0.60", "--lora", LORA_PATH]
    },
    {
        "id": "arch_5_aiec",
        "name": "Arch 5: Aggressive Innovation Exploration Core (AIEC-2)",
        "flags": ["--recurrent-t", "2", "--recurrent-layer", "13", "--recurrent-layer-b", "14", "--recurrent-a", "0.75", "--recurrent-b", "0.25", "--recurrent-gate", "1.00", "--lora", LORA_PATH]
    },
    {
        "id": "arch_6_lslc",
        "name": "Arch 6: Late-Semantic Pre-Logit Core (LSLC-2)",
        "flags": ["--recurrent-t", "2", "--recurrent-layer", "20", "--recurrent-layer-b", "21", "--recurrent-a", "0.92", "--recurrent-b", "0.08", "--recurrent-gate", "0.70", "--lora", LORA_PATH]
    },
    {
        "id": "arch_7_ebsc",
        "name": "Arch 7: Early-Bridge Syntactic-Semantic Core (EBSC-2)",
        "flags": ["--recurrent-t", "2", "--recurrent-layer", "7", "--recurrent-layer-b", "8", "--recurrent-a", "0.88", "--recurrent-b", "0.12", "--recurrent-gate", "0.90", "--lora", LORA_PATH]
    },
    {
        "id": "arch_8_slfb",
        "name": "Arch 8: Single-Layer Focused Bottleneck (SLFB-1)",
        "flags": ["--recurrent-t", "2", "--recurrent-layer", "14", "--recurrent-layer-b", "14", "--recurrent-a", "0.90", "--recurrent-b", "0.10", "--recurrent-gate", "1.00", "--lora", LORA_PATH]
    },
    {
        "id": "arch_9_bcdc",
        "name": "Arch 9: Balanced Critical Damping Core (BCDC-2)",
        "flags": ["--recurrent-t", "2", "--recurrent-layer", "13", "--recurrent-layer-b", "14", "--recurrent-a", "0.85", "--recurrent-b", "0.15", "--recurrent-gate", "0.85", "--lora", LORA_PATH]
    },
    {
        "id": "arch_10_afac",
        "name": "Arch 10: Asymmetric Frozen-Anchor Dominant Core (AFAC-2)",
        "flags": ["--recurrent-t", "2", "--recurrent-layer", "13", "--recurrent-layer-b", "14", "--recurrent-a", "0.80", "--recurrent-b", "0.20", "--recurrent-gate", "0.75", "--lora", LORA_PATH]
    }
]

def main():
    print("=" * 80)
    print("BENCHMARKING 10 RECURRENT / COGNITIVE ARCHITECTURES (+ BASELINE)")
    print(f"Base Model: {MODEL_PATH}")
    print(f"LoRA Adapter: {LORA_PATH}")
    print("=" * 80)

    all_results = []

    for arch_idx, arch in enumerate(ARCHITECTURES):
        arch_id = arch["id"]
        arch_name = arch["name"]
        flags = arch["flags"]

        print(f"\n[{arch_idx+1}/{len(ARCHITECTURES)}] Testing: {arch_name}")
        print(f"Flags: {' '.join(flags) if flags else '(Baseline T=1)'}")

        arch_summary = {
            "id": arch_id,
            "name": arch_name,
            "passed": 0,
            "total": len(LOGIC_QUESTIONS),
            "total_latency": 0.0,
            "details": []
        }

        for q in LOGIC_QUESTIONS:
            qid = q["id"]
            qtext = q["question"]
            expected = q["expected"]

            res = execute_llama_cli(MODEL_PATH, qtext, flags, max_tokens=32)
            out = res.get("output", "")
            dur = res.get("latency_sec", 0.0)
            arch_summary["total_latency"] += dur

            clean_out = out.lower().replace("$", "").replace(",", "")
            # Check for exact or contained expected number
            is_pass = bool(re.search(rf"\b{re.escape(expected.lower())}\b", clean_out)) or (expected.lower() in clean_out)
            if is_pass:
                arch_summary["passed"] += 1

            status_mark = "PASS" if is_pass else "FAIL"
            print(f"  - {qid:15s}: [{status_mark}] ({dur:.2f}s) | Expected: '{expected}' | Got: '{out[:40]}'")

            arch_summary["details"].append({
                "id": qid,
                "expected": expected,
                "output": out,
                "passed": is_pass,
                "latency": dur
            })

        arch_summary["score_pct"] = round((arch_summary["passed"] / len(LOGIC_QUESTIONS)) * 100, 1)
        arch_summary["total_latency"] = round(arch_summary["total_latency"], 2)
        print(f"  ==> Result: {arch_summary['passed']}/{len(LOGIC_QUESTIONS)} ({arch_summary['score_pct']}%) in {arch_summary['total_latency']}s")
        all_results.append(arch_summary)

    # Output Summary Table
    print("\n" + "=" * 80)
    print("FINAL SUMMARY MATRIX: 10 ARCHITECTURES + BASELINE")
    print("=" * 80)
    print(f"{'Architecture':<45} | {'Score':<10} | {'Pct':<8} | {'Latency':<8}")
    print("-" * 80)
    for res in all_results:
        print(f"{res['name']:<45} | {res['passed']}/{res['total']:<8} | {res['score_pct']:<6}% | {res['total_latency']}s")
    print("=" * 80)

    # Save to JSON
    out_file = "/home/cune/llama.cpp/benchmarks/results_10_architectures.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {out_file}")

if __name__ == "__main__":
    main()
