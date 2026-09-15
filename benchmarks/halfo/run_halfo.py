"""
run_halfo.py
Execution engine and scoring harness for the Halfo Benchmark Suite.
Tests logic, tool-calling schema compliance, and multilingual idiomatic translation.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from typing import Dict, Any, List

def run_inference(cli_bin: str, model_path: str, extra_flags: List[str], prompt: str, max_tokens: int = 256) -> str:
    cmd = [
        cli_bin,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", "0.0",
        "-ngl", "99",
        "-c", "2048",
        "--log-disable",
        "-st",
        "--simple-io"
    ] + extra_flags
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, input="", timeout=120)
        output = res.stdout.strip()
        if prompt in output:
            output = output.split(prompt, 1)[-1]
        if ">" in output:
            output = output.split(">", 1)[-1]
        if "[" in output and "t/s" in output:
            output = output.split("[", 1)[0]
        if "Exiting..." in output:
            output = output.replace("Exiting...", "")
        return output.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def evaluate_logic(expected: str, output: str) -> bool:
    output_lower = output.lower().replace("$", "").replace(",", "")
    return expected.lower() in output_lower

def evaluate_tool_calling(task: Dict[str, Any], output: str) -> bool:
    try:
        # Find json block
        start_idx = output.find("{")
        end_idx = output.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return False
        json_str = output[start_idx:end_idx+1]
        data = json.loads(json_str)
        
        name = data.get("name") or data.get("tool") or data.get("function")
        if name != task["expected_tool"]:
            return False
            
        args = data.get("arguments") or data.get("parameters") or {}
        for k in task.get("required_keys", []):
            if k not in args:
                return False
        for k, v in task.get("required_values", {}).items():
            if str(v).lower() not in str(args.get(k, "")).lower():
                return False
        return True
    except Exception:
        return False

def evaluate_translation(keywords: List[str], output: str) -> bool:
    out_lower = output.lower()
    return any(kw.lower() in out_lower for kw in keywords)

def main():
    parser = argparse.ArgumentParser(description="Halfo Benchmark Evaluation Harness")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--cli", default="/home/cune/llama.cpp/build-cuda-vnni/bin/llama-cli", help="Path to llama-cli")
    parser.add_argument("--dataset", default="/home/cune/llama.cpp/benchmarks/halfo/dataset.json", help="Path to dataset.json")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[], help="Extra arguments for llama-cli (e.g. recurrent flags)")
    parser.add_argument("--category", choices=["all", "logic", "tool_calling", "multilingual_translation"], default="all")
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    tasks_dict = bench_data["tasks"]
    categories = [args.category] if args.category != "all" else ["logic", "tool_calling", "multilingual_translation"]

    results = {}
    print("=" * 78)
    print(f"HALFO BENCHMARK SUITE: {args.model}")
    print(f"CLI Flags: {' '.join(args.extra_args)}")
    print("=" * 78)

    total_passed = 0
    total_tasks = 0

    for cat in categories:
        cat_tasks = tasks_dict.get(cat, [])
        passed = 0
        print(f"\n--- Category: {cat.upper()} ({len(cat_tasks)} items) ---")
        for item in cat_tasks:
            total_tasks += 1
            tid = item["id"]
            prompt = item["prompt"]
            t0 = time.time()
            output = run_inference(args.cli, args.model, args.extra_args, prompt)
            latency = time.time() - t0

            is_correct = False
            if cat == "logic":
                is_correct = evaluate_logic(item["expected"], output)
            elif cat == "tool_calling":
                is_correct = evaluate_tool_calling(item, output)
            elif cat == "multilingual_translation":
                is_correct = evaluate_translation(item["keywords"], output)

            if is_correct:
                passed += 1
                total_passed += 1
                status_str = "PASS"
            else:
                status_str = "FAIL"

            print(f"[{status_str}] {tid} ({latency:.2f}s) | Output snippet: {output[:65]}...")
            if not is_correct:
                expected_str = item.get("expected") or item.get("expected_tool") or item.get("keywords")
                print(f"       Expected: {expected_str}")

        cat_acc = (passed / len(cat_tasks) * 100) if cat_tasks else 0
        results[cat] = {"passed": passed, "total": len(cat_tasks), "accuracy": cat_acc}
        print(f"Subtotal {cat}: {passed}/{len(cat_tasks)} ({cat_acc:.1f}%)")

    total_acc = (total_passed / total_tasks * 100) if total_tasks else 0
    print("\n" + "=" * 78)
    print(f"HALFO FINAL RESULT: {total_passed}/{total_tasks} ({total_acc:.1f}% accuracy)")
    print("=" * 78)

if __name__ == "__main__":
    main()
