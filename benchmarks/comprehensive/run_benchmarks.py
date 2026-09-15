#!/usr/bin/env python3
"""
run_benchmarks.py
Execution engine for the Comprehensive Multi-Domain Benchmark Suite.
Evaluates Tool Calling, Logic Reasoning, Python Algorithmic Coding (unit tests), and Web Development.
"""

import os
import sys
import json
import time
import re
import argparse
import subprocess
from typing import Dict, Any, List, Optional

def clean_output(prompt: str, raw_out: str) -> str:
    ans = raw_out.strip()
    if prompt in ans:
        ans = ans.split(prompt, 1)[-1]
    if ans.startswith(">"):
        ans = ans[1:].lstrip()
    # Remove trailing telemetry banner safely without cutting valid code brackets
    ans = re.sub(r"\[\s*Prompt:.*?t/s\s*\]", "", ans, flags=re.DOTALL)
    if "Exiting..." in ans:
        ans = ans.split("Exiting...", 1)[0]
    return ans.strip()

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
        return clean_output(prompt, res.stdout)
    except Exception as e:
        return f"ERROR: {str(e)}"

def evaluate_json_tool(task: Dict[str, Any], output: str) -> bool:
    try:
        start_idx = output.find("{")
        end_idx = output.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return False
        payload = json.loads(output[start_idx:end_idx+1])
        name = payload.get("name") or payload.get("function")
        if name != task["expected_tool"]:
            return False
        args = payload.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        for req_k in task.get("required_keys", []):
            if req_k not in args:
                return False
        for req_k, req_v in task.get("required_values", {}).items():
            actual_v = str(args.get(req_k, "")).lower().lstrip(".")
            expected_v = str(req_v).lower().lstrip(".")
            if actual_v != expected_v:
                return False
        return True
    except Exception:
        return False

def evaluate_contains(expected: str, output: str) -> bool:
    cleaned = output.lower().replace("$", "").replace(",", "").replace("%", "")
    return expected.lower() in cleaned

def evaluate_keywords(keywords: List[str], output: str) -> bool:
    output_lower = output.lower()
    return any(kw.lower() in output_lower for kw in keywords)

def evaluate_python_code(task: Dict[str, Any], output: str) -> bool:
    code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", output, re.DOTALL)
    code_to_run = code_match.group(1) if code_match else output
    
    # Strip any potential command lines or markdown headers
    lines = [l for l in code_to_run.split("\n") if not l.startswith("#!") and not l.startswith("```")]
    sanitized_code = "\n".join(lines)
    
    test_code = task.get("test_code", "")
    full_script = f"{sanitized_code}\n\n{test_code}"
    
    try:
        env = {}
        exec(full_script, env, env)
        return True
    except Exception as e:
        return False

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Benchmark Suite")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--cli", default="/home/cune/llama.cpp/build-cuda-vnni/bin/llama-cli", help="Path to llama-cli")
    parser.add_argument("--dataset", default="/home/cune/llama.cpp/benchmarks/comprehensive/dataset.json", help="Path to dataset.json")
    parser.add_argument("--category", choices=["all", "agentic_loop", "tool_calling", "logic", "coding", "web_dev"], default="all")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[], help="Extra arguments for llama-cli")

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset {args.dataset} not found.")
        sys.exit(1)

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks_dict = data.get("tasks", {})
    categories = [args.category] if args.category != "all" else list(tasks_dict.keys())

    print("=" * 78)
    print(f"COMPREHENSIVE BENCHMARK: {os.path.basename(args.model)}")
    print(f"CLI Flags: {' '.join(args.extra_args)}")
    print("=" * 78)

    results = {}
    total_passed = 0
    total_tasks = 0

    for cat in categories:
        cat_tasks = tasks_dict.get(cat, [])
        passed = 0
        print(f"\n--- Category: {cat.upper()} ({len(cat_tasks)} items) ---")
        for item in cat_tasks:
            total_tasks += 1
            tid = item["id"]
            prompt = item.get("prompt", item.get("turn1_prompt", ""))
            vtype = item.get("validation_type", "contains")
            max_tok = 512 if cat in ["coding", "web_dev"] else 128

            t0 = time.time()
            output = run_inference(args.cli, args.model, args.extra_args, prompt, max_tokens=max_tok)
            latency = time.time() - t0

            is_correct = False
            if vtype == "agentic_multi_turn":
                turn1_prompt = item["turn1_prompt"]
                turn1_out = run_inference(args.cli, args.model, args.extra_args, turn1_prompt, max_tokens=128)
                tool_called = (item["tool_name"].lower() in turn1_out.lower())
                turn2_prompt = item["turn2_prompt"]
                output = run_inference(args.cli, args.model, args.extra_args, turn2_prompt, max_tokens=128)
                grounded = any(exp.lower() in output.lower() for exp in item["expected_final"])
                is_correct = tool_called and grounded
            elif vtype == "json_tool_call":
                is_correct = evaluate_json_tool(item, output)
            elif vtype == "contains":
                is_correct = evaluate_contains(item["expected"], output)
            elif vtype == "keywords":
                is_correct = evaluate_keywords(item["keywords"], output)
            elif vtype == "python_unit_test":
                is_correct = evaluate_python_code(item, output)

            if is_correct:
                passed += 1
                total_passed += 1
                status = "PASS"
            else:
                status = "FAIL"

            snippet = output.replace("\n", " ")[:60]
            print(f"[{status}] {tid} ({latency:.2f}s) | Snippet: {snippet}...")
            if not is_correct and cat in ["logic", "tool_calling"]:
                exp = item.get("expected") or item.get("expected_tool") or item.get("keywords")
                print(f"       Expected: {exp}")

        cat_acc = (passed / len(cat_tasks) * 100) if cat_tasks else 0
        results[cat] = {"passed": passed, "total": len(cat_tasks), "accuracy": cat_acc}
        print(f"Subtotal {cat}: {passed}/{len(cat_tasks)} ({cat_acc:.1f}%)")

    total_acc = (total_passed / total_tasks * 100) if total_tasks else 0
    print("\n" + "=" * 78)
    print(f"FINAL SCORE: {total_passed}/{total_tasks} ({total_acc:.1f}% accuracy)")
    print("=" * 78)

if __name__ == "__main__":
    main()
