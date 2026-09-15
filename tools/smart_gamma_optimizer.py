#!/usr/bin/env python3
"""
smart_gamma_optimizer.py
Intelligent Adaptive Optimizer for Recurrent Deliberation Scale (gamma).
Uses Canary Early-Stopping and Pivot Boundary Tracking to find the optimal gamma on real GPU silicon.
"""

import os
import sys
import json
import time
import re
import subprocess
from typing import Dict, Any, List, Tuple

MODEL_PATH = "/home/cune/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
CLI_BIN = "/home/cune/llama.cpp/build-cuda-vnni/bin/llama-cli"
DATASET_PATH = "/home/cune/llama.cpp/benchmarks/comprehensive/dataset.json"

CANARY_IDS = ["code_04_lisp_interpreter", "code_06_diff_patch_engine", "code_08_transactional_key_value"]
FRONTIER_IDS = ["code_12_tarjan_scc"]
PIVOT_IDS = CANARY_IDS + FRONTIER_IDS

def clean_output(prompt: str, raw_out: str) -> str:
    ans = raw_out.strip()
    if prompt in ans:
        ans = ans.split(prompt, 1)[-1]
    if ans.startswith(">"):
        ans = ans[1:].lstrip()
    ans = re.sub(r"\[\s*Prompt:.*?t/s\s*\]", "", ans, flags=re.DOTALL)
    if "Exiting..." in ans:
        ans = ans.split("Exiting...", 1)[0]
    return ans.strip()

def run_task(task: Dict[str, Any], gamma: float, mode: int = 1) -> Tuple[bool, float]:
    cmd = [
        CLI_BIN,
        "-m", MODEL_PATH,
        "-p", task["prompt"],
        "-n", "512",
        "--temp", "0.0",
        "-ngl", "99",
        "-c", "2048",
        "--log-disable",
        "-st",
        "--simple-io",
        "--recurrent-t", "2",
        "--recurrent-layer", "13",
        "--recurrent-layer-b", "14",
        "--recurrent-gate", f"{gamma:.4f}",
        "--recurrent-mode", str(mode)
    ]
    t0 = time.time()
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, input="", timeout=90)
        dt = time.time() - t0
        output = clean_output(task["prompt"], res.stdout)
        
        m = re.search(r"```(?:python)?\s*(.*?)\s*```", output, re.DOTALL)
        code = m.group(1) if m else output
        full_code = code + "\n" + task["test_code"]
        
        test_res = subprocess.run([sys.executable, "-c", full_code], capture_output=True, text=True, timeout=15)
        passed = (test_res.returncode == 0)
        return passed, dt
    except Exception as e:
        return False, time.time() - t0

def evaluate_gamma(tasks_by_id: Dict[str, Any], gamma: float, mode: int = 1) -> Dict[str, Any]:
    print(f"\n==================================================")
    print(f"[*] Testing candidate gamma = {gamma:.4f} (Mode {mode})")
    print(f"==================================================")
    
    results = {}
    
    # Phase 1: Test Canaries (Fail-Fast)
    print("[1/2] Evaluating Stability Canaries & Frontier Unlocker...")
    pivots_passed = 0
    for tid in PIVOT_IDS:
        task = tasks_by_id[tid]
        ok, dt = run_task(task, gamma, mode)
        results[tid] = ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tid} ({dt:.2f}s)")
        if ok: pivots_passed += 1
    
    canaries_ok = all(results[cid] for cid in CANARY_IDS)
    frontier_ok = results["code_12_tarjan_scc"]
    
    print(f"  -> Pivot score: {pivots_passed}/{len(PIVOT_IDS)} (Canaries OK: {canaries_ok}, Tarjan OK: {frontier_ok})")
    
    # Phase 2: If stable, evaluate the remaining tasks
    remaining_ids = [tid for tid in tasks_by_id.keys() if tid not in PIVOT_IDS]
    if canaries_ok:
        print("[2/2] Canaries intact! Evaluating remaining 9 tasks...")
        for tid in remaining_ids:
            task = tasks_by_id[tid]
            ok, dt = run_task(task, gamma, mode)
            results[tid] = ok
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {tid} ({dt:.2f}s)")
    else:
        print(f"  [!] Canary failure detected. Skipping remaining tasks to conserve time.")
        for tid in remaining_ids:
            results[tid] = False
            
    total_passed = sum(1 for v in results.values() if v)
    total_tasks = len(tasks_by_id)
    acc = total_passed / total_tasks * 100.0
    print(f"[*] Result for gamma={gamma:.4f}: {total_passed}/{total_tasks} ({acc:.1f}%)")
    
    return {
        "gamma": gamma,
        "mode": mode,
        "score": total_passed,
        "total": total_tasks,
        "accuracy": acc,
        "results": results
    }

def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tasks = data["tasks"]["coding"]
    tasks_by_id = {t["id"]: t for t in tasks}
    
    # Candidate gamma values focused around the golden bracket [0.06 .. 0.12]
    candidates = [0.06, 0.07, 0.08, 0.09, 0.10, 0.12]
    
    summary = []
    
    for gamma in candidates:
        res = evaluate_gamma(tasks_by_id, gamma, mode=1)
        summary.append(res)
        with open("gamma_optimization_summary.json", "w", encoding="utf-8") as out_f:
            json.dump(summary, out_f, indent=2)
            
    print("\n" + "="*60)
    print("FINAL OPTIMIZATION SUMMARY TABLE")
    print("="*60)
    print(f"{'Gamma':<10} | {'Score':<10} | {'Accuracy':<10} | {'Tarjan SCC':<12} | {'Stability'}")
    print("-"*60)
    for s in summary:
        tarjan = "PASS" if s["results"].get("code_12_tarjan_scc") else "FAIL"
        stab = "STABLE" if all(s["results"].get(c) for c in CANARY_IDS) else "DEGRADED"
        print(f"{s['gamma']:<10.4f} | {s['score']}/{s['total']:<8} | {s['accuracy']:<9.1f}% | {tarjan:<12} | {stab}")

if __name__ == "__main__":
    main()
