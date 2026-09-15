#!/usr/bin/env python3
"""
sweep_gamma_failed_tasks.py
Systematic sweep of gamma (recurrent gate) on tasks that failed at gamma=0.12.
Evaluates gamma in [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30] at T=2 and T=3
with the trained Qwen Recurrent LoRA adapter.
"""

import os
import sys
import json
import time
import re
import subprocess
from typing import Dict, Any, List, Tuple

CLI_BIN = "/home/cune/llama.cpp/build-cuda-vnni/bin/llama-cli"
MODEL_PATH = "/home/cune/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
LORA_PATH = "/home/cune/training_recurrent/checkpoints_qwen/step-100/qwen_recurrent_step100.gguf"
COMPREHENSIVE_PATH = "/home/cune/llama.cpp/benchmarks/comprehensive/dataset.json"
RESULTS_FILE = "/home/cune/llama.cpp/benchmarks/gamma_sweep_failed_results.json"

TARGET_FAILED_TASKS = [
    "code_01_regex_nfa",
    "code_03_lazy_segment_tree",
    "code_05_dinic_max_flow",
    "code_06_diff_patch_engine",
    "code_07_avl_tree_invariants",
    "code_08_transactional_key_value",
    "code_09_expression_calculator_shunting_yard",
    "code_10_interval_tree_overlap",
    "agent_10_forex_triangular"
]

GAMMA_VALUES = [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20, 0.25]

def load_target_tasks() -> List[Dict[str, Any]]:
    with open(COMPREHENSIVE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    task_map = {}
    for cat, tasks in data.get("tasks", {}).items():
        for t in tasks:
            t["_cat"] = cat
            task_map[t["id"]] = t
            
    return [task_map[tid] for tid in TARGET_FAILED_TASKS if tid in task_map]

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

def run_inference(gamma: float, t_iter: int, prompt: str, max_tokens: int = 512) -> str:
    cmd = [
        CLI_BIN,
        "-m", MODEL_PATH,
        "--lora", LORA_PATH,
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", "0.0",
        "-ngl", "99",
        "-c", "2048",
        "--log-disable",
        "-st",
        "--simple-io",
        "-fa", "on",
        "-b", "512",
        "-ub", "256",
        "--no-warmup",
        "--recurrent-t", str(t_iter),
        "--recurrent-layer", "13",
        "--recurrent-gate", f"{gamma:.4f}",
        "--recurrent-mode", "1"
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, input="", timeout=120)
        return clean_output(prompt, res.stdout)
    except Exception as e:
        return f"ERROR: {e}"

def evaluate_task(task: Dict[str, Any], output: str) -> Tuple[bool, str]:
    cat = task.get("_cat", "")
    if cat == "coding":
        code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", output, re.DOTALL)
        code_to_run = code_match.group(1) if code_match else output
        lines = [l for l in code_to_run.split("\n") if not l.startswith("#!") and not l.startswith("```")]
        sanitized_code = "\n".join(lines)
        test_code = task.get("test_code", "")
        test_script = sanitized_code + "\n\n" + test_code
        try:
            r = subprocess.run([sys.executable, "-c", test_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            if r.returncode == 0:
                return True, "PASSED"
            else:
                err_line = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "Non-zero exit"
                return False, err_line[:60]
        except subprocess.TimeoutExpired:
            return False, "TimeoutExpired (>10s, infinite loop)"
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:60]}"
            
    elif cat == "agentic_loop":
        # agent_10_forex_triangular: multi-turn
        # Note: in sweep we evaluate single prompt or test if final grounded answer is correct
        exp_list = task.get("expected_final", [])
        passed = any(exp.lower() in output.lower() for exp in exp_list)
        return passed, ("PASSED" if passed else "Missing expected grounded answer")
        
    return False, "Unknown validation"

def main():
    print("=" * 80)
    print("RECURRENT GAMMA (γ) TUNING HARNESS ON FAILED REASONING TASKS")
    print(f"Target Tasks ({len(TARGET_FAILED_TASKS)}): {', '.join(TARGET_FAILED_TASKS)}")
    print(f"Gamma Grid: {GAMMA_VALUES}")
    print("=" * 80)

    tasks = load_target_tasks()
    
    results = {}
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    solved_at_least_once = set()

    for gamma in GAMMA_VALUES:
        key = f"gamma_{gamma:.2f}_t2"
        if key not in results:
            results[key] = {}
            
        print(f"\n>>> Evaluating γ = {gamma:.2f} (T=2, ORSD, Layer 13)")
        passed_in_cfg = 0
        
        for task in tasks:
            tid = task["id"]
            if tid in results[key]:
                if results[key][tid]["passed"]:
                    passed_in_cfg += 1
                    solved_at_least_once.add(tid)
                continue
                
            prompt = task.get("prompt")
            if not prompt:
                # agentic task turn 1 or turn 2
                prompt = task.get("turn1_prompt", "")
                
            full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            t0 = time.time()
            out = run_inference(gamma, 2, full_prompt, max_tokens=512)
            dt = time.time() - t0
            
            passed, reason = evaluate_task(task, out)
            if passed:
                passed_in_cfg += 1
                solved_at_least_once.add(tid)
                status_icon = "✅ PASS"
            else:
                status_icon = "❌ FAIL"
                
            print(f"  [{status_icon}] {tid:<40} | γ={gamma:.2f} | {dt:4.1f}s | {reason}")
            
            results[key][tid] = {
                "passed": passed,
                "reason": reason,
                "duration": dt
            }
            
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)
                
        print(f"--- γ = {gamma:.2f} Summary: {passed_in_cfg}/{len(tasks)} passed ({passed_in_cfg/len(tasks)*100:.1f}%) ---")

    print("\n" + "=" * 80)
    print("GAMMA SWEEP COMPLETE!")
    print(f"Total Unique Tasks Solved Across Any γ: {len(solved_at_least_once)} / {len(tasks)}")
    for tid in TARGET_FAILED_TASKS:
        sol_gammas = [g for g in GAMMA_VALUES if results.get(f"gamma_{g:.2f}_t2", {}).get(tid, {}).get("passed")]
        if sol_gammas:
            print(f"  🎯 {tid:<40}: SOLVED at γ = {sol_gammas}")
        else:
            print(f"  ❌ {tid:<40}: Unsolved across all tested gammas")
    print("=" * 80)

if __name__ == "__main__":
    main()
