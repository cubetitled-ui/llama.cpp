#!/usr/bin/env python3
"""
optuna_recurrent_tuning.py
Optuna-driven Bayesian optimization of recurrent hyperparameters:
  - layer: [10, 11, 12, 13, 14, 15] (focusing around reasoning centroid)
  - gamma: [0.01, 0.30] (log or uniform continuous scale)
  - recurrent_mode: [1 (ORSD), 2 (SNC-MD), 3 (REG-CAV)]
  - t_iterations: [2, 3]

Focus set (10 tasks):
  1. Failed at Baseline T=1 (Need breakthroughs):
     - code_01_regex_nfa
     - code_03_lazy_segment_tree
     - code_07_avl_tree_invariants
     - code_09_expression_calculator_shunting_yard
     - code_10_interval_tree_overlap
  2. Regressed at T=2 gamma=0.12 (Must protect/recover):
     - code_05_dinic_max_flow
     - code_06_diff_patch_engine
     - code_08_transactional_key_value
  3. Stable Anchor Verification:
     - code_02_bytecode_vm
     - code_04_lisp_interpreter
"""

import os
import sys
import json
import time
import re
import subprocess
import optuna
from optuna.samplers import TPESampler

CLI_BIN = "/home/cune/llama.cpp/build-cuda-vnni/bin/llama-cli"
MODEL_PATH = "/home/cune/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
COMPREHENSIVE_PATH = "/home/cune/llama.cpp/benchmarks/comprehensive/dataset.json"
STORAGE_DB = "sqlite:////home/cune/llama.cpp/benchmarks/optuna_recurrent.db"
STUDY_NAME = "recurrent_hypers_focus_set_v1"

TARGET_TASKS = [
    # Breakthrough targets
    "code_01_regex_nfa",
    "code_03_lazy_segment_tree",
    "code_07_avl_tree_invariants",
    "code_09_expression_calculator_shunting_yard",
    "code_10_interval_tree_overlap",
    # Regression protection targets
    "code_05_dinic_max_flow",
    "code_06_diff_patch_engine",
    "code_08_transactional_key_value",
    # Invariant stability anchors
    "code_02_bytecode_vm",
    "code_04_lisp_interpreter"
]

def load_focus_tasks():
    with open(COMPREHENSIVE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    task_map = {}
    for cat, tasks in data.get("tasks", {}).items():
        for t in tasks:
            task_map[t["id"]] = t
    return [task_map[tid] for tid in TARGET_TASKS if tid in task_map]

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

def run_inference(layer: int, gamma: float, mode: int, t_iter: int, prompt: str, max_tokens: int = 512) -> str:
    cmd = [
        CLI_BIN,
        "-m", MODEL_PATH,
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
        "--recurrent-layer", str(layer),
        "--recurrent-gate", f"{gamma:.4f}",
        "--recurrent-mode", str(mode)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, input="", timeout=90)
        return clean_output(prompt, res.stdout)
    except Exception as e:
        return f"ERROR: {e}"

def evaluate_code_task(task, output: str):
    code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", output, re.DOTALL)
    code_to_run = code_match.group(1) if code_match else output
    lines = [l for l in code_to_run.split("\n") if not l.startswith("#!") and not l.startswith("```")]
    sanitized_code = "\n".join(lines)
    test_code = task.get("test_code", "")
    test_script = sanitized_code + "\n\n" + test_code
    try:
        r = subprocess.run([sys.executable, "-c", test_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        return (r.returncode == 0)
    except Exception:
        return False

def objective(trial: optuna.Trial) -> float:
    layer = trial.suggest_int("layer", 11, 15)
    gamma = trial.suggest_float("gamma", 0.02, 0.25, log=True)
    mode = trial.suggest_categorical("mode", [1, 2]) # 1: ORSD, 2: SNC-MD (momentum)
    t_iter = trial.suggest_int("t_iterations", 2, 3)

    tasks = load_focus_tasks()
    passed_count = 0
    
    # Weights: breakthrough tasks get slightly higher reward to incentivize novel discoveries
    breakthrough_ids = {"code_01_regex_nfa", "code_03_lazy_segment_tree", "code_07_avl_tree_invariants", "code_09_expression_calculator_shunting_yard", "code_10_interval_tree_overlap"}
    regression_ids = {"code_05_dinic_max_flow", "code_06_diff_patch_engine", "code_08_transactional_key_value"}
    
    score = 0.0
    for idx, task in enumerate(tasks):
        prompt = f"<|im_start|>user\n{task['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        out = run_inference(layer, gamma, mode, t_iter, prompt, max_tokens=512)
        passed = evaluate_code_task(task, out)
        
        if passed:
            passed_count += 1
            if task["id"] in breakthrough_ids:
                score += 1.5
            elif task["id"] in regression_ids:
                score += 1.2
            else:
                score += 1.0
                
        # Pruning check: if after 5 tasks score is 0, prune early to save GPU time
        if idx == 4 and passed_count == 0:
            raise optuna.TrialPruned()

    trial.set_user_attr("passed_count", passed_count)
    trial.set_user_attr("total_tasks", len(tasks))
    print(f"Trial #{trial.number}: layer={layer}, gamma={gamma:.4f}, mode={mode}, T={t_iter} -> Passed: {passed_count}/{len(tasks)} (Score: {score:.1f})")
    return score

def main():
    print("=" * 80)
    print("OPTUNA RECURRENT BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print(f"Target Tasks ({len(TARGET_TASKS)}): {', '.join(TARGET_TASKS)}")
    print(f"Storage: {STORAGE_DB}")
    print("=" * 80)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_DB,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True
    )

    print(f"Existing trials in study: {len(study.trials)}")
    
    # Run 30 optimization trials
    study.optimize(objective, n_trials=30, timeout=7200)

    print("\n" + "=" * 80)
    print("OPTIMIZATION FINISHED!")
    print(f"Best Trial #{study.best_trial.number}: Score = {study.best_value}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Passed count in best trial: {study.best_trial.user_attrs.get('passed_count')}/{len(TARGET_TASKS)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
