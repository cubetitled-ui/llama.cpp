#!/usr/bin/env python3
"""
run_multi_model_swe_benchmark.py
Comprehensive Autonomous Benchmark for Latent Recurrence on RTX 3050 Laptop GPU.
Evaluates 3 Frontier 7B Architectures across 3 Configurations:
  1. Baseline Vanilla (T=1)
  2. Recurrent ORSD Core (T=2, gamma*=0.12, mode=1)
  3. Deep Recurrence Core (T=4, gamma*=0.12, mode=1)

Benchmark Suite: 100 Tasks
- Terminal-Bench Lite (39 real-world sysadmin/terminal tasks)
- Comprehensive Hardened Systems (52 tasks: 13 Agentic, 13 Systems Coding Unit Tests, 13 Logic, 13 Web)
- Harbor Terminal-Bench (9 tasks)
"""

import os
import sys
import json
import time
import re
import subprocess
import shutil
from typing import Dict, Any, List, Tuple

CLI_BIN = "/home/cune/llama.cpp/build-cuda-vnni/bin/llama-cli"
TBENCH_LITE_DIR = "/home/cune/llama.cpp/benchmarks/terminal-bench-lite"
COMPREHENSIVE_PATH = "/home/cune/llama.cpp/benchmarks/comprehensive/dataset.json"
HARBOR_DIR = "/home/cune/llama.cpp/benchmarks/harbor-terminal-bench/tasks"
RESULTS_FILE = "/home/cune/llama.cpp/benchmarks/multi_model_swe_benchmark_results.json"
SANDBOX_DIR = "/tmp/swe_eval_sandbox"
PYTEST_BIN = "/home/cune/.local/bin/pytest"

MODELS = [
    {
        "id": "qwen25_coder_7b",
        "name": "Qwen2.5-Coder-7B-Instruct",
        "path": "/home/cune/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "layer_a": 13,
        "layer_b": 14,
        "extra_flags": []
    },
    {
        "id": "falcon_h1r_7b",
        "name": "Falcon-H1R-7B-IQ4_XS",
        "path": "/home/cune/models/Falcon-H1R-7B-IQ4_XS.gguf",
        "layer_a": 13,
        "layer_b": 14,
        "extra_flags": []
    },
    {
        "id": "mistral_7b",
        "name": "Mistral-7B-Instruct-v0.2",
        "path": "/home/cune/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "layer_a": 15,
        "layer_b": 16,
        "extra_flags": []
    }
]

CONFIGS = [
    {"id": "t1_baseline", "name": "Baseline (T=1)", "t": 1, "gamma": 0.0},
    {"id": "t2_orsd", "name": "Recurrent ORSD (T=2, g=0.12)", "t": 2, "gamma": 0.12},
    {"id": "t4_deep", "name": "Deep Recurrence (T=4, g=0.12)", "t": 4, "gamma": 0.12}
]

def clean_model_output(prompt: str, raw_out: str) -> str:
    ans = raw_out.strip()
    if prompt in ans:
        ans = ans.split(prompt, 1)[-1]
    if ans.startswith(">"):
        ans = ans[1:].lstrip()
    ans = re.sub(r"\[\s*Prompt:.*?t/s\s*\]", "", ans, flags=re.DOTALL)
    if "Exiting..." in ans:
        ans = ans.split("Exiting...", 1)[0]
    return ans.strip()

def extract_code_or_bash(text: str) -> str:
    m = re.search(r"```(?:python|bash|sh)?\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def execute_model(model_info: Dict[str, Any], config_info: Dict[str, Any], prompt: str, max_tokens: int = 512) -> Tuple[str, float, float]:
    cmd = [
        CLI_BIN,
        "-m", model_info["path"],
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
        "--no-warmup"
    ] + model_info.get("extra_flags", [])

    t = config_info["t"]
    if t > 1:
        cmd.extend([
            "--recurrent-t", str(t),
            "--recurrent-layer", str(model_info["layer_a"]),
            "--recurrent-layer-b", str(model_info["layer_b"]),
            "--recurrent-gate", f"{config_info['gamma']:.4f}",
            "--recurrent-mode", "1"
        ])

    t0 = time.time()
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        dt = time.time() - t0
        output = clean_model_output(prompt, res.stdout)
        
        speed = 25.0
        sm = re.search(r"Generation:\s*([\d\.]+)\s*t/s", res.stdout)
        if sm:
            speed = float(sm.group(1))
        elif dt > 0:
            speed = round(len(output.split()) / dt, 1)
        return output, dt, speed
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 120.0, 0.0
    except Exception as e:
        return f"ERROR: {e}", 0.0, 0.0

def collect_100_tasks() -> List[Dict[str, Any]]:
    all_tasks = []

    # 1. 52 tasks from comprehensive dataset
    if os.path.exists(COMPREHENSIVE_PATH):
        with open(COMPREHENSIVE_PATH, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
        for cat, task_list in comp_data.get("tasks", {}).items():
            for t in task_list:
                all_tasks.append({
                    "id": t["id"],
                    "category": f"systems_{cat}",
                    "source": "comprehensive",
                    "raw": t,
                    "eval_type": f"comp_{cat}"
                })

    # 2. 39 tasks from Terminal-Bench Lite
    if os.path.exists(TBENCH_LITE_DIR):
        dirs = sorted([d for d in os.listdir(TBENCH_LITE_DIR) if os.path.isdir(os.path.join(TBENCH_LITE_DIR, d)) and not d.startswith(".")])
        for d in dirs:
            instr_path = os.path.join(TBENCH_LITE_DIR, d, "instruction.md")
            if os.path.exists(instr_path):
                with open(instr_path, "r", encoding="utf-8") as f:
                    instr = f.read()
                all_tasks.append({
                    "id": f"tbench_{d}",
                    "category": "terminal_agent",
                    "source": "terminal-bench-lite",
                    "task_dir": os.path.join(TBENCH_LITE_DIR, d),
                    "prompt": f"You are a Linux terminal agent. Execute the necessary commands to solve the following task:\n\n{instr}\n\nProvide your bash solution inside a markdown ```bash block.",
                    "eval_type": "terminal_bench"
                })

    # 3. 9 additional tasks from Harbor Terminal-Bench
    if os.path.exists(HARBOR_DIR):
        harbor_dirs = sorted([d for d in os.listdir(HARBOR_DIR) if os.path.isdir(os.path.join(HARBOR_DIR, d)) and not d.startswith(".")])
        count_needed = 100 - len(all_tasks)
        for d in harbor_dirs[:count_needed]:
            instr_path = os.path.join(HARBOR_DIR, d, "instruction.md")
            if not os.path.exists(instr_path):
                instr_path = os.path.join(HARBOR_DIR, d, "README.md")
            instr = ""
            if os.path.exists(instr_path):
                with open(instr_path, "r", encoding="utf-8") as f:
                    instr = f.read()
            all_tasks.append({
                "id": f"harbor_{d}",
                "category": "heavy_systems",
                "source": "harbor",
                "task_dir": os.path.join(HARBOR_DIR, d),
                "prompt": f"You are a low-level systems engineer. Solve the following problem:\n\n{instr}\n\nProvide your solution inside a markdown block.",
                "eval_type": "harbor_bench"
            })

    return all_tasks[:100]

def run_task_evaluation(model_info: Dict[str, Any], config_info: Dict[str, Any], task: Dict[str, Any]) -> Tuple[bool, float, float]:
    eval_type = task["eval_type"]

    # Category: Multi-turn Agentic Loop
    if eval_type == "comp_agentic_loop":
        t = task["raw"]
        turn1_prompt = t["turn1_prompt"]
        out1, dt1, sp1 = execute_model(model_info, config_info, turn1_prompt, max_tokens=128)
        tool_called = (t["tool_name"].lower() in out1.lower())
        
        turn2_prompt = t["turn2_prompt"]
        out2, dt2, sp2 = execute_model(model_info, config_info, turn2_prompt, max_tokens=128)
        grounded = any(exp.lower() in out2.lower() for exp in t.get("expected_final", []))
        passed = tool_called and grounded
        return passed, dt1 + dt2, (sp1 + sp2) / 2.0

    # Category: Systems Coding
    elif eval_type == "comp_coding":
        t = task["raw"]
        prompt = t["prompt"]
        out, dt, sp = execute_model(model_info, config_info, prompt, max_tokens=512)
        code = extract_code_or_bash(out)
        lines = [l for l in code.split("\n") if not l.startswith("#!") and not l.startswith("```")]
        sanitized = "\n".join(lines)
        test_script = sanitized + "\n\n" + t.get("test_code", "")
        try:
            r = subprocess.run(["python3", "-c", test_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            passed = (r.returncode == 0)
        except Exception:
            passed = False
        return passed, dt, sp

    # Category: Logic Reasoning
    elif eval_type == "comp_logic":
        t = task["raw"]
        prompt = t["prompt"]
        out, dt, sp = execute_model(model_info, config_info, prompt, max_tokens=256)
        expected = str(t.get("expected", "")).strip().lower()
        cleaned = out.lower().replace("$", "").replace(",", "").replace("%", "")
        passed = expected in cleaned
        return passed, dt, sp

    # Category: Web Dev / Fullstack
    elif eval_type == "comp_web_dev":
        t = task["raw"]
        prompt = t["prompt"]
        out, dt, sp = execute_model(model_info, config_info, prompt, max_tokens=512)
        keywords = t.get("keywords", [])
        passed = any(kw.lower() in out.lower() for kw in keywords)
        return passed, dt, sp

    # Category: Terminal-Bench & Harbor
    elif eval_type in ["terminal_bench", "harbor_bench"]:
        task_dir = task.get("task_dir")
        prompt = task["prompt"]
        out, dt, sp = execute_model(model_info, config_info, prompt, max_tokens=512)
        
        if not task_dir or not os.path.exists(task_dir):
            return False, dt, sp

        # Setup sandbox
        if os.path.exists(SANDBOX_DIR):
            shutil.rmtree(SANDBOX_DIR, ignore_errors=True)
        os.makedirs(SANDBOX_DIR, exist_ok=True)

        bash_code = extract_code_or_bash(out)
        script_path = os.path.join(SANDBOX_DIR, "solution.sh")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\nset -e\n" + bash_code)
        os.chmod(script_path, 0o755)

        # Execute agent commands in sandbox
        try:
            subprocess.run(["bash", "solution.sh"], cwd=SANDBOX_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        except Exception:
            pass

        # Check solution vs task tests
        tests_dir = os.path.join(task_dir, "tests")
        passed = False
        if os.path.exists(tests_dir):
            for test_file in os.listdir(tests_dir):
                if test_file.endswith(".sh"):
                    t_path = os.path.join(tests_dir, test_file)
                    try:
                        tr = subprocess.run(["bash", t_path], cwd=SANDBOX_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
                        if tr.returncode == 0:
                            passed = True
                            break
                    except Exception:
                        pass
                elif test_file.endswith(".py"):
                    t_path = os.path.join(tests_dir, test_file)
                    try:
                        tr = subprocess.run([sys.executable, t_path], cwd=SANDBOX_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
                        if tr.returncode == 0:
                            passed = True
                            break
                    except Exception:
                        pass
        return passed, dt, sp

    return False, 0.0, 0.0

def main():
    print("=" * 80)
    print("AUTONOMOUS MULTI-MODEL SWE-BENCH LITE / ACADEMIC 100 HARNESS")
    print("Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, GA107, sm_86)")
    print("Evaluating 3 Architectures x 3 Configurations (T=1, T=2 g=0.12, T=4 g=0.12)")
    print("=" * 80)

    # Filter models to only those that exist
    active_models = [m for m in MODELS if os.path.exists(m["path"])]
    if not active_models:
        print("ERROR: No valid models found on disk!")
        sys.exit(1)

    print(f"Active Models ({len(active_models)}):")
    for m in active_models:
        print(f"  - {m['name']} ({m['path']})")

    tasks = collect_100_tasks()
    print(f"\nTotal Collected Tasks: {len(tasks)}")

    # Load existing results if any to resume gracefully
    results = {}
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {}

    for m in active_models:
        mid = m["id"]
        if mid not in results:
            results[mid] = {}

        print(f"\n{'='*40}\nMODEL: {m['name']}\n{'='*40}")

        for cfg in CONFIGS:
            cid = cfg["id"]
            if cid not in results[mid]:
                results[mid][cid] = {"tasks": {}, "passed_count": 0, "total_count": len(tasks), "total_duration": 0.0}

            cfg_res = results[mid][cid]
            passed_count = cfg_res.get("passed_count", 0)
            total_duration = cfg_res.get("total_duration", 0.0)

            print(f"\n>>> Configuration: {cfg['name']} ({cid})")

            for idx, task in enumerate(tasks):
                tid = task["id"]
                if tid in cfg_res["tasks"]:
                    continue  # already evaluated

                t_start = time.time()
                passed, dt, sp = run_task_evaluation(m, cfg, task)
                
                if passed:
                    passed_count += 1
                total_duration += dt

                cfg_res["tasks"][tid] = {
                    "passed": passed,
                    "duration": dt,
                    "speed": sp,
                    "category": task["category"]
                }
                cfg_res["passed_count"] = passed_count
                cfg_res["total_duration"] = total_duration

                status_str = "PASS" if passed else "FAIL"
                print(f"  [{idx+1:03d}/{len(tasks):03d}] {tid:<35} | {status_str} | {dt:5.1f}s | {sp:4.1f} t/s | Progress: {passed_count}/{idx+1}")

                # Save checkpoint after every task
                with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

            score = (passed_count / len(tasks)) * 100 if tasks else 0.0
            print(f"\n--- {m['name']} [{cfg['name']}] Summary: {passed_count}/{len(tasks)} ({score:.1f}%) | Time: {total_duration:.1f}s ---")

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED ACROSS ALL MODELS AND CONFIGURATIONS!")
    print(f"FINAL RESULTS SAVED TO: {RESULTS_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()
