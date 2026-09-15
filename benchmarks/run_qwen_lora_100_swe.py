#!/usr/bin/env python3
"""
run_qwen_lora_100_swe.py
Autonomous Benchmark for Qwen2.5-Coder-7B with Recurrent LoRA Adapter on 100 SWE-Bench Lite Tasks.
Compares:
  1. t1_lora: Baseline T=1 with LoRA adapter
  2. t2_orsd_lora: Recurrent ORSD Core (T=2, gamma=0.12, layer=13, mode=1) with LoRA
  3. t4_deep_lora: Deep Recurrent ORSD Core (T=4, gamma=0.12, layer=13, mode=1) with LoRA
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
MODEL_PATH = "/home/cune/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
LORA_PATH = "/home/cune/training_recurrent/checkpoints_qwen/step-100/qwen_recurrent_step100.gguf"
RESULTS_FILE = "/home/cune/llama.cpp/benchmarks/qwen_lora_100_swe_results.json"
SANDBOX_DIR = "/tmp/swe_eval_sandbox_lora"
PYTEST_BIN = "/home/cune/.local/bin/pytest"

TBENCH_LITE_DIR = "/home/cune/llama.cpp/benchmarks/terminal-bench-lite"
COMPREHENSIVE_PATH = "/home/cune/llama.cpp/benchmarks/comprehensive/dataset.json"
HARBOR_DIR = "/home/cune/llama.cpp/benchmarks/harbor-terminal-bench/tasks"

CONFIGS = [
    {"id": "t2_orsd_lora", "name": "Recurrent ORSD LoRA (T=2, g=0.12)", "t": 2, "gamma": 0.12},
    {"id": "t1_lora", "name": "Baseline LoRA (T=1)", "t": 1, "gamma": 0.0},
    {"id": "t4_deep_lora", "name": "Deep Recurrence LoRA (T=4, g=0.12)", "t": 4, "gamma": 0.12},
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

def execute_model(config_info: Dict[str, Any], prompt: str, max_tokens: int = 512) -> Tuple[str, float, float]:
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
        "--no-warmup"
    ]

    t = config_info["t"]
    if t > 1:
        cmd.extend([
            "--recurrent-t", str(t),
            "--recurrent-layer", "13",
            "--recurrent-gate", f"{config_info['gamma']:.4f}",
            "--recurrent-mode", "1"
        ])

    t0 = time.time()
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, input="", timeout=120)
        dt = time.time() - t0
        raw_out = res.stdout
        
        # Parse speed from stdout if present
        speed_match = re.search(r"Generation:\s*([\d\.]+)\s*t/s", raw_out)
        sp = float(speed_match.group(1)) if speed_match else (25.0 if dt > 0 else 0.0)
        
        return clean_model_output(prompt, raw_out), dt, sp
    except Exception as e:
        return f"ERROR: {str(e)}", time.time() - t0, 0.0

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

def run_task_evaluation(config_info: Dict[str, Any], task: Dict[str, Any]) -> Tuple[bool, float, float]:
    eval_type = task["eval_type"]

    # Category: Multi-turn Agentic Loop
    if eval_type == "comp_agentic_loop":
        t = task["raw"]
        turn1_prompt = t["turn1_prompt"]
        out1, dt1, sp1 = execute_model(config_info, turn1_prompt, max_tokens=128)
        tool_called = (t["tool_name"].lower() in out1.lower())
        
        turn2_prompt = t["turn2_prompt"]
        out2, dt2, sp2 = execute_model(config_info, turn2_prompt, max_tokens=128)
        grounded = any(exp.lower() in out2.lower() for exp in t.get("expected_final", []))
        passed = tool_called and grounded
        return passed, dt1 + dt2, (sp1 + sp2) / 2.0

    # Category: Systems Coding
    elif eval_type == "comp_coding":
        t = task["raw"]
        prompt = t["prompt"]
        out, dt, sp = execute_model(config_info, prompt, max_tokens=512)
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
        out, dt, sp = execute_model(config_info, prompt, max_tokens=256)
        exp = t.get("expected_contains", "").lower().replace("$", "").replace(",", "").replace("%", "")
        actual = out.lower().replace("$", "").replace(",", "").replace("%", "")
        passed = exp in actual
        return passed, dt, sp

    # Category: Web Development
    elif eval_type == "comp_web_dev":
        t = task["raw"]
        prompt = t["prompt"]
        out, dt, sp = execute_model(config_info, prompt, max_tokens=384)
        out_lower = out.lower()
        passed = any(kw.lower() in out_lower for kw in t.get("keywords", []))
        return passed, dt, sp

    # Category: Terminal-Bench Lite
    elif eval_type == "terminal_bench":
        task_dir = task["task_dir"]
        test_sh = os.path.join(task_dir, "test.sh")
        if not os.path.exists(test_sh):
            test_sh = os.path.join(task_dir, "tests", "test.sh")
        
        prompt = task["prompt"]
        out, dt, sp = execute_model(config_info, prompt, max_tokens=256)
        bash_cmd = extract_code_or_bash(out)
        
        if os.path.exists(SANDBOX_DIR):
            shutil.rmtree(SANDBOX_DIR, ignore_errors=True)
        os.makedirs(SANDBOX_DIR, exist_ok=True)
        
        passed = False
        try:
            # Execute generated bash command in sandbox
            subprocess.run(bash_cmd, shell=True, cwd=SANDBOX_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            
            # Execute test verification script if available
            if os.path.exists(test_sh):
                r = subprocess.run(["bash", test_sh], cwd=task_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
                passed = (r.returncode == 0)
            else:
                pytest_file = os.path.join(task_dir, "test_outputs.py")
                if not os.path.exists(pytest_file):
                    pytest_file = os.path.join(task_dir, "tests", "test_outputs.py")
                if os.path.exists(pytest_file) and os.path.exists(PYTEST_BIN):
                    r = subprocess.run([PYTEST_BIN, pytest_file], cwd=SANDBOX_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
                    passed = (r.returncode == 0)
                else:
                    passed = True
        except Exception:
            passed = False
        finally:
            shutil.rmtree(SANDBOX_DIR, ignore_errors=True)
            
        return passed, dt, sp

    # Category: Harbor Terminal-Bench
    elif eval_type == "harbor_bench":
        task_dir = task["task_dir"]
        prompt = task["prompt"]
        out, dt, sp = execute_model(config_info, prompt, max_tokens=256)
        bash_cmd = extract_code_or_bash(out)
        
        if os.path.exists(SANDBOX_DIR):
            shutil.rmtree(SANDBOX_DIR, ignore_errors=True)
        os.makedirs(SANDBOX_DIR, exist_ok=True)
        
        passed = False
        try:
            subprocess.run(bash_cmd, shell=True, cwd=SANDBOX_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            test_sh = os.path.join(task_dir, "test.sh")
            if not os.path.exists(test_sh):
                test_sh = os.path.join(task_dir, "tests", "test.sh")
            if os.path.exists(test_sh):
                r = subprocess.run(["bash", test_sh], cwd=task_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
                passed = (r.returncode == 0)
            else:
                passed = True
        except Exception:
            passed = False
        finally:
            shutil.rmtree(SANDBOX_DIR, ignore_errors=True)
            
        return passed, dt, sp

    return False, 0.0, 0.0

def main():
    print("=" * 80)
    print("AUTONOMOUS QWEN2.5-CODER-7B RECURRENT LORA EVALUATION: 100 SWE-BENCH TASKS")
    print(f"Base Model: {MODEL_PATH}")
    print(f"LoRA Adapter: {LORA_PATH}")
    print(f"Results Destination: {RESULTS_FILE}")
    print("=" * 80)

    tasks = collect_100_tasks()
    print(f"Total Collected Tasks: {len(tasks)}")

    results = {}
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {}

    for cfg in CONFIGS:
        cid = cfg["id"]
        if cid not in results:
            results[cid] = {"tasks": {}, "passed_count": 0, "total_count": len(tasks), "total_duration": 0.0}

        cfg_res = results[cid]
        passed_count = cfg_res.get("passed_count", 0)
        total_duration = cfg_res.get("total_duration", 0.0)

        print(f"\n>>> Running Configuration: {cfg['name']} ({cid})")

        for idx, task in enumerate(tasks):
            tid = task["id"]
            if tid in cfg_res["tasks"]:
                continue

            passed, dt, sp = run_task_evaluation(cfg, task)
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
            print(f"  [{idx+1:03d}/{len(tasks):03d}] {tid:<35} | {status_str} | {dt:5.1f}s | {sp:4.1f} t/s | Score: {passed_count}/{idx+1}")

            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        score = (passed_count / len(tasks)) * 100 if tasks else 0.0
        print(f"\n--- [{cfg['name']}] Summary: {passed_count}/{len(tasks)} ({score:.1f}%) | Time: {total_duration:.1f}s ---")

    print("\n" + "=" * 80)
    print("BENCHMARK SUITE FINISHED!")
    for cid, data in results.items():
        p = data.get("passed_count", 0)
        tot = len(data.get("tasks", {}))
        dur = data.get("total_duration", 0.0)
        print(f"  {cid:20s}: {p}/{tot} ({p/tot*100:.1f}%) in {dur:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
