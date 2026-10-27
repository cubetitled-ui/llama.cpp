#!/usr/bin/env python3
"""
Arm: recurrent mode 2 (snc -- EMA momentum + RMS rescale, experimental), layer 14, gate 0.0331,
T=2, on the 52-task suite.

Same grader and same per-category token budget as the reference arm
(benchmarks/eval_baseline_t1.py): any deviation is refused by bench_contract.check_arm() before
inference starts, and the results file records the contract so a cached verdict can be attributed
to this arm. See benchmarks/README-BENCHMARKS.md -- a single run of one arm on one GPU is not a
result, it is one sample.
"""

import re
import subprocess
import sys
import time
from typing import Dict, Any, Tuple

import bench_contract
from bench_contract import CLI_BIN, MODEL_PATH, OUTPUT_FILE_DIR, REF_BUDGET, REF_LOGIC_ACCEPT

DATASET_PATH = bench_contract.DATASET_PATH
OUTPUT_FILE = OUTPUT_FILE_DIR + "champion_momentum_benchmark_results.json"

CONFIG = {
    "id": "champion_momentum_t2",
    "name": "Optuna Champion (Mode 2 snc, L=14, gate=0.0331, T=2)",
    "layer": 14,
    "gamma": 0.0331,
    "mode": 2,
    "t": 2
}

# Reference grader and reference budget: this arm must stay comparable with the baseline.
BUDGET = dict(REF_BUDGET)
LOGIC_ACCEPT = dict(REF_LOGIC_ACCEPT)

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

def extract_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def run_inference(prompt: str, max_tokens: int = 512) -> Tuple[str, float]:
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
        "--recurrent-t", str(CONFIG["t"]),
        "--recurrent-layer", str(CONFIG["layer"]),
        "--recurrent-gate", f"{CONFIG['gamma']:.4f}",
        "--recurrent-mode", str(CONFIG["mode"])
    ]
    t0 = time.time()
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, input="", timeout=90)
        dur = time.time() - t0
        return clean_output(prompt, res.stdout), dur
    except subprocess.TimeoutExpired:
        return "", 90.0
    except Exception as e:
        return f"Error: {e}", 0.0

def eval_task(cat: str, task: dict) -> Tuple[bool, float, str]:
    if cat == "agentic_loop":
        t1_prompt = task["turn1_prompt"]
        out1, d1 = run_inference(t1_prompt, max_tokens=BUDGET[cat])
        tool_called = (task["tool_name"].lower() in out1.lower())

        t2_prompt = task["turn2_prompt"]
        out2, d2 = run_inference(t2_prompt, max_tokens=BUDGET[cat])
        grounded = any(exp.lower() in out2.lower() for exp in task.get("expected_final", []))
        return (tool_called and grounded), d1 + d2, out2[:120]

    elif cat == "coding":
        prompt = task["prompt"]
        out, dur = run_inference(prompt, max_tokens=BUDGET[cat])
        code = extract_code(out)
        lines = [l for l in code.split("\n") if not l.startswith("#!") and not l.startswith("```")]
        sanitized = "\n".join(lines)
        test_script = sanitized + "\n\n" + task.get("test_code", "")
        try:
            r = subprocess.run([sys.executable, "-c", test_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            ok = (r.returncode == 0)
        except Exception:
            ok = False
        return ok, dur, out[:120]

    elif cat == "logic":
        prompt = task["prompt"]
        out, dur = run_inference(prompt, max_tokens=BUDGET[cat])
        exp = str(task.get("expected", "")).strip().lower()
        cleaned = out.lower().replace("$", "").replace(",", "").replace("%", "")
        ok = exp in cleaned
        return ok, dur, out[:120]

    elif cat == "web_dev":
        prompt = task["prompt"]
        out, dur = run_inference(prompt, max_tokens=BUDGET[cat])
        keywords = task.get("keywords", [])
        ok = any(kw.lower() in out.lower() for kw in keywords)
        return ok, dur, out[:120]

    return False, 0.0, ""

def main():
    bench_contract.run_suite(CONFIG, BUDGET, LOGIC_ACCEPT, OUTPUT_FILE, eval_task)

if __name__ == "__main__":
    main()
