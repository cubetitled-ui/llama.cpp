#!/usr/bin/env python3
"""
Arm that used to claim "Cascade Dual-Core MSRD", layers A=14 / B=16, mode 5 (41/52).

Two reasons that number is void:

1. mode 5 has no branch in build_recurrent_core, so the arm ran vanilla LTI with a two-layer core;
   llama.cpp now refuses mode 5 with a non-zero exit and bench_contract.check_arm() aborts the run.
2. the arm was not comparable with the baseline: it accepted extra logic answers ("0.4" for
   "2/5", "0.375", "1.94%") and spent 3x the tokens on logic (768 instead of 256) and 1.5x on
   coding (768 instead of 512).

A and B at distance 2 is a supported topology in itself (sandwich: loop {14, 16} + bridge 15,
chosen automatically in build_recurrent_core), and llama-context now validates an explicitly set
bridge to be strictly inside (lo, hi) so no layer can run twice. cascade_dual_core_results.json was
deleted: its verdicts were vanilla under a different grader and budget.
"""

import re
import subprocess
import sys
import time
from typing import Dict, Any, Tuple

import bench_contract
from bench_contract import CLI_BIN, MODEL_PATH, OUTPUT_FILE_DIR, REF_BUDGET, REF_LOGIC_ACCEPT

DATASET_PATH = bench_contract.DATASET_PATH
OUTPUT_FILE = OUTPUT_FILE_DIR + "cascade_dual_core_results.json"

CONFIG = {
    "id": "cascade_dual_core_14_16",
    "name": "Cascade Dual-Core (mode 5 -- NOT IMPLEMENTED, refused by llama.cpp), A=14 B=16, gate=0.0331, T=2",
    "layer": 14,
    "layer_b": 16,
    "gamma": 0.0331,
    "mode": 5,
    "t": 2
}

# Reference grader and reference budget, restored (see the module docstring).
BUDGET = dict(REF_BUDGET)
LOGIC_ACCEPT = dict(REF_LOGIC_ACCEPT)

def clean_output(prompt: str, raw_out: str) -> str:
    idx = raw_out.find(prompt)
    if idx != -1:
        ans = raw_out[idx + len(prompt):]
    else:
        ans = raw_out
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
        "--recurrent-layer-b", str(CONFIG["layer_b"]),
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
        # Reference grading (identical to eval_baseline_t1.py). The LaTeX-fraction and percentage
        # accepts this arm used to have are removed: they are extra accepts that this arm alone
        # was given, so its score was not comparable with the baseline's.
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
    bench_contract.run_suite(CONFIG, BUDGET, LOGIC_ACCEPT, OUTPUT_FILE, eval_task, throttle=0.5)

if __name__ == "__main__":
    main()
