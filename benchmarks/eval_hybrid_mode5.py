#!/usr/bin/env python3
"""
Arm that used to claim "Hybrid Ortho-Momentum, mode 5" (43/52 vs baseline 40/52).

Mode 5 never existed in build_recurrent_core: --recurrent-mode 5 was accepted by the argument
parser, had no branch in the graph, and ran plain vanilla LTI, while the context log printed
"Vanilla LTI" for it. On top of that this arm was not comparable with the baseline: it accepted
extra logic answers ("0.4" for "2/5", "0.375", "1.94%") and it spent 3x the tokens on logic
(768 instead of 256) and 1.5x on coding (768 instead of 512), so the "gain" was grader plus budget,
not the model.

Both defects are now closed: llama.cpp refuses mode 5 with a non-zero exit, the arm carries the
reference grader and the reference budget, and bench_contract.check_arm() aborts the run if either
is changed. hybrid_ortho_momentum_benchmark_results.json was deleted: its 43 tasks were vanilla
under a different grader and budget and cannot be attributed to any arm. Re-run this script only
after picking an implemented mode.
"""

import re
import subprocess
import sys
import time
from typing import Dict, Any, Tuple

import bench_contract
from bench_contract import CLI_BIN, MODEL_PATH, OUTPUT_FILE_DIR, REF_BUDGET, REF_LOGIC_ACCEPT

DATASET_PATH = bench_contract.DATASET_PATH
OUTPUT_FILE = OUTPUT_FILE_DIR + "hybrid_ortho_momentum_benchmark_results.json"

CONFIG = {
    "id": "hybrid_ortho_momentum_t2",
    "name": "Hybrid Ortho-Momentum (mode 5 -- NOT IMPLEMENTED, refused by llama.cpp)",
    "layer": 14,
    "gamma": 0.0331,
    "mode": 5,
    "t": 2
}

# Reference grader and reference budget, restored. The extra logic accepts and the 768-token
# budgets that this arm used to carry are gone; see the module docstring.
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
        # was given, worth several tasks of the reported 40 -> 43 "gain".
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
