#!/usr/bin/env python3
"""
run_qwen35_reap_eval.py
Autonomous verification of Flagstone8878/Qwen3.5-18B-REAP-A3B-Coding-Q4_K_M.gguf
Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, GA107, sm_86) + CPU MoE Offloading (-cmoe)
Evaluates Baseline (T=1) vs Recurrent ORSD (T=2) on Silicon.
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
MODEL_PATH = "/home/cune/models/Qwen3.5-18B-REAP-A3B-Coding-Q4_K_M.gguf"
RESULTS_FILE = "/home/cune/llama.cpp/benchmarks/qwen35_reap_results.json"

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

def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def execute_qwen35(prompt: str, recurrent: bool = False, gamma: float = 0.08, max_tokens: int = 160) -> Tuple[str, float, float]:
    cmd = [
        CLI_BIN,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", "0.0",
        "-ngl", "99",
        "-cmoe",
        "-t", "6",
        "-c", "1024",
        "--log-disable",
        "-st",
        "--simple-io",
        "--reasoning", "off",
        "-fa", "on",
        "-fgu",
        "-b", "512",
        "-ub", "256",
        "--no-warmup"
    ]
    if recurrent:
        # Qwen3.5 MoE has 40 layers. Middle layers: 20, 21
        cmd.extend([
            "--recurrent-t", "2",
            "--recurrent-layer", "20",
            "--recurrent-layer-b", "21",
            "--recurrent-gate", f"{gamma:.4f}",
            "--recurrent-mode", "1"
        ])

    t0 = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        dt = time.time() - t0
        raw_out = res.stdout
        speed = 0.0
        m = re.search(r"([0-9.]+)\s+tokens per second", res.stderr + "\n" + res.stdout)
        if m:
            speed = float(m.group(1))
        elif dt > 0:
            tokens = len(raw_out.split())
            speed = round(tokens / dt, 1)
        cleaned = clean_model_output(prompt, raw_out)
        return cleaned, dt, speed
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 600.0, 0.0
    except Exception as e:
        return f"ERROR: {e}", 0.0, 0.0

def eval_tarjan(code: str) -> bool:
    test_harness = """
import sys

# Extracted code
{{CODE}}

def test_tarjan():
    # Graph: 0->1, 1->2, 2->0, 1->3, 3->4, 4->5, 5->3
    # SCCs: {0,1,2}, {3,4,5}
    adj = {0: [1], 1: [2, 3], 2: [0], 3: [4], 4: [5], 5: [3]}
    res = tarjan_scc(adj)
    # Normalize sets
    sccs = [frozenset(c) for c in res]
    expected = [frozenset([0, 1, 2]), frozenset([3, 4, 5])]
    assert len(sccs) == 2, f"Expected 2 SCCs, got {len(sccs)}"
    for exp in expected:
        assert exp in sccs, f"Missing SCC {exp}"

    # DAG: 0->1, 1->2
    adj_dag = {0: [1], 1: [2], 2: []}
    res_dag = [frozenset(c) for c in tarjan_scc(adj_dag)]
    assert len(res_dag) == 3, f"Expected 3 SCCs in DAG, got {len(res_dag)}"
    print("ALL_TESTS_PASSED")

if __name__ == "__main__":
    test_tarjan()
"""
    clean = extract_code(code)
    test_script = test_harness.replace("{{CODE}}", clean)
    p = subprocess.run([sys.executable, "-c", test_script], capture_output=True, text=True, timeout=10)
    return "ALL_TESTS_PASSED" in p.stdout

def eval_monty_hall_4doors(ans: str) -> bool:
    # 4 doors, 1 car, 3 goats. Contestant picks 1 door (P=1/4). Host opens 1 goat door.
    # 2 remaining closed doors share 3/4 probability -> 3/8 (37.5%) each.
    # Check for 3/8, 0.375, or 37.5%
    ans_lower = ans.lower()
    return "3/8" in ans or "37.5" in ans or "0.375" in ans

def eval_lisp(code: str) -> bool:
    test_harness = """
import sys
{{CODE}}

def test_lisp():
    assert evaluate("(+ 1 2)") == 3
    assert evaluate("(* (+ 2 3) 4)") == 20
    assert evaluate("(if (> 5 3) 10 20)") == 10
    print("ALL_TESTS_PASSED")

if __name__ == "__main__":
    test_lisp()
"""
    clean = extract_code(code)
    test_script = test_harness.replace("{{CODE}}", clean)
    p = subprocess.run([sys.executable, "-c", test_script], capture_output=True, text=True, timeout=10)
    return "ALL_TESTS_PASSED" in p.stdout

def eval_kv_store(code: str) -> bool:
    test_harness = """
import sys
{{CODE}}

def test_kv():
    kv = KVStore()
    kv.set("a", 10)
    assert kv.get("a") == 10
    kv.begin()
    kv.set("a", 20)
    assert kv.get("a") == 20
    kv.rollback()
    assert kv.get("a") == 10
    kv.begin()
    kv.set("a", 30)
    kv.commit()
    assert kv.get("a") == 30
    print("ALL_TESTS_PASSED")

if __name__ == "__main__":
    test_kv()
"""
    clean = extract_code(code)
    test_script = test_harness.replace("{{CODE}}", clean)
    p = subprocess.run([sys.executable, "-c", test_script], capture_output=True, text=True, timeout=10)
    return "ALL_TESTS_PASSED" in p.stdout

def main():
    print("=" * 70)
    print("VERIFYING QWEN3.5-18B-REAP-A3B-Coding-Q4_K_M (MoE CPU Offloading)")
    print("=" * 70)

    # 1. Warmup / Basic Inference test
    print("\n[Step 1] Basic Inference Verification:")
    out, dt, sp = execute_qwen35("Write a Python function to compute the factorial of n recursively.", recurrent=False, max_tokens=64)
    print(f"Time: {dt:.2f}s | Speed: {sp:.1f} t/s")
    print(f"Output preview:\n{out[:200]}...")

    if "ERROR" in out or "TIMEOUT" in out:
        print("FAILED TO EXECUTE INFERENCE!")
        sys.exit(1)

    print("\n[Step 2] Recurrent Graph Execution Verification (T=2, Layer 20-21, Mode=1):")
    out_rec, dt_rec, sp_rec = execute_qwen35("Write a Python function to compute the factorial of n recursively.", recurrent=True, gamma=0.08, max_tokens=64)
    print(f"Time: {dt_rec:.2f}s | Speed: {sp_rec:.1f} t/s")
    print(f"Output preview:\n{out_rec[:200]}...")

    # Tasks to evaluate
    tasks = [
        {
            "id": "code_tarjan_scc",
            "prompt": "Implement Tarjan's strongly connected components algorithm in Python: def tarjan_scc(adj): -> List[List[int]]. adj is a dict mapping node to list of neighbor nodes. Return list of SCCs. Output only the code.",
            "validator": eval_tarjan
        },
        {
            "id": "logic_monty_hall_4doors",
            "prompt": "Suppose you're on a game show with 4 doors: behind 1 is a car, behind the other 3 are goats. You pick Door 1. The host, Monty Hall, who knows what's behind the doors, opens Door 3 to reveal a goat. There are now 3 closed doors: Door 1 (your choice), Door 2, and Door 4. The host offers you the option to switch to either Door 2 or Door 4. What is the probability of winning the car if you switch to Door 2? Give the exact fractional or decimal probability and explain your reasoning concisely.",
            "validator": eval_monty_hall_4doors
        },
        {
            "id": "canary_lisp_interpreter",
            "prompt": "Implement a minimal Lisp evaluator in Python: def evaluate(expr: str) -> int. Support +, *, >, and if. Example: evaluate('(+ 1 2)') -> 3, evaluate('(* (+ 2 3) 4)') -> 20, evaluate('(if (> 5 3) 10 20)') -> 10. Output only python code.",
            "validator": eval_lisp
        },
        {
            "id": "canary_transactional_kv",
            "prompt": "Implement an in-memory Key-Value store with nested transactions in Python: class KVStore with methods set(k, v), get(k), begin(), commit(), rollback(). Rollback restores previous state. Output only python code.",
            "validator": eval_kv_store
        }
    ]

    results = {}

    print("\n[Step 3] Comparative Evaluation (Baseline T=1 vs Recurrent T=2 gamma=0.12):")
    for t in tasks:
        tid = t["id"]
        print(f"\n--- Task: {tid} ---")
        
        # Baseline T=1
        out_base, dt_base, sp_base = execute_qwen35(t["prompt"], recurrent=False, max_tokens=220)
        pass_base = t["validator"](out_base)
        print(f"  [T=1 Baseline]       Pass: {pass_base} | {dt_base:.1f}s | {sp_base:.1f} t/s")

        # Recurrent T=2 (gamma=0.12 - Pareto Optimum)
        out_rec12, dt_rec12, sp_rec12 = execute_qwen35(t["prompt"], recurrent=True, gamma=0.12, max_tokens=220)
        pass_rec12 = t["validator"](out_rec12)
        print(f"  [T=2 g=0.12 Pareto]  Pass: {pass_rec12} | {dt_rec12:.1f}s | {sp_rec12:.1f} t/s")

        results[tid] = {
            "baseline": {"passed": pass_base, "time": dt_base, "speed": sp_base, "output_preview": out_base[:150]},
            "recurrent_g12": {"passed": pass_rec12, "time": dt_rec12, "speed": sp_rec12, "output_preview": out_rec12[:150]}
        }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"RESULTS SAVED TO {RESULTS_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    main()
