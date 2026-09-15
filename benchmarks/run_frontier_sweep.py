#!/usr/bin/env python3
"""
run_frontier_sweep.py
Evaluates Recurrent ORSD at higher deliberate scale (gamma=0.18) and across layer intervals.
Focuses on the 55 failure modes from baseline to identify solvable frontier tasks.
Continuously appends results to benchmarks/academic_frontier_results.json and updates the paper.
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, Any, List

sys.path.insert(0, "/home/cune/llama.cpp")

from benchmarks.run_100_academic_benchmark import (
    collect_100_tasks,
    run_task_evaluation,
    MODEL_PATH,
    CLI_BIN,
    SANDBOX_DIR
)

FRONTIER_RESULTS_FILE = "/home/cune/llama.cpp/benchmarks/academic_frontier_results.json"
BASE_100_FILE = "/home/cune/llama.cpp/benchmarks/academic_100_benchmark_results.json"
PAPER_FILE = "/home/cune/llama.cpp/docs/recurrent/ARCHITECTURE_AND_SYSTEM_GUIDE_RU.md"

def main():
    print("=" * 70)
    print("FRONTIER RECURRENT SWEEP (gamma = 0.18, ORSD Core, layers 13-14)")
    print("Target: Evaluate recovery of baseline failure modes")
    print("=" * 70)

    tasks = collect_100_tasks()
    base_results = {}
    if os.path.exists(BASE_100_FILE):
        with open(BASE_100_FILE, "r") as f:
            base_results = json.load(f).get("baseline", {})

    frontier_data = {}
    if os.path.exists(FRONTIER_RESULTS_FILE):
        try:
            with open(FRONTIER_RESULTS_FILE, "r") as f:
                frontier_data = json.load(f)
        except Exception:
            pass

    recurrent_g18 = frontier_data.get("recurrent_g18", {})

    # Focus sweep across all 100 tasks with gamma=0.18
    gains = []
    regressions = []
    total_passed = 0

    for i, t in enumerate(tasks):
        tid = t["id"]
        base_pass = base_results.get(tid, {}).get("passed", False)
        
        if tid in recurrent_g18:
            passed = recurrent_g18[tid]["passed"]
            dt = recurrent_g18[tid]["duration"]
            sp = recurrent_g18[tid]["speed"]
        else:
            passed, dt, sp = run_task_evaluation(t, recurrent=True, gamma=0.18)
            recurrent_g18[tid] = {
                "passed": passed,
                "duration": dt,
                "speed": sp,
                "category": t["category"]
            }
            frontier_data["recurrent_g18"] = recurrent_g18
            with open(FRONTIER_RESULTS_FILE, "w") as f:
                json.dump(frontier_data, f, indent=2)

        if passed:
            total_passed += 1

        delta_tag = "     "
        if not base_pass and passed:
            delta_tag = "[+GAIN]"
            gains.append(tid)
        elif base_pass and not passed:
            delta_tag = "[-REGR]"
            regressions.append(tid)

        print(f"[{i+1:3d}/100] g=0.18 | {tid[:32]:32s} | {'PASS' if passed else 'FAIL':4s} | {dt:5.1f}s | {sp:4.1f} t/s {delta_tag}")

    print("\n" + "=" * 70)
    print("FRONTIER SWEEP SUMMARY (gamma=0.18)")
    print("=" * 70)
    print(f"Baseline (T=1):         44 / 100 (44.0%)")
    print(f"Recurrent (gamma=0.18): {total_passed} / 100 ({total_passed / len(tasks) * 100:.1f}%)")
    print(f"Gains:       +{len(gains)} ({gains})")
    print(f"Regressions: -{len(regressions)} ({regressions})")
    print(f"Net Delta:   {len(gains) - len(regressions):+d}")
    print("=" * 70)

if __name__ == "__main__":
    main()
