#!/usr/bin/env python3
"""
run_boundary_sweep.py
Fine-grained sweep of recurrent gate gamma in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
Targeting the 4 critical boundary tasks on NVIDIA GeForce RTX 3050:
1. code_12_tarjan_scc (Gain at 0.18, Fail at 0.08)
2. logic_10_monty_hall_4doors (Gain at 0.08 and 0.18)
3. code_08_transactional_key_value (Pass at 0.08, Regr at 0.18)
4. logic_01_bat_ball (Pass at 0.08, Regr at 0.18)
Goal: Discover the exact Pareto-optimal gamma* achieving net +2 gains with 0 regressions.
"""

import sys
import os
import json
import time

sys.path.insert(0, "/home/cune/llama.cpp")
from benchmarks.run_100_academic_benchmark import (
    collect_100_tasks,
    run_task_evaluation,
    MODEL_PATH,
    CLI_BIN
)

BOUNDARY_TASK_IDS = [
    "code_12_tarjan_scc",
    "logic_10_monty_hall_4doors",
    "code_08_transactional_key_value",
    "logic_01_bat_ball"
]

GAMMAS = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18]

def main():
    print("=" * 70)
    print("PARETO FRONTIER BOUNDARY SWEEP ON RTX 3050 (sm_86)")
    print("Searching for gamma* with maximal gains and zero regressions")
    print("=" * 70)

    all_tasks = {t["id"]: t for t in collect_100_tasks()}
    selected = [all_tasks[tid] for tid in BOUNDARY_TASK_IDS if tid in all_tasks]

    matrix = {tid: {} for tid in BOUNDARY_TASK_IDS}

    for gamma in GAMMAS:
        print(f"\nEvaluating gamma = {gamma:.2f} ...")
        for t in selected:
            tid = t["id"]
            passed, dt, sp = run_task_evaluation(t, recurrent=True, gamma=gamma)
            matrix[tid][f"{gamma:.2f}"] = passed
            status = "PASS" if passed else "FAIL"
            print(f"  gamma={gamma:.2f} | {tid:34s} | {status} ({dt:.1f}s, {sp:.1f} t/s)")

    print("\n" + "=" * 70)
    print("BOUNDARY MATRIX:")
    print(f"{'Task ID':34s} | " + " | ".join([f"g={g:.2f}" for g in GAMMAS]))
    print("-" * 70)
    for tid in BOUNDARY_TASK_IDS:
        row = [f"{'PASS' if matrix[tid].get(f'{g:.2f}') else 'FAIL':6s}" for g in GAMMAS]
        print(f"{tid:34s} | " + " | ".join(row))
    print("=" * 70)

    out_file = "/home/cune/llama.cpp/benchmarks/pareto_gamma_sweep.json"
    with open(out_file, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
