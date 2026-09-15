#!/usr/bin/env python3
"""
rlang/pipeline.py
Single-Command Automated Recurrent Build & Benchmark Pipeline for llama.cpp.

Workflow:
1. Parse user's RLang theory script (.rlang) via custom Lexer & Parser.
2. Validate mathematical stability bounds (-1 <= a < 1) and layer topologies.
3. Emit optimized C++ GGML computational graph routines.
4. Atomically patch `src/llama-graph.cpp` with safety backups.
5. Rebuild `llama.cpp` (`llama-cli`, `llama-bench`).
6. Execute automated benchmark (Fast Logic Suite, Halfo Suite, or custom prompt).
7. Collect outputs, verify answers, measure latency / throughput, and report metrics.
"""

import os
import sys
import time
import argparse
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recurrent_engine.rlang.compiler import RLangCompiler
from recurrent_engine.rlang.parser import Program, LoopStage

LOGIC_QUESTIONS = [
    {"id": "bat_and_ball", "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents? Respond with only the final number.", "expected": "5"},
    {"id": "train_speed", "question": "A train travels 120 km in 2.5 hours. What is its average speed in km/h? Respond with only the number.", "expected": "48"},
    {"id": "snail_in_well", "question": "A snail is at the bottom of a 10-meter well. Each day it climbs up 3 meters, and each night it slips back down 2 meters. On which day does the snail reach the top of the well? Respond with only the number.", "expected": "8"},
    {"id": "lily_pads", "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days would it take for the patch to cover half of the lake? Respond with only the number.", "expected": "47"},
    {"id": "widget_machines", "question": "If 5 machines take 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets? Respond with only the number.", "expected": "5"},
    {"id": "sheep_puzzle", "question": "Farmer John has 17 sheep. All but 9 run away. How many sheep does Farmer John have left? Respond with only the number.", "expected": "9"},
    {"id": "sister_age", "question": "When I was 4 years old, my sister was half my age. Now I am 100 years old. How old is my sister? Respond with only the number.", "expected": "98"}
]

def find_default_model(models_dir: str) -> Optional[str]:
    p = Path(models_dir)
    if not p.exists():
        return None
    ggufs = list(p.glob("**/*.gguf"))
    if not ggufs:
        return None
    for g in ggufs:
        if "falcon" in g.name.lower():
            return str(g)
    for g in ggufs:
        if "qwen" in g.name.lower():
            return str(g)
    return str(ggufs[0])

def run_fast_logic_suite(llama_cli: str, model_path: str, timeout_per_q: int = 40) -> Dict[str, Any]:
    results = []
    correct_count = 0
    total_time = 0.0

    print(f"\n================================================================================")
    print(f" FAST LOGIC BENCHMARK SUITE (Silicon Execution on RTX 3050)")
    print(f" Model: {Path(model_path).name}")
    print(f" Binary: {llama_cli}")
    print(f"================================================================================\n")

    for idx, item in enumerate(LOGIC_QUESTIONS, 1):
        q_id = item["id"]
        prompt = item["question"]
        expected = item["expected"]

        cmd = [
            llama_cli,
            "-m", model_path,
            "-p", prompt,
            "-n", "32",
            "--temp", "0.0",
            "-ngl", "16",
            "-c", "1024",
            "-st",
            "--simple-io"
        ]

        t0 = time.perf_counter()
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_per_q)
            dt = time.perf_counter() - t0
            total_time += dt

            raw_out = res.stdout.strip()
            # Clean up prompt echo if present
            if raw_out.startswith(prompt):
                ans = raw_out[len(prompt):].strip()
            else:
                ans = raw_out

            ans_tokens = [tok.strip(".,;:?!()[]'\"") for tok in ans.split()]
            is_correct = (expected in ans_tokens) or (expected == ans.strip())
            if is_correct:
                correct_count += 1

            status_str = "PASS" if is_correct else "FAIL"
            print(f"[{idx}/{len(LOGIC_QUESTIONS)}] {q_id:<16} | Result: {status_str:<4} | Latency: {dt:5.2f}s | Expected: {expected:<4} | Got: {ans[:40]}")

            results.append({
                "id": q_id,
                "expected": expected,
                "response": ans,
                "correct": is_correct,
                "latency_s": round(dt, 2)
            })

        except subprocess.TimeoutExpired:
            print(f"[{idx}/{len(LOGIC_QUESTIONS)}] {q_id:<16} | Result: TIMEOUT (> {timeout_per_q}s)")
            results.append({
                "id": q_id,
                "expected": expected,
                "response": "TIMEOUT",
                "correct": False,
                "latency_s": timeout_per_q
            })

    total_q = len(LOGIC_QUESTIONS)
    accuracy_pct = (correct_count / total_q) * 100.0
    print(f"\n--------------------------------------------------------------------------------")
    print(f" SUMMARY: {correct_count}/{total_q} Correct ({accuracy_pct:.1f}%) | Total Time: {total_time:.2f}s")
    print(f"--------------------------------------------------------------------------------\n")

    return {
        "accuracy": accuracy_pct,
        "correct": correct_count,
        "total": total_q,
        "total_time_s": round(total_time, 2),
        "details": results
    }

def main():
    parser = argparse.ArgumentParser(description="RLang Single-Command Build & Benchmark Pipeline")
    parser.add_argument("--script", type=str, help="Path to RLang script (.rlang)")
    parser.add_argument("--model", type=str, help="Path to GGUF model file (auto-detected if omitted)")
    parser.add_argument("--build-dir", type=str, default="build-cuda-vnni", help="CMake build directory")
    parser.add_argument("--bench", choices=["fast_logic", "halfo", "throughput", "prompt", "none"], default="fast_logic", help="Benchmark to execute")
    parser.add_argument("--prompt", type=str, default="Explain the difference between momentum and standard gradient descent in 2 sentences.", help="Custom prompt for --bench prompt")
    parser.add_argument("--n-predict", type=int, default=64, help="Tokens to predict")
    parser.add_argument("--dry-run", action="store_true", help="Emit C++ code without patching or compiling")
    parser.add_argument("--restore", action="store_true", help="Restore original llama-graph.cpp and rebuild")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="Build parallelism")

    args = parser.parse_args()

    compiler = RLangCompiler(llama_cpp_root=str(PROJECT_ROOT))
    build_path = PROJECT_ROOT / args.build_dir
    llama_cli = str(build_path / "bin" / "llama-cli")
    llama_bench = str(build_path / "bin" / "llama-bench")

    # Mode: Restore
    if args.restore:
        print("[Pipeline] Restoring original src/llama-graph.cpp...")
        if compiler.restore_graph_cpp():
            print("[Pipeline] Restored successfully. Rebuilding binaries...")
            cmd = ["cmake", "--build", str(build_path), "--target", "llama-cli", "llama-bench", f"-j{args.jobs}"]
            res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if res.returncode == 0:
                print("[Pipeline] Rebuild successful.")
                sys.exit(0)
            else:
                print("[Pipeline] Rebuild FAILED.")
                sys.exit(1)
        else:
            print("[Pipeline] No backup found at src/llama-graph.cpp.orig.")
            sys.exit(1)

    if not args.script:
        parser.print_help()
        print("\nError: --script is required unless --restore is specified.")
        sys.exit(1)

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"Error: Script file '{args.script}' not found.")
        sys.exit(1)

    # 1. Parse RLang Script
    print(f"[Pipeline] 1. Parsing RLang script: {script_path.name}")
    try:
        prog = compiler.compile_file(str(script_path))
    except Exception as e:
        print(f"[Pipeline] Parse Error: {e}")
        sys.exit(1)

    # 2. Validate Topology
    print(f"[Pipeline] 2. Validating mathematical topology...")
    topo = compiler.program_to_topology(prog)
    errors = topo.validate()
    if errors:
        print(f"[Pipeline] Validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    print(f"[Pipeline] Topology is mathematically valid (Arch: {topo.architecture}, Layers: {topo.total_layers}, Stages: {len(prog.stages)})")

    # 3. Generate C++ GGML Code
    print(f"[Pipeline] 3. Emitting C++ GGML computational graph...")
    cpp_code = compiler.generate_cpp_core(prog)

    if args.dry_run:
        print("\n--- GENERATED C++ CORE (Dry Run) ---")
        print(cpp_code)
        print("------------------------------------")
        sys.exit(0)

    # 4. Patch src/llama-graph.cpp
    print(f"[Pipeline] 4. Patching src/llama-graph.cpp...")
    compiler.patch_graph_cpp(cpp_code)
    print(f"[Pipeline] Successfully patched src/llama-graph.cpp.")

    # 5. Build Binaries
    print(f"[Pipeline] 5. Compiling llama.cpp targets (llama-cli, llama-bench) with {args.jobs} threads...")
    build_cmd = ["cmake", "--build", str(build_path), "--target", "llama-cli", "llama-bench", f"-j{args.jobs}"]
    t_b0 = time.perf_counter()
    build_res = subprocess.run(build_cmd, cwd=str(PROJECT_ROOT))
    dt_build = time.perf_counter() - t_b0

    if build_res.returncode != 0:
        print(f"[Pipeline] Compilation FAILED. Reverting patch...")
        compiler.restore_graph_cpp()
        sys.exit(1)
    print(f"[Pipeline] Compilation succeeded in {dt_build:.2f}s.")

    # 6. Resolve Model
    model_path = args.model
    if not model_path:
        model_path = find_default_model(str(PROJECT_ROOT / "models"))
    if not model_path or not Path(model_path).exists():
        print(f"[Pipeline] Error: No GGUF model found. Specify --model <path.gguf>")
        sys.exit(1)

    # 7. Benchmark Execution
    if args.bench == "none":
        print("[Pipeline] Pipeline completed without running benchmarks.")
        sys.exit(0)

    elif args.bench == "fast_logic":
        run_fast_logic_suite(llama_cli, model_path)

    elif args.bench == "throughput":
        print(f"\n[Pipeline] Running throughput test (llama-bench)...")
        cmd = [llama_bench, "-m", model_path, "-p", "512", "-n", "128", "-ngl", "16", "-r", "2"]
        subprocess.run(cmd)

    elif args.bench == "prompt":
        print(f"\n[Pipeline] Running single inference prompt: '{args.prompt}'")
        cmd = [
            llama_cli,
            "-m", model_path,
            "-p", args.prompt,
            "-n", str(args.n_predict),
            "--temp", "0.0",
            "-ngl", "16",
            "-c", "2048",
            "-st",
            "--simple-io"
        ]
        subprocess.run(cmd)

    elif args.bench == "halfo":
        halfo_script = PROJECT_ROOT / "benchmarks" / "halfo" / "run_halfo.py"
        if halfo_script.exists():
            print(f"\n[Pipeline] Running Halfo benchmark suite...")
            cmd = [sys.executable, str(halfo_script), "--model", model_path]
            subprocess.run(cmd)
        else:
            print(f"[Pipeline] Halfo runner not found at {halfo_script}")

    print("\n[Pipeline] End-to-end RLang pipeline execution complete.")

if __name__ == "__main__":
    main()
