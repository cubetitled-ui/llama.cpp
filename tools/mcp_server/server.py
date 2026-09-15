#!/usr/bin/env python3
"""
mcp_server/server.py
Model Context Protocol (MCP) Server for Fast Logic Benchmarking & Recurrent LLM Evaluation.
Exposes MCP tools for AI agents to benchmark models, query hardware telemetry, and run recurrent graphs.
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, Any, List, Optional
from mcp.server.fastmcp import FastMCP

# Standard fast logic questions
LOGIC_QUESTIONS = [
    {
        "id": "bat_and_ball",
        "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents? Respond with only the final number.",
        "expected": "5"
    },
    {
        "id": "train_speed",
        "question": "A train travels 120 km in 2.5 hours. What is its average speed in km/h? Respond with only the number.",
        "expected": "48"
    },
    {
        "id": "snail_in_well",
        "question": "A snail is at the bottom of a 10-meter well. Each day it climbs up 3 meters, and each night it slips back down 2 meters. On which day does the snail reach the top of the well? Respond with only the number.",
        "expected": "8"
    },
    {
        "id": "lily_pads",
        "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days would it take for the patch to cover half of the lake? Respond with only the number.",
        "expected": "47"
    },
    {
        "id": "widget_machines",
        "question": "If 5 machines take 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets? Respond with only the number.",
        "expected": "5"
    },
    {
        "id": "sheep_puzzle",
        "question": "Farmer John has 17 sheep. All but 9 run away. How many sheep does Farmer John have left? Respond with only the number.",
        "expected": "9"
    },
    {
        "id": "sister_age",
        "question": "When I was 4 years old, my sister was half my age. Now I am 100 years old. How old is my sister? Respond with only the number.",
        "expected": "98"
    }
]

# Initialize FastMCP Server
mcp = FastMCP("model-logic-bench")

LLAMA_CLI = "/home/cune/llama.cpp/build-cuda-vnni/bin/llama-cli"
MODELS_DIR = "/home/cune/llama.cpp/models"

def execute_llama_cli(model_path: str, prompt: str, cli_flags: List[str], max_tokens: int = 128) -> Dict[str, Any]:
    # Check free VRAM to decide on GPU layers (to prevent OOM during active training)
    ngl = "99"
    try:
        t_stat = get_hardware_status()
        if t_stat.get("vram_free_mib", 6000) < 2500:
            ngl = "16"
    except Exception:
        ngl = "16"

    cmd = [
        LLAMA_CLI,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", "0.0",
        "-ngl", ngl,
        "-c", "2048",
        "--log-disable",
        "-st",
        "--simple-io"
    ] + cli_flags

    t0 = time.time()
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90)
        dt = time.time() - t0
        raw_out = res.stdout.strip()
        
        # Parse generation output between prompt and exit/telemetry
        ans = raw_out
        if "... (truncated)" in ans:
            ans = ans.split("... (truncated)", 1)[-1]
        elif prompt in ans:
            ans = ans.split(prompt, 1)[-1]
        elif "\n> " in ans:
            # Banner ends with \n> prompt
            # Split off the banner, then if prompt tail is found, split that
            ans = ans.split("\n> ", 1)[-1]
            prompt_tail = prompt.strip().splitlines()[-1] if prompt.strip().splitlines() else ""
            if prompt_tail and prompt_tail in ans:
                ans = ans.split(prompt_tail, 1)[-1]

        if "[" in ans and "t/s" in ans:
            ans = ans.split("[", 1)[0]
        if "Exiting..." in ans:
            ans = ans.replace("Exiting...", "")
        ans = ans.strip()

        return {"success": True, "output": ans, "latency_sec": round(dt, 2), "error": None}
    except Exception as e:
        return {"success": False, "output": "", "latency_sec": round(time.time() - t0, 2), "error": str(e)}

@mcp.tool()
def get_hardware_status() -> Dict[str, Any]:
    """Query real-time NVIDIA GPU VRAM usage, utilization, and temperature."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        parts = [p.strip() for p in res.stdout.strip().split(",")]
        return {
            "gpu_utilization_pct": float(parts[0]),
            "vram_used_mib": float(parts[1]),
            "vram_total_mib": float(parts[2]),
            "temperature_c": float(parts[3]),
            "vram_free_mib": float(parts[2]) - float(parts[1])
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def list_models() -> List[Dict[str, Any]]:
    """Scan and list available GGUF model files for inference and benchmarking."""
    models = []
    if os.path.exists(MODELS_DIR):
        for root, _, files in os.walk(MODELS_DIR):
            for f in files:
                if f.endswith(".gguf") and not f.startswith("ggml-vocab"):
                    p = os.path.join(root, f)
                    sz = os.path.getsize(p) / (1024**3)
                    models.append({"name": f, "path": p, "size_gb": round(sz, 2)})
    return models

@mcp.tool()
def run_model_inference(
    model_path: str,
    prompt: str,
    recurrent_t: int = 1,
    recurrent_layer: int = -1,
    recurrent_layer_b: int = -1,
    recurrent_a: float = 0.90,
    recurrent_b: float = 0.10,
    recurrent_gate: float = 1.00,
    recurrent_config: Optional[str] = None,
    lora_path: Optional[str] = None,
    max_tokens: int = 128
) -> Dict[str, Any]:
    """Execute single prompt inference with configurable recurrent parameters or config/rscript."""
    flags = []
    if recurrent_config:
        flags.extend(["-rc", recurrent_config])
    elif recurrent_t > 1 and recurrent_layer >= 0:
        flags.extend([
            "--recurrent-layer", str(recurrent_layer),
            "--recurrent-t", str(recurrent_t),
            "--recurrent-a", f"{recurrent_a:.2f}",
            "--recurrent-b", f"{recurrent_b:.2f}",
            "--recurrent-gate", f"{recurrent_gate:.2f}",
        ])
        if recurrent_layer_b >= 0:
            flags.extend(["--recurrent-layer-b", str(recurrent_layer_b)])

    if lora_path:
        flags.extend(["--lora", lora_path])

    return execute_llama_cli(model_path, prompt, flags, max_tokens)

@mcp.tool()
def benchmark_logic_fast(
    model_path: str,
    recurrent_t: int = 1,
    recurrent_layer: int = -1,
    recurrent_layer_b: int = -1,
    recurrent_a: float = 0.90,
    recurrent_b: float = 0.10,
    recurrent_gate: float = 1.00,
    recurrent_config: Optional[str] = None,
    lora_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run standardized 7-question logic and arithmetic benchmark suite on a model.
    Checks Bat & Ball, Train speed, Snail in well, Lily pads, Machines, Sheep, and Sister age.
    """
    flags = []
    if recurrent_config:
        flags.extend(["-rc", recurrent_config])
    elif recurrent_t > 1 and recurrent_layer >= 0:
        flags.extend([
            "--recurrent-layer", str(recurrent_layer),
            "--recurrent-t", str(recurrent_t),
            "--recurrent-a", f"{recurrent_a:.2f}",
            "--recurrent-b", f"{recurrent_b:.2f}",
            "--recurrent-gate", f"{recurrent_gate:.2f}",
        ])
        if recurrent_layer_b >= 0:
            flags.extend(["--recurrent-layer-b", str(recurrent_layer_b)])

    if lora_path:
        flags.extend(["--lora", lora_path])

    passed_count = 0
    details = []
    t_start = time.time()

    for item in LOGIC_QUESTIONS:
        qid = item["id"]
        qtext = item["question"]
        expected = item["expected"]

        res = execute_llama_cli(model_path, qtext, flags, max_tokens=64)
        out = res.get("output", "")
        clean_out = out.lower().replace("$", "").replace(",", "")
        
        # Check if expected answer is present
        is_pass = expected.lower() in clean_out
        if is_pass:
            passed_count += 1

        details.append({
            "id": qid,
            "passed": is_pass,
            "expected": expected,
            "output_snippet": out[:80],
            "latency_sec": res.get("latency_sec", 0.0)
        })

    total_time = round(time.time() - t_start, 2)
    score_pct = round((passed_count / len(LOGIC_QUESTIONS)) * 100, 1)

    return {
        "model": os.path.basename(model_path),
        "passed": passed_count,
        "total": len(LOGIC_QUESTIONS),
        "score_pct": score_pct,
        "total_latency_sec": total_time,
        "recurrent_config": {
            "t": recurrent_t,
            "layer_lo": recurrent_layer,
            "layer_hi": recurrent_layer_b,
            "a": recurrent_a,
            "b": recurrent_b,
            "gate": recurrent_gate
        },
        "questions": details
    }

@mcp.tool()
def compile_and_run_rlang(
    rlang_code: str,
    model_path: Optional[str] = None,
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Parses, validates, and compiles an RLang recurrent theory script.
    If dry_run=False, patches src/llama-graph.cpp, rebuilds binaries, and executes the fast logic benchmark on silicon.
    """
    from recurrent_engine.rlang.compiler import RLangCompiler
    compiler = RLangCompiler(llama_cpp_root="/home/cune/llama.cpp")

    try:
        prog = compiler.compile_source(rlang_code)
        topo = compiler.program_to_topology(prog)
        errs = topo.validate()
        if errs:
            return {"status": "error", "message": "Mathematical validation failed", "errors": errs}
        cpp_code = compiler.generate_cpp_core(prog)

        if dry_run:
            return {
                "status": "success",
                "mode": "dry_run",
                "architecture": topo.architecture,
                "total_layers": topo.total_layers,
                "stages_count": len(prog.stages),
                "cpp_code": cpp_code
            }

        # Full compile and bench
        compiler.patch_graph_cpp(cpp_code)
        build_cmd = ["cmake", "--build", "/home/cune/llama.cpp/build-cuda-vnni", "--target", "llama-cli", "-j4"]
        b_res = subprocess.run(build_cmd, cwd="/home/cune/llama.cpp", capture_output=True, text=True)
        if b_res.returncode != 0:
            compiler.restore_graph_cpp()
            return {"status": "error", "message": "C++ build failed", "stderr": b_res.stderr}

        target_model = model_path
        if not target_model:
            models = list_models().get("models", [])
            target_model = models[0]["path"] if models else None

        if not target_model:
            return {"status": "error", "message": "No model found to run benchmark"}

        from recurrent_engine.rlang.pipeline import run_fast_logic_suite
        bench_res = run_fast_logic_suite("/home/cune/llama.cpp/build-cuda-vnni/bin/llama-cli", target_model)
        return {
            "status": "success",
            "mode": "silicon_benchmark",
            "model": target_model,
            "benchmark_results": bench_res
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    mcp.run()
