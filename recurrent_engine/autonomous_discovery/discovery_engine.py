"""
discovery_engine.py
Universal Autonomous Discovery & Zero-Tuning Engine for Recurrent LLMs.
Allows users to run, benchmark, synthesize, and explore new cognitive loop architectures
with ZERO manual hyperparameter tuning.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from typing import Dict, Any, List, Optional

# Ensure project root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(SCRIPT_DIR)
LLAMA_ROOT = os.path.dirname(ENGINE_DIR)
if LLAMA_ROOT not in sys.path:
    sys.path.insert(0, LLAMA_ROOT)

from recurrent_engine.autonomous_discovery.auto_tuner import AutoTuner
from recurrent_engine.autonomous_discovery.theory_synthesizer import TheorySynthesizer, ARCHETYPES
from recurrent_engine.rlang.compiler import RLangCompiler
from tools.mcp_server.server import LOGIC_QUESTIONS, execute_llama_cli

CLI_BIN = os.path.join(LLAMA_ROOT, "build-cuda-vnni/bin/llama-cli")
DEFAULT_MODEL = "/home/cune/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
DEFAULT_LORA = "/home/cune/llama.cpp/models/recurrent_core_lora/checkpoint_step_200.gguf"

class DiscoveryEngine:
    """
    Main orchestrator for autonomous zero-tuning and architectural exploration.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL, lora_path: Optional[str] = DEFAULT_LORA):
        self.model_path = model_path
        self.lora_path = lora_path if (lora_path and os.path.exists(lora_path)) else None
        self.auto_tuner = AutoTuner()
        self.compiler = RLangCompiler(llama_cpp_root=LLAMA_ROOT)

    def run_zero_tuning(self, prompt: str, mode: str = "balanced", max_tokens: int = 64) -> Dict[str, Any]:
        """
        Executes inference on prompt with ZERO manual tuning.
        Automatically calculates bounds, centroids, and flags.
        """
        cfg = self.auto_tuner.auto_configure(
            model_path=self.model_path,
            mode=mode,
            target_stage="semantic",
            lora_path=self.lora_path
        )
        print(f"[AutoTuner] Calibrated parameters for {self.model_path}:")
        print(f"  - Layers: {cfg['recurrent_layer']}..{cfg['recurrent_layer_b']}")
        print(f"  - Stability: a={cfg['recurrent_a']:.2f}, b={cfg['recurrent_b']:.2f}, gate={cfg['recurrent_gate']:.2f} (spectral_radius={cfg['spectral_radius']})")
        print(f"  - LoRA: {cfg['lora_path']}\n")

        res = execute_llama_cli(self.model_path, prompt, cfg["cli_flags"], max_tokens=max_tokens)
        return {"config": cfg, "result": res}

    def benchmark_zero_tuning(self, mode: str = "balanced") -> Dict[str, Any]:
        """
        Runs the full 7-task logic benchmark with zero manual tuning.
        """
        cfg = self.auto_tuner.auto_configure(
            model_path=self.model_path,
            mode=mode,
            target_stage="semantic",
            lora_path=self.lora_path
        )

        passed = 0
        details = []
        t0 = time.time()

        for q in LOGIC_QUESTIONS:
            qid = q["id"]
            qtext = q["question"]
            expected = q["expected"]

            res = execute_llama_cli(self.model_path, qtext, cfg["cli_flags"], max_tokens=32)
            out = res.get("output", "")
            dur = res.get("latency_sec", 0.0)

            clean_out = out.lower().replace("$", "").replace(",", "")
            is_pass = (expected.lower() in clean_out)
            if is_pass:
                passed += 1

            details.append({
                "id": qid,
                "expected": expected,
                "output": out,
                "passed": is_pass,
                "latency": dur
            })

        elapsed = round(time.time() - t0, 2)
        score_pct = round((passed / len(LOGIC_QUESTIONS)) * 100, 1)

        return {
            "mode": mode,
            "config": cfg,
            "passed": passed,
            "total": len(LOGIC_QUESTIONS),
            "score_pct": score_pct,
            "elapsed_sec": elapsed,
            "details": details
        }

    def synthesize_and_validate_all(self) -> List[Dict[str, Any]]:
        """
        Synthesizes all 5 revolutionary archetypes, compiles them, and validates
        their formal mathematical contractive stability.
        """
        results = []
        examples_dir = os.path.join(LLAMA_ROOT, "recurrent_engine/rlang/examples")

        for key, meta in ARCHETYPES.items():
            path = TheorySynthesizer.save_synthesized_rlang(key, output_dir=examples_dir)
            try:
                prog = self.compiler.compile_file(path)
                topo = self.compiler.program_to_topology(prog)
                errs = topo.validate()
                valid = (len(errs) == 0)
            except Exception as e:
                valid = False
                errs = [str(e)]

            results.append({
                "archetype": key,
                "name": meta["name"],
                "description": meta["description"],
                "proof": meta["proof"],
                "path": path,
                "valid": valid,
                "stages": len(prog.stages) if valid else 0,
                "errors": errs if not valid else []
            })
        return results

def main():
    import time
    parser = argparse.ArgumentParser(description="Autonomous Discovery & Zero-Tuning Recurrent Engine")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # auto-run
    p_run = subparsers.add_parser("auto-run", help="Run prompt inference with zero manual tuning")
    p_run.add_argument("-p", "--prompt", required=True, help="Prompt text")
    p_run.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Path to GGUF model")
    p_run.add_argument("--lora", default=DEFAULT_LORA, help="Path to LoRA adapter")
    p_run.add_argument("--mode", choices=["balanced", "exploratory", "conservative", "high_order_deep"], default="balanced")
    p_run.add_argument("-n", "--max-tokens", type=int, default=64)

    # auto-bench
    p_bench = subparsers.add_parser("auto-bench", help="Run logic benchmark with zero manual tuning")
    p_bench.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Path to GGUF model")
    p_bench.add_argument("--lora", default=DEFAULT_LORA, help="Path to LoRA adapter")
    p_bench.add_argument("--mode", choices=["balanced", "exploratory", "conservative", "high_order_deep"], default="balanced")

    # synthesize
    p_synth = subparsers.add_parser("synthesize", help="Synthesize and validate a novel architecture archetype")
    p_synth.add_argument("--archetype", choices=list(ARCHETYPES.keys()), default="hamiltonian")

    # explore-all
    p_explore = subparsers.add_parser("explore-all", help="Synthesize and validate all 5 revolutionary archetypes")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "auto-run":
        engine = DiscoveryEngine(args.model, args.lora)
        res = engine.run_zero_tuning(args.prompt, mode=args.mode, max_tokens=args.max_tokens)
        print("Model Response:")
        print(res["result"].get("output", ""))
        print(f"\nLatency: {res['result'].get('latency_sec', 0.0)}s")

    elif args.command == "auto-bench":
        engine = DiscoveryEngine(args.model, args.lora)
        res = engine.benchmark_zero_tuning(mode=args.mode)
        print("=" * 80)
        print(f"ZERO-TUNING LOGIC BENCHMARK RESULT (Mode: {args.mode})")
        print(f"Score: {res['passed']}/{res['total']} ({res['score_pct']}%) in {res['elapsed_sec']}s")
        print("=" * 80)
        for item in res["details"]:
            status = "PASS" if item["passed"] else "FAIL"
            print(f"  - {item['id']:15s}: [{status}] ({item['latency']:.2f}s) | Got: '{item['output'][:40]}'")

    elif args.command == "synthesize":
        path = TheorySynthesizer.save_synthesized_rlang(args.archetype)
        print(f"Synthesized archetype '{args.archetype}' to: {path}")
        compiler = RLangCompiler(llama_cpp_root=LLAMA_ROOT)
        prog = compiler.compile_file(path)
        topo = compiler.program_to_topology(prog)
        errs = topo.validate()
        if errs:
            print("Validation FAILED:", errs)
        else:
            print(f"Validation PASSED (Architecture: {topo.architecture}, Layers: {topo.total_layers}, Stages: {len(prog.stages)})")

    elif args.command == "explore-all":
        engine = DiscoveryEngine()
        results = engine.synthesize_and_validate_all()
        print("=" * 80)
        print("AUTONOMOUS EXPLORATION: 5 REVOLUTIONARY RECURRENT ARCHETYPES")
        print("=" * 80)
        for r in results:
            status = "VALID" if r["valid"] else "INVALID"
            print(f"\n* Archetype: {r['name']} [{status}]")
            print(f"  Description: {r['description']}")
            print(f"  Mathematical Proof: {r['proof']}")
            print(f"  Generated File: {r['path']}")
            print(f"  Stages: {r['stages']}")
        print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
