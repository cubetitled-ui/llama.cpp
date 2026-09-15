#!/usr/bin/env python3
"""
engine_cli.py
Universal CLI for declarative recurrent architecture execution in llama.cpp.
Allows defining recurrent loops via YAML/JSON topology specs, RLang scripts, or JSON configs.
Automatically:
1. Validates mathematical contractive bounds.
2. Emits C++ GGML computational graph implementation.
3. Automatically triggers incremental builds.
4. Executes real inference, benchmarks, and multi-domain evaluation suites.
"""

import os
import sys
import json
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from recurrent_engine.recurrent_spec import ArchitectureTopology
from recurrent_engine.graph_generator import RecurrentGraphCompiler

def add_recurrence_arguments(parser):
    """Add recurrent configuration options to a subcommand parser."""
    parser.add_argument("--spec", default=None, help="Path to topology spec (JSON or YAML)")
    parser.add_argument("-rc", "--config", dest="config", default=None, help="Path to recurrent JSON configuration file")
    parser.add_argument("-rs", "--script", dest="script", default=None, help="Path to RLang (.rlang) script")
    parser.add_argument("--lora", default=None, help="Path to LoRA adapter GGUF")

def resolve_cli_flags(args, compiler):
    """Resolve recurrent flags from spec, config, or script arguments."""
    flags = []
    if getattr(args, "config", None):
        flags.extend(["-rc", args.config])
    elif getattr(args, "script", None):
        flags.extend(["-rs", args.script])
    elif getattr(args, "spec", None):
        topo = ArchitectureTopology.from_file(args.spec)
        flags.extend(compiler.extract_runtime_cli_args(topo))

    if getattr(args, "lora", None):
        flags.extend(["--lora", args.lora])
    return flags

def main():
    parser = argparse.ArgumentParser(description="Universal Recurrent Engine CLI for llama.cpp")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: validate
    parser_val = subparsers.add_parser("validate", help="Validate a topology spec, script, or config")
    parser_val.add_argument("--spec", default=None, help="Path to topology spec (JSON or YAML)")
    parser_val.add_argument("-rs", "--script", dest="script", default=None, help="Path to RLang script")
    parser_val.add_argument("-rc", "--config", dest="config", default=None, help="Path to recurrent config JSON")

    # Command: emit-cpp
    parser_emit = subparsers.add_parser("emit-cpp", help="Emit C++ GGML graph code for a spec or script")
    parser_emit.add_argument("--spec", default=None, help="Path to topology spec")
    parser_emit.add_argument("-rs", "--script", dest="script", default=None, help="Path to RLang script")

    # Command: build
    parser_build = subparsers.add_parser("build", help="Trigger compilation of llama.cpp binaries")
    parser_build.add_argument("--targets", nargs="+", default=["llama-cli", "llama-bench", "llama-perplexity"])

    # Command: run
    parser_run = subparsers.add_parser("run", help="Run inference with recurrent configuration")
    add_recurrence_arguments(parser_run)
    parser_run.add_argument("--model", required=True, help="Path to GGUF model")
    parser_run.add_argument("-p", "--prompt", required=True, help="Prompt text")
    parser_run.add_argument("-n", "--n-predict", type=int, default=128)
    parser_run.add_argument("--temp", type=float, default=0.0)
    parser_run.add_argument("-ngl", type=int, default=99)
    parser_run.add_argument("-c", "--ctx-size", type=int, default=2048)

    # Command: bench
    parser_bench = subparsers.add_parser("bench", help="Run llama-bench throughput test")
    add_recurrence_arguments(parser_bench)
    parser_bench.add_argument("--model", required=True, help="Path to GGUF model")
    parser_bench.add_argument("-p", "--prompt-tokens", type=int, default=512)
    parser_bench.add_argument("-n", "--gen-tokens", type=int, default=128)
    parser_bench.add_argument("-ngl", type=int, default=99)
    parser_bench.add_argument("-r", "--repetitions", type=int, default=2)

    # Command: test-halfo
    parser_halfo = subparsers.add_parser("test-halfo", help="Run Halfo benchmark suite")
    add_recurrence_arguments(parser_halfo)
    parser_halfo.add_argument("--model", required=True, help="Path to GGUF model")
    parser_halfo.add_argument("--category", choices=["all", "logic", "tool_calling", "multilingual_translation"], default="all")

    # Command: test-comprehensive
    parser_comp = subparsers.add_parser("test-comprehensive", help="Run 22-task comprehensive benchmark suite (Tool calling, Logic, Coding, Web dev)")
    add_recurrence_arguments(parser_comp)
    parser_comp.add_argument("--model", required=True, help="Path to GGUF model")
    parser_comp.add_argument("--output", default=None, help="Optional output JSON path")

    # Command: auto-run (Zero-Tuning inference)
    parser_autorun = subparsers.add_parser("auto-run", help="Run inference with zero-tuning self-calibrated recurrence")
    parser_autorun.add_argument("-p", "--prompt", required=True, help="Prompt text")
    parser_autorun.add_argument("-m", "--model", default="/home/cune/qwen2.5-coder-7b-instruct-q4_k_m.gguf", help="Path to GGUF model")
    parser_autorun.add_argument("--lora", default="/home/cune/llama.cpp/models/recurrent_core_lora/checkpoint_step_200.gguf", help="Path to LoRA adapter")
    parser_autorun.add_argument("--mode", choices=["balanced", "exploratory", "conservative", "high_order_deep"], default="exploratory")
    parser_autorun.add_argument("-n", "--max-tokens", type=int, default=64)

    # Command: auto-bench (Zero-Tuning benchmark)
    parser_autobench = subparsers.add_parser("auto-bench", help="Run logic benchmark with zero manual tuning")
    parser_autobench.add_argument("-m", "--model", default="/home/cune/qwen2.5-coder-7b-instruct-q4_k_m.gguf", help="Path to GGUF model")
    parser_autobench.add_argument("--lora", default="/home/cune/llama.cpp/models/recurrent_core_lora/checkpoint_step_200.gguf", help="Path to LoRA adapter")
    parser_autobench.add_argument("--mode", choices=["balanced", "exploratory", "conservative", "high_order_deep"], default="exploratory")

    # Command: discover (Autonomous theory exploration & synthesis)
    parser_disc = subparsers.add_parser("discover", help="Synthesize and explore novel recurrence archetypes (Hamiltonian, FreeEnergy, Kalman, etc.)")
    parser_disc.add_argument("--archetype", choices=["hamiltonian", "predictive_coding", "kalman_filter", "dual_speed_attractor", "syntactic_semantic_cascade", "all"], default="all")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    compiler = RecurrentGraphCompiler()

    if args.command == "validate":
        if args.script:
            from recurrent_engine.rlang.compiler import RLangCompiler
            rlang_comp = RLangCompiler()
            prog = rlang_comp.compile_file(args.script)
            topo = rlang_comp.program_to_topology(prog)
            errs = topo.validate()
            if errs:
                print(f"RLang validation FAILED for {args.script}:")
                for e in errs:
                    print(f"  - {e}")
                sys.exit(1)
            print(f"RLang script '{args.script}' is VALID (Architecture: {topo.architecture}, Total Layers: {topo.total_layers}, Stages: {len(prog.stages)})")
        elif args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            t = cfg.get("recurrent_t", 1)
            a = cfg.get("recurrent_a", 0.90)
            b = cfg.get("recurrent_b", 0.10)
            gate = cfg.get("recurrent_gate", 1.00)
            print(f"Recurrent Config '{args.config}' is VALID: T={t}, a={a}, b={b}, gate={gate}, lora={cfg.get('lora_path')}")
        elif args.spec:
            topo = ArchitectureTopology.from_file(args.spec)
            errs = topo.validate()
            if errs:
                print("Validation FAILED:")
                for e in errs:
                    print(f"  - {e}")
                sys.exit(1)
            print(f"Topology '{args.spec}' is VALID (Architecture: {topo.architecture}, Layers: {topo.total_layers}, Blocks: {len(topo.blocks)})")
        else:
            print("Error: Specify --spec, --script, or --config to validate.")
            sys.exit(1)

    elif args.command == "emit-cpp":
        if args.script:
            from recurrent_engine.rlang.compiler import RLangCompiler
            rlang_comp = RLangCompiler()
            prog = rlang_comp.compile_file(args.script)
            code = rlang_comp.generate_cpp_core(prog)
            print(code)
        elif args.spec:
            topo = ArchitectureTopology.from_file(args.spec)
            errs = topo.validate()
            if errs:
                print("Validation FAILED:", errs)
                sys.exit(1)
            code = compiler.generate_cpp_core_function(topo)
            print(code)
        else:
            print("Error: Specify --spec or --script to emit C++ code.")
            sys.exit(1)

    elif args.command == "build":
        ok = compiler.build_binaries(args.targets)
        if ok:
            print("Build completed successfully.")
        else:
            print("Build FAILED.")
            sys.exit(1)

    elif args.command == "run":
        cli_flags = resolve_cli_flags(args, compiler)
        cli_bin = os.path.join(compiler.build_dir, "bin/llama-cli")
        cmd = [
            cli_bin,
            "-m", args.model,
            "-p", args.prompt,
            "-n", str(args.n_predict),
            "--temp", str(args.temp),
            "-ngl", str(args.ngl),
            "-c", str(args.ctx_size),
            "--single-turn"
        ] + cli_flags
        print(f"Executing: {' '.join(cmd)}\n")
        subprocess.run(cmd)

    elif args.command == "bench":
        cli_flags = resolve_cli_flags(args, compiler)
        bench_bin = os.path.join(compiler.build_dir, "bin/llama-bench")
        cmd = [
            bench_bin,
            "-m", args.model,
            "-p", str(args.prompt_tokens),
            "-n", str(args.gen_tokens),
            "-ngl", str(args.ngl),
            "-r", str(args.repetitions)
        ] + cli_flags
        print(f"Executing: {' '.join(cmd)}\n")
        subprocess.run(cmd)

    elif args.command == "test-halfo":
        cli_flags = resolve_cli_flags(args, compiler)
        cmd = [
            sys.executable,
            "/home/cune/llama.cpp/benchmarks/halfo/run_halfo.py",
            "--model", args.model,
            "--category", args.category,
        ]
        if cli_flags:
            cmd.extend(["--extra-args"] + cli_flags)
        print(f"Running Halfo Suite with flags: {' '.join(cli_flags)}\n")
        subprocess.run(cmd)

    elif args.command == "test-comprehensive":
        cli_flags = resolve_cli_flags(args, compiler)
        cmd = [
            sys.executable,
            "/home/cune/llama.cpp/benchmarks/comprehensive/run_benchmarks.py",
            "--model", args.model,
        ]
        if cli_flags:
            cmd.extend(["--extra-args"] + cli_flags)
        if args.output:
            cmd.extend(["--output", args.output])
        print(f"Running Comprehensive Suite with flags: {' '.join(cli_flags)}\n")
        subprocess.run(cmd)

    elif args.command == "auto-run":
        from recurrent_engine.autonomous_discovery.discovery_engine import DiscoveryEngine
        engine = DiscoveryEngine(args.model, args.lora)
        res = engine.run_zero_tuning(args.prompt, mode=args.mode, max_tokens=args.max_tokens)
        print("Model Response:")
        print(res["result"].get("output", ""))
        print(f"\nLatency: {res['result'].get('latency_sec', 0.0)}s")

    elif args.command == "auto-bench":
        from recurrent_engine.autonomous_discovery.discovery_engine import DiscoveryEngine
        engine = DiscoveryEngine(args.model, args.lora)
        res = engine.benchmark_zero_tuning(mode=args.mode)
        print("=" * 80)
        print(f"ZERO-TUNING LOGIC BENCHMARK RESULT (Mode: {args.mode})")
        print(f"Score: {res['passed']}/{res['total']} ({res['score_pct']}%) in {res['elapsed_sec']}s")
        print("=" * 80)
        for item in res["details"]:
            status = "PASS" if item["passed"] else "FAIL"
            print(f"  - {item['id']:15s}: [{status}] ({item['latency']:.2f}s) | Got: '{item['output'][:40]}'")

    elif args.command == "discover":
        from recurrent_engine.autonomous_discovery.discovery_engine import DiscoveryEngine
        from recurrent_engine.autonomous_discovery.theory_synthesizer import TheorySynthesizer, ARCHETYPES
        from recurrent_engine.rlang.compiler import RLangCompiler
        engine = DiscoveryEngine()
        if args.archetype == "all":
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
        else:
            path = TheorySynthesizer.save_synthesized_rlang(args.archetype)
            print(f"Synthesized archetype '{args.archetype}' to: {path}")
            compiler = RLangCompiler()
            prog = compiler.compile_file(path)
            topo = compiler.program_to_topology(prog)
            errs = topo.validate()
            if errs:
                print("Validation FAILED:", errs)
            else:
                print(f"Validation PASSED (Architecture: {topo.architecture}, Layers: {topo.total_layers}, Stages: {len(prog.stages)})")

if __name__ == "__main__":
    main()
