#!/usr/bin/env python3
import json
import re
import subprocess
import sys
import time

DATASET_PATH = "/home/cune/llama.cpp/benchmarks/comprehensive/dataset.json"
MODEL_PATH = "/home/cune/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
LORA_PATH = "/home/cune/training_recurrent/checkpoints_qwen/step-100/qwen_recurrent_step100.gguf"
CLI_BIN = "/home/cune/llama.cpp/build-cuda-vnni/bin/llama-cli"

TARGET_TASKS = [
    "code_01_regex_nfa",
    "code_02_bytecode_vm",
    "code_03_lazy_segment_tree",
    "code_04_lisp_interpreter",
    "code_07_avl_tree_invariants",
    "code_09_expression_calculator_shunting_yard",
    "code_10_interval_tree_overlap",
    "code_12_tarjan_scc",
    "logic_10_monty_hall_4doors"
]

def load_tasks():
    with open(DATASET_PATH, "r") as f:
        data = json.load(f)
    task_map = {}
    tasks_dict = data.get("tasks", data)
    for cat, tasks in tasks_dict.items():
        if isinstance(tasks, list):
            for t in tasks:
                task_map[t["id"]] = t
    return [task_map[tid] for tid in TARGET_TASKS if tid in task_map]

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

def run_inference(extra_flags, prompt, max_tokens=768):
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
        "--simple-io"
    ] + extra_flags

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, input="", timeout=120)
        return clean_output(prompt, res.stdout)
    except Exception as e:
        return f"ERROR: {e}"

def evaluate_task(task, output):
    vtype = task.get("validation_type")
    if vtype == "python_unit_test":
        code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", output, re.DOTALL)
        code_to_run = code_match.group(1) if code_match else output
        lines = [l for l in code_to_run.split("\n") if not l.startswith("#!") and not l.startswith("```")]
        sanitized_code = "\n".join(lines)
        test_code = task.get("test_code", "")
        full_script = f"{sanitized_code}\n\n{test_code}"
        try:
            env = {}
            exec(full_script, env, env)
            return True, "PASSED"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
    elif vtype == "contains":
        exp = task.get("expected_contains", "").lower().replace("$", "").replace(",", "").replace("%", "")
        actual = output.lower().replace("$", "").replace(",", "").replace("%", "")
        passed = exp in actual
        return passed, ("PASSED" if passed else f"Expected '{exp}'")
    return False, "Unknown validation type"

def main():
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} target tasks.")

    configs = [
        ("Base_T1", []),
        ("LoRA_T1", ["--lora", LORA_PATH]),
        ("LoRA_Recurrent_T2_Gate02_ORSD", ["--lora", LORA_PATH, "--recurrent-t", "2", "--recurrent-layer", "13", "--recurrent-gate", "0.2", "--recurrent-mode", "1"]),
        ("LoRA_Recurrent_T2_Gate05_Vanilla", ["--lora", LORA_PATH, "--recurrent-t", "2", "--recurrent-layer", "13", "--recurrent-gate", "0.5", "--recurrent-mode", "0"]),
    ]

    results = {}
    for cfg_name, flags in configs:
        print(f"\n==========================================")
        print(f"Running config: {cfg_name}")
        print(f"Flags: {' '.join(flags)}")
        print(f"==========================================")
        cfg_results = {}
        passed_count = 0
        t0 = time.time()
        for t in tasks:
            tid = t["id"]
            prompt = f"<|im_start|>user\n{t['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            out = run_inference(flags, prompt, max_tokens=768)
            passed, reason = evaluate_task(t, out)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"[{status}] {tid}: {reason}")
            cfg_results[tid] = {"passed": passed, "reason": reason}
            if passed:
                passed_count += 1
        elapsed = time.time() - t0
        print(f"--- Summary for {cfg_name}: {passed_count}/{len(tasks)} passed ({passed_count/len(tasks)*100:.1f}%) in {elapsed:.1f}s ---")
        results[cfg_name] = {"passed": passed_count, "total": len(tasks), "details": cfg_results, "time": elapsed}

    print("\n\nFINAL COMPARISON:")
    for cfg_name, res in results.items():
        print(f"{cfg_name:35s}: {res['passed']}/{res['total']} ({res['passed']/res['total']*100:.1f}%) in {res['time']:.1f}s")

if __name__ == "__main__":
    main()
