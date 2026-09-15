#!/usr/bin/env python3
import json
import subprocess
import re

d = json.load(open("benchmarks/comprehensive/dataset.json"))
tasks = {t['id']: t for t in d['tasks']['coding']}
task = tasks['code_12_tarjan_scc']

prompt = task['prompt']
test_code = task['test_code']
CLI_BIN = "/home/cune/llama.cpp/build-cuda-vnni/bin/llama-cli"
MODEL_PATH = "/home/cune/qwen2.5-coder-7b-instruct-q4_k_m.gguf"

print("=" * 60)
print("RUNNING T=1 (VANILLA BASELINE)")
print("=" * 60)

cmd_t1 = [
    CLI_BIN, "-m", MODEL_PATH, "-p", prompt,
    "-n", "512", "--temp", "0.0", "-ngl", "99", "-c", "2048",
    "--log-disable", "-st", "--simple-io"
]
res_t1 = subprocess.run(cmd_t1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
out_t1 = res_t1.stdout.strip()
if prompt in out_t1:
    out_t1 = out_t1.split(prompt, 1)[-1]

m = re.search(r"```(?:python)?\s*(.*?)\s*```", out_t1, re.DOTALL)
code_t1 = m.group(1) if m else out_t1

print("--- GENERATED CODE (T=1) ---")
print(code_t1)

test_script_t1 = code_t1 + "\n" + test_code
r = subprocess.run(["python3", "-c", test_script_t1], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print("--- TEST EXECUTION (T=1) ---")
print("Return code:", r.returncode)
print("STDOUT:", r.stdout)
print("STDERR:", r.stderr)

print("\n" + "=" * 60)
print("RUNNING T=2 (ORSD RECURRENT)")
print("=" * 60)

cmd_t2 = [
    CLI_BIN, "-m", MODEL_PATH, "-p", prompt,
    "-n", "512", "--temp", "0.0", "-ngl", "99", "-c", "2048",
    "--log-disable", "-st", "--simple-io",
    "--recurrent-t", "2", "--recurrent-layer", "13", "--recurrent-layer-b", "14",
    "--recurrent-gate", "0.20", "--recurrent-mode", "1"
]
res_t2 = subprocess.run(cmd_t2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
out_t2 = res_t2.stdout.strip()
if prompt in out_t2:
    out_t2 = out_t2.split(prompt, 1)[-1]

m = re.search(r"```(?:python)?\s*(.*?)\s*```", out_t2, re.DOTALL)
code_t2 = m.group(1) if m else out_t2

print("--- GENERATED CODE (T=2) ---")
print(code_t2)

test_script_t2 = code_t2 + "\n" + test_code
r2 = subprocess.run(["python3", "-c", test_script_t2], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print("--- TEST EXECUTION (T=2) ---")
print("Return code:", r2.returncode)
print("STDOUT:", r2.stdout)
print("STDERR:", r2.stderr)

