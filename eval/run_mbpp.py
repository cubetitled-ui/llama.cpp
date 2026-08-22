#!/usr/bin/env python3
"""
Rigorous MBPP / MBPP+ Unit Test Execution Evaluation Harness for llama.cpp.
Executes generated Python code directly against standard unit test assertions in an isolated subprocess.
"""

import os
import sys
import re
import json
import time
import argparse
import subprocess
import urllib.request
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="MBPP Unit Test Execution Benchmark Harness")
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "macro_recurrence", "dualstream"],
                        help="Evaluation mode profile")
    parser.add_argument("--limit", type=int, default=100,
                        help="Number of problems to evaluate (default: 100)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port of running llama-server (default: 8080)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output JSON")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 for deterministic greedy decoding)")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Max tokens to generate")
    parser.add_argument("--timeout", type=int, default=5,
                        help="Execution timeout per unit test in seconds (default: 5s)")
    return parser.parse_args()

def extract_python_code(text: str) -> str:
    """
    Extracts executable Python code from markdown blocks or raw text.
    """
    matches = re.findall(r"```(?:python|py)?\s*\n(.*?)\n```", text, re.DOTALL)
    if matches:
        for m in reversed(matches):
            if "def " in m:
                return m.strip()
        return matches[-1].strip()
    
    match = re.search(r"(def\s+[a-zA-Z0-9_]+\s*\(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    return text.strip()

def run_unit_tests(code: str, test_imports: list, test_list: list, timeout: int = 5) -> tuple[bool, str]:
    """
    Executes code against unit test assertions in an isolated Python subprocess.
    Returns (passed: bool, output_or_error: str).
    """
    import_block = "\n".join(test_imports)
    
    # Indent tests for try block
    indented_tests = "\n".join(f"    {t.strip()}" for t in test_list if t.strip())
    
    full_script = f"""
import sys
{import_block}

{code}

# --- Unit Test Assertions ---
try:
{indented_tests}
    sys.exit(0)
except AssertionError as ae:
    print(f"AssertionError: {{ae}}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"RuntimeError: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(2)
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", full_script],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if proc.returncode == 0:
            return True, "PASSED"
        else:
            err = proc.stderr.strip() or proc.stdout.strip() or f"Exited with code {proc.returncode}"
            return False, err
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: Execution exceeded {timeout} seconds"
    except Exception as ex:
        return False, f"Execution failed: {ex}"

def query_server(port: int, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
    
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        msg = res["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        return f"{reasoning}\n\n{content}".strip()

def main():
    args = parse_args()
    out_file = args.output or f"eval/mbpp_{args.mode}_{args.limit}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    
    print("=" * 75)
    print(f"MBPP UNIT TEST EXECUTION HARNESS: Mode={args.mode.upper()}, Limit={args.limit}")
    print(f"Sampling: Temperature={args.temperature}, MaxTokens={args.max_tokens}, Timeout={args.timeout}s")
    print("=" * 75)
    
    print("Loading evalplus/mbppplus dataset...")
    ds = load_dataset("evalplus/mbppplus", split="test")
    samples = list(ds)
    if args.limit and args.limit < len(samples):
        samples = samples[:args.limit]
        
    print(f"Loaded {len(samples)} MBPP tasks with ground-truth unit tests.")
    
    system_prompt = (
        "You are an expert Python programmer. Write a clean, complete, self-contained Python function "
        "that solves the requested task. Do not include example usage or explanations. Enclose your Python "
        "code in a markdown code block ```python\n...\n```."
    )
    
    results = []
    passed_count = 0
    total_latency = 0.0
    
    for idx, sample in enumerate(samples, 1):
        task_id = sample.get("task_id", idx)
        prompt_text = sample["prompt"]
        test_imports = sample.get("test_imports", [])
        test_list = sample.get("test_list", [])
        
        user_prompt = f"Problem:\n{prompt_text}\n\n"
        if test_list:
            user_prompt += f"Your code should pass assertions such as:\n{test_list[0]}\n\n"
        user_prompt += "Write the complete Python function:"
        
        t0 = time.time()
        try:
            raw_response = query_server(args.port, system_prompt, user_prompt, args.max_tokens, args.temperature)
            elapsed = time.time() - t0
            extracted_code = extract_python_code(raw_response)
        except Exception as e:
            elapsed = time.time() - t0
            raw_response = f"ERROR: {e}"
            extracted_code = ""
            
        passed, test_msg = run_unit_tests(extracted_code, test_imports, test_list, args.timeout)
        if passed:
            passed_count += 1
            
        total_latency += elapsed
        
        status_str = "PASSED" if passed else f"FAILED ({test_msg[:30]})"
        running_pass_rate = (passed_count / idx) * 100.0
        print(f"[{idx:3d}/{len(samples):3d}] Task #{task_id:3d} | {status_str:35s} | Time: {elapsed:.2f}s | Pass@1: {passed_count}/{idx} ({running_pass_rate:.1f}%)", flush=True)
        
        results.append({
            "index": idx,
            "task_id": task_id,
            "prompt": prompt_text,
            "passed": passed,
            "test_message": test_msg,
            "test_list": test_list,
            "latency_seconds": elapsed,
            "extracted_code": extracted_code,
            "raw_response": raw_response
        })
        
        if idx % 5 == 0 or idx == len(samples):
            interim = {
                "benchmark": "MBPP_Unit_Tests",
                "mode": args.mode,
                "completed_samples": idx,
                "total_samples": len(samples),
                "passed_count": passed_count,
                "pass_rate_pct": running_pass_rate,
                "mean_latency_seconds": total_latency / idx,
                "results": results
            }
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(interim, f, indent=2, ensure_ascii=False)
                
    pass_rate = (passed_count / len(samples)) * 100.0 if samples else 0.0
    mean_latency = total_latency / len(samples) if samples else 0.0
    
    summary = {
        "benchmark": "MBPP_Unit_Tests",
        "mode": args.mode,
        "total_samples": len(samples),
        "passed_count": passed_count,
        "pass_rate_pct": pass_rate,
        "mean_latency_seconds": mean_latency,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "results": results
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print("=" * 75)
    print(f"EVALUATION COMPLETE: {args.mode.upper()}")
    print(f"Ground-Truth Unit Tests Passed: {passed_count}/{len(samples)} ({pass_rate:.2f}%)")
    print(f"Mean Latency: {mean_latency:.2f}s per problem")
    print(f"Results saved to: {out_file}")
    print("=" * 75)

if __name__ == "__main__":
    main()
