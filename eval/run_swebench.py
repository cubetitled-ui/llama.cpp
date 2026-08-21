#!/usr/bin/env python3
"""
Reproducible SWE-bench Lite Evaluation Harness for llama.cpp.

Usage:
  python3 eval/run_swebench.py --mode baseline --limit 50 --port 8080 --output eval/swebench_baseline.json
  python3 eval/run_swebench.py --mode dualstream --limit 50 --port 8080 --output eval/swebench_dualstream.json
"""

import os
import sys
import re
import json
import time
import argparse
import urllib.request
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Reproducible SWE-bench Lite Benchmark Harness")
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "dualstream", "ablation_no_counter"],
                        help="Evaluation mode profile")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max samples to evaluate (default: 50)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port of the running llama-server (default: 8080)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output results JSON file")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 for deterministic greedy decoding)")
    parser.add_argument("--max_tokens", type=int, default=1536,
                        help="Maximum generation tokens (default: 1536)")
    return parser.parse_args()

def extract_diff(text: str) -> str:
    """
    Extracts unified git diff from model output.
    """
    match = re.search(r"```(?:diff|patch)?\s*\n(diff --git.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"(diff --git.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def validate_diff_syntax(diff_text: str) -> bool:
    """
    Checks if the extracted diff conforms to standard unified git diff syntax.
    """
    if not diff_text:
        return False
    has_diff_header = bool(re.search(r"diff --git a/.* b/.*", diff_text))
    has_hunk_header = bool(re.search(r"@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", diff_text))
    has_changes = bool(re.search(r"^[\+\-]", diff_text, re.MULTILINE))
    return has_diff_header and has_hunk_header and has_changes

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
    
    with urllib.request.urlopen(req, timeout=180) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        msg = res["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        return f"{reasoning}\n\n{content}".strip()

def main():
    args = parse_args()
    
    out_file = args.output or f"eval/swebench_{args.mode}_{args.limit}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    
    print("=" * 70)
    print(f"SWE-BENCH LITE EVALUATION HARNESS: Mode={args.mode.upper()}, Limit={args.limit}, Port={args.port}")
    print(f"Sampling: Temperature={args.temperature}, MaxTokens={args.max_tokens}")
    print("=" * 70)
    
    print("Loading princeton-nlp/SWE-bench_Lite...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    samples = list(ds)
    if args.limit and args.limit < len(samples):
        samples = samples[:args.limit]
        
    print(f"Loaded {len(samples)} issue instances.")
    
    system_prompt = (
        "You are an expert principal software engineer. You are given a repository issue description. "
        "Analyze the root cause and provide an exact, minimal, unified git diff that resolves the issue. "
        "Enclose your git diff inside a markdown code block ```diff\n...\n```."
    )
    
    results = []
    valid_syntax_count = 0
    file_match_count = 0
    total_latency = 0.0
    
    for idx, sample in enumerate(samples, 1):
        instance_id = sample["instance_id"]
        repo = sample["repo"]
        problem_statement = sample["problem_statement"]
        gold_patch = sample["patch"]
        
        # Extract target files from gold patch
        gold_files = set(re.findall(r"diff --git a/(.*?) b/", gold_patch))
        
        prompt_text = f"Repository: {repo}\nInstance: {instance_id}\n\nProblem Description:\n{problem_statement}\n\nProvide the unified git diff fix."
        
        start_t = time.time()
        try:
            raw_response = query_server(args.port, system_prompt, prompt_text, args.max_tokens, args.temperature)
            elapsed = time.time() - start_t
            pred_diff = extract_diff(raw_response)
        except Exception as e:
            elapsed = time.time() - start_t
            raw_response = f"ERROR: {e}"
            pred_diff = ""
            
        is_valid_syntax = validate_diff_syntax(pred_diff)
        pred_files = set(re.findall(r"diff --git a/(.*?) b/", pred_diff))
        target_file_match = bool(gold_files and pred_files and (gold_files & pred_files))
        
        if is_valid_syntax:
            valid_syntax_count += 1
        if target_file_match:
            file_match_count += 1
            
        total_latency += elapsed
        
        status_str = f"Syntax={'VALID' if is_valid_syntax else 'INVALID'} | FileMatch={'YES' if target_file_match else 'NO'}"
        print(f"[{idx:2d}/{len(samples):2d}] {instance_id[:25]:25s} | {status_str} | Time: {elapsed:.2f}s")
        
        results.append({
            "index": idx,
            "instance_id": instance_id,
            "repo": repo,
            "is_valid_syntax": is_valid_syntax,
            "target_file_match": target_file_match,
            "gold_files": list(gold_files),
            "pred_files": list(pred_files),
            "latency_seconds": elapsed,
            "extracted_diff": pred_diff,
            "raw_response": raw_response
        })
        
    syntax_rate = (valid_syntax_count / len(samples)) * 100.0 if samples else 0.0
    match_rate = (file_match_count / len(samples)) * 100.0 if samples else 0.0
    mean_latency = total_latency / len(samples) if samples else 0.0
    
    summary = {
        "benchmark": "SWE-bench_Lite",
        "mode": args.mode,
        "total_samples": len(samples),
        "valid_syntax_count": valid_syntax_count,
        "valid_syntax_rate_pct": syntax_rate,
        "target_file_match_count": file_match_count,
        "target_file_match_rate_pct": match_rate,
        "mean_latency_seconds": mean_latency,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "results": results
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print("=" * 70)
    print(f"EVALUATION COMPLETE: {args.mode.upper()}")
    print(f"Valid Diff Syntax: {valid_syntax_count}/{len(samples)} ({syntax_rate:.2f}%)")
    print(f"Target File Match: {file_match_count}/{len(samples)} ({match_rate:.2f}%)")
    print(f"Mean Latency: {mean_latency:.2f}s per issue")
    print(f"Results saved to: {out_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
