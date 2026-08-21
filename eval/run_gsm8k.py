#!/usr/bin/env python3
"""
Reproducible GSM8K Benchmark Harness for llama.cpp Dual-Stream vs Baseline.

Usage:
  python3 run_gsm8k.py --mode baseline --limit 100 --port 8080 --output baseline_100.json
  python3 run_gsm8k.py --mode dualstream --limit 100 --port 8080 --output dualstream_100.json
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
    parser = argparse.ArgumentParser(description="Reproducible GSM8K Benchmark Harness")
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "dualstream", "ablation_no_counter"],
                        help="Evaluation mode profile")
    parser.add_argument("--limit", type=int, default=100,
                        help="Number of samples from the test split to evaluate (default: 100)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port of the running llama-server (default: 8080)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output results JSON file")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 for deterministic greedy decoding)")
    parser.add_argument("--max_tokens", type=int, default=1536,
                        help="Maximum generation tokens per problem (default: 1536)")
    return parser.parse_args()

def extract_numerical_answer(text: str):
    """
    Extracts the final numerical answer from the model output.
    Matches standard GSM8K format '#### <number>' or fallback to last parsed number.
    """
    match = re.search(r"####\s*([0-9\.\,\-]+)", text)
    if match:
        raw = match.group(1).replace(",", "").strip()
        try:
            return float(raw)
        except ValueError:
            pass

    match = re.search(r"(?:final answer|answer is|result is)[:\s]*\$?\s*([0-9\.\,\-]+)", text, re.IGNORECASE)
    if match:
        raw = match.group(1).replace(",", "").strip()
        try:
            return float(raw)
        except ValueError:
            pass

    nums = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass

    return None

def query_server(port: int, question: str, max_tokens: int, temperature: float) -> str:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful and precise math assistant. Solve the problem step by step and conclude your response with '#### <number>' where <number> is the exact final numeric answer."
            },
            {
                "role": "user",
                "content": question
            }
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
    
    out_file = args.output or f"gsm8k_{args.mode}_{args.limit}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    
    print("=" * 70)
    print(f"GSM8K EVALUATION HARNESS: Mode={args.mode.upper()}, Limit={args.limit}, Port={args.port}")
    print(f"Sampling: Temperature={args.temperature}, MaxTokens={args.max_tokens}")
    print("=" * 70)
    
    print("Loading official openai/gsm8k test split...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    correct_count = 0
    records = []
    
    t_start = time.time()
    
    for i in range(min(args.limit, len(dataset))):
        item = dataset[i]
        q = item["question"]
        gold_text = item["answer"]
        gold_ans = extract_numerical_answer(gold_text)
        
        t0 = time.time()
        try:
            raw_response = query_server(args.port, q, args.max_tokens, args.temperature)
            gen_time = round(time.time() - t0, 2)
            pred_ans = extract_numerical_answer(raw_response)
            
            is_correct = (pred_ans is not None and gold_ans is not None and abs(pred_ans - gold_ans) < 1e-4)
            if is_correct:
                correct_count += 1
                
            status_str = "PASSED" if is_correct else "FAILED"
            print(f"[{i+1:03d}/{args.limit:03d}] {status_str} | Pred: {pred_ans} | Gold: {gold_ans} | {gen_time}s")
            
            records.append({
                "sample_id": i + 1,
                "question": q,
                "gold_answer_raw": gold_text,
                "gold_numeric": gold_ans,
                "pred_response_raw": raw_response,
                "pred_numeric": pred_ans,
                "is_correct": is_correct,
                "latency_seconds": gen_time
            })
        except Exception as e:
            print(f"[{i+1:03d}/{args.limit:03d}] ERROR: {e}")
            records.append({
                "sample_id": i + 1,
                "question": q,
                "error": str(e),
                "is_correct": False
            })
            
    total_time = round(time.time() - t_start, 2)
    accuracy = round((correct_count / len(records)) * 100, 2) if records else 0.0
    
    summary = {
        "benchmark": "GSM8K",
        "mode": args.mode,
        "sample_count": len(records),
        "correct_count": correct_count,
        "accuracy_pct": accuracy,
        "total_latency_seconds": total_time,
        "average_latency_per_sample": round(total_time / len(records), 2) if records else 0.0,
        "evaluation_config": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "port": args.port
        },
        "records": records
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE: {correct_count}/{len(records)} Correct ({accuracy}%)")
    print(f"Total Time: {total_time}s | Output: {out_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
