#!/usr/bin/env python3
"""
Reproducible GPQA (Google-Proof Q&A) Diamond Evaluation Harness for llama.cpp.

Usage:
  python3 eval/run_gpqa.py --mode baseline --limit 100 --port 8080 --output eval/gpqa_diamond_baseline.json
  python3 eval/run_gpqa.py --mode dualstream --limit 100 --port 8080 --output eval/gpqa_diamond_dualstream.json
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
    parser = argparse.ArgumentParser(description="Reproducible GPQA Benchmark Harness")
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "dualstream", "ablation_no_counter"],
                        help="Evaluation mode profile")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples to evaluate (default: full split)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port of the running llama-server (default: 8080)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output results JSON file")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 for deterministic greedy decoding)")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum generation tokens (default: 2048)")
    return parser.parse_args()

def extract_choice(response: str) -> str:
    """
    Extracts choice (A, B, C, D) from response.
    """
    match = re.search(r"####\s*([A-D])", response)
    if match:
        return match.group(1).upper()
        
    match = re.search(r"(?:final answer|answer is|correct option is|correct choice is)[:\s]*\(?([A-D])\)?", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([A-D])\b", response[-60:])
    if match:
        return match.group(1).upper()
        
    return None

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
    
    out_file = args.output or f"eval/gpqa_diamond_{args.mode}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    
    print("=" * 70)
    print(f"GPQA DIAMOND EVALUATION HARNESS: Mode={args.mode.upper()}, Port={args.port}")
    print(f"Sampling: Temperature={args.temperature}, MaxTokens={args.max_tokens}")
    print("=" * 70)
    
    print("Loading GPQA Diamond dataset (fingertap/GPQA-Diamond)...")
    ds = load_dataset("fingertap/GPQA-Diamond", split="test")
    
    samples = list(ds)
    if args.limit and args.limit < len(samples):
        samples = samples[:args.limit]
        
    print(f"Loaded {len(samples)} questions.")
    
    system_prompt = (
        "You are an expert scientist answering a high-difficulty multiple choice question. "
        "Think step-by-step through the scientific reasoning, eliminate invalid options, and conclude your response "
        "with '#### <LETTER>' where <LETTER> is exactly one of A, B, C, or D."
    )
    
    results = []
    correct_count = 0
    total_latency = 0.0
    
    for idx, sample in enumerate(samples, 1):
        question_text = sample["question"]
        correct_letter = sample["answer"].strip().upper()
        
        prompt_text = f"{question_text}\n\nProvide step-by-step reasoning and end your answer with '#### <LETTER>'."
        
        start_t = time.time()
        try:
            raw_response = query_server(args.port, system_prompt, prompt_text, args.max_tokens, args.temperature)
            elapsed = time.time() - start_t
            pred_letter = extract_choice(raw_response)
        except Exception as e:
            elapsed = time.time() - start_t
            raw_response = f"ERROR: {e}"
            pred_letter = None
            
        is_correct = (pred_letter == correct_letter)
        if is_correct:
            correct_count += 1
        total_latency += elapsed
        
        status_str = "CORRECT" if is_correct else f"WRONG (Got {pred_letter}, Expected {correct_letter})"
        print(f"[{idx:3d}/{len(samples):3d}] {status_str} | Time: {elapsed:.2f}s | Running Acc: {correct_count/idx*100:.1f}%")
        
        results.append({
            "index": idx,
            "question": question_text,
            "correct_letter": correct_letter,
            "predicted_letter": pred_letter,
            "is_correct": is_correct,
            "latency_seconds": elapsed,
            "raw_response": raw_response
        })
        
    final_accuracy = (correct_count / len(samples)) * 100.0 if samples else 0.0
    mean_latency = total_latency / len(samples) if samples else 0.0
    
    summary = {
        "benchmark": "GPQA-Diamond",
        "mode": args.mode,
        "total_samples": len(samples),
        "correct_count": correct_count,
        "accuracy_pct": final_accuracy,
        "mean_latency_seconds": mean_latency,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "results": results
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print("=" * 70)
    print(f"EVALUATION COMPLETE: {args.mode.upper()}")
    print(f"Accuracy: {correct_count}/{len(samples)} ({final_accuracy:.2f}%)")
    print(f"Mean Latency: {mean_latency:.2f}s per problem")
    print(f"Results saved to: {out_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
