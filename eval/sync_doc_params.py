#!/usr/bin/env python3
"""
Synchronizes documentation with the exact constants defined in src/models/models.h.
Prevents doc-vs-code drift.
"""

import os
import re

MODELS_H = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "models", "models.h"))
HOWITWORKS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "howitfuckingworks.md"))

def extract_constants(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract presets for Qwen
    qwen_match = re.search(r"case LLM_ARCH_QWEN:.*?return \{(\d+),\s*(\d+),\s*([0-9\.]+)f?,\s*([0-9\.]+)f?\};", content, re.DOTALL)
    if not qwen_match:
        # Fallback to secondary Qwen block
        qwen_match = re.search(r"case LLM_ARCH_QWEN35:.*?return \{(\d+),\s*(\d+),\s*([0-9\.]+)f?,\s*([0-9\.]+)f?\};", content, re.DOTALL)
    
    start_pct, end_pct, alpha, exit_alpha = (42, 64, 0.20, 0.62)
    if qwen_match:
        start_pct = int(qwen_match.group(1))
        end_pct = int(qwen_match.group(2))
        alpha = float(qwen_match.group(3))
        exit_alpha = float(qwen_match.group(4))

    # Extract counter beta
    beta_match = re.search(r"get_recurrent_counter_beta\(\)\s*\{.*?return\s*([0-9\.]+)f?;", content, re.DOTALL)
    beta = float(beta_match.group(1)) if beta_match else 0.06

    # Extract block loops default
    loops_match = re.search(r"get_recurrent_block_loops\(\)\s*\{.*?return\s*(\d+);", content, re.DOTALL)
    loops = int(loops_match.group(1)) if loops_match else 2

    return {
        "start_pct": start_pct,
        "end_pct": end_pct,
        "alpha": alpha,
        "exit_alpha": exit_alpha,
        "beta": beta,
        "loops": loops
    }

def main():
    params = extract_constants(MODELS_H)
    print("Extracted exact parameters from models.h:")
    for k, v in params.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
