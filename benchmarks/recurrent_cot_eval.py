#!/usr/bin/env python3
"""
benchmarks/recurrent_cot_eval.py
Evaluates Qwen3.5-4B-Q4_K_M.gguf across varied, fresh reasoning problems
comparing Baseline (T=1) vs Recurrent Configurations (T=2) in pair with CoT.
Measures:
  - Accuracy (% correct)
  - CoT token count (reasoning_content length)
  - Generation Speed (t/s)
  - Failure/Degeneration modes
"""

import subprocess, os, sys, time, json, re
import urllib.request, urllib.error

MODEL = "/home/cune/Qwen3.5-4B-Q4_K_M.gguf"
BIN   = "/home/cune/llama.cpp/build-cuda-vnni/bin/llama-server"
PORT  = 8195

TASKS = [
    {
        "id": "math_lily_pads_54",
        "category": "Math Invariant",
        "prompt": "A patch of water lilies doubles in surface area every day. If it takes 54 days for the lilies to cover a pond entirely, on what day was the pond exactly 25% covered? State the final day number clearly.",
        "check": lambda ans: "52" in ans and not any(f"day {w}" in ans.lower() for w in ["27", "13", "14"]),
        "expected": "52"
    },
    {
        "id": "math_invariant_age",
        "category": "Math Invariant",
        "prompt": "When Marcus was 9 years old, his younger sister Elena was one-third of his age. Today Marcus is 45 years old. How old is Elena? State Elena's age clearly.",
        "check": lambda ans: "39" in ans and "15" not in re.findall(r"\b15\b", ans),
        "expected": "39"
    },
    {
        "id": "math_machine_rate",
        "category": "Rate / Scaling",
        "prompt": "If 8 identical industrial 3D printers take 8 hours to produce 8 engine manifolds, how many hours will it take 24 identical printers to produce 24 engine manifolds? State the number of hours clearly.",
        "check": lambda ans: bool(re.search(r"\b8\b", ans)) and not bool(re.search(r"\b(24|64|1)\s*(hours|hrs)?\b", ans.lower())),
        "expected": "8"
    },
    {
        "id": "math_keyboard_cable",
        "category": "Algebraic Trap",
        "prompt": "A high-end mechanical keyboard and custom cable cost $210 in total. The keyboard costs $200 more than the cable. How much does the custom cable cost in dollars? State the exact price of the cable.",
        "check": lambda ans: bool(re.search(r"\b5(\.00)?\b|\$5\b", ans)) and not bool(re.search(r"\b10(\.00)?\b|\$10\b", ans)),
        "expected": "5"
    },
    {
        "id": "math_brick_weight",
        "category": "Algebraic Balance",
        "prompt": "A specialized refractory brick weighs exactly 4 kilograms plus the weight of half a brick. What is the total weight of one whole brick in kilograms? State the number of kilograms.",
        "check": lambda ans: bool(re.search(r"\b8\b|\b8\s*kg\b", ans)) and not bool(re.search(r"\b6\b|\b6\s*kg\b", ans)),
        "expected": "8"
    },
    {
        "id": "logic_sheep_storm",
        "category": "Linguistic / Logic",
        "prompt": "A shepherd has 28 sheep in a pasture. A sudden storm hits, and all but 11 find shelter in the barn. How many sheep are still outside in the pasture? State the exact number.",
        "check": lambda ans: bool(re.search(r"\b11\b", ans)) and not bool(re.search(r"\b17\b", ans)),
        "expected": "11"
    },
    {
        "id": "logic_pill_schedule",
        "category": "Discrete Scheduling",
        "prompt": "A physician prescribes a patient 5 tablets to be taken one tablet every 45 minutes starting immediately at 12:00 PM. At what exact time will the patient take the last tablet? Specify the time (e.g. 3:00 PM).",
        "check": lambda ans: bool(re.search(r"3:00\s*(pm)?|15:00|3\s*pm", ans.lower())) and not bool(re.search(r"3:45|4:00", ans.lower())),
        "expected": "3:00 PM"
    },
    {
        "id": "logic_race_overtake",
        "category": "Order / Relativity",
        "prompt": "You are competing in a marathon and you overtake the runner in second place right before the finish line. What place did you finish in? State your finishing place.",
        "check": lambda ans: bool(re.search(r"\b(second|2nd|2)\b", ans.lower())) and not bool(re.search(r"\b(first|1st|1)\b", ans.lower())),
        "expected": "second / 2nd"
    },
    {
        "id": "algo_bracket_depth",
        "category": "Algorithmic Trace",
        "prompt": "What is the maximum nesting depth of parentheses in the following expression: `((a + (b * (c - d))) / (e + (f * (g + h))))`? State the single integer representing the maximum nesting depth.",
        "check": lambda ans: bool(re.search(r"\b4\b", ans)),
        "expected": "4"
    },
    {
        "id": "algo_cycle_detection",
        "category": "Graph Reasoning",
        "prompt": "Consider a directed graph with 5 vertices labelled 0, 1, 2, 3, 4 and the following directed edges: 0->1, 1->2, 2->3, 3->4, 4->1. Does this graph contain a directed cycle? Answer YES or NO, and identify which vertices are in the cycle.",
        "check": lambda ans: "yes" in ans.lower() and all(v in ans for v in ["1", "2", "3", "4"]),
        "expected": "YES (1, 2, 3, 4)"
    },
    {
        "id": "algo_stack_evaluation",
        "category": "Stack / RPN",
        "prompt": "Evaluate the following postfix (Reverse Polish Notation) arithmetic expression: `15 7 1 1 + - / 3 * 2 1 1 + + -`. What is the exact final integer result? State only the integer.",
        "check": lambda ans: bool(re.search(r"\b5\b", ans)),
        "expected": "5"
    },
    {
        "id": "deductive_knights_knaves",
        "category": "Deductive Logic",
        "prompt": "On an island, Knights always tell the truth and Knaves always lie. You meet two inhabitants, Alice and Bob. Alice says: 'At least one of us is a Knave.' What are Alice and Bob? State clearly whether Alice is a Knight or Knave, and whether Bob is a Knight or Knave.",
        "check": lambda ans: ("alice" in ans.lower() and "knight" in ans.lower()) and ("bob" in ans.lower() and "knave" in ans.lower()),
        "expected": "Alice=Knight, Bob=Knave"
    },
    {
        "id": "deductive_transitive_height",
        "category": "Transitive Ordering",
        "prompt": "Five colleagues (A, B, C, D, E) compare heights: A is taller than B. C is shorter than B. D is taller than A. E is shorter than A but taller than B. Who is the second tallest person among the five? State the single letter.",
        "check": lambda ans: bool(re.search(r"\bA\b", ans)) and not bool(re.search(r"\b(second tallest is|person is)\s*[BCDE]\b", ans, re.IGNORECASE)),
        "expected": "A"
    },
    {
        "id": "code_loop_invariant",
        "category": "Code Trace",
        "prompt": "Consider the following Python snippet:\nx = 0\nfor i in range(1, 11):\n    if i % 3 == 0:\n        x += i * 2\n    else:\n        x += 1\nWhat is the final integer value of x after the loop completes? State the final integer value.",
        "check": lambda ans: bool(re.search(r"\b43\b", ans)),
        "expected": "43"
    },
    {
        "id": "math_modular_arithmetic",
        "category": "Number Theory",
        "prompt": "What is the remainder when 7^2026 is divided by 10? State the single digit remainder.",
        "check": lambda ans: bool(re.search(r"\b9\b", ans)),
        "expected": "9"
    },
    {
        "id": "logic_water_jug",
        "category": "State Search",
        "prompt": "You have an empty 3-liter jug and an empty 5-liter jug, and an unlimited supply of water. Neither jug has markings. What is the minimum number of steps (filling, emptying, or pouring between jugs) required to obtain exactly 4 liters in the 5-liter jug? State the minimum number of steps as an integer.",
        "check": lambda ans: bool(re.search(r"\b6\b", ans)),
        "expected": "6"
    }
]

CONFIGS = [
    {
        "name": "Baseline (T=1 Vanilla)",
        "plan": None,
    },
    {
        "name": "Recurrent T=2 (ORSD γ=0.0331, L12-14, B13, M1)",
        "plan": "loop: 12, 14\nbridge: 13\niterations: 2\ngate: 0.0331\nmode: 1\n",
    },
    {
        "name": "Recurrent T=2 (DeltaNet γ=0.10, L12-14, B13, M1)",
        "plan": "loop: 12, 14\nbridge: 13\niterations: 2\ngate: 0.10\nmode: 1\n",
    },
    {
        "name": "Recurrent T=2 (DeltaNet γ=0.20, L12-14, B13, M1)",
        "plan": "loop: 12, 14\nbridge: 13\niterations: 2\ngate: 0.20\nmode: 1\n",
    }
]

def wait_health(timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.4)
    return False

def query_model(prompt: str):
    payload = json.dumps({
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode())
    msg = res["choices"][0]["message"]
    reasoning = msg.get("reasoning_content") or ""
    content   = msg.get("content") or ""
    usage     = res.get("usage", {})
    timings   = res.get("timings", {})
    t_speed   = timings.get("predicted_per_second", 0.0)
    cot_toks  = len(reasoning.split())  # approximate token proxy
    comp_toks = usage.get("completion_tokens", 0)
    return reasoning, content, comp_toks, t_speed

def run_config(cfg):
    print(f"\n=================================================================")
    print(f"EVALUATING CONFIG: {cfg['name']}")
    print(f"=================================================================")
    
    plan_file = "/home/cune/benchmarks/temp_plan.txt"
    extra_flags = []
    if cfg["plan"]:
        os.makedirs("/home/cune/benchmarks", exist_ok=True)
        with open(plan_file, "w") as f:
            f.write(cfg["plan"])
        extra_flags = ["-rc", plan_file]
    
    cmd = [
        BIN,
        "-m", MODEL,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "-ngl", "99",
        "-c", "4096",
        "--log-disable",
        "--no-warmup",
        "-fa", "on",
    ] + extra_flags

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = []
    
    try:
        if not wait_health(timeout=45):
            print("ERROR: Server failed to become healthy!")
            return None
        
        passed = 0
        total_cot_toks = 0
        speeds = []

        for i, t in enumerate(TASKS, 1):
            t0 = time.time()
            try:
                reasoning, content, c_toks, speed = query_model(t["prompt"])
                full_text = (reasoning + "\n" + content).strip()
                ok = t["check"](full_text)
                elapsed = time.time() - t0
                if ok:
                    passed += 1
                total_cot_toks += c_toks
                if speed > 0:
                    speeds.append(speed)
                
                status_str = "PASS" if ok else "FAIL"
                print(f"[{i:02d}/{len(TASKS):02d}] [{status_str}] {t['id']} ({t['category']}) | {c_toks} tok | {elapsed:.2f}s")
                results.append({
                    "id": t["id"],
                    "category": t["category"],
                    "passed": ok,
                    "tokens": c_toks,
                    "elapsed": elapsed,
                    "content_snippet": content[:80].replace("\n", " ")
                })
            except Exception as e:
                print(f"[{i:02d}/{len(TASKS):02d}] [EXC] {t['id']}: {e}")
                results.append({
                    "id": t["id"],
                    "category": t["category"],
                    "passed": False,
                    "error": str(e)
                })

        acc = (passed / len(TASKS)) * 100.0
        avg_tokens = total_cot_toks / len(TASKS)
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

        print(f"--- Summary for {cfg['name']} ---")
        print(f"Accuracy: {passed}/{len(TASKS)} ({acc:.1f}%)")
        print(f"Avg Tokens per response: {avg_tokens:.1f}")
        print(f"Avg Speed: {avg_speed:.2f} t/s")
        return {
            "name": cfg["name"],
            "passed": passed,
            "total": len(TASKS),
            "accuracy": acc,
            "avg_tokens": avg_tokens,
            "avg_speed": avg_speed,
            "results": results
        }

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        time.sleep(1)

def main():
    all_summaries = []
    for cfg in CONFIGS:
        res = run_config(cfg)
        if res:
            all_summaries.append(res)
    
    # Save results
    out_path = "/home/cune/benchmarks/recurrent_cot_results.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    
    print("\n" + "="*70)
    print("FINAL COMPARATIVE REPORT: CO-EXISTENCE OF COT + LATENT RECURRENCE")
    print("="*70)
    print(f"{'Configuration':<45} | {'Score':<10} | {'Avg Tok':<9} | {'Speed (t/s)':<10}")
    print("-" * 80)
    for s in all_summaries:
        score_str = f"{s['passed']}/{s['total']} ({s['accuracy']:.1f}%)"
        print(f"{s['name']:<45} | {score_str:<10} | {s['avg_tokens']:<9.1f} | {s['avg_speed']:<10.2f}")
    print("="*70)

if __name__ == "__main__":
    main()
