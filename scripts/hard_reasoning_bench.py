import subprocess, os, json

cli = '/home/cune/llama.cpp/build-cuda-vnni/bin/llama-cli'
model = '/home/cune/qwen2.5-coder-7b-instruct-q4_k_m.gguf'
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = '/home/cune/llama.cpp/build-cuda-vnni/bin:' + env.get('LD_LIBRARY_PATH', '')

HARD_QUESTIONS = [
    {
        "id": "cows_chickens",
        "q": "A farmer has chickens (2 legs) and cows (4 legs). There are 30 heads and 74 legs in total. Exactly how many cows does the farmer have? Output just the number.",
        "target": "7"
    },
    {
        "id": "order_of_ops",
        "q": "Compute the exact value of 7 * (8 - 3) + 12 / 4. Output just the numerical result.",
        "target": "38"
    },
    {
        "id": "calendar_modulo",
        "q": "If yesterday was Tuesday, what day of the week will it be exactly 100 days from today? Answer only the name of the day.",
        "target": "Friday"
    },
    {
        "id": "boxes_logic",
        "q": "There are 3 boxes labeled Apples, Oranges, and Both. Every box is labeled incorrectly. You draw an Apple from the box labeled Both. What is in the box labeled Oranges? Answer strictly Apples, Oranges, or Both.",
        "target": "Both"
    },
    {
        "id": "knights_knaves",
        "q": "On an island, Knights always tell the truth and Knaves always lie. A says: 'Both of us are knaves.' What are A and B? Answer Knight or Knave for each.",
        "target": "A is Knave, B is Knight"
    },
    {
        "id": "monty_hall",
        "q": "In the Monty Hall problem with 3 doors (1 car, 2 goats), after you pick Door 1 and the host opens Door 3 showing a goat, what is the probability of winning if you switch to Door 2? Output as a fraction like 1/3 or 2/3.",
        "target": "2/3"
    },
    {
        "id": "rope_burning",
        "q": "You have two ropes, each takes 60 minutes to burn completely, but burn at non-uniform rates. Can you measure exactly 45 minutes using these ropes? Answer strictly Yes or No.",
        "target": "Yes"
    },
    {
        "id": "water_jug_exact",
        "q": "You have an empty 3-liter jug and an empty 5-liter jug, and infinite water. How many steps minimum (filling, pouring, or emptying) are required to obtain exactly 4 liters in the 5-liter jug? Output just the number.",
        "target": "6"
    }
]

def run_prompt(t, layer_a=13, layer_b=14):
    results = {}
    for item in HARD_QUESTIONS:
        prompt = f"Answer concisely with the final result:\nQuestion: {item['q']}\nAnswer:"
        cmd = [
            cli, "-m", model, "-ngl", "28", "-t", "6", "-c", "1024", "-n", "256",
            "--temp", "0", "--seed", "7", "--no-conversation", "-p", prompt
        ]
        if t > 1:
            cmd.extend([
                "--recurrent-t", str(t),
                "--recurrent-layer", str(layer_a),
                "--recurrent-layer-b", str(layer_b)
            ])
            
        res = subprocess.run(cmd, env=env, input="/exit\n", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        raw = res.stdout.split("Answer:\n")[-1].split("[ Prompt:")[0].strip().replace("\n", " ")
        results[item['id']] = raw
    return results

print("=== Running Hard Benchmarks: Baseline T=1 vs 2-Layer Recurrent T=2 ===")
print("Evaluating Baseline (T=1)...")
base_res = run_prompt(1)

print("Evaluating 2-Layer Recurrent (T=2, layers 13, 14)...")
rec_res = run_prompt(2)

print("\n| ID | Expected | Baseline T=1 | 2-Layer Recurrent T=2 |")
print("|---|---|---|---|")
for item in HARD_QUESTIONS:
    qid = item['id']
    exp = item['target']
    b_ans = (base_res[qid][:35] + '..') if len(base_res[qid]) > 35 else base_res[qid]
    r_ans = (rec_res[qid][:35] + '..') if len(rec_res[qid]) > 35 else rec_res[qid]
    print(f"| {qid} | `{exp}` | {b_ans} | {r_ans} |")

with open("/home/cune/llama.cpp/hard_bench_results.json", "w") as f:
    json.dump({"baseline": base_res, "recurrent_2layer": rec_res}, f, indent=2)

