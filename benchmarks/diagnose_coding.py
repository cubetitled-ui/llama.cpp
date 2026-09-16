import json
import subprocess
import re

with open('benchmarks/comprehensive/dataset.json') as f:
    d = json.load(f)

cli = '/home/cune/llama.cpp/build-cuda-vnni/bin/llama-cli'
model = '/home/cune/qwen2.5-coder-7b-instruct-q4_k_m.gguf'

target_ids = [
    'code_03_lazy_segment_tree',
    'code_07_avl_tree_invariants',
    'code_09_expression_calculator_shunting_yard',
    'code_12_tarjan_scc'
]

coding_tasks = [t for t in d['tasks']['coding'] if t['id'] in target_ids]

for t in coding_tasks:
    print(f"=== Task: {t['id']} ===")
    prompt = t['prompt']
    cmd = [cli, '-m', model, '-p', prompt, '-n', '600', '--temp', '0.0', '-ngl', '99', '-c', '2048', '--log-disable', '-st', '--simple-io', '-fa', 'on', '--no-warmup']
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = res.stdout
    if prompt in out:
        out = out.split(prompt, 1)[1]
    out = out.replace('Exiting...', '').strip()
    
    m = re.search(r'```(?:python|py)?\s*(.*?)\s*```', out, re.DOTALL)
    code = m.group(1).strip() if m else out.strip()
    lines = [l for l in code.split('\n') if not l.startswith('#!') and not l.startswith('```')]
    sanitized = '\n'.join(lines)
    test_script = sanitized + '\n\n' + t.get('test_code', '')
    
    try:
        p = subprocess.run(['python3', '-c', test_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        print('Returncode:', p.returncode)
        if p.returncode != 0:
            print('Error tail:\n', p.stderr.strip()[-400:])
        else:
            print('PASS!')
    except Exception as e:
        print('Exception running test:', e)
