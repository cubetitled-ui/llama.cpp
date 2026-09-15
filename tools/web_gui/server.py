#!/usr/bin/env python3
"""
web_gui/server.py
Lightweight FastAPI Web Dashboard for Fast Model Logic Benchmarking and Recurrent Parameter Tuning.
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Add project root
sys.path.insert(0, "/home/cune/llama.cpp")
from tools.mcp_server.server import LOGIC_QUESTIONS, execute_llama_cli, get_hardware_status, list_models

app = FastAPI(title="Recurrent LLM Logic Benchmark & Evaluation Studio")

class BenchmarkRequest(BaseModel):
    model_path: str
    recurrent_t: int = 1
    recurrent_layer: int = -1
    recurrent_layer_b: int = -1
    recurrent_a: float = 0.90
    recurrent_b: float = 0.10
    recurrent_gate: float = 1.00
    recurrent_config: Optional[str] = None
    lora_path: Optional[str] = None

class GenerateRequest(BaseModel):
    model_path: str
    prompt: str
    recurrent_t: int = 1
    recurrent_layer: int = -1
    recurrent_layer_b: int = -1
    recurrent_a: float = 0.90
    recurrent_b: float = 0.10
    recurrent_gate: float = 1.00
    recurrent_config: Optional[str] = None
    lora_path: Optional[str] = None
    max_tokens: int = 128

@app.get("/api/telemetry")
async def api_telemetry():
    return get_hardware_status()

@app.get("/api/models")
async def api_models():
    return list_models()

@app.get("/api/topologies")
async def api_topologies():
    topos_dir = "/home/cune/llama.cpp/recurrent_engine/topologies"
    items = []
    if os.path.exists(topos_dir):
        for f in os.listdir(topos_dir):
            if f.endswith(".json"):
                p = os.path.join(topos_dir, f)
                with open(p, "r") as fp:
                    try:
                        data = json.load(fp)
                        items.append({"filename": f, "data": data})
                    except Exception:
                        pass
    return items

class RLangCompileRequest(BaseModel):
    script_code: str
    model_path: Optional[str] = None
    dry_run: bool = True

@app.get("/api/rlang/examples")
async def api_rlang_examples():
    examples_dir = "/home/cune/llama.cpp/recurrent_engine/rlang/examples"
    items = []
    if os.path.exists(examples_dir):
        for f in os.listdir(examples_dir):
            if f.endswith(".rlang"):
                p = os.path.join(examples_dir, f)
                with open(p, "r", encoding="utf-8") as fp:
                    items.append({"filename": f, "content": fp.read()})
    return items

@app.post("/api/rlang/compile_and_run")
async def api_rlang_compile_and_run(req: RLangCompileRequest):
    from recurrent_engine.rlang.compiler import RLangCompiler
    compiler = RLangCompiler(llama_cpp_root="/home/cune/llama.cpp")
    try:
        prog = compiler.compile_source(req.script_code)
        topo = compiler.program_to_topology(prog)
        errs = topo.validate()
        if errs:
            return {"status": "error", "message": "Validation failed", "errors": errs}
        cpp_code = compiler.generate_cpp_core(prog)
        if req.dry_run:
            return {
                "status": "success",
                "mode": "dry_run",
                "architecture": topo.architecture,
                "total_layers": topo.total_layers,
                "stages": len(prog.stages),
                "cpp_code": cpp_code
            }
        compiler.patch_graph_cpp(cpp_code)
        build_cmd = ["cmake", "--build", "/home/cune/llama.cpp/build-cuda-vnni", "--target", "llama-cli", "-j4"]
        b_res = subprocess.run(build_cmd, cwd="/home/cune/llama.cpp", capture_output=True, text=True)
        if b_res.returncode != 0:
            compiler.restore_graph_cpp()
            return {"status": "error", "message": "Compilation failed", "stderr": b_res.stderr}
        target_model = req.model_path or "/home/cune/llama.cpp/models/Falcon-H1R-7B-IQ4_XS.gguf"
        from recurrent_engine.rlang.pipeline import run_fast_logic_suite
        bench_res = run_fast_logic_suite("/home/cune/llama.cpp/build-cuda-vnni/bin/llama-cli", target_model)
        return {"status": "success", "mode": "benchmark", "results": bench_res}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/benchmark")
async def api_benchmark(req: BenchmarkRequest):
    flags = []
    if req.recurrent_config:
        flags.extend(["-rc", req.recurrent_config])
    elif req.recurrent_t > 1 and req.recurrent_layer >= 0:
        flags.extend([
            "--recurrent-layer", str(req.recurrent_layer),
            "--recurrent-t", str(req.recurrent_t),
            "--recurrent-a", f"{req.recurrent_a:.2f}",
            "--recurrent-b", f"{req.recurrent_b:.2f}",
            "--recurrent-gate", f"{req.recurrent_gate:.2f}",
        ])
        if req.recurrent_layer_b >= 0:
            flags.extend(["--recurrent-layer-b", str(req.recurrent_layer_b)])
    if req.lora_path:
        flags.extend(["--lora", req.lora_path])

    passed_count = 0
    results = []
    t0 = time.time()

    for item in LOGIC_QUESTIONS:
        qid = item["id"]
        qtext = item["question"]
        expected = item["expected"]

        res = execute_llama_cli(req.model_path, qtext, flags, max_tokens=64)
        out = res.get("output", "")
        clean_out = out.lower().replace("$", "").replace(",", "")
        
        is_pass = expected.lower() in clean_out
        if is_pass:
            passed_count += 1

        results.append({
            "id": qid,
            "question": qtext,
            "expected": expected,
            "output": out,
            "passed": is_pass,
            "latency": res.get("latency_sec", 0.0)
        })

    elapsed = round(time.time() - t0, 2)
    score_pct = round((passed_count / len(LOGIC_QUESTIONS)) * 100, 1)

    return {
        "passed": passed_count,
        "total": len(LOGIC_QUESTIONS),
        "score_pct": score_pct,
        "elapsed_sec": elapsed,
        "results": results
    }

@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    flags = []
    if req.recurrent_config:
        flags.extend(["-rc", req.recurrent_config])
    elif req.recurrent_t > 1 and req.recurrent_layer >= 0:
        flags.extend([
            "--recurrent-layer", str(req.recurrent_layer),
            "--recurrent-t", str(req.recurrent_t),
            "--recurrent-a", f"{req.recurrent_a:.2f}",
            "--recurrent-b", f"{req.recurrent_b:.2f}",
            "--recurrent-gate", f"{req.recurrent_gate:.2f}",
        ])
        if req.recurrent_layer_b >= 0:
            flags.extend(["--recurrent-layer-b", str(req.recurrent_layer_b)])
    if req.lora_path:
        flags.extend(["--lora", req.lora_path])

    res = execute_llama_cli(req.model_path, req.prompt, flags, max_tokens=req.max_tokens)
    return res

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Recurrent LLM Evaluation Studio</title>
<style>
  :root {
    --bg: #0f172a;
    --card-bg: #1e293b;
    --border: #334155;
    --accent: #38bdf8;
    --accent-hover: #0284c7;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --success: #22c55e;
    --fail: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
  body { background: var(--bg); color: var(--text); padding: 24px; }
  .header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
  .title { font-size: 24px; font-weight: 700; color: var(--accent); }
  .telemetry-bar { display: flex; gap: 16px; background: var(--card-bg); padding: 10px 16px; border-radius: 8px; border: 1px solid var(--border); font-size: 13px; }
  .badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; }
  .badge-pass { background: rgba(34, 197, 94, 0.2); color: var(--success); border: 1px solid var(--success); }
  .badge-fail { background: rgba(239, 68, 68, 0.2); color: var(--fail); border: 1px solid var(--fail); }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .card h2 { font-size: 18px; margin-bottom: 16px; color: var(--accent); }
  .form-group { margin-bottom: 14px; }
  label { display: block; font-size: 12px; color: var(--text-muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
  input, select, textarea { width: 100%; background: #0f172a; border: 1px solid var(--border); color: var(--text); padding: 10px; border-radius: 6px; font-size: 14px; }
  .param-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
  button { background: var(--accent); color: #000; font-weight: 600; padding: 12px 20px; border: none; border-radius: 6px; cursor: pointer; transition: 0.2s; width: 100%; margin-top: 10px; }
  button:hover { background: var(--accent-hover); color: #fff; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 13px; }
  th, td { text-align: left; padding: 10px; border-bottom: 1px solid var(--border); }
  th { color: var(--text-muted); font-size: 12px; }
  .score-card { background: #090d16; border: 1px solid var(--accent); border-radius: 8px; padding: 16px; text-align: center; margin-top: 16px; display: none; }
  .score-val { font-size: 32px; font-weight: 800; color: var(--accent); }
  .output-box { background: #090d16; padding: 14px; border-radius: 6px; border: 1px solid var(--border); min-height: 80px; font-family: monospace; white-space: pre-wrap; font-size: 13px; color: #38bdf8; }
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="title">RECURRENT LLM BENCHMARK STUDIO</div>
    <div style="color: var(--text-muted); font-size: 13px;">Fast Logic Evaluation & Latent Architecture Exploration</div>
  </div>
  <div class="telemetry-bar" id="telemetry">
    <div>VRAM: <span id="vram-val" style="color:var(--accent);">-- / -- MB</span></div>
    <div>GPU Util: <span id="gpu-val" style="color:var(--accent);">--%</span></div>
    <div>Temp: <span id="temp-val" style="color:var(--accent);">--°C</span></div>
  </div>
</div>

<div class="grid">
  <!-- Benchmark Panel -->
  <div class="card">
    <h2>Fast Logic Benchmark (7 Verification Tests)</h2>
    <div class="form-group">
      <label>Target Model</label>
      <select id="bench-model"></select>
    </div>
    <div class="param-row">
      <div class="form-group">
        <label>Recurrence (T)</label>
        <input type="number" id="bench-t" value="2" min="1" max="16">
      </div>
      <div class="form-group">
        <label>Layer Lo</label>
        <input type="number" id="bench-lo" value="13">
      </div>
      <div class="form-group">
        <label>Layer Hi</label>
        <input type="number" id="bench-hi" value="14">
      </div>
    </div>
    <div class="param-row">
      <div class="form-group">
        <label>State Preserve (a)</label>
        <input type="number" step="0.05" id="bench-a" value="0.90">
      </div>
      <div class="form-group">
        <label>Anchor Inject (b)</label>
        <input type="number" step="0.05" id="bench-b" value="0.10">
      </div>
      <div class="form-group">
        <label>Delta Gate (γ)</label>
        <input type="number" step="0.05" id="bench-gate" value="1.00">
      </div>
    </div>
    <button id="btn-run-bench" onclick="runBenchmark()">EXECUTE LOGIC BENCHMARK</button>

    <div class="score-card" id="score-box">
      <div style="font-size: 13px; color: var(--text-muted); text-transform: uppercase;">Benchmark Score</div>
      <div class="score-val" id="score-pct">--%</div>
      <div style="font-size: 12px; color: var(--text-muted);" id="score-details">Passed 0 / 7 questions in 0.0s</div>
    </div>

    <table id="results-table" style="display:none;">
      <thead>
        <tr>
          <th>Question</th>
          <th>Expected</th>
          <th>Model Output</th>
          <th>Result</th>
        </tr>
      </thead>
      <tbody id="results-body"></tbody>
    </table>
  </div>

  <!-- Interactive Inference Playground -->
  <div class="card">
    <h2>Interactive Inference & Verification Playground</h2>
    <div class="form-group">
      <label>Prompt</label>
      <textarea id="gen-prompt" rows="4">Solve step by step: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?</textarea>
    </div>
    <div class="param-row">
      <div class="form-group">
        <label>Recurrence (T)</label>
        <input type="number" id="gen-t" value="2" min="1" max="16">
      </div>
      <div class="form-group">
        <label>Layer Lo</label>
        <input type="number" id="gen-lo" value="13">
      </div>
      <div class="form-group">
        <label>Layer Hi</label>
        <input type="number" id="gen-hi" value="14">
      </div>
    </div>
    <div class="param-row">
      <div class="form-group">
        <label>Preserve (a)</label>
        <input type="number" step="0.05" id="gen-a" value="0.90">
      </div>
      <div class="form-group">
        <label>Anchor (b)</label>
        <input type="number" step="0.05" id="gen-b" value="0.10">
      </div>
      <div class="form-group">
        <label>Gate (γ)</label>
        <input type="number" step="0.05" id="gen-gate" value="1.00">
      </div>
    </div>
    <button id="btn-run-gen" onclick="runGenerate()">RUN INFERENCE</button>

    <div style="margin-top: 20px;">
      <label>Generated Response (<span id="gen-latency">0.0s</span>)</label>
      <div class="output-box" id="gen-output">Output will appear here...</div>
    </div>
  </div>
</div>

<script>
  async function loadTelemetry() {
    try {
      const res = await fetch('/api/telemetry');
      const data = await res.json();
      if (!data.error) {
        document.getElementById('vram-val').innerText = `${data.vram_used_mib} / ${data.vram_total_mib} MB`;
        document.getElementById('gpu-val').innerText = `${data.gpu_utilization_pct}%`;
        document.getElementById('temp-val').innerText = `${data.temperature_c}°C`;
      }
    } catch (e) {}
  }
  setInterval(loadTelemetry, 3000);
  loadTelemetry();

  async function loadModels() {
    try {
      const res = await fetch('/api/models');
      const models = await res.json();
      const select = document.getElementById('bench-model');
      select.innerHTML = '';
      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.path;
        opt.innerText = `${m.name} (${m.size_gb} GB)`;
        select.appendChild(opt);
      });
    } catch (e) {}
  }
  loadModels();

  async function runBenchmark() {
    const btn = document.getElementById('btn-run-bench');
    btn.disabled = true;
    btn.innerText = 'EVALUATING MODEL...';

    const payload = {
      model_path: document.getElementById('bench-model').value,
      recurrent_t: parseInt(document.getElementById('bench-t').value),
      recurrent_layer: parseInt(document.getElementById('bench-lo').value),
      recurrent_layer_b: parseInt(document.getElementById('bench-hi').value),
      recurrent_a: parseFloat(document.getElementById('bench-a').value),
      recurrent_b: parseFloat(document.getElementById('bench-b').value),
      recurrent_gate: parseFloat(document.getElementById('bench-gate').value)
    };

    try {
      const res = await fetch('/api/benchmark', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      document.getElementById('score-box').style.display = 'block';
      document.getElementById('score-pct').innerText = `${data.score_pct}%`;
      document.getElementById('score-details').innerText = `Passed ${data.passed} / ${data.total} tests in ${data.elapsed_sec}s`;

      const tbody = document.getElementById('results-body');
      tbody.innerHTML = '';
      data.results.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${r.question.slice(0, 50)}...</td>
          <td><b>${r.expected}</b></td>
          <td>${r.output.slice(0, 60)}...</td>
          <td><span class="badge ${r.passed ? 'badge-pass' : 'badge-fail'}">${r.passed ? 'PASS' : 'FAIL'}</span></td>
        `;
        tbody.appendChild(tr);
      });
      document.getElementById('results-table').style.display = 'table';
    } catch (e) {
      alert('Benchmark error: ' + e);
    } finally {
      btn.disabled = false;
      btn.innerText = 'EXECUTE LOGIC BENCHMARK';
    }
  }

  async function runGenerate() {
    const btn = document.getElementById('btn-run-gen');
    btn.disabled = true;
    btn.innerText = 'GENERATING...';

    const payload = {
      model_path: document.getElementById('bench-model').value,
      prompt: document.getElementById('gen-prompt').value,
      recurrent_t: parseInt(document.getElementById('gen-t').value),
      recurrent_layer: parseInt(document.getElementById('gen-lo').value),
      recurrent_layer_b: parseInt(document.getElementById('gen-hi').value),
      recurrent_a: parseFloat(document.getElementById('gen-a').value),
      recurrent_b: parseFloat(document.getElementById('gen-b').value),
      recurrent_gate: parseFloat(document.getElementById('gen-gate').value),
      max_tokens: 128
    };

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      document.getElementById('gen-output').innerText = data.output || data.error;
      document.getElementById('gen-latency').innerText = `${data.latency_sec}s`;
    } catch (e) {
      document.getElementById('gen-output').innerText = 'Error: ' + e;
    } finally {
      btn.disabled = false;
      btn.innerText = 'RUN INFERENCE';
    }
  }
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML_CONTENT)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8088)
