#!/usr/bin/env python3
"""
hardened_agentic_harness.py
Hardened Multi-Hop Agentic Benchmark Harness for LLMs.
Implements a strict, budget-constrained multi-turn tool execution loop with:
- Decoy tools and schema distraction
- Multi-hop causal dependencies (Step 1 -> Output 1 -> Step 2 -> Output 2 -> Grounded Answer)
- Strict turn budget (max 3 turns) and hard termination
- Real sandboxed tool execution and evaluation
"""

import os
import sys
import json
import time
import re
import ast
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, "/home/cune/llama.cpp")
from tools.mcp_server.server import execute_llama_cli

# ==============================================================================
# Real Tool Sandbox Implementations
# ==============================================================================

def tool_calculator(expression: str) -> float:
    """Safe mathematical calculator using ast."""
    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            if isinstance(node.op, ast.Pow): return left ** right
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.USub): return -operand
        raise ValueError(f"Unsupported expression node: {node}")
    tree = ast.parse(expression, mode='eval')
    return float(_eval(tree.body))

TOOL_REGISTRY = {
    "get_incident_alert": lambda incident_id: {"incident_id": int(incident_id), "service": "auth-gateway", "target_node": "worker-04", "priority": "CRITICAL"},
    "get_node_metrics": lambda node: {"node": str(node), "cpu_pct": 99.4, "status": "OOMKilled", "exit_code": 137} if str(node) == "worker-04" else {"error": "unknown node"},
    "get_user_by_email": lambda email: {"user_id": 4091, "name": "Bob Vance", "tier": "enterprise"} if "bob" in str(email).lower() else {"error": "user not found"},
    "get_last_transaction": lambda user_id: {"user_id": int(user_id), "tx_id": "TX-9912", "amount": 1499.00, "status": "failed", "reason": "card_declined"} if int(user_id) == 4091 else {"error": "no tx"},
    "get_market_price": lambda pair, exchange: 64200.0 if "binance" in str(exchange).lower() else (64550.0 if "coinbase" in str(exchange).lower() else 0.0),
    "lookup_warehouse": lambda city: {"warehouse_id": 7, "city": "Berlin", "active": True} if "berlin" in str(city).lower() else {"error": "warehouse not found"},
    "query_inventory": lambda warehouse_id, sku: {"warehouse_id": int(warehouse_id), "sku": str(sku), "available_stock": 14} if int(warehouse_id) == 7 else {"error": "not found"},
    "get_flight_details": lambda flight_no: {"flight_no": str(flight_no), "origin_iata": "MUC", "destination_iata": "JFK"} if "402" in str(flight_no) else {"error": "unknown flight"},
    "get_airport_weather": lambda iata: {"iata": "MUC", "visibility": "800m", "fog": True, "delay_min": 90} if "muc" in str(iata).lower() else {"error": "unknown airport"},
    "dns_lookup": lambda domain: {"domain": str(domain), "resolved_ip": "10.0.4.15"} if "service" in str(domain) else {"error": "nxdomain"},
    "threat_intel": lambda ip: {"ip": str(ip), "threat_score": 85, "compromised": True, "tag": "c2_beacon"} if str(ip) == "10.0.4.15" else {"threat_score": 0},
    "get_employee_payroll": lambda emp_id: {"emp_id": int(emp_id), "gross_monthly": 8000.0, "tax_rate": 0.28} if int(emp_id) == 303 else {"error": "not found"},
    "grep_logs": lambda pattern: {"file": "src/auth.cpp", "line": 142, "exception": "NullPointerException"} if "null" in str(pattern).lower() else {"matches": 0},
    "read_file_lines": lambda file, line: {"file": str(file), "line": int(line), "code": "if (token->is_valid()) // WARNING: token pointer not null-checked before dereference"} if "auth.cpp" in str(file) else {"error": "file not found"},
    "calculator": tool_calculator,
    # Decoy tool implementations
    "reboot_node": lambda **kw: {"error": "Action Denied: Node reboot prohibited during active triage."},
    "list_all_alerts": lambda **kw: {"error": "Rate limit exceeded on list_all_alerts."},
    "place_limit_order": lambda **kw: {"error": "Action Denied: Trading disabled in audit mode."},
    "transfer_funds": lambda **kw: {"error": "Action Denied: Transfer disabled."},
    "transfer_inventory": lambda **kw: {"error": "Action Denied: Warehouse inventory locked."},
    "list_suppliers": lambda **kw: {"error": "Suppliers database unavailable."},
    "cancel_booking": lambda **kw: {"error": "Action Denied: Flight modification locked."},
    "search_flights": lambda **kw: {"error": "Search quota exhausted."},
    "block_firewall_rule": lambda **kw: {"error": "Action Denied: Security triage is read-only."},
    "flush_dns_cache": lambda **kw: {"error": "DNS cache flush failed."},
    "update_salary": lambda **kw: {"error": "Action Denied: HR payroll locked."},
    "send_tax_form": lambda **kw: {"error": "Action Denied: Mailer service offline."},
    "git_revert_commit": lambda **kw: {"error": "Action Denied: Production branch is protected."},
    "format_codebase": lambda **kw: {"error": "Formatter unavailable."},
    "calculate_shipping_freight": lambda **kw: {"error": "Freight calculation service offline."},
    "print_customs_declaration": lambda **kw: {"error": "Printer queue empty."},
    "estimate_flops": lambda **kw: {"error": "FLOP estimation tool deprecated."}
}

# ==============================================================================
# Hardened Scenarios with Decoys & Multi-Hop Chaining
# ==============================================================================

HARDENED_SCENARIOS = [
    {
        "id": "hard_01_k8s_incident",
        "description": "Multi-hop incident diagnosis: Query incident -> fetch worker metrics -> report root cause",
        "system_tools": """Available Tools:
- get_incident_alert(incident_id: int) -> dict (Returns incident service and worker node)
- get_node_metrics(node: str) -> dict (Returns node CPU and failure status)
- reboot_node(node: str) -> bool [DECOY: Do not reboot during triage]
- list_all_alerts(severity: str) -> list [DECOY]
Call tools in valid JSON: {"tool": "...", "args": {...}}""",
        "user_goal": "Investigate incident #8821. Identify the target node and report the exact termination status.",
        "hop1_tool": "get_incident_alert",
        "hop1_arg_check": lambda args: args.get("incident_id") in (8821, "8821"),
        "hop2_tool": "get_node_metrics",
        "hop2_arg_check": lambda args: "worker-04" in str(args.get("node")),
        "expected_final_keywords": ["oomkilled", "worker-04"]
    },
    {
        "id": "hard_02_failed_payment",
        "description": "Multi-hop billing lookup: Lookup user ID by email -> query failed transaction -> state exact amount",
        "system_tools": """Available Tools:
- get_user_by_email(email: str) -> dict (Returns user_id and profile)
- get_last_transaction(user_id: int) -> dict (Returns transaction amount and status)
- send_email_receipt(user_id: int) -> bool [DECOY]
- search_crm_tickets(query: str) -> list [DECOY]
Call tools in valid JSON: {"tool": "...", "args": {...}}""",
        "user_goal": "Customer 'bob@corp.com' reported a failed charge. Find their user ID and state the exact amount of the failed transaction.",
        "hop1_tool": "get_user_by_email",
        "hop1_arg_check": lambda args: "bob@corp.com" in str(args.get("email")).lower(),
        "hop2_tool": "get_last_transaction",
        "hop2_arg_check": lambda args: args.get("user_id") in (4091, "4091"),
        "expected_final_keywords": ["1499", "4091"]
    },
    {
        "id": "hard_03_crypto_spread",
        "description": "Multi-hop arbitrage: Query Binance price -> query Coinbase price -> compute spread",
        "system_tools": """Available Tools:
- get_market_price(pair: str, exchange: str) -> float (Returns spot price on exchange)
- place_limit_order(pair: str, amount: float) -> dict [DECOY]
- transfer_funds(from_acc: str, to_acc: str) -> bool [DECOY]
- calculator(expression: str) -> float (Evaluates math)
Call tools in valid JSON: {"tool": "...", "args": {...}}""",
        "user_goal": "Query the price of BTC/USDT on 'binance' and on 'coinbase'. Calculate the exact spread (Coinbase price minus Binance price).",
        "hop1_tool": "get_market_price",
        "hop1_arg_check": lambda args: "binance" in str(args.get("exchange")).lower(),
        "hop2_tool": "get_market_price",
        "hop2_arg_check": lambda args: "coinbase" in str(args.get("exchange")).lower(),
        "expected_final_keywords": ["350"]
    },
    {
        "id": "hard_04_inventory_trace",
        "description": "Multi-hop warehouse inventory: Locate Berlin warehouse ID -> inspect stock for SKU-101",
        "system_tools": """Available Tools:
- lookup_warehouse(city: str) -> dict (Returns warehouse_id for city)
- query_inventory(warehouse_id: int, sku: str) -> dict (Returns available stock)
- transfer_inventory(sku: str, from_w: int, to_w: int) -> bool [DECOY]
- list_suppliers(country: str) -> list [DECOY]
Call tools in valid JSON: {"tool": "...", "args": {...}}""",
        "user_goal": "Find the warehouse ID for 'Berlin', then check the inventory for product 'SKU-101' at that warehouse. State available stock.",
        "hop1_tool": "lookup_warehouse",
        "hop1_arg_check": lambda args: "berlin" in str(args.get("city")).lower(),
        "hop2_tool": "query_inventory",
        "hop2_arg_check": lambda args: args.get("warehouse_id") in (7, "7") and "sku-101" in str(args.get("sku")).lower(),
        "expected_final_keywords": ["14"]
    },
    {
        "id": "hard_05_flight_weather",
        "description": "Multi-hop aviation dispatch: Lookup flight departure origin -> query airport weather -> state delay",
        "system_tools": """Available Tools:
- get_flight_details(flight_no: str) -> dict (Returns flight route and departure airport code)
- get_airport_weather(iata: str) -> dict (Returns weather, fog, and delay minutes)
- cancel_booking(pnr: str) -> bool [DECOY]
- search_flights(origin: str, dest: str) -> list [DECOY]
Call tools in valid JSON: {"tool": "...", "args": {...}}""",
        "user_goal": "Lookup flight LH-402 to find its departure airport IATA code, then check weather at that airport. State the expected delay in minutes.",
        "hop1_tool": "get_flight_details",
        "hop1_arg_check": lambda args: "402" in str(args.get("flight_no")),
        "hop2_tool": "get_airport_weather",
        "hop2_arg_check": lambda args: "muc" in str(args.get("iata")).lower(),
        "expected_final_keywords": ["90"]
    },
    {
        "id": "hard_06_threat_investigation",
        "description": "Multi-hop cyber security triage: Resolve internal domain -> scan threat intelligence",
        "system_tools": """Available Tools:
- dns_lookup(domain: str) -> dict (Returns resolved IP address)
- threat_intel(ip: str) -> dict (Returns threat risk score and compromise flag)
- block_firewall_rule(ip: str) -> bool [DECOY]
- flush_dns_cache() -> bool [DECOY]
Call tools in valid JSON: {"tool": "...", "args": {...}}""",
        "user_goal": "Resolve domain 'api.service.internal' to its IP address, then inspect threat intel for that IP. State the risk score and compromise status.",
        "hop1_tool": "dns_lookup",
        "hop1_arg_check": lambda args: "api.service.internal" in str(args.get("domain")).lower(),
        "hop2_tool": "threat_intel",
        "hop2_arg_check": lambda args: "10.0.4.15" in str(args.get("ip")),
        "expected_final_keywords": ["85", "compromised"]
    },
    {
        "id": "hard_07_payroll_net_pay",
        "description": "Multi-hop payroll math: Lookup salary & tax bracket -> compute net monthly payout",
        "system_tools": """Available Tools:
- get_employee_payroll(emp_id: int) -> dict (Returns gross_monthly and tax_rate)
- calculator(expression: str) -> float (Evaluates math)
- update_bank_iban(emp_id: int, iban: str) -> bool [DECOY]
Call tools in valid JSON: {"tool": "...", "args": {...}}""",
        "user_goal": "Fetch the payroll data for employee 303. Calculate their net monthly salary after deducting tax (gross * (1 - tax_rate)).",
        "hop1_tool": "get_employee_payroll",
        "hop1_arg_check": lambda args: args.get("emp_id") in (303, "303"),
        "hop2_tool": "calculator",
        "hop2_arg_check": lambda args: "8000" in str(args.get("expression")),
        "expected_final_keywords": ["5760"]
    },
    {
        "id": "hard_08_null_deref_debugger",
        "description": "Multi-hop crash debugger: Grep exception stack -> inspect source line -> identify root cause",
        "system_tools": """Available Tools:
- grep_logs(pattern: str) -> dict (Returns file and line of exception)
- read_file_lines(file: str, line: int) -> dict (Reads source code at line)
- git_revert_commit(commit: str) -> bool [DECOY]
Call tools in valid JSON: {"tool": "...", "args": {...}}""",
        "user_goal": "Search logs for 'NullPointerException' to find the crash location, then read the code at that line to diagnose the exact issue.",
        "hop1_tool": "grep_logs",
        "hop1_arg_check": lambda args: "null" in str(args.get("pattern")).lower(),
        "hop2_tool": "read_file_lines",
        "hop2_arg_check": lambda args: "auth.cpp" in str(args.get("file")).lower() and args.get("line") in (142, "142"),
        "expected_final_keywords": ["null", "token"]
    },
    {
        "id": "hard_09_customs_tariff",
        "description": "Multi-hop trade compliance: Lookup HS code -> calculate tariff percentage",
        "system_tools": """Available Tools:
- get_hs_tariff(hs_code: str, country: str) -> dict (Returns customs duty_rate and vat)
- calculate_shipping_freight(weight_kg: float) -> float [DECOY]
- print_customs_declaration(doc_id: str) -> bool [DECOY]
Call tools in valid JSON: {"tool": "...", "args": {...}}""",
        "user_goal": "Check the customs duty rate for HS code '8507.60' imported into country 'DE'. State the duty rate as a percentage or decimal.",
        "hop1_tool": "get_hs_tariff",
        "hop1_arg_check": lambda args: "8507" in str(args.get("hs_code")),
        "hop2_tool": None,
        "expected_final_keywords": ["2.7%", "0.027", "2.7"]
    },
    {
        "id": "hard_10_kv_cache_memory",
        "description": "Multi-hop architecture math: Compute exact KV cache memory consumption for 32k context",
        "system_tools": """Available Tools:
- calculator(expression: str) -> float (Evaluates math expressions)
- estimate_flops(tokens: int) -> float [DECOY]
Call tools in valid JSON: {"tool": "...", "args": {...}}""",
        "user_goal": "A model has 28 layers, 28 heads, head_dim 128. For context 32768 tokens in FP16 (2 bytes), calculate the total KV cache memory in megabytes using: 2 * 28 * 28 * 128 * 32768 * 2 / (1024 * 1024).",
        "hop1_tool": "calculator",
        "hop1_arg_check": lambda args: "32768" in str(args.get("expression")),
        "hop2_tool": None,
        "expected_final_keywords": ["12544"]
    }
]

# ==============================================================================
# Hardened Multi-Turn Execution Engine
# ==============================================================================

def parse_json_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract tool call JSON from model output handling arbitrary nesting and markdown."""
    idx = 0
    decoder = json.JSONDecoder()
    while idx < len(text):
        pos = text.find("{", idx)
        if pos == -1:
            break
        try:
            obj, end_pos = decoder.raw_decode(text[pos:])
            if isinstance(obj, dict):
                if "tool" in obj:
                    return obj
                if "name" in obj and "arguments" in obj:
                    return {"tool": obj["name"], "args": obj["arguments"]}
            idx = pos + 1
        except Exception:
            idx = pos + 1
    return None

def run_hardened_agentic_scenario(
    scenario: Dict[str, Any],
    model_path: str,
    extra_flags: List[str],
    max_turns: int = 3
) -> Dict[str, Any]:
    """
    Executes a multi-turn agentic scenario under strict budget constraints.
    """
    sid = scenario["id"]
    sys_prompt = scenario["system_tools"]
    user_goal = scenario["user_goal"]

    conversation_history = [
        f"{sys_prompt}\n\nTask: {user_goal}\n\nAssistant:"
    ]

    cmd_flags = list(extra_flags)
    if "-r" not in cmd_flags:
        cmd_flags.extend(["-r", "<|im_end|>", "-r", "Tool Observation:"])

    hop1_passed = False
    hop2_passed = False
    final_grounded = False
    turn_details = []

    for turn in range(1, max_turns + 1):
        prompt = "\n".join(conversation_history)
        t0 = time.time()
        res = execute_llama_cli(model_path, prompt, cmd_flags, max_tokens=128)
        dur = round(time.time() - t0, 2)
        out = res.get("output", "")

        turn_details.append({"turn": turn, "latency": dur, "model_output": out})

        # Check if model produced a tool call
        call = parse_json_tool_call(out)
        if call:
            t_name = call.get("tool")
            t_args = call.get("args", {})
            if isinstance(t_args, str):
                try: t_args = json.loads(t_args)
                except: t_args = {}

            # Check Hop 1
            if not hop1_passed and t_name == scenario["hop1_tool"]:
                if scenario["hop1_arg_check"](t_args):
                    hop1_passed = True

            # Check Hop 2
            if hop1_passed and not hop2_passed and scenario.get("hop2_tool"):
                if t_name == scenario["hop2_tool"]:
                    if scenario["hop2_arg_check"](t_args):
                        hop2_passed = True

            # Execute tool in sandbox
            if t_name in TOOL_REGISTRY:
                try:
                    tool_fn = TOOL_REGISTRY[t_name]
                    # Pass args dict or unpack
                    if isinstance(t_args, dict):
                        t_result = tool_fn(**t_args)
                    else:
                        t_result = tool_fn(t_args)
                except Exception as e:
                    t_result = f"Execution Error: {e}"
            else:
                t_result = f"Error: Tool '{t_name}' not found."

            # Inject Tool Observation into context for next turn
            obs_str = json.dumps(t_result) if not isinstance(t_result, str) else t_result
            conversation_history.append(f"{out}\n\nTool Observation: {obs_str}\n\nAssistant:")
        else:
            # Model emitted final natural language answer
            out_clean = out.lower().replace("$", "").replace(",", "")
            expected_kws = scenario["expected_final_keywords"]
            grounded = any(kw.lower() in out_clean for kw in expected_kws)
            if grounded:
                final_grounded = True
            break

    # Determine overall success:
    # Requires Hop 1 + Hop 2 (if specified) + Final grounded answer
    req_hop2 = scenario.get("hop2_tool") is not None
    if req_hop2:
        success = hop1_passed and hop2_passed and final_grounded
    else:
        success = hop1_passed and final_grounded

    return {
        "id": sid,
        "success": success,
        "hop1_passed": hop1_passed,
        "hop2_passed": hop2_passed if req_hop2 else True,
        "final_grounded": final_grounded,
        "turns": len(turn_details),
        "total_latency": round(sum(t["latency"] for t in turn_details), 2),
        "details": turn_details
    }

def main():
    model_path = "/home/cune/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    lora_path = "/home/cune/llama.cpp/models/recurrent_core_lora/checkpoint_step_200.gguf"
    flags = ["--recurrent-t", "2", "--recurrent-layer", "13", "--recurrent-layer-b", "14", "--lora", lora_path]

    print("=" * 80)
    print("HARDENED MULTI-HOP AGENTIC BENCHMARK (10 SCENARIOS)")
    print(f"Model: {model_path}")
    print(f"Flags: {' '.join(flags)}")
    print("=" * 80)

    all_results = []
    passed_count = 0
    total_latency = 0.0

    for idx, sc in enumerate(HARDENED_SCENARIOS):
        sid = sc["id"]
        desc = sc["description"]
        print(f"\n[{idx+1}/10] Testing: {sid}")
        print(f"Goal: {desc}")

        res = run_hardened_agentic_scenario(sc, model_path, flags, max_turns=3)
        all_results.append(res)
        total_latency += res["total_latency"]

        status = "PASS" if res["success"] else "FAIL"
        if res["success"]:
            passed_count += 1

        print(f"  Result: [{status}] in {res['turns']} turns ({res['total_latency']}s)")
        print(f"  Hop 1 Validated: {res['hop1_passed']} | Hop 2 Validated: {res['hop2_passed']} | Grounded: {res['final_grounded']}")

    score_pct = round((passed_count / len(HARDENED_SCENARIOS)) * 100, 1)
    print("\n" + "=" * 80)
    print(f"HARDENED AGENTIC SCORE: {passed_count}/{len(HARDENED_SCENARIOS)} ({score_pct}%) in {round(total_latency, 2)}s")
    print("=" * 80)

    out_path = "/home/cune/llama.cpp/benchmarks/comprehensive/hardened_agentic_results.json"
    summary_data = {
        "score_pct": score_pct,
        "passed": passed_count,
        "total": len(HARDENED_SCENARIOS),
        "total_latency_s": round(total_latency, 2),
        "results": all_results
    }
    with open(out_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved hardened benchmark results to: {out_path}")

if __name__ == "__main__":
    main()
