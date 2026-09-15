#!/usr/bin/env python3
"""
generate_52_tasks.py
Generates the 52-task Hardened Comprehensive Multi-Domain Benchmark Suite:
- 13 Multi-Turn Agentic Tasks (strict schema, parameter chaining, verification)
- 13 Algorithmic Coding Tasks (LeetCode Hard/Medium with live assertions)
- 13 Frontend Web Engineering Tasks (FSM, virtual scrolling, signals, concurrency)
- 13 Probabilistic & Deductive Logic Tasks (Bayes, Diophantine, game theory, permutations)
"""

import json
import os

tasks = {
  "agentic_loop": [
    {
      "id": "agent_01_calc",
      "description": "Calculate discounted price with calculator tool and ground final answer",
      "turn1_prompt": "Available Tools:\n- calculator(expression: str) -> float\nUser Request: Calculate the total bill for 14 licenses at $39.50 each with a 15% discount. Call the tool as valid JSON: {\"tool\": \"calculator\", \"args\": {\"expression\": ...}}",
      "tool_name": "calculator",
      "tool_output": 470.05,
      "turn2_prompt": "Tool Observation: 470.05\nNow state the final total bill amount clearly in dollars.",
      "expected_final": ["470.05", "470"],
      "validation_type": "agentic_multi_turn"
    },
    {
      "id": "agent_02_sql",
      "description": "Query database and extract record count",
      "turn1_prompt": "Available Tools:\n- sql_query(query: str) -> list[dict]\nUser Request: Find how many active users exist in department 'AI'. Call the tool as valid JSON: {\"tool\": \"sql_query\", \"args\": {\"query\": ...}}",
      "tool_name": "sql_query",
      "tool_output": "[{\"count\": 37}]",
      "turn2_prompt": "Tool Observation: [{\"count\": 37}]\nHow many active users are in the AI department? Answer with only the number.",
      "expected_final": ["37"],
      "validation_type": "agentic_multi_turn"
    },
    {
      "id": "agent_03_file",
      "description": "Read server configuration and locate port",
      "turn1_prompt": "Available Tools:\n- file_read(path: str) -> str\nUser Request: Inspect 'config/server.yaml' to find the listening port. Call the tool as valid JSON: {\"tool\": \"file_read\", \"args\": {\"path\": ...}}",
      "tool_name": "file_read",
      "tool_output": "host: 127.0.0.1\nport: 8088\ntls: false",
      "turn2_prompt": "Tool Observation: host: 127.0.0.1\nport: 8088\ntls: false\nWhat port is the server configured to listen on? Answer with only the port number.",
      "expected_final": ["8088"],
      "validation_type": "agentic_multi_turn"
    },
    {
      "id": "agent_04_weather",
      "description": "Fetch weather observation and make clothing recommendation",
      "turn1_prompt": "Available Tools:\n- get_weather(city: str) -> dict\nUser Request: Check current weather in Reykjavik to see if a coat is needed. Call the tool as valid JSON: {\"tool\": \"get_weather\", \"args\": {\"city\": ...}}",
      "tool_name": "get_weather",
      "tool_output": "{\"city\": \"Reykjavik\", \"temp_c\": 4.2, \"condition\": \"Rainy\"}",
      "turn2_prompt": "Tool Observation: {\"city\": \"Reykjavik\", \"temp_c\": 4.2, \"condition\": \"Rainy\"}\nBased on the 4.2C temperature and rain, is a coat needed? Answer clearly with Yes or No and explain.",
      "expected_final": ["yes", "coat"],
      "validation_type": "agentic_multi_turn"
    },
    {
      "id": "agent_05_stock",
      "description": "Look up stock ticker price and multiply by share count",
      "turn1_prompt": "Available Tools:\n- stock_quote(ticker: str) -> dict\nUser Request: Get the current price of NVDA. Call the tool as valid JSON: {\"tool\": \"stock_quote\", \"args\": {\"ticker\": ...}}",
      "tool_name": "stock_quote",
      "tool_output": "{\"ticker\": \"NVDA\", \"price\": 125.00}",
      "turn2_prompt": "Tool Observation: {\"ticker\": \"NVDA\", \"price\": 125.00}\nIf an investor holds 50 shares of NVDA at $125.00, what is the total position value? Answer with the exact dollar number.",
      "expected_final": ["6250", "6,250"],
      "validation_type": "agentic_multi_turn"
    },
    {
      "id": "agent_06_db_update",
      "description": "Dispatch record update to relational table",
      "turn1_prompt": "Available Tools:\n- db_update(table: str, id: int, fields: dict) -> bool\nUser Request: Upgrade customer 442 to 'enterprise' tier in table 'users'. Call the tool as valid JSON: {\"tool\": \"db_update\", \"args\": {\"table\": ..., \"id\": ..., \"fields\": ...}}",
      "tool_name": "db_update",
      "tool_output": "true",
      "turn2_prompt": "Tool Observation: true\nHas customer 442 been successfully updated? Answer clearly with status.",
      "expected_final": ["success", "updated", "yes", "true"],
      "validation_type": "agentic_multi_turn"
    },
    {
      "id": "agent_07_unit_convert",
      "description": "Convert velocity units and round to two decimal places",
      "turn1_prompt": "Available Tools:\n- unit_convert(value: float, from_unit: str, to_unit: str) -> float\nUser Request: Convert 120 km/h to meters per second. Call the tool as valid JSON: {\"tool\": \"unit_convert\", \"args\": {\"value\": ..., \"from_unit\": ..., \"to_unit\": ...}}",
      "tool_name": "unit_convert",
      "tool_output": 33.333333,
      "turn2_prompt": "Tool Observation: 33.333333\nWhat is 120 km/h in meters per second (m/s)? Answer with the number rounded to one or two decimal places.",
      "expected_final": ["33.33", "33.3"],
      "validation_type": "agentic_multi_turn"
    },
    {
      "id": "agent_08_kb_search",
      "description": "Perform knowledge base semantic vector retrieval",
      "turn1_prompt": "Available Tools:\n- search_kb(query: str, top_k: int) -> list[dict]\nUser Request: Search knowledge base for 'recurrent lora stability bounds'. Call the tool as valid JSON: {\"tool\": \"search_kb\", \"args\": {\"query\": ..., \"top_k\": ...}}",
      "tool_name": "search_kb",
      "tool_output": "[{\"doc_id\": \"chunk-88\", \"score\": 0.94, \"text\": \"Lyapunov condition: a + b*gamma <= 1.0 ensures contractive stability.\"}]",
      "turn2_prompt": "Tool Observation: [{\"doc_id\": \"chunk-88\", \"score\": 0.94, \"text\": \"Lyapunov condition: a + b*gamma <= 1.0 ensures contractive stability.\"}]\nWhich doc_id contained the top stability result?",
      "expected_final": ["chunk-88", "88"],
      "validation_type": "agentic_multi_turn"
    },
    {
      "id": "agent_09_k8s_status",
      "description": "Inspect Kubernetes pod health and report failure code",
      "turn1_prompt": "Available Tools:\n- k8s_get_pod(namespace: str, pod: str) -> dict\nUser Request: Check status of pod 'api-gateway-0' in namespace 'prod'. Call tool in JSON: {\"tool\": \"k8s_get_pod\", \"args\": {\"namespace\": ..., \"pod\": ...}}",
      "tool_name": "k8s_get_pod",
      "tool_output": "{\"pod\": \"api-gateway-0\", \"status\": \"CrashLoopBackOff\", \"restart_count\": 5, \"exit_code\": 137}",
      "turn2_prompt": "Tool Observation: {\"pod\": \"api-gateway-0\", \"status\": \"CrashLoopBackOff\", \"restart_count\": 5, \"exit_code\": 137}\nWhat is the pod status and its exit code?",
      "expected_final": ["crashloopbackoff", "137"],
      "validation_type": "agentic_multi_turn"
    },
    {
      "id": "agent_10_forex_triangular",
      "description": "Calculate forex triangular cross rate",
      "turn1_prompt": "Available Tools:\n- get_fx_rate(pair: str) -> float\nUser Request: Look up the EUR/USD exchange rate. Call tool in JSON: {\"tool\": \"get_fx_rate\", \"args\": {\"pair\": ...}}",
      "tool_name": "get_fx_rate",
      "tool_output": 1.0850,
      "turn2_prompt": "Tool Observation: 1.0850\nIf EUR/USD is 1.0850 and USD/JPY is 155.00, what is the cross rate of EUR/JPY (1.0850 * 155.00)? State the rate.",
      "expected_final": ["168.17", "168.18", "168.2", "168"],
      "validation_type": "agentic_multi_turn"
    },
    {
      "id": "agent_11_auth_jwt",
      "description": "Inspect token payload and extract user identity",
      "turn1_prompt": "Available Tools:\n- decode_jwt_token(token: str) -> dict\nUser Request: Decode auth token 'eyJhbGciOiJIUzI1NiJ9.eyJ1aWQiOjkwMjEsInJvbGUiOiJhZG1pbiJ9' to find user identity. Call tool in JSON: {\"tool\": \"decode_jwt_token\", \"args\": {\"token\": ...}}",
      "tool_name": "decode_jwt_token",
      "tool_output": "{\"uid\": 9021, \"role\": \"admin\", \"exp\": 1735689600}",
      "turn2_prompt": "Tool Observation: {\"uid\": 9021, \"role\": \"admin\", \"exp\": 1735689600}\nWhat is the user ID and role in this token?",
      "expected_final": ["9021", "admin"],
      "validation_type": "agentic_multi_turn"
    },
    {
      "id": "agent_12_cve_severity",
      "description": "Query National Vulnerability Database for CVSS score",
      "turn1_prompt": "Available Tools:\n- query_nvd_cve(cve_id: str) -> dict\nUser Request: Check severity of CVE-2024-38077. Call tool in JSON: {\"tool\": \"query_nvd_cve\", \"args\": {\"cve_id\": ...}}",
      "tool_name": "query_nvd_cve",
      "tool_output": "{\"cve\": \"CVE-2024-38077\", \"score\": 9.8, \"severity\": \"CRITICAL\"}",
      "turn2_prompt": "Tool Observation: {\"cve\": \"CVE-2024-38077\", \"score\": 9.8, \"severity\": \"CRITICAL\"}\nWhat is the CVSS base score and severity level?",
      "expected_final": ["9.8", "critical"],
      "validation_type": "agentic_multi_turn"
    },
    {
      "id": "agent_13_disk_cleanup",
      "description": "Find large disk directories for server cleanup",
      "turn1_prompt": "Available Tools:\n- inspect_disk_usage(directory: str) -> list[dict]\nUser Request: Inspect disk usage under '/var/log'. Call tool in JSON: {\"tool\": \"inspect_disk_usage\", \"args\": {\"directory\": ...}}",
      "tool_name": "inspect_disk_usage",
      "tool_output": "[{\"path\": \"/var/log/journal\", \"size_gb\": 4.2}, {\"path\": \"/var/log/nginx\", \"size_gb\": 0.3}]",
      "turn2_prompt": "Tool Observation: [{\"path\": \"/var/log/journal\", \"size_gb\": 4.2}, {\"path\": \"/var/log/nginx\", \"size_gb\": 0.3}]\nWhich path consumes the most disk space and how much?",
      "expected_final": ["/var/log/journal", "4.2"],
      "validation_type": "agentic_multi_turn"
    }
  ],

  "coding": [
    {
      "id": "code_01_two_sum",
      "description": "Two Sum returning 0-indexed indices",
      "prompt": "Write a Python function `two_sum(nums: list[int], target: int) -> list[int]` that returns the indices of the two numbers in `nums` that add up to `target`. Assume exactly one solution exists. Provide only Python code inside a markdown block.",
      "test_code": "assert two_sum([2, 7, 11, 15], 9) in ([0, 1], [1, 0])\nassert two_sum([3, 2, 4], 6) in ([1, 2], [2, 1])\nassert two_sum([3, 3], 6) in ([0, 1], [1, 0])",
      "validation_type": "python_unit_test"
    },
    {
      "id": "code_02_valid_parens",
      "description": "Bracket validator for (), [], {}",
      "prompt": "Write a Python function `is_valid(s: str) -> bool` that determines if input string of brackets `()[]{}` is valid. Provide only Python code inside a markdown block.",
      "test_code": "assert is_valid('()[]{}') is True\nassert is_valid('(]') is False\nassert is_valid('([)]') is False\nassert is_valid('{[]}') is True\nassert is_valid('') is True",
      "validation_type": "python_unit_test"
    },
    {
      "id": "code_03_merge_intervals",
      "description": "Merge overlapping intervals",
      "prompt": "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges all overlapping intervals. Provide only Python code inside a markdown block.",
      "test_code": "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]]) == [[1,5]]\nassert merge_intervals([]) == []",
      "validation_type": "python_unit_test"
    },
    {
      "id": "code_04_flatten_list",
      "description": "Arbitrarily nested list flattener",
      "prompt": "Write a Python function `flatten(lst: list) -> list` that flattens an arbitrarily deep nested list into a 1D list. Provide only Python code inside a markdown block.",
      "test_code": "assert flatten([1, [2, [3, [4, 5]], 6], 7]) == [1, 2, 3, 4, 5, 6, 7]\nassert flatten([]) == []\nassert flatten([[1], [2], [3]]) == [1, 2, 3]",
      "validation_type": "python_unit_test"
    },
    {
      "id": "code_05_binary_search",
      "description": "Binary search in sorted array",
      "prompt": "Write a Python function `binary_search(nums: list[int], target: int) -> int` that returns index of target in sorted list or -1 if not found. Provide only Python code inside a markdown block.",
      "test_code": "assert binary_search([-1,0,3,5,9,12], 9) == 4\nassert binary_search([-1,0,3,5,9,12], 2) == -1\nassert binary_search([5], 5) == 0",
      "validation_type": "python_unit_test"
    },
    {
      "id": "code_06_longest_unique_substr",
      "description": "Length of longest substring without repeating characters",
      "prompt": "Write a Python function `length_of_longest_substring(s: str) -> int` returning the length of the longest substring without duplicate characters. Provide only Python code inside a markdown block.",
      "test_code": "assert length_of_longest_substring('abcabcbb') == 3\nassert length_of_longest_substring('bbbbb') == 1\nassert length_of_longest_substring('pwwkew') == 3\nassert length_of_longest_substring('') == 0",
      "validation_type": "python_unit_test"
    },
    {
      "id": "code_07_coin_change_min",
      "description": "Minimum coins for change or -1",
      "prompt": "Write a Python function `coin_change(coins: list[int], amount: int) -> int` that computes the fewest coins needed to make up amount. Return -1 if not possible. Provide only Python code inside a markdown block.",
      "test_code": "assert coin_change([1, 2, 5], 11) == 3\nassert coin_change([2], 3) == -1\nassert coin_change([1], 0) == 0",
      "validation_type": "python_unit_test"
    },
    {
      "id": "code_08_longest_consecutive",
      "description": "Longest consecutive elements sequence in O(n)",
      "prompt": "Write a Python function `longest_consecutive(nums: list[int]) -> int` that returns length of longest consecutive elements sequence in O(n) time. Provide only Python code inside a markdown block.",
      "test_code": "assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4\nassert longest_consecutive([0,3,7,2,5,8,4,6,0,1]) == 9\nassert longest_consecutive([]) == 0",
      "validation_type": "python_unit_test"
    },
    {
      "id": "code_09_trapping_rain_water",
      "description": "Calculate total trapped rainwater",
      "prompt": "Write a Python function `trap_rain_water(height: list[int]) -> int` computing total trapped rainwater. Provide only Python code inside a markdown block.",
      "test_code": "assert trap_rain_water([0,1,0,2,1,0,1,3,2,1,2,1]) == 6\nassert trap_rain_water([4,2,0,3,2,5]) == 9\nassert trap_rain_water([1,2,3]) == 0",
      "validation_type": "python_unit_test"
    },
    {
      "id": "code_10_word_break",
      "description": "Word break dynamic programming segmentation",
      "prompt": "Write a Python function `word_break(s: str, wordDict: list[str]) -> bool` determining if string s can be segmented into words from wordDict. Provide only Python code inside a markdown block.",
      "test_code": "assert word_break('leetcode', ['leet', 'code']) is True\nassert word_break('applepenapple', ['apple', 'pen']) is True\nassert word_break('catsandog', ['cats', 'dog', 'sand', 'and', 'cat']) is False",
      "validation_type": "python_unit_test"
    },
    {
      "id": "code_11_course_schedule",
      "description": "Topological cycle detection for course prerequisites",
      "prompt": "Write a Python function `can_finish(numCourses: int, prerequisites: list[list[int]]) -> bool` that returns True if all courses can be finished without cyclical prerequisites. Provide only Python code inside a markdown block.",
      "test_code": "assert can_finish(2, [[1, 0]]) is True\nassert can_finish(2, [[1, 0], [0, 1]]) is False\nassert can_finish(4, [[1, 0], [2, 0], [3, 1], [3, 2]]) is True",
      "validation_type": "python_unit_test"
    },
    {
      "id": "code_12_lru_cache",
      "description": "LRUCache class with get and put in O(1)",
      "prompt": "Implement an `LRUCache` class with `__init__(capacity: int)`, `get(key: int) -> int` (returns value or -1), and `put(key: int, value: int)` with O(1) operations and LRU eviction. Provide only Python code inside a markdown block.",
      "test_code": "c = LRUCache(2)\nc.put(1, 1)\nc.put(2, 2)\nassert c.get(1) == 1\nc.put(3, 3)\nassert c.get(2) == -1\nc.put(4, 4)\nassert c.get(1) == -1\nassert c.get(3) == 3\nassert c.get(4) == 4",
      "validation_type": "python_unit_test"
    },
    {
      "id": "code_13_max_product_subarray",
      "description": "Maximum product contiguous subarray",
      "prompt": "Write a Python function `max_product(nums: list[int]) -> int` that finds the contiguous subarray with the largest product. Provide only Python code inside a markdown block.",
      "test_code": "assert max_product([2, 3, -2, 4]) == 6\nassert max_product([-2, 0, -1]) == 0\nassert max_product([-2, 3, -4]) == 24",
      "validation_type": "python_unit_test"
    }
  ],

  "web_dev": [
    {
      "id": "web_01_countdown",
      "description": "Countdown timer with setInterval and clearInterval",
      "prompt": "Write a JavaScript function `startCountdown(seconds, displayElem, onComplete)` that decrements every second using setInterval, updates displayElem.textContent, stops at 0 with clearInterval, and calls onComplete.",
      "keywords": ["setInterval", "clearInterval", "textContent"],
      "validation_type": "keywords"
    },
    {
      "id": "web_02_dark_mode",
      "description": "Dark mode theme toggle with localStorage persistence",
      "prompt": "Write a JavaScript function `toggleDarkMode()` that toggles class 'dark-mode' on document.body and saves state in localStorage under key 'theme'.",
      "keywords": ["classList.toggle", "localStorage.setItem", "localStorage.getItem"],
      "validation_type": "keywords"
    },
    {
      "id": "web_03_modal_esc",
      "description": "Modal dismiss on Escape key listener",
      "prompt": "Write a JavaScript function `attachModalDismiss(modalElem)` that adds a keydown event listener to window closing modalElem when Escape key is pressed.",
      "keywords": ["Escape", "addEventListener", "keydown"],
      "validation_type": "keywords"
    },
    {
      "id": "web_04_responsive_nav",
      "description": "Mobile hamburger navbar toggle with aria-expanded",
      "prompt": "Write a JavaScript function `setupMobileNav(buttonElem, menuElem)` that toggles class 'active' and updates aria-expanded attribute between 'true' and 'false' on click.",
      "keywords": ["aria-expanded", "classList.toggle"],
      "validation_type": "keywords"
    },
    {
      "id": "web_05_debounce",
      "description": "Debounce utility with timeout cancellation",
      "prompt": "Implement a generic JavaScript `debounce(fn, delay)` function returning debounced wrapper with clearTimeout.",
      "keywords": ["clearTimeout", "setTimeout", "apply"],
      "validation_type": "keywords"
    },
    {
      "id": "web_06_fetch_retry",
      "description": "Async fetch with exponential backoff retries",
      "prompt": "Write an async JavaScript function `fetchWithRetry(url, retries, delay)` using async/await that retries fetch upon error with delay multiplier.",
      "keywords": ["async", "await", "fetch", "retries"],
      "validation_type": "keywords"
    },
    {
      "id": "web_07_custom_tabs",
      "description": "Accessible tab navigation controller with aria-selected",
      "prompt": "Write JavaScript `activateTab(clickedTab, allTabs, allPanels)` that updates aria-selected, tabindex, and toggles hidden attribute on associated panels.",
      "keywords": ["aria-selected", "tabindex", "hidden"],
      "validation_type": "keywords"
    },
    {
      "id": "web_08_css_grid",
      "description": "Responsive CSS Grid rule with auto-fit and minmax",
      "prompt": "Write CSS rules for a class `.product-grid` using display: grid, repeat, auto-fit, minmax(280px, 1fr), and gap: 1.5rem.",
      "keywords": ["grid-template-columns", "repeat", "auto-fit", "minmax"],
      "validation_type": "keywords"
    },
    {
      "id": "web_09_promise_pool",
      "description": "Promise concurrency worker pool",
      "prompt": "Write an async JavaScript function `promisePool(tasks, limit)` executing an array of async task functions with at most `limit` tasks running concurrently.",
      "keywords": ["Promise", "async", "Promise.all"],
      "validation_type": "keywords"
    },
    {
      "id": "web_10_virtual_list_calc",
      "description": "Virtual scroller slice index and translateY offset calculator",
      "prompt": "Write a JavaScript function `calcVirtualSlice(scrollTop, viewportHeight, itemHeight, totalItems, buffer)` returning `{startIndex, endIndex, offsetY}`.",
      "keywords": ["scrollTop", "itemHeight", "Math.floor"],
      "validation_type": "keywords"
    },
    {
      "id": "web_11_fsm_state_machine",
      "description": "Finite State Machine with transition map",
      "prompt": "Implement a JavaScript `createFSM(initialState, transitions)` function returning `{getState, transition(event)}` that validates allowed transitions.",
      "keywords": ["transition", "state", "transitions"],
      "validation_type": "keywords"
    },
    {
      "id": "web_12_svg_arc_path",
      "description": "SVG pie/donut slice arc path generator",
      "prompt": "Write a JavaScript function `polarToCartesian(cx, cy, r, angleDeg)` and `describeArc(cx, cy, r, startAngle, endAngle)` returning SVG 'd' path string using Math.cos and Math.sin.",
      "keywords": ["Math.cos", "Math.sin", "path", "d"],
      "validation_type": "keywords"
    },
    {
      "id": "web_13_reactive_signal",
      "description": "Fine-grained reactive signal getter and setter",
      "prompt": "Write a JavaScript function `createSignal(initialVal)` returning `[get, set]` where set notifies registered subscribers.",
      "keywords": ["signal", "subscribers", "notify"],
      "validation_type": "keywords"
    }
  ],

  "logic": [
    {
      "id": "logic_01_bat_ball",
      "description": "Cognitive reflection: Bat and ball total $1.10",
      "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Answer with only the exact numerical price in dollars or cents.",
      "expected": "0.05",
      "validation_type": "contains"
    },
    {
      "id": "logic_02_train_speed",
      "description": "Harmonic mean: Average speed of 40 and 60 km/h",
      "prompt": "A train travels 60 km at 40 km/h and returns 60 km at 60 km/h. What is the average speed for the round trip? Answer with only the number in km/h.",
      "expected": "48",
      "validation_type": "contains"
    },
    {
      "id": "logic_03_snail_well",
      "description": "Discrete state: Snail climbing 10m well (+3m, -2m)",
      "prompt": "A snail is at the bottom of a 10-meter well. Each day it climbs up 3 meters, but each night it slips back 2 meters. On which day will the snail first reach the top? Answer with only the number of days.",
      "expected": "8",
      "validation_type": "contains"
    },
    {
      "id": "logic_04_lily_pads",
      "description": "Exponential growth: Lily pads doubling daily",
      "prompt": "A patch of lily pads doubles in size every day. If it takes 48 days for the patch to completely cover the entire lake, on what day was the lake exactly half covered? Answer with only the day number.",
      "expected": "47",
      "validation_type": "contains"
    },
    {
      "id": "logic_05_machines",
      "description": "Rate problem: 5 machines, 5 widgets, 5 minutes",
      "prompt": "If 5 machines take 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets? Answer with only the number of minutes.",
      "expected": "5",
      "validation_type": "contains"
    },
    {
      "id": "logic_06_sheep",
      "description": "Literal phrasing: 17 sheep, all but 9 die",
      "prompt": "A farmer has 17 sheep, and all but 9 run away. How many sheep does the farmer have left? Answer with only the number of sheep.",
      "expected": "9",
      "validation_type": "contains"
    },
    {
      "id": "logic_07_sister_age",
      "description": "Invariant difference: Sister age problem",
      "prompt": "When I was 6 years old, my sister was half my age. Now I am 70 years old. How old is my sister? Answer with only the sister's age.",
      "expected": "67",
      "validation_type": "contains"
    },
    {
      "id": "logic_08_monty_hall",
      "description": "Monty Hall 3 doors switching probability",
      "prompt": "In the standard Monty Hall problem with 3 doors and 1 car, after the host reveals a goat behind an unchosen door, what is the exact probability of winning if you switch? Answer as a fraction or percentage.",
      "expected": "2/3",
      "validation_type": "contains"
    },
    {
      "id": "logic_09_bayesian_disease",
      "description": "Bayesian posterior probability calculation",
      "prompt": "A disease has a prevalence of 1 in 1000 (0.001). A diagnostic test has 99% sensitivity (true positive) and a 5% false positive rate (0.05). If a randomly selected person tests positive, what is the approximate posterior probability that they actually have the disease? State the percentage or decimal.",
      "expected": "2%",
      "validation_type": "contains"
    },
    {
      "id": "logic_10_monty_hall_4doors",
      "description": "Monty Hall 4-door variant probability",
      "prompt": "Consider a 4-door Monty Hall problem: 1 car, 3 goats. You pick Door 1. The host reveals 1 goat behind one of the other doors. You then switch by choosing randomly between the remaining 2 unopened doors. What is your probability of winning? State as a fraction (e.g. 3/8).",
      "expected": "3/8",
      "validation_type": "contains"
    },
    {
      "id": "logic_11_water_jug_4L",
      "description": "Diophantine water jug 5L and 3L measuring 4L",
      "prompt": "You have a 5-liter jug and a 3-liter jug, with an unlimited water supply and no markings. What is the minimum number of steps (fills, empties, or transfers) needed to measure exactly 4 liters in the 5-liter jug? Answer with only the number of steps.",
      "expected": "6",
      "validation_type": "contains"
    },
    {
      "id": "logic_12_two_dice_sum8",
      "description": "Conditional probability of two fair dice given sum 8",
      "prompt": "Two standard fair 6-sided dice are rolled. Given that the sum of their faces is 8, what is the probability that at least one of the dice shows a 5? Answer as a fraction (e.g. 2/5).",
      "expected": "2/5",
      "validation_type": "contains"
    },
    {
      "id": "logic_13_clock_angle_315",
      "description": "Analog clock hand angle at 3:15",
      "prompt": "What is the acute angle in degrees between the hour hand and the minute hand of a standard analog clock at exactly 3:15? Answer with only the number in degrees.",
      "expected": "7.5",
      "validation_type": "contains"
    }
  ]
}

out_path = "/home/cune/llama.cpp/benchmarks/comprehensive/dataset.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "benchmark_name": "Comprehensive Multi-Domain Benchmark Suite (52 Hardened Tasks)",
        "version": "4.0.0",
        "description": "Exhaustive evaluation suite covering 52 non-saturated tasks across Multi-Turn Agentic Loops, LeetCode Hard Algorithmic Coding, Advanced Web Engineering, and Probabilistic Deductive Logic.",
        "tasks": tasks
    }, f, indent=2)

total = sum(len(v) for v in tasks.values())
print(f"Successfully generated {total} tasks in {out_path}!")
for k, v in tasks.items():
    print(f"  - {k}: {len(v)} tasks")
