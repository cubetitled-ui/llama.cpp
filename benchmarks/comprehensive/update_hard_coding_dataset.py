#!/usr/bin/env python3
"""
update_hard_coding_dataset.py
Replaces the memorized LeetCode-easy coding tasks in dataset.json with 13 brutal,
systems-level algorithmic tasks (Bytecode VM, Lisp closures, Dinic Max Flow,
Transactional KV with nested rollback, Lazy Segment Tree, Shunting Yard, Tarjan SCC, etc.)
"""

import json

HARD_CODING_TASKS = [
    {
        "id": "code_01_regex_nfa",
        "description": "Custom recursive regex engine supporting ., *, +, and [a-z] classes without re module",
        "prompt": "Write a Python function `regex_match(pattern: str, text: str) -> bool` implementing pattern matching supporting: `.` (any single char), `*` (zero or more of preceding char/class), `+` (one or more of preceding char/class), and character classes like `[0-9]` or `[a-z]`. Do not import or use Python's `re` module. Provide only Python code inside a markdown block.",
        "test_code": """
assert regex_match("a*b", "b") is True
assert regex_match("a*b", "aaaaab") is True
assert regex_match("a+b", "b") is False
assert regex_match("a+b", "ab") is True
assert regex_match(".*c", "xyzc") is True
assert regex_match("[0-9]+", "4821") is True
assert regex_match("[0-9]+", "abc") is False
""",
        "validation_type": "python_unit_test"
    },
    {
        "id": "code_02_bytecode_vm",
        "description": "Stack-based Bytecode Virtual Machine interpreter with jumps and ALU",
        "prompt": "Write a Python function `execute_vm(bytecode: list[tuple]) -> int` that simulates a stack machine with an instruction pointer. Instructions: `('PUSH', val)`, `('ADD',)`, `('SUB',)` (pops b then a, pushes a - b), `('MUL',)`, `('DUP',)`, `('JMP_IF_ZERO', target_idx)` (pops cond; if cond == 0, pc = target_idx), and `('HALT',)` (returns top of stack). Provide only Python code inside a markdown block.",
        "test_code": """
# 5 * (3 + 4) = 35
prog1 = [('PUSH', 5), ('PUSH', 3), ('PUSH', 4), ('ADD',), ('MUL',), ('HALT',)]
assert execute_vm(prog1) == 35

# 10 - 3 = 7
prog2 = [('PUSH', 10), ('PUSH', 3), ('SUB',), ('HALT',)]
assert execute_vm(prog2) == 7

# Conditional jump test: if cond==0 jump to halt returning 42, else returning 99
# PUSH 0, JMP_IF_ZERO 4, PUSH 99, HALT, PUSH 42, HALT
prog3 = [('PUSH', 0), ('JMP_IF_ZERO', 4), ('PUSH', 99), ('HALT',), ('PUSH', 42), ('HALT',)]
assert execute_vm(prog3) == 42
""",
        "validation_type": "python_unit_test"
    },
    {
        "id": "code_03_lazy_segment_tree",
        "description": "Segment tree with lazy propagation for range updates and range sums",
        "prompt": "Implement a Python class `SegmentTree` for an array of integers with lazy propagation supporting: `__init__(nums: list[int])`, `range_add(l: int, r: int, val: int)`, and `query_sum(l: int, r: int) -> int` returning the sum of elements from index l to r inclusive modulo 1000000007. Provide only Python code inside a markdown block.",
        "test_code": """
st = SegmentTree([1, 2, 3, 4, 5])
assert st.query_sum(0, 4) == 15
assert st.query_sum(1, 3) == 9
st.range_add(1, 3, 10)
# Array becomes [1, 12, 13, 14, 5]
assert st.query_sum(1, 3) == 39
assert st.query_sum(0, 4) == 45
assert st.query_sum(0, 0) == 1
""",
        "validation_type": "python_unit_test"
    },
    {
        "id": "code_04_lisp_interpreter",
        "description": "Mini Lisp AST Evaluator with Lexical Scoping and Closures",
        "prompt": "Write a Python function `evaluate_lisp(expr, env=None)` that evaluates a Lisp AST. Expressions are numbers, strings (variable names), or lists. Supported forms: `['+', a, b]`, `['*', a, b]`, `['-', a, b]`, `['define', name, val]`, `['if', cond, then_b, else_b]`, and `['lambda', params, body]`. User functions can be called as `[fn, arg1, ...]`. Must support lexical closure environment. Provide only Python code inside a markdown block.",
        "test_code": """
env = {}
evaluate_lisp(['define', 'x', 10], env)
assert evaluate_lisp('x', env) == 10
assert evaluate_lisp(['+', 'x', 5], env) == 15
evaluate_lisp(['define', 'square', ['lambda', ['n'], ['*', 'n', 'n']]], env)
assert evaluate_lisp(['square', 6], env) == 36
assert evaluate_lisp(['if', 1, 42, 0], env) == 42
assert evaluate_lisp(['if', 0, 42, 99], env) == 99
""",
        "validation_type": "python_unit_test"
    },
    {
        "id": "code_05_dinic_max_flow",
        "description": "Dinic or Edmonds-Karp maximum network flow algorithm",
        "prompt": "Write a Python function `max_flow(n: int, source: int, sink: int, edges: list[tuple[int, int, int]]) -> int` that computes the maximum network flow from source to sink. `edges` is a list of `(u, v, capacity)`. Provide only Python code inside a markdown block.",
        "test_code": """
# Diamond graph: 0 -> 1 (cap 3), 0 -> 2 (cap 2), 1 -> 3 (cap 2), 2 -> 3 (cap 3), 1 -> 2 (cap 1)
edges = [(0, 1, 3), (0, 2, 2), (1, 3, 2), (2, 3, 3), (1, 2, 1)]
assert max_flow(4, 0, 3, edges) == 5
assert max_flow(3, 0, 2, [(0, 1, 10)]) == 0
""",
        "validation_type": "python_unit_test"
    },
    {
        "id": "code_06_diff_patch_engine",
        "description": "Shortest edit script unified diff generator (LCS-based)",
        "prompt": "Write a Python function `compute_diff(a: list[str], b: list[str]) -> list[str]` that computes the shortest edit script between two lists of string lines. The output list must contain lines prefixed with `'  '` for common lines, `'- '` for deletions from a, and `'+ '` for insertions from b. Provide only Python code inside a markdown block.",
        "test_code": """
a = ["apple", "banana", "cherry"]
b = ["apple", "blueberry", "cherry"]
diff = compute_diff(a, b)
assert "  apple" in diff
assert "- banana" in diff
assert "+ blueberry" in diff
assert "  cherry" in diff
""",
        "validation_type": "python_unit_test"
    },
    {
        "id": "code_07_avl_tree_invariants",
        "description": "Self-balancing AVL tree maintaining strict balance factor <= 1",
        "prompt": "Implement a self-balancing `AVLTree` class in Python with `insert(val: int)` and `is_balanced() -> bool` (checks that for every node, |height(left) - height(right)| <= 1 and BST ordering holds). Provide only Python code inside a markdown block.",
        "test_code": """
tree = AVLTree()
for x in [10, 20, 30, 40, 50, 25]:
    tree.insert(x)
    assert tree.is_balanced() is True
""",
        "validation_type": "python_unit_test"
    },
    {
        "id": "code_08_transactional_key_value",
        "description": "Transactional in-memory KV store with nested rollback and commit frames",
        "prompt": "Implement a Python class `TransactionalKV` providing an in-memory key-value store with nested transactions: `set(k, v)`, `get(k) -> v or None`, `delete(k)`, `begin()`, `commit() -> bool`, and `rollback() -> bool`. `rollback()` reverts changes since the last `begin()`. `commit()` applies changes to parent or root. Returns False if no active transaction. Provide only Python code inside a markdown block.",
        "test_code": """
kv = TransactionalKV()
kv.set("a", 10)
assert kv.get("a") == 10
kv.begin()
kv.set("a", 20)
kv.set("b", 30)
assert kv.get("a") == 20
kv.begin()
kv.delete("a")
assert kv.get("a") is None
assert kv.rollback() is True
assert kv.get("a") == 20
assert kv.commit() is True
assert kv.get("a") == 20
assert kv.get("b") == 30
assert kv.rollback() is False
""",
        "validation_type": "python_unit_test"
    },
    {
        "id": "code_09_expression_calculator_shunting_yard",
        "description": "Math expression parser with right-associative power and unary minus without eval",
        "prompt": "Write a Python function `evaluate_math(expr: str) -> float` that evaluates an arithmetic expression string without using `eval()`. Must support `+`, `-`, `*`, `/`, `^` (exponentiation, right-associative), parentheses `()`, and negative numbers. Return float rounded to 4 decimals. Provide only Python code inside a markdown block.",
        "test_code": """
assert round(evaluate_math("2 ^ 3 ^ 2"), 4) == 512.0
assert round(evaluate_math("-5 + 3 * 4"), 4) == 7.0
assert round(evaluate_math("(2 + 3) * -4"), 4) == -20.0
""",
        "validation_type": "python_unit_test"
    },
    {
        "id": "code_10_interval_tree_overlap",
        "description": "Interval tree data structure for range overlap queries",
        "prompt": "Implement an `IntervalTree` class in Python with `add_interval(start: int, end: int, val: str)` and `query_overlaps(start: int, end: int) -> list[str]` returning sorted list of `val` for all intervals overlapping with `[start, end]`. Provide only Python code inside a markdown block.",
        "test_code": """
it = IntervalTree()
it.add_interval(1, 5, "A")
it.add_interval(6, 10, "B")
it.add_interval(4, 8, "C")
assert sorted(it.query_overlaps(2, 3)) == ["A"]
assert sorted(it.query_overlaps(4, 7)) == ["A", "B", "C"]
assert sorted(it.query_overlaps(11, 15)) == []
""",
        "validation_type": "python_unit_test"
    },
    {
        "id": "code_11_topological_lexical_kahn",
        "description": "Lexicographically smallest topological sort with cycle detection using min-heap",
        "prompt": "Write a Python function `lexical_topo_sort(num_nodes: int, edges: list[tuple[int, int]]) -> list[int]` that returns the lexicographically smallest topological ordering of a directed graph (nodes labeled 0 to num_nodes - 1). Return empty list `[]` if a cycle exists. Provide only Python code inside a markdown block.",
        "test_code": """
# 0 -> 2, 1 -> 2, 0 -> 1. Valid: [0, 1, 2]
assert lexical_topo_sort(3, [(0, 2), (1, 2), (0, 1)]) == [0, 1, 2]
# Independent: 0, 1, 2 -> should return [0, 1, 2]
assert lexical_topo_sort(3, []) == [0, 1, 2]
# Cycle: 0 -> 1 -> 2 -> 0
assert lexical_topo_sort(3, [(0, 1), (1, 2), (2, 0)]) == []
""",
        "validation_type": "python_unit_test"
    },
    {
        "id": "code_12_tarjan_scc",
        "description": "Tarjan's strongly connected components algorithm for directed graphs",
        "prompt": "Write a Python function `find_scc(num_nodes: int, edges: list[tuple[int, int]]) -> list[list[int]]` that finds all Strongly Connected Components of a directed graph. Each component must be sorted, and the outer list must be sorted by the smallest element in each component. Provide only Python code inside a markdown block.",
        "test_code": """
edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
res = find_scc(4, edges)
assert sorted(res) == [[0, 1, 2], [3]]
""",
        "validation_type": "python_unit_test"
    },
    {
        "id": "code_13_knapsack_with_reconstruction",
        "description": "0/1 Knapsack with dynamic programming backtrace path reconstruction",
        "prompt": "Write a Python function `knapsack_reconstruct(weights: list[int], values: list[int], capacity: int) -> tuple[int, list[int]]` that solves 0/1 knapsack and returns `(max_value, chosen_indices)` where `chosen_indices` is the sorted list of 0-indexed item indices selected. Provide only Python code inside a markdown block.",
        "test_code": """
weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
val, chosen = knapsack_reconstruct(weights, values, 5)
assert val == 7 and chosen == [0, 1]
val, chosen = knapsack_reconstruct([10], [100], 5)
assert val == 0 and chosen == []
""",
        "validation_type": "python_unit_test"
    }
]

ds_path = "/home/cune/llama.cpp/benchmarks/comprehensive/dataset.json"
with open(ds_path, "r", encoding="utf-8") as f:
    data = json.load(f)

data["tasks"]["coding"] = HARD_CODING_TASKS

with open(ds_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Successfully updated {len(HARD_CODING_TASKS)} brutal coding tasks in {ds_path}!")
