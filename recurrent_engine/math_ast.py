"""
math_ast.py
Mathematical AST parser and GGML code generator.
Compiles arbitrary algebraic/tensor formulas into valid C++ ggml graph builder routines.
Allows testing ANY recurrence theory, non-linear gating, momentum, or dual-stream coupling.
"""

import ast
from typing import Dict, Any, Set, Tuple, List

class GGMLMathCompiler:
    def __init__(self, ctx_name: str = "gf.ctx0", default_eps: float = 1e-6):
        self.ctx = ctx_name
        self.default_eps = default_eps
        self.known_tensor_vars: Set[str] = set()
        self.scalar_params: Dict[str, float] = {}
        self.scheduled_vars: Set[str] = set()
        self.lines: List[str] = []

    def set_environment(self, tensor_vars: Set[str], scalar_params: Dict[str, float], scheduled_vars: Set[str] = None):
        self.known_tensor_vars = set(tensor_vars)
        self.scalar_params = dict(scalar_params)
        self.scheduled_vars = set(scheduled_vars or [])
        self.lines = []

    def is_scalar(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float))
        if isinstance(node, ast.Name):
            return (node.id in self.scalar_params) or (node.id in self.scheduled_vars)
        if isinstance(node, ast.UnaryOp):
            return self.is_scalar(node.operand)
        if isinstance(node, ast.BinOp):
            return self.is_scalar(node.left) and self.is_scalar(node.right)
        return False

    def eval_scalar_expr(self, node: ast.AST) -> str:
        """Evaluates or emits C++ float literal for scalar math."""
        if isinstance(node, ast.Constant):
            val = float(node.value)
            return f"{val:.6f}f"
        if isinstance(node, ast.Name):
            # Priority: dynamic scheduled variables ALWAYS take precedence over static scalar defaults
            if node.id in self.scheduled_vars:
                return f"{node.id}"
            if node.id in self.scalar_params:
                val = float(self.scalar_params[node.id])
                return f"{val:.6f}f /* {node.id} */"
            return f"{node.id}"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return f"(-{self.eval_scalar_expr(node.operand)})"
        if isinstance(node, ast.BinOp):
            left_s = self.eval_scalar_expr(node.left)
            right_s = self.eval_scalar_expr(node.right)
            if isinstance(node.op, ast.Pow):
                return f"powf({left_s}, {right_s})"
            op_s = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/"
            }.get(type(node.op), "+")
            return f"({left_s} {op_s} {right_s})"
        return "1.0f"

    def compile_expr(self, node: ast.AST) -> str:
        """Recursively compiles an AST expression into GGML C++ code."""
        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Constant):
            # Scalar constant in tensor context: represented as float
            return f"{float(node.value):.6f}f"

        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else str(node.func)
            args = [self.compile_expr(arg) for arg in node.args]

            # GGML functional primitives
            if func_name in ("rms_norm", "ggml_rms_norm"):
                arg0 = args[0]
                eps_val = f"{self.default_eps}f"
                if len(args) > 1:
                    eps_val = args[1]
                # Check keywords
                for kw in node.keywords:
                    if kw.arg == "eps":
                        eps_val = self.compile_expr(kw.value)
                return f"ggml_rms_norm({self.ctx}, {arg0}, {eps_val})"

            elif func_name in ("norm", "ggml_norm"):
                arg0 = args[0]
                eps_val = f"{self.default_eps}f"
                if len(args) > 1:
                    eps_val = args[1]
                return f"ggml_norm({self.ctx}, {arg0}, {eps_val})"

            elif func_name in ("silu", "tanh", "gelu", "relu", "sigmoid", "swish"):
                f_mapped = "silu" if func_name == "swish" else func_name
                return f"ggml_{f_mapped}({self.ctx}, {args[0]})"

            elif func_name in ("l2_norm", "l2_normalize"):
                return f"ggml_norm({self.ctx}, {args[0]}, {self.default_eps}f)"

            elif func_name in ("lerp", "blend"):
                # lerp(x, y, t) = (1 - t) * x + t * y
                x, y, t = args[0], args[1], args[2]
                node_t = node.args[2]
                if self.is_scalar(node_t):
                    s_t = self.eval_scalar_expr(node_t)
                    return f"ggml_add({self.ctx}, ggml_scale({self.ctx}, {x}, (1.0f - ({s_t}))), ggml_scale({self.ctx}, {y}, {s_t}))"
                else:
                    return f"ggml_add({self.ctx}, ggml_mul({self.ctx}, ggml_scale_bias({self.ctx}, {t}, -1.0f, 1.0f), {x}), ggml_mul({self.ctx}, {t}, {y}))"

            elif func_name in ("gate_highway", "highway"):
                # highway(thought, residual, gate) = gate * thought + (1 - gate) * residual
                th, res, g = args[0], args[1], args[2]
                node_g = node.args[2]
                if self.is_scalar(node_g):
                    s_g = self.eval_scalar_expr(node_g)
                    return f"ggml_add({self.ctx}, ggml_scale({self.ctx}, {th}, {s_g}), ggml_scale({self.ctx}, {res}, (1.0f - ({s_g}))))"
                else:
                    return f"ggml_add({self.ctx}, ggml_mul({self.ctx}, {g}, {th}), ggml_mul({self.ctx}, ggml_scale_bias({self.ctx}, {g}, -1.0f, 1.0f), {res}))"

            elif func_name == "clamp":
                min_val = args[1] if len(args) > 1 else "0.0f"
                max_val = args[2] if len(args) > 2 else "1.0f"
                return f"ggml_clamp({self.ctx}, {args[0]}, {min_val}, {max_val})"

            elif func_name == "scale":
                return f"ggml_scale({self.ctx}, {args[0]}, {args[1]})"

            # State Space Model (Mamba) & Linear Attention (DeltaNet) primitives
            elif func_name == "exp":
                return f"ggml_exp({self.ctx}, {args[0]})"

            elif func_name == "log":
                return f"ggml_log({self.ctx}, {args[0]})"

            elif func_name == "softplus":
                # softplus(x) = log(1 + exp(x))
                return f"ggml_log({self.ctx}, ggml_scale_bias({self.ctx}, ggml_exp({self.ctx}, {args[0]}), 1.0f, 1.0f))"

            elif func_name in ("silu_gate", "gated"):
                # Mamba output gate: x * silu(z)
                return f"ggml_mul({self.ctx}, {args[0]}, ggml_silu({self.ctx}, {args[1]}))"

            elif func_name == "matvec":
                # Matrix-vector multiplication S * v
                return f"ggml_mul_mat({self.ctx}, {args[0]}, {args[1]})"

            elif func_name == "outer_product":
                # Outer product of two vectors (v (x) k^T) produces a 2D matrix
                return f"ggml_out_prod({self.ctx}, {args[0]}, {args[1]})"

            elif func_name in ("view", "slice"):
                dim_raw = args[1].rstrip("f")
                try:
                    dim = str(int(float(dim_raw)))
                except Exception:
                    dim = args[1]
                offset = args[2] if len(args) > 2 else "0"
                return f"ggml_view_1d({self.ctx}, {args[0]}, {dim}, {offset})"

            elif func_name == "repeat":
                return f"ggml_repeat({self.ctx}, {args[0]}, {args[1]})"

            else:
                # Custom / generic call fallback
                return f"ggml_{func_name}({self.ctx}, {', '.join(args)})"

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                inner = self.compile_expr(node.operand)
                return f"ggml_scale({self.ctx}, {inner}, -1.0f)"

        if isinstance(node, ast.BinOp):
            # Check for scalar multiplication: scalar * tensor or tensor * scalar
            if isinstance(node.op, ast.Mult):
                if self.is_scalar(node.left) and not self.is_scalar(node.right):
                    scalar_val = self.eval_scalar_expr(node.left)
                    tensor_val = self.compile_expr(node.right)
                    return f"ggml_scale({self.ctx}, {tensor_val}, {scalar_val})"
                elif self.is_scalar(node.right) and not self.is_scalar(node.left):
                    scalar_val = self.eval_scalar_expr(node.right)
                    tensor_val = self.compile_expr(node.left)
                    return f"ggml_scale({self.ctx}, {tensor_val}, {scalar_val})"
                elif self.is_scalar(node.left) and self.is_scalar(node.right):
                    return self.eval_scalar_expr(node)
                else:
                    # Tensor * Tensor (Hadamard product)
                    left_t = self.compile_expr(node.left)
                    right_t = self.compile_expr(node.right)
                    return f"ggml_mul({self.ctx}, {left_t}, {right_t})"

            elif isinstance(node.op, ast.Div):
                if self.is_scalar(node.right) and not self.is_scalar(node.left):
                    scalar_val = self.eval_scalar_expr(node.right)
                    tensor_val = self.compile_expr(node.left)
                    return f"ggml_scale({self.ctx}, {tensor_val}, 1.0f / ({scalar_val}))"
                else:
                    left_t = self.compile_expr(node.left)
                    right_t = self.compile_expr(node.right)
                    return f"ggml_div({self.ctx}, {left_t}, {right_t})"

            elif isinstance(node.op, ast.Add):
                if self.is_scalar(node.left) and not self.is_scalar(node.right):
                    scalar_val = self.eval_scalar_expr(node.left)
                    tensor_val = self.compile_expr(node.right)
                    return f"ggml_scale_bias({self.ctx}, {tensor_val}, 1.0f, {scalar_val})"
                elif self.is_scalar(node.right) and not self.is_scalar(node.left):
                    scalar_val = self.eval_scalar_expr(node.right)
                    tensor_val = self.compile_expr(node.left)
                    return f"ggml_scale_bias({self.ctx}, {tensor_val}, 1.0f, {scalar_val})"
                else:
                    left_t = self.compile_expr(node.left)
                    right_t = self.compile_expr(node.right)
                    return f"ggml_add({self.ctx}, {left_t}, {right_t})"

            elif isinstance(node.op, ast.Sub):
                if self.is_scalar(node.left) and not self.is_scalar(node.right):
                    scalar_val = self.eval_scalar_expr(node.left)
                    tensor_val = self.compile_expr(node.right)
                    return f"ggml_scale_bias({self.ctx}, {tensor_val}, -1.0f, {scalar_val})"
                elif self.is_scalar(node.right) and not self.is_scalar(node.left):
                    scalar_val = self.eval_scalar_expr(node.right)
                    tensor_val = self.compile_expr(node.left)
                    return f"ggml_scale_bias({self.ctx}, {tensor_val}, 1.0f, -({scalar_val}))"
                else:
                    left_t = self.compile_expr(node.left)
                    right_t = self.compile_expr(node.right)
                    return f"ggml_sub({self.ctx}, {left_t}, {right_t})"

        raise NotImplementedError(f"Unsupported AST node: {ast.dump(node)}")

    def compile_statement(self, stmt_str: str) -> List[str]:
        """Parses a single assignment or expression statement and returns C++ lines."""
        tree = ast.parse(stmt_str.strip())
        out_lines = []
        for item in tree.body:
            if isinstance(item, ast.Assign):
                target = item.targets[0].id
                cpp_expr = self.compile_expr(item.value)
                if target not in self.known_tensor_vars:
                    self.known_tensor_vars.add(target)
                    out_lines.append(f"    ggml_tensor * {target} = {cpp_expr};")
                else:
                    out_lines.append(f"    {target} = {cpp_expr};")
            elif isinstance(item, ast.Expr):
                cpp_expr = self.compile_expr(item.value)
                out_lines.append(f"    {cpp_expr};")
        return out_lines

    def compile_block(self, formulas: List[str]) -> str:
        """Compiles a sequence of mathematical formulas into C++ block."""
        all_lines = []
        for f in formulas:
            f = f.strip()
            if not f or f.startswith("#"):
                continue
            all_lines.extend(self.compile_statement(f))
        return "\n".join(all_lines)
