"""
rlang/compiler.py
Universal Compiler for Recurrent Language (RLang).
Compiles high-level recurrent language specifications into:
1. ArchitectureTopology objects (for declarative validation).
2. Optimized C++ GGML computational graphs.
3. Live patches for llama.cpp's `src/llama-graph.cpp`.
"""

import os
import re
import math
import shutil
from typing import Dict, Any, List, Optional
from recurrent_engine.rlang.lexer import Lexer
from recurrent_engine.rlang.parser import Parser, Program, ModelDecl, FeedforwardStage, LoopStage, Range
from recurrent_engine.recurrent_spec import ArchitectureTopology, RecurrentLoopBlock, FeedForwardBlock
from recurrent_engine.math_ast import GGMLMathCompiler

MARKER_BEGIN = "// >>> RLANG_RECURRENT_CORE_BEGIN"
MARKER_END   = "// <<< RLANG_RECURRENT_CORE_END"

class RLangCompiler:
    def __init__(self, llama_cpp_root: str = "/home/cune/llama.cpp"):
        self.root = os.path.abspath(llama_cpp_root)
        self.graph_cpp_path = os.path.join(self.root, "src/llama-graph.cpp")
        self.graph_backup_path = os.path.join(self.root, "src/llama-graph.cpp.orig")

    def compile_source(self, source: str) -> Program:
        """Lexes and parses RLang source code into an AST Program."""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        return parser.parse()

    def compile_file(self, filepath: str) -> Program:
        """Reads and parses an .rlang file into an AST Program."""
        with open(filepath, "r", encoding="utf-8") as f:
            return self.compile_source(f.read())

    def program_to_topology(self, prog: Program) -> ArchitectureTopology:
        """Converts an RLang AST Program into an ArchitectureTopology specification."""
        arch = prog.model.architecture if prog.model else "qwen2"
        total_layers = prog.model.total_layers if prog.model else 28
        desc = prog.model.description if prog.model else ""
        blocks = []

        if prog.pipeline:
            from recurrent_engine.rlang.parser import RunStmt, RepeatStmt
            run_layers = []
            for stmt in prog.pipeline.stmts:
                if isinstance(stmt, RunStmt):
                    run_layers.extend([stmt.layers.start, stmt.layers.end])
                elif isinstance(stmt, RepeatStmt):
                    for sub in stmt.stmts:
                        if isinstance(sub, RunStmt):
                            run_layers.extend([sub.layers.start, sub.layers.end])
            max_l = max(run_layers) + 1 if run_layers else total_layers
            blocks.append(FeedForwardBlock(layers=[0, max_l - 1]))
            return ArchitectureTopology(
                architecture=arch,
                total_layers=max(total_layers, max_l),
                blocks=blocks,
                description="Universal Layer Execution Pipeline"
            )

        for stage in prog.stages:
            if isinstance(stage, FeedforwardStage):
                blocks.append(FeedForwardBlock(layers=[stage.layers.start, stage.layers.end]))
            elif isinstance(stage, LoopStage):
                a_val = stage.params.get("a", stage.params.get("recurrent_a", 0.90))
                b_val = stage.params.get("b", stage.params.get("recurrent_b", 0.10))
                g_val = stage.params.get("gamma", stage.params.get("recurrent_gate", 1.00))

                blocks.append(RecurrentLoopBlock(
                    name=stage.name,
                    layers=[stage.layers.start, stage.layers.end],
                    iterations=stage.iterations,
                    recurrent_a=a_val,
                    recurrent_b=b_val,
                    recurrent_gate=g_val,
                    delta_mode=stage.delta_mode,
                    norm_eps=prog.model.epsilon if prog.model else 1e-6,
                    kv_mode=stage.kv_cache,
                    custom_formulas=stage.formulas,
                    custom_params=stage.params,
                    state_tensors=list(stage.registers.keys()),
                    schedules=stage.schedules,
                    anchors=stage.anchors,
                ))

        return ArchitectureTopology(
            architecture=arch,
            total_layers=total_layers,
            blocks=blocks,
            description=desc
        )

    def _evaluate_schedule(self, sched_str: str, T: int) -> List[float]:
        """Precomputes dynamic parameter schedules."""
        s = sched_str.strip()
        if s.startswith("linear"):
            args = [float(x.strip()) for x in s[s.find("(")+1:s.rfind(")")].split(",")]
            start, end = args[0], args[1]
            return [start + t * (end - start) / max(1, T - 1) for t in range(T)]
        elif s.startswith("cosine"):
            args = [float(x.strip()) for x in s[s.find("(")+1:s.rfind(")")].split(",")]
            start, end = args[0], args[1]
            return [end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t / max(1, T - 1))) for t in range(T)]
        elif s.startswith("exponential") or s.startswith("exp"):
            args = [float(x.strip()) for x in s[s.find("(")+1:s.rfind(")")].split(",")]
            start, decay = args[0], args[1]
            return [start * (decay ** t) for t in range(T)]
        return [float(s)] * T

    def generate_cpp_core(self, prog: Program) -> str:
        """
        Generates the complete C++ build_recurrent_core implementation
        from an RLang AST Program.
        """
        eps = prog.model.epsilon if prog.model else 1e-6
        arch = prog.model.architecture if prog.model else "generic"

        lines = []
        lines.append(f"// Compiled from RLang specification ({arch})")
        lines.append("ggml_tensor * build_recurrent_core(")
        lines.append("        llm_graph_context & gf,")
        lines.append("        ggml_tensor * inpL,")
        lines.append("        const std::function<ggml_tensor * (int il, ggml_tensor * input)> & decoder,")
        lines.append("        const std::function<void (int il, ggml_tensor * input)> & on_entry) {")

        # Runtime bypass check: if caller explicitly set recurrent_t == 1 via CLI flags, bypass recurrence
        lines.append("    if (gf.cparams.recurrent_t == 1) {")
        lines.append("        for (int il = 0; il < gf.n_layer; ++il) {")
        lines.append("            if (on_entry) on_entry(il, inpL);")
        lines.append("            inpL = decoder(il, inpL);")
        lines.append("        }")
        lines.append("        return inpL;")
        lines.append("    }")
        lines.append("")

        if prog.pipeline:
            from recurrent_engine.rlang.parser import RunStmt, AssignStmt, RepeatStmt
            math_comp = GGMLMathCompiler(ctx_name="gf.ctx0", default_eps=eps)
            declared_vars = set(["inpL"])
            last_var = "inpL"

            def emit_stmts(stmts, indent="    "):
                nonlocal last_var
                sub_lines = []
                for stmt in stmts:
                    if isinstance(stmt, RunStmt):
                        dest = stmt.dest
                        lo, hi = stmt.layers.start, stmt.layers.end
                        in_var = stmt.input_var
                        decl = "" if dest in declared_vars else "ggml_tensor * "
                        declared_vars.add(dest)
                        last_var = dest
                        sub_lines.append(f"{indent}// {dest} = run({lo}..{hi}, {in_var})")
                        sub_lines.append(f"{indent}{decl}{dest};")
                        sub_lines.append(f"{indent}{{")
                        sub_lines.append(f"{indent}    ggml_tensor * cur = {in_var};")
                        sub_lines.append(f"{indent}    for (int il = {lo}; il <= {hi}; ++il) {{")
                        sub_lines.append(f"{indent}        if (on_entry) on_entry(il, cur);")
                        sub_lines.append(f"{indent}        cur = decoder(il, cur);")
                        sub_lines.append(f"{indent}    }}")
                        sub_lines.append(f"{indent}    {dest} = cur;")
                        sub_lines.append(f"{indent}}}")
                    elif isinstance(stmt, AssignStmt):
                        dest = stmt.dest
                        expr = stmt.expr
                        decl = "" if dest in declared_vars else "ggml_tensor * "
                        declared_vars.add(dest)
                        last_var = dest
                        math_comp.set_environment(tensor_vars=declared_vars, scalar_params={})
                        compiled_expr = math_comp.compile_block([f"{dest} = {expr}"])
                        sub_lines.append(f"{indent}// {dest} = {expr}")
                        for cl in compiled_expr.split("\n"):
                            if cl.strip():
                                if decl and cl.strip().startswith(f"{dest} ="):
                                    cl = decl + cl.strip()
                                sub_lines.append(f"{indent}{cl}")
                    elif isinstance(stmt, RepeatStmt):
                        sub_lines.append(f"{indent}for (int rep = 0; rep < {stmt.count}; ++rep) {{")
                        sub_lines.extend(emit_stmts(stmt.stmts, indent + "    "))
                        sub_lines.append(f"{indent}}}")
                return sub_lines

            lines.extend(emit_stmts(prog.pipeline.stmts))
            lines.append(f"    return {last_var};")
            lines.append("}")
            return "\n".join(lines)

        loop_idx = 0
        for stage in prog.stages:
            if isinstance(stage, FeedforwardStage):
                lo, hi = stage.layers.start, stage.layers.end
                lines.append(f"    // Stage: {stage.name} (Feedforward [{lo}..{hi}])")
                lines.append(f"    for (int il = {lo}; il <= {hi}; ++il) {{")
                lines.append("        if (on_entry) on_entry(il, inpL);")
                lines.append("        inpL = decoder(il, inpL);")
                lines.append("    }")
                lines.append("")

            elif isinstance(stage, LoopStage):
                loop_idx += 1
                lo, hi = stage.layers.start, stage.layers.end
                T = stage.iterations
                a_val = stage.params.get("a", stage.params.get("recurrent_a", 0.90))
                b_val = stage.params.get("b", stage.params.get("recurrent_b", 0.10))
                g_val = stage.params.get("gamma", stage.params.get("recurrent_gate", 1.00))

                lines.append(f"    // Stage: {stage.name} (Recurrent Loop [{lo}..{hi}], T={T})")
                lines.append("    {")
                lines.append("        ggml_tensor * anchor = inpL;")
                lines.append("        ggml_tensor * h = inpL;")
                if getattr(stage, "merge", "last") != "last":
                    lines.append(f"        ggml_tensor * pass_history[{T}];")

                # Dynamic schedules arrays
                for p_name, p_sched in stage.schedules.items():
                    vals = self._evaluate_schedule(p_sched, T)
                    c_vals = ", ".join(f"{v:.6f}f" for v in vals)
                    lines.append(f"        const float {p_name}_sched[{T}] = {{ {c_vals} }}; // {p_sched}")

                # Auxiliary state registers (supports 2D matrices, 1D states/buffers, and zero vectors)
                for r_name, r_init in stage.registers.items():
                    r_clean = r_init.replace(" ", "").lower()
                    if r_clean.startswith("matrix(") and r_clean.endswith(")"):
                        dims = [int(x.strip()) for x in r_clean[7:-1].split(",")]
                        r1, c1 = dims[0], dims[1] if len(dims) > 1 else dims[0]
                        lines.append(f"        ggml_tensor * {r_name} = ggml_new_tensor_2d(gf.ctx0, GGML_TYPE_F32, {r1}, {c1}); ggml_set_zero({r_name}); // 2D matrix: {r1}x{c1}")
                    elif (r_clean.startswith("state(") or r_clean.startswith("buffer(")) and r_clean.endswith(")"):
                        dim = int(r_clean[r_clean.find("(")+1:-1].strip())
                        lines.append(f"        ggml_tensor * {r_name} = ggml_new_tensor_1d(gf.ctx0, GGML_TYPE_F32, {dim}); ggml_set_zero({r_name}); // 1D state: dim={dim}")
                    elif r_clean in ("0.0", "0", "zero", "0.0f"):
                        lines.append(f"        ggml_tensor * {r_name} = ggml_scale(gf.ctx0, inpL, 0.0f); // zero-initialized register: {r_name}")
                    elif r_clean in ("anchor", "inpl"):
                        lines.append(f"        ggml_tensor * {r_name} = inpL; // register: {r_name}")
                    else:
                        lines.append(f"        ggml_tensor * {r_name} = ggml_scale(gf.ctx0, inpL, 0.0f); // register: {r_name}")

                if stage.system == "deltanet" and "S" not in stage.registers:
                    lines.append("        ggml_tensor * S = ggml_new_tensor_2d(gf.ctx0, GGML_TYPE_F32, 64, 64); ggml_set_zero(S); // DeltaNet Associative Memory Matrix: 64x64")
                elif stage.system == "mamba" and "h_ssm" not in stage.registers:
                    lines.append("        ggml_tensor * h_ssm = ggml_new_tensor_1d(gf.ctx0, GGML_TYPE_F32, 64); ggml_set_zero(h_ssm); // Mamba SSM State Vector: 64")

                lines.append(f"        for (int t = 0; t < {T}; ++t) {{")
                # Bind scheduled parameters
                for p_name in stage.schedules.keys():
                    lines.append(f"            const float {p_name} = {p_name}_sched[t];")

                lines.append(f"            ggml_tensor * combined = ggml_rms_norm(gf.ctx0,")
                lines.append(f"                    ggml_add(gf.ctx0, h, anchor), {eps:.6e}f);")
                lines.append("            ggml_tensor * cur = combined;")
                lines.append(f"            for (int il = {lo}; il <= {hi}; ++il) {{")
                lines.append("                if (on_entry) on_entry(il, cur);")
                lines.append("                cur = decoder(il, cur);")
                lines.append("            }")
                lines.append("            ggml_tensor * block_out = cur;")
                lines.append("            GGML_UNUSED(block_out);")

                if stage.formulas:
                    math_comp = GGMLMathCompiler(ctx_name="gf.ctx0", default_eps=eps)
                    env_tensors = {"h", "anchor", "combined", "block_out"}.union(stage.registers.keys())
                    scalar_params = {
                        "a": a_val,
                        "b": b_val,
                        "gamma": g_val,
                        "recurrent_a": a_val,
                        "recurrent_b": b_val,
                        "recurrent_gate": g_val,
                    }
                    scalar_params.update(stage.params)
                    for sp in stage.schedules.keys():
                        scalar_params.pop(sp, None)
                    math_comp.set_environment(
                        tensor_vars=env_tensors,
                        scalar_params=scalar_params,
                        scheduled_vars=set(stage.schedules.keys())
                    )
                    compiled_math = math_comp.compile_block(stage.formulas)
                    lines.append("            // Custom Theory Formulas:")
                    for c_line in compiled_math.split("\n"):
                        lines.append("            " + c_line)
                elif stage.system == "mamba":
                    lines.append("            // True Continuous-to-Discrete Mamba Selective State Space Core")
                    lines.append("            const int64_t d_state = 64;")
                    lines.append("            ggml_tensor * x_in = ggml_view_1d(gf.ctx0, cur, d_state, 0);")
                    lines.append("            ggml_tensor * delta_ssm = ggml_log(gf.ctx0, ggml_scale_bias(gf.ctx0, ggml_exp(gf.ctx0, x_in), 1.0f, 1.0f)); // softplus(x)")
                    lines.append("            ggml_tensor * A_bar = ggml_exp(gf.ctx0, ggml_scale(gf.ctx0, delta_ssm, -0.500000f)); // exp(delta * A_diag)")
                    lines.append("            ggml_tensor * B_bar = ggml_scale(gf.ctx0, delta_ssm, 0.100000f); // delta * B")
                    lines.append("            h_ssm = ggml_add(gf.ctx0, ggml_mul(gf.ctx0, A_bar, h_ssm), ggml_mul(gf.ctx0, B_bar, x_in)); // h = A_bar * h + B_bar * x")
                    lines.append("            ggml_tensor * y_ssm = ggml_scale(gf.ctx0, h_ssm, 0.250000f); // C * h")
                    lines.append("            ggml_tensor * y_full = ggml_repeat(gf.ctx0, y_ssm, inpL);")
                    lines.append("            ggml_tensor * y_gated = ggml_mul(gf.ctx0, y_full, ggml_silu(gf.ctx0, cur)); // Multiplicative SiLU Gating")
                    lines.append(f"            h = ggml_add(gf.ctx0, ggml_add(gf.ctx0, ggml_scale(gf.ctx0, h, {a_val:.6f}f), ggml_scale(gf.ctx0, anchor, {b_val:.6f}f)), ggml_scale(gf.ctx0, y_gated, {g_val:.6f}f));")
                elif stage.system == "deltanet":
                    lines.append("            // True DeltaNet Linear Attention with Error-Correcting Delta Rule")
                    lines.append("            const int64_t d_k = 64;")
                    lines.append("            ggml_tensor * q = ggml_view_1d(gf.ctx0, cur, d_k, 0);")
                    lines.append("            ggml_tensor * k = ggml_norm(gf.ctx0, ggml_view_1d(gf.ctx0, combined, d_k, 0), 1e-6f);")
                    lines.append("            ggml_tensor * v = ggml_view_1d(gf.ctx0, block_out, d_k, 0);")
                    lines.append("            // 1. Associative Retrieval: v_pred = S * k")
                    lines.append("            ggml_tensor * v_pred = ggml_mul_mat(gf.ctx0, S, k);")
                    lines.append("            // 2. Innovation Error: v_err = v - v_pred")
                    lines.append("            ggml_tensor * v_err = ggml_sub(gf.ctx0, v, v_pred);")
                    lines.append("            // 3. Associative Memory Matrix Write: S = alpha * S + beta * (v_err (x) k^T)")
                    lines.append(f"            ggml_tensor * delta_S = ggml_scale(gf.ctx0, ggml_out_prod(gf.ctx0, v_err, k), 0.250000f);")
                    lines.append(f"            S = ggml_add(gf.ctx0, ggml_scale(gf.ctx0, S, 0.950000f), delta_S);")
                    lines.append("            // 4. Associative Readout: y_readout = S * q")
                    lines.append("            ggml_tensor * y_readout = ggml_mul_mat(gf.ctx0, S, q);")
                    lines.append("            // 5. Broadcast / Repeat across hidden dimensions and fuse")
                    lines.append("            ggml_tensor * y_full = ggml_repeat(gf.ctx0, y_readout, inpL);")
                    lines.append("            ggml_tensor * gate = ggml_sigmoid(gf.ctx0, ggml_scale(gf.ctx0, y_full, 1.500000f));")
                    lines.append(f"            h = ggml_add(gf.ctx0, ggml_add(gf.ctx0, ggml_scale(gf.ctx0, h, {a_val:.6f}f), ggml_scale(gf.ctx0, anchor, {b_val:.6f}f)), ggml_scale(gf.ctx0, ggml_mul(gf.ctx0, gate, y_full), {g_val:.6f}f));")
                else:
                    if stage.delta_mode in ("pure_delta", "double_residual"):
                        lines.append("            ggml_tensor * delta_thought = ggml_sub(gf.ctx0, block_out, combined);")
                    else:
                        lines.append("            ggml_tensor * delta_thought = block_out;")
                    lines.append(f"            h = ggml_add(gf.ctx0,")
                    lines.append(f"                    ggml_add(gf.ctx0, ggml_scale(gf.ctx0, h, {a_val:.6f}f), ggml_scale(gf.ctx0, anchor, {b_val:.6f}f)),")
                    lines.append(f"                    ggml_scale(gf.ctx0, delta_thought, {g_val:.6f}f));")

                if getattr(stage, "merge", "last") != "last":
                    lines.append("            pass_history[t] = h;")

                lines.append("        }")

                merge_strategy = getattr(stage, "merge", "last")
                if merge_strategy == "weighted_sum":
                    weights = stage.merge_weights if len(stage.merge_weights) == T else [(i+1)/sum(range(1, T+1)) for i in range(T)]
                    lines.append("        // Multi-Pass Historical Thought Merging (weighted_sum):")
                    lines.append(f"        ggml_tensor * merged = ggml_scale(gf.ctx0, pass_history[0], {weights[0]:.6f}f);")
                    for i in range(1, T):
                        lines.append(f"        merged = ggml_add(gf.ctx0, merged, ggml_scale(gf.ctx0, pass_history[{i}], {weights[i]:.6f}f));")
                    lines.append("        h = merged;")
                elif merge_strategy == "average":
                    inv_T = 1.0 / T
                    lines.append("        // Multi-Pass Historical Thought Merging (average):")
                    lines.append(f"        ggml_tensor * merged = ggml_scale(gf.ctx0, pass_history[0], {inv_T:.6f}f);")
                    for i in range(1, T):
                        lines.append(f"        merged = ggml_add(gf.ctx0, merged, ggml_scale(gf.ctx0, pass_history[{i}], {inv_T:.6f}f));")
                    lines.append("        h = merged;")
                elif merge_strategy == "highway":
                    lines.append("        // Multi-Pass Historical Thought Merging (residual highway):")
                    lines.append("        ggml_tensor * merged = pass_history[0];")
                    for i in range(1, T):
                        lines.append(f"        merged = ggml_add(gf.ctx0, merged, ggml_scale(gf.ctx0, pass_history[{i}], 0.500000f));")
                    lines.append("        h = merged;")
                elif merge_strategy == "gated":
                    lines.append("        // Multi-Pass Historical Thought Merging (gated first vs last):")
                    lines.append(f"        ggml_tensor * merged = ggml_add(gf.ctx0, ggml_scale(gf.ctx0, pass_history[0], 0.300000f), ggml_scale(gf.ctx0, pass_history[{T-1}], 0.700000f));")
                    lines.append("        h = merged;")

                lines.append("        inpL = h;")
                lines.append("    }")
                lines.append("")
                lines.append("")

        lines.append("    return inpL;")
        lines.append("}")
        return "\n".join(lines)

    def patch_graph_cpp(self, cpp_code: str) -> bool:
        """
        Injects the generated C++ core into src/llama-graph.cpp with safety backup.
        """
        if not os.path.exists(self.graph_cpp_path):
            raise FileNotFoundError(f"Cannot find {self.graph_cpp_path}")

        # Ensure backup exists
        if not os.path.exists(self.graph_backup_path):
            shutil.copyfile(self.graph_cpp_path, self.graph_backup_path)

        with open(self.graph_cpp_path, "r", encoding="utf-8") as f:
            content = f.read()

        marked_block = f"{MARKER_BEGIN}\n{cpp_code}\n{MARKER_END}"

        if MARKER_BEGIN in content and MARKER_END in content:
            # Replace existing marked section
            pattern = re.compile(rf"{re.escape(MARKER_BEGIN)}.*?{re.escape(MARKER_END)}", re.DOTALL)
            new_content = pattern.sub(marked_block, content)
        else:
            # First time injection: locate build_recurrent_core
            fn_start = content.find("ggml_tensor * build_recurrent_core(")
            if fn_start == -1:
                raise ValueError("Could not find build_recurrent_core in src/llama-graph.cpp")
            
            brace_open = content.find("{", fn_start)
            if brace_open == -1:
                raise ValueError("Could not find opening brace for build_recurrent_core")
            depth = 0
            fn_end = -1
            for i in range(brace_open, len(content)):
                if content[i] == '{':
                    depth += 1
                elif content[i] == '}':
                    depth -= 1
                    if depth == 0:
                        fn_end = i + 1
                        break
            if fn_end == -1:
                raise ValueError("Could not find closing brace of build_recurrent_core in src/llama-graph.cpp")
            new_content = content[:fn_start] + marked_block + content[fn_end:]

        with open(self.graph_cpp_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return True

    def restore_graph_cpp(self) -> bool:
        """Restores src/llama-graph.cpp from original backup."""
        if os.path.exists(self.graph_backup_path):
            shutil.copyfile(self.graph_backup_path, self.graph_cpp_path)
            return True
        return False
