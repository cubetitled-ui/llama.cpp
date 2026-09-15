"""
recurrent_spec.py
Declarative specification for universal recurrent loop topologies in llama.cpp.
Supports Qwen2, Focal, and generic transformer architectures.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class RecurrentLoopBlock:
    name: str
    layers: List[int]                     # e.g. [13, 14] (contiguous or explicit list)
    iterations: int = 2                   # T: recurrence steps
    recurrent_a: float = 0.90             # LTI state preservation factor (contractive if -1 <= a < 1)
    recurrent_b: float = 0.10             # Anchor input injection factor
    recurrent_gate: float = 1.00          # Delta thought scaling gate (gamma)
    delta_mode: str = "pure_delta"        # "pure_delta" (block_out - combined, isolates G(x)) or "raw_unsubtracted" (buggy double-residual)
    anchor_mode: str = "prelude_hidden"   # "prelude_hidden" (output of prelude), "zero", or "previous_loop"
    norm_eps: float = 1e-6                # RMSNorm epsilon for state combination
    kv_mode: str = "step_overwrite"       # "step_overwrite" (final iteration overwrites token KV slot), "frozen_kv" (only t=0 writes KV), "virtual_depth"
    save_state: bool = True               # State retained between steps
    custom_formulas: List[str] = field(default_factory=list)  # Arbitrary algebraic/tensor formulas compiled by GGMLMathCompiler
    custom_params: Dict[str, float] = field(default_factory=dict) # User-defined scalar hyperparameters (e.g. mu, alpha, beta)
    state_tensors: List[str] = field(default_factory=list)    # Additional state registers (e.g. ["v", "momentum"])
    schedules: Dict[str, str] = field(default_factory=dict)   # Dynamic parameter schedules (e.g. {"a": "cosine(0.95, 0.7)", "gamma": "linear(1.0, 0.2)"})
    anchors: Dict[str, str] = field(default_factory=dict)     # Additional anchor bindings (e.g. {"token_emb": "token_embedding", "skip": "layer_output(6)"})
    early_stop_norm: Optional[float] = None                  # Optional convergence threshold for adaptive early exit
    type: str = "recurrent_loop"

@dataclass
class FeedForwardBlock:
    layers: List[int]                     # e.g. [0, 12] (range) or explicit list
    type: str = "feedforward"

@dataclass
class ArchitectureTopology:
    architecture: str                     # "qwen2", "focal", or "generic"
    total_layers: int
    blocks: List[Any] = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchitectureTopology":
        arch = data.get("architecture", "qwen2").lower()
        total_layers = data.get("total_layers", 28)
        desc = data.get("description", "")
        blocks = []
        for b in data.get("blocks", []):
            b_type = b.get("type", "feedforward")
            if b_type == "feedforward":
                blocks.append(FeedForwardBlock(layers=b.get("layers", [])))
            elif b_type == "recurrent_loop":
                delta_m = b.get("delta_mode", "pure_delta")
                # Normalize legacy naming if present
                if delta_m == "double_residual":
                    delta_m = "pure_delta"
                blocks.append(RecurrentLoopBlock(
                    name=b.get("name", "recurrent_core"),
                    layers=b.get("layers", []),
                    iterations=b.get("iterations", 2),
                    recurrent_a=b.get("recurrent_a", 0.90),
                    recurrent_b=b.get("recurrent_b", 0.10),
                    recurrent_gate=b.get("recurrent_gate", 1.00),
                    delta_mode=delta_m,
                    anchor_mode=b.get("anchor_mode", "prelude_hidden"),
                    norm_eps=b.get("norm_eps", 1e-6),
                    kv_mode=b.get("kv_mode", "step_overwrite"),
                    save_state=b.get("save_state", True),
                    custom_formulas=b.get("custom_formulas", []),
                    custom_params=b.get("custom_params", {}),
                    state_tensors=b.get("state_tensors", []),
                    schedules=b.get("schedules", {}),
                    anchors=b.get("anchors", {}),
                    early_stop_norm=b.get("early_stop_norm", None)
                ))
            else:
                raise ValueError(f"Unknown block type: {b_type}")
        return cls(architecture=arch, total_layers=total_layers, blocks=blocks, description=desc)

    @classmethod
    def from_json(cls, json_str: str) -> "ArchitectureTopology":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: str) -> "ArchitectureTopology":
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".json"):
                return cls.from_json(f.read())
            elif path.endswith(".yaml") or path.endswith(".yml"):
                import yaml
                return cls.from_dict(yaml.safe_load(f))
            else:
                # Try json then python eval
                content = f.read()
                try:
                    return cls.from_json(content)
                except Exception:
                    import yaml
                    return cls.from_dict(yaml.safe_load(content))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> List[str]:
        """Validate topological correctness and contractive bounds."""
        errors = []
        if self.total_layers <= 0:
            errors.append(f"Invalid total_layers: {self.total_layers}")
        
        covered_layers = set()
        for b in self.blocks:
            if isinstance(b, FeedForwardBlock):
                l = b.layers
                if len(l) == 2 and isinstance(l[0], int) and isinstance(l[1], int) and l[0] <= l[1]:
                    layer_set = set(range(l[0], l[1] + 1))
                else:
                    layer_set = set(l)
                covered_layers.update(layer_set)
            elif isinstance(b, RecurrentLoopBlock):
                l = b.layers
                if len(l) == 2 and isinstance(l[0], int) and isinstance(l[1], int) and l[0] <= l[1]:
                    layer_set = set(range(l[0], l[1] + 1))
                else:
                    layer_set = set(l)
                covered_layers.update(layer_set)
                
                # Check contractive stability
                if not (-1.0 <= b.recurrent_a < 1.0):
                    errors.append(f"recurrent_a must satisfy -1 <= a < 1 for contractive stability, got {b.recurrent_a}")
                if b.iterations < 1:
                    errors.append(f"iterations must be >= 1, got {b.iterations}")
        
        # Check layer bounds
        for lyr in covered_layers:
            if lyr < 0 or lyr >= self.total_layers:
                errors.append(f"Layer index {lyr} out of bounds for total_layers={self.total_layers}")
        
        return errors
