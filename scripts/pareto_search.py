"""
Pareto Search: Finding the Exact Hyperplane where Premise Error stays bounded <= 4.0
while Deduction Error beats OpenMythos by > 20%.
"""

import numpy as np
from definitive_falsification import ChallengingLogicTask, rms_norm

def run_pareto_search(n_trials=400):
    candidates = {
        "Vanilla T=1": [],
        "OpenMythos Baseline (T=3)": [],
        "Dual-Track (Scale Guard delta_scale=0.10)": [],
        "Dual-Track (Scale Guard delta_scale=0.05)": [],
        "Dual-Track (Scale Guard delta_scale=0.03)": [],
        "Dual-Track Projector (Anchor Orthogonalization)": []
    }
    
    for trial in range(n_trials):
        task = ChallengingLogicTask(dim=128, n_premises=8, seed=trial)
        e = task.e
        
        # 1. Vanilla
        h1 = e + task.core_forward(e)
        candidates["Vanilla T=1"].append(task.score(h1))
        
        # 2. OpenMythos
        h = e.copy()
        for _ in range(3):
            comb = rms_norm(h + e)
            h = 0.9 * h + 0.1 * e + 0.2 * task.core_forward(comb)
        candidates["OpenMythos Baseline (T=3)"].append(task.score(h))
        
        # 3. Scale Guards
        for scale, name in [(0.10, "Dual-Track (Scale Guard delta_scale=0.10)"),
                            (0.05, "Dual-Track (Scale Guard delta_scale=0.05)"),
                            (0.03, "Dual-Track (Scale Guard delta_scale=0.03)")]:
            delta = np.zeros_like(e)
            for _ in range(3):
                comb = rms_norm(e + delta)
                step = task.core_forward(comb)
                delta = 0.9 * delta + scale * step
            candidates[name].append(task.score(e + delta))
            
        # 4. Orthogonalized Delta:
        # Prevent delta from projecting onto the anchor direction: delta = delta - (delta . e_norm) * e_norm
        # In LTT2/Linear Attention, this is projecting the update onto the nullspace of the anchor!
        e_unit = e / np.linalg.norm(e)
        delta_orth = np.zeros_like(e)
        for _ in range(3):
            comb = rms_norm(e + delta_orth)
            step = task.core_forward(comb)
            delta_orth = 0.9 * delta_orth + 0.10 * step
            # Nullspace projection (Zero Premise Interference!)
            delta_orth = delta_orth - np.dot(delta_orth, e_unit) * e_unit
        candidates["Dual-Track Projector (Anchor Orthogonalization)"].append(task.score(e + delta_orth))

    print("="*95)
    print(f"PARETO FRONTIER SEARCH ({n_trials} Trials)")
    print("="*95)
    print(f"{'Method':55s} | {'Ded Err (v)':11s} | {'Prem Err (v)':12s} | {'Total Cos (^)':13s}")
    print("-"*95)
    for k, v in candidates.items():
        ds = [x[0] for x in v]
        ps = [x[1] for x in v]
        cs = [x[2] for x in v]
        print(f"{k:55s} | {np.mean(ds):.4f}     | {np.mean(ps):.4f}      | {np.mean(cs):+.4f}")
    print("="*95)

if __name__ == "__main__":
    run_pareto_search(400)
