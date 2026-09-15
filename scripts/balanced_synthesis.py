"""
Balanced Synthesis Benchmark:
Testing Anchor Protection vs Reasoning Gain to achieve the ultimate Pareto-optimal state:
Maximum Deduction Power WITH Zero Premise Corruption.
"""

import numpy as np
from definitive_falsification import ChallengingLogicTask, rms_norm

def run_balanced_benchmark(n_trials=500):
    task_scores = {
        "Vanilla T=1": [],
        "OpenMythos (A=0.9, B=0.1, T=3)": [],
        "Dual-Track (Delta + Anchor Injection)": [],
        "Pareto-Optimal LTT2 Hybrid (Selective Delta + Anchor Lock)": []
    }
    
    for trial in range(n_trials):
        task = ChallengingLogicTask(dim=128, n_premises=8, seed=trial)
        e = task.e
        
        # 1. Vanilla
        h1 = e + task.core_forward(e)
        d, p, c = task.score(h1)
        task_scores["Vanilla T=1"].append((d, p, c))
        
        # 2. OpenMythos Baseline
        h = e.copy()
        for _ in range(3):
            comb = rms_norm(h + e)
            h = 0.9 * h + 0.1 * e + 0.2 * task.core_forward(comb)
        d, p, c = task.score(h)
        task_scores["OpenMythos (A=0.9, B=0.1, T=3)"].append((d, p, c))
        
        # 3. Dual Track Delta
        delta = np.zeros_like(e)
        for _ in range(3):
            comb = rms_norm(e + delta)
            delta = 0.85 * delta + 0.15 * task.core_forward(comb)
        h = e + delta
        d, p, c = task.score(h)
        task_scores["Dual-Track (Delta + Anchor Injection)"].append((d, p, c))
        
        # 4. Pareto-Optimal LTT2 Hybrid
        # Anchor Lock: High energy channels in anchor are protected from perturbation
        e_norm = rms_norm(e)
        channel_importance = np.abs(e_norm)
        # Gating: if anchor strongly activates channel, limit delta growth in that channel
        anchor_guard = 1.0 / (1.0 + np.exp(2.5 * (channel_importance - np.mean(channel_importance))))
        
        delta = np.zeros_like(e)
        for t in range(3):
            comb = rms_norm(e + delta)
            step = task.core_forward(comb)
            # Channel-selective delta update with anchor guard (LTT2 Selective State Space)
            delta = 0.85 * delta + 0.20 * (anchor_guard * step)
            
        h_pareto = e + delta
        d, p, c = task.score(h_pareto)
        task_scores["Pareto-Optimal LTT2 Hybrid (Selective Delta + Anchor Lock)"].append((d, p, c))

    print("="*95)
    print(f"PARETO FRONTIER ANALYSIS ({n_trials} Trials)")
    print("="*95)
    print(f"{'Method':55s} | {'Ded Err (v)':11s} | {'Prem Err (v)':12s} | {'Total Cos (^)':13s}")
    print("-"*95)
    for k, v in task_scores.items():
        ds = [x[0] for x in v]
        ps = [x[1] for x in v]
        cs = [x[2] for x in v]
        print(f"{k:55s} | {np.mean(ds):.4f}     | {np.mean(ps):.4f}      | {np.mean(cs):+.4f}")
    print("="*95)

if __name__ == "__main__":
    run_balanced_benchmark(500)
