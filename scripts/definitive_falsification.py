"""
Engineering Toy Experiment 3: Definitive Multi-Hypothesis Falsification and Discovery

We test 4 competing paradigms for upgrading OpenMythos on a hybrid SSM/Transformer core:
1. Baseline: OpenMythos (RMSNorm(h+e) + Scalar LTI A=0.9, B=0.1, G=0.2)
2. Hypothesis 1: Pure ACT (Early halting based on Angular Velocity in latent space)
3. Hypothesis 2: LTT2 Diagonal Selective Gating (Channel-dependent decay alpha_c based on anchor energy)
4. Hypothesis 3: Decoupled Dual-Track Delta Formulation (h = e + delta, iterating only on delta)
5. Hypothesis 4 (The Hybrid Synthesis):
   "Selective Dual-Track with Angular Velocity Halting & Subspace Energy Verification"
   - Anchor e is invariant (Dual Track)
   - Perturbation delta is updated with data-dependent diagonal decay (LTT2)
   - Loop halts adaptively when angular velocity stabilizes (ACT)
   - Intermediate state doesn't mutate permanent memory
"""

import numpy as np

def rms_norm(x, eps=1e-6):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms

class ChallengingLogicTask:
    def __init__(self, dim=128, n_premises=8, seed=42):
        self.rng = np.random.default_rng(seed)
        self.dim = dim
        self.n_premises = n_premises
        
        # Orthogonal basis for premise space and deduction space
        Q, _ = np.linalg.qr(self.rng.normal(size=(dim, dim)))
        self.premise_basis = Q[:, :n_premises]          # True facts / premises
        self.deduction_basis = Q[:, n_premises:2*n_premises] # Hidden deductions to solve
        
        # Truth assignment for this problem
        self.premise_weights = self.rng.normal(size=(n_premises,))
        # The true deduction requires a linear-algebraic transformation of premises
        T_matrix = self.rng.normal(scale=0.5, size=(n_premises, n_premises))
        self.deduction_weights = T_matrix @ self.premise_weights
        
        # Ground truth solution state (both premises preserved AND deductions solved)
        self.sol_premise = self.premise_basis @ self.premise_weights
        self.sol_deduction = self.deduction_basis @ self.deduction_weights
        self.ground_truth = self.sol_premise + self.sol_deduction
        self.ground_truth = self.ground_truth / np.linalg.norm(self.ground_truth)
        
        # Anchor e: Contains only the premises + noise (deductions are unsolved, zeroed)
        self.e = self.sol_premise + 0.2 * self.rng.normal(size=(dim,))
        self.e = self.e / np.linalg.norm(self.e)
        
        # Transformer core parameters (simulates one layer of hybrid reasoning)
        # It calculates a step towards deduction_basis from premise_basis, but has finite step size and non-linear noise
        self.W_deduce = self.deduction_basis @ T_matrix @ self.premise_basis.T
        self.W_noise1 = self.rng.normal(scale=0.08, size=(dim, dim))
        self.W_noise2 = self.rng.normal(scale=0.08, size=(dim, dim))

    def core_forward(self, x):
        x_n = rms_norm(x)
        # Deduction advancement step
        step = 0.35 * (self.W_deduce @ x_n)
        # Non-linear transformer noise/saturation
        noise = 0.05 * np.tanh(x_n @ self.W_noise1) @ self.W_noise2
        return step + noise

    def score(self, state):
        s_n = rms_norm(state)
        # 1. Deduction accuracy (cosine similarity with true deductions)
        d_pred = self.deduction_basis.T @ s_n
        d_true = self.deduction_basis.T @ self.ground_truth
        ded_err = np.linalg.norm(d_pred - d_true) / (np.linalg.norm(d_true) + 1e-6)
        
        # 2. Premise corruption (how much did we distort original true premises)
        p_pred = self.premise_basis.T @ s_n
        p_true = self.premise_basis.T @ self.ground_truth
        prem_err = np.linalg.norm(p_pred - p_true) / (np.linalg.norm(p_true) + 1e-6)
        
        # 3. Overall Cosine similarity to complete ground truth
        total_cos = np.dot(s_n, self.ground_truth)
        return ded_err, prem_err, total_cos

def run_falsification_suite():
    n_trials = 500
    
    metrics = {
        "1. Vanilla T=1": {"ded_err": [], "prem_err": [], "cos": []},
        "2. OpenMythos Scalar LTI (T=3)": {"ded_err": [], "prem_err": [], "cos": []},
        "3. OpenMythos Scalar LTI (T=6) [Overshoot]": {"ded_err": [], "prem_err": [], "cos": []},
        "4. Hypothesis 1: Naive ACT (T_max=6)": {"ded_err": [], "prem_err": [], "cos": []},
        "5. Hypothesis 2: LTT2 Selective Decay (T=3)": {"ded_err": [], "prem_err": [], "cos": []},
        "6. Hypothesis 3: Decoupled Dual-Track Delta (T=3)": {"ded_err": [], "prem_err": [], "cos": []},
        "7. Final Synthesis: LTT2 Selective Dual-Track + Angular ACT": {"ded_err": [], "prem_err": [], "cos": []}
    }
    
    synth_steps = []
    
    for trial in range(n_trials):
        task = ChallengingLogicTask(dim=128, n_premises=8, seed=trial)
        e = task.e
        
        # 1. Vanilla T=1
        h1 = e + task.core_forward(e)
        d, p, c = task.score(h1)
        metrics["1. Vanilla T=1"]["ded_err"].append(d)
        metrics["1. Vanilla T=1"]["prem_err"].append(p)
        metrics["1. Vanilla T=1"]["cos"].append(c)
        
        # 2. OpenMythos Scalar LTI (T=3)
        h = e.copy()
        for _ in range(3):
            comb = rms_norm(h + e)
            h = 0.9 * h + 0.1 * e + 0.2 * task.core_forward(comb)
        d, p, c = task.score(h)
        metrics["2. OpenMythos Scalar LTI (T=3)"]["ded_err"].append(d)
        metrics["2. OpenMythos Scalar LTI (T=3)"]["prem_err"].append(p)
        metrics["2. OpenMythos Scalar LTI (T=3)"]["cos"].append(c)
        
        # 3. OpenMythos Scalar LTI (T=6)
        h = e.copy()
        for _ in range(6):
            comb = rms_norm(h + e)
            h = 0.9 * h + 0.1 * e + 0.2 * task.core_forward(comb)
        d, p, c = task.score(h)
        metrics["3. OpenMythos Scalar LTI (T=6) [Overshoot]"]["ded_err"].append(d)
        metrics["3. OpenMythos Scalar LTI (T=6) [Overshoot]"]["prem_err"].append(p)
        metrics["3. OpenMythos Scalar LTI (T=6) [Overshoot]"]["cos"].append(c)
        
        # 4. Hypothesis 1: Naive ACT
        h = e.copy()
        for t in range(6):
            h_prev = h.copy()
            comb = rms_norm(h + e)
            h = 0.9 * h + 0.1 * e + 0.2 * task.core_forward(comb)
            if t >= 1 and np.linalg.norm(h - h_prev)/np.linalg.norm(h_prev) < 0.03:
                break
        d, p, c = task.score(h)
        metrics["4. Hypothesis 1: Naive ACT (T_max=6)"]["ded_err"].append(d)
        metrics["4. Hypothesis 1: Naive ACT (T_max=6)"]["prem_err"].append(p)
        metrics["4. Hypothesis 1: Naive ACT (T_max=6)"]["cos"].append(c)
        
        # 5. Hypothesis 2: LTT2 Selective Decay
        # Channel sensitivity based on anchor energy
        e_norm = rms_norm(e)
        alpha = 0.70 + 0.25 / (1.0 + np.exp(- 2.0 * np.abs(e_norm)))
        beta = 1.0 - alpha
        h = e.copy()
        for _ in range(3):
            comb = rms_norm(h + e)
            h = alpha * h + beta * e + 0.2 * task.core_forward(comb)
        d, p, c = task.score(h)
        metrics["5. Hypothesis 2: LTT2 Selective Decay (T=3)"]["ded_err"].append(d)
        metrics["5. Hypothesis 2: LTT2 Selective Decay (T=3)"]["prem_err"].append(p)
        metrics["5. Hypothesis 2: LTT2 Selective Decay (T=3)"]["cos"].append(c)
        
        # 6. Hypothesis 3: Decoupled Dual-Track Delta
        # State: h = e + delta. Anchor e is 100% immune to scalar decay erosion!
        delta = np.zeros_like(e)
        for _ in range(3):
            comb = rms_norm(e + delta)
            delta = 0.9 * delta + 0.2 * task.core_forward(comb)
        h_track = e + delta
        d, p, c = task.score(h_track)
        metrics["6. Hypothesis 3: Decoupled Dual-Track Delta (T=3)"]["ded_err"].append(d)
        metrics["6. Hypothesis 3: Decoupled Dual-Track Delta (T=3)"]["prem_err"].append(p)
        metrics["6. Hypothesis 3: Decoupled Dual-Track Delta (T=3)"]["cos"].append(c)
        
        # 7. Final Synthesis: LTT2 Selective Dual-Track + Angular ACT
        delta = np.zeros_like(e)
        t_used = 0
        for t in range(6):
            t_used += 1
            delta_prev = delta.copy()
            comb = rms_norm(e + delta)
            delta = alpha * delta + 0.2 * task.core_forward(comb)
            
            # Angular change check:
            if t >= 1:
                cur_full = rms_norm(e + delta)
                prev_full = rms_norm(e + delta_prev)
                angular_cos = np.dot(cur_full, prev_full)
                if angular_cos > 0.997: # Angular velocity saturated
                    break
        synth_steps.append(t_used)
        h_synth = e + delta
        d, p, c = task.score(h_synth)
        metrics["7. Final Synthesis: LTT2 Selective Dual-Track + Angular ACT"]["ded_err"].append(d)
        metrics["7. Final Synthesis: LTT2 Selective Dual-Track + Angular ACT"]["prem_err"].append(p)
        metrics["7. Final Synthesis: LTT2 Selective Dual-Track + Angular ACT"]["cos"].append(c)

    print("="*95)
    print(f"DEFINITIVE FALSIFICATION BENCHMARK: {n_trials} TRIALS")
    print("="*95)
    print(f"{'Method':58s} | {'Ded Err (v)':11s} | {'Prem Err (v)':12s} | {'Total Cos (^)':13s}")
    print("-"*95)
    for k, v in metrics.items():
        d_m = np.mean(v["ded_err"])
        p_m = np.mean(v["prem_err"])
        c_m = np.mean(v["cos"])
        print(f"{k:58s} | {d_m:.4f}     | {p_m:.4f}      | {c_m:+.4f}")
    print("="*95)
    print(f"Average adaptive steps executed in Synthesis: {np.mean(synth_steps):.2f} / 6.00")
    print("="*95)

if __name__ == "__main__":
    run_falsification_suite()
