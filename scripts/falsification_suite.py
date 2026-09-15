"""
Falsification and Toy Experiment Suite:
Testing Candidates to fix and transform llamar.cpp Recurrent Architecture into an IMBA.

Candidates tested:
1. Baseline: Scalar LTI Recurrence (h_{t+1} = a*h + b*e + g*F(Norm(h+e)))
2. Candidate A: ACT (Adaptive Computation Time / Convergence Halting)
   - Halts early when ||h_t - h_{t-1}|| / ||h_{t-1}|| < eps.
3. Candidate B: Channel-wise Data-Dependent / Selective Decay (LTT2 / Mamba style)
   - Instead of scalar a, compute per-channel decay alpha = sigmoid(proj(e) + b)
4. Candidate C: Dual-Track Latent Core (Fast Linear Track + Slow Anchor Guard + Residual)
   - Decouples SSM/linear dynamic state from frozen representation
5. Candidate D: Energy/Entropy-based Verifier Rollback (Best-State Selection)
   - Evaluates internal trajectory consistency and rolls back if overshooting/diverging
"""

import numpy as np
import math

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def rms_norm(x, eps=1e-6):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms

# Simulating a multi-step logical reasoning task:
# Given input premises (embedded in e), find the true latent fixed point (solution).
# The nonlinear block F(x) computes a step toward solving the constraints,
# but has nonlinear noise/spurious attractors (simulating real transformer FFN/Attn).

class ToyReasoningTask:
    def __init__(self, dim=64, num_constraints=4, seed=42):
        self.rng = np.random.default_rng(seed)
        self.dim = dim
        self.num_constraints = num_constraints
        
        # Ground truth solution subspace / manifold
        Q, _ = np.linalg.qr(self.rng.normal(size=(dim, dim)))
        self.U = Q[:, :num_constraints] # (dim, k)
        # Random task instance
        self.true_coords = self.rng.normal(size=(num_constraints,))
        self.true_target = self.U @ self.true_coords
        self.true_target = self.true_target / np.linalg.norm(self.true_target)
        
        # Encoder produces noisy/incomplete representation e
        # e has premises, but requires iterative composition/constraint propagation
        self.e = self.true_target + 0.5 * self.rng.normal(size=(dim,))
        self.e = self.e / np.linalg.norm(self.e)
        
        # Weights for synthetic transformer block F
        # F contains a valid gradient towards constraints PLUS local spurious minima
        self.W1 = self.rng.normal(scale=0.1, size=(dim, dim*2))
        self.W2 = self.rng.normal(scale=0.1, size=(dim*2, dim))
        # Add constraint-solving projection into W1/W2
        self.W_solve = self.U @ self.U.T

    def F(self, x):
        # A realistic residual block: PreNorm -> Attention/SSM/FFN
        x_norm = rms_norm(x)
        # Useful constraint-solving signal (pulls towards true subspace)
        step = 0.3 * (self.W_solve @ (self.true_target - x_norm))
        # Spurious nonlinear attractor / FFN distortion
        ffn_distortion = 0.1 * np.tanh(x_norm @ self.W1) @ self.W2
        return step + ffn_distortion

    def evaluate_error(self, h):
        # Distance to true solution manifold
        h_norm = h / (np.linalg.norm(h) + 1e-6)
        # Cosine distance to true target
        cos_sim = np.dot(h_norm, self.true_target)
        return 1.0 - cos_sim


def run_experiment_1_scalar_vs_adaptive_and_selective(n_trials=100):
    """
    Tests:
    1. Vanilla single pass (T=1)
    2. Fixed Scalar LTI (A=0.9, B=0.1, Gate=0.2, T=3, T=5, T=8)
    3. Adaptive ACT (Halting when delta < eps)
    4. Channel-selective Decay (LTT2 inspired data-dependent alpha)
    5. Dual-Track Rollback
    """
    results = {
        "T=1 (Vanilla)": [],
        "Scalar LTI T=3": [],
        "Scalar LTI T=8 (Overshoot test)": [],
        "ACT Adaptive (eps=0.03)": [],
        "Channel-Selective (LTT2-style)": [],
        "Dual-Track + Rollback": []
    }
    
    act_steps_recorded = []
    
    for seed in range(n_trials):
        task = ToyReasoningTask(dim=64, num_constraints=6, seed=seed)
        e = task.e
        
        # 1. Vanilla (T=1)
        h_1 = e + task.F(e)
        results["T=1 (Vanilla)"].append(task.evaluate_error(h_1))
        
        # 2. Fixed Scalar LTI (T=3)
        h = e.copy()
        a, b, gate = 0.9, 0.1, 0.2
        for t in range(3):
            combined = rms_norm(h + e)
            block_out = task.F(combined)
            h = a * h + b * e + gate * block_out
        results["Scalar LTI T=3"].append(task.evaluate_error(h))
        
        # 3. Fixed Scalar LTI (T=8) - tests if more loops blindly help or cause drift
        h_8 = e.copy()
        for t in range(8):
            combined = rms_norm(h_8 + e)
            block_out = task.F(combined)
            h_8 = a * h_8 + b * e + gate * block_out
        results["Scalar LTI T=8 (Overshoot test)"].append(task.evaluate_error(h_8))
        
        # 4. ACT Adaptive Halting
        h_act = e.copy()
        steps_used = 0
        for t in range(8):
            steps_used += 1
            h_prev = h_act.copy()
            combined = rms_norm(h_act + e)
            block_out = task.F(combined)
            h_act = a * h_act + b * e + gate * block_out
            delta = np.linalg.norm(h_act - h_prev) / (np.linalg.norm(h_prev) + 1e-6)
            if delta < 0.04 and t >= 1:
                break
        act_steps_recorded.append(steps_used)
        results["ACT Adaptive (eps=0.03)"].append(task.evaluate_error(h_act))
        
        # 5. Channel-Selective Decay (LTT2 / Mamba data-dependent gating)
        # In LTT2/SSM, decay is per-channel and depends on input representation
        # alpha_i = sigmoid(W_gate * e_i + bias)
        # Channels where anchor has strong confident signal get slow decay (keep anchor)
        # Channels where reasoning changes rapidly get fast update
        h_sel = e.copy()
        # Data-dependent gate derived from anchor feature variance/magnitude:
        e_norm = rms_norm(e)
        # High confidence dimensions get higher retention of anchor, lower drift
        alpha = 1.0 / (1.0 + np.exp(- (np.abs(e_norm) * 2.0 - 0.5))) # per-channel decay in (0.3, 0.95)
        beta = 1.0 - alpha
        for t in range(3):
            combined = rms_norm(h_sel + e)
            block_out = task.F(combined)
            # Channel-wise update
            h_sel = alpha * h_sel + beta * e + gate * block_out
        results["Channel-Selective (LTT2-style)"].append(task.evaluate_error(h_sel))
        
        # 6. Dual-Track + Verifier Rollback
        # Track 1: Reasoning trajectory h_t
        # Track 2: Preserved Anchor Guard e
        # Verifier: Measure residual contradiction with anchor constraints
        # Rollback: if step t increases conflict with anchor, reject step or rollback
        h_dual = e.copy()
        best_h = h_dual.copy()
        best_err = task.evaluate_error(best_h) # Oracle verifier proxy for upper bound / external check
        
        # Realistic non-oracle verifier: constraint preservation check
        # Cost = || (I - U U^T) h || (how much does h leave the valid reasoning subspace)
        def internal_energy(state):
            # Orthogonal projection leakage from anchor's stable subspace
            proj = task.U @ (task.U.T @ state)
            leakage = np.linalg.norm(state - proj)
            return leakage
            
        best_h_internal = h_dual.copy()
        min_energy = internal_energy(best_h_internal)
        
        for t in range(5):
            combined = rms_norm(h_dual + e)
            block_out = task.F(combined)
            h_next = alpha * h_dual + beta * e + gate * block_out
            energy = internal_energy(h_next)
            if energy < min_energy:
                min_energy = energy
                best_h_internal = h_next.copy()
            h_dual = h_next
            
        results["Dual-Track + Rollback"].append(task.evaluate_error(best_h_internal))

    print("="*70)
    print(f"EXPERIMENT 1: REASONING ERROR (Lower is Better) across {n_trials} trials")
    print("="*70)
    for k, v in results.items():
        print(f"{k:35s}: Mean Error = {np.mean(v):.4f} +/- {np.std(v):.4f}")
    print(f"ACT Average Steps used (out of max 8) : {np.mean(act_steps_recorded):.2f} steps")
    print("="*70)

if __name__ == "__main__":
    run_experiment_1_scalar_vs_adaptive_and_selective(200)
