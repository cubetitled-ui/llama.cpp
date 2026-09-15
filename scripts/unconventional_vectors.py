"""
Unconventional Vectors Exploration Suite:
Testing 5 Non-Obvious Vectors for Latent Reasoning:

1. Vector A: Momentum Neural ODE (Hamiltonian / Heavy-Ball Latent Dynamics)
   - v_{t+1} = mu * v_t + (1 - mu) * F(RMSNorm(h_t + e))
   - h_{t+1} = h_t + dt * v_{t+1}
   - Hypothesis: Prevents limit cycles and dead attractors; smooths convergence like Nesterov acceleration.

2. Vector B: Symplectic Leapfrog / Predictor-Corrector (Alternating Layers A and B)
   - Layer A acts as Position update, Layer B acts as Momentum/Correction update.
   - Hypothesis: Preserves energy/symplectic volume in latent space, prevents explosive/collapsing eigenvalues.

3. Vector C: High/Low Frequency Split (Wavelet / Spectral Residual)
   - Splits h into Low-frequency (global semantic manifold) and High-frequency (local deduction delta).
   - Only High-frequency components undergo non-linear recurrence; Low-frequency is hard-anchored.

4. Vector D: Annealed Langevin Dynamics (Thermodynamic Noise Diffusion)
   - Adds decaying Brownian perturbation: h_{t+1} = ... + sigma_t * N(0, I)
   - Hypothesis: Allows escaping shallow incorrect intuitive basins to reach deeper logical solutions.

5. Vector E: Active Re-Attention / Dynamic Head Gating
   - Simulates attention redistribution across prompt tokens during latent recurrence.
"""

import numpy as np

def rms_norm(x, eps=1e-6):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms

class HardMultiHopEnvironment:
    """
    Complex non-convex reasoning landscape with:
    1. A shallow 'intuitive' trap (spurious attractor - e.g. the common wrong heuristic answer)
    2. A deep 'logical' global minimum (correct multi-step deduction)
    3. Premise anchor containing initial problem setup
    """
    def __init__(self, dim=128, seed=42):
        self.rng = np.random.default_rng(seed)
        self.dim = dim
        
        # Orthonormal basis
        Q, _ = np.linalg.qr(self.rng.normal(size=(dim, dim)))
        self.basis_premise = Q[:, :16]    # Dimensions holding premise facts
        self.basis_heuristic = Q[:, 16:32] # Dimensions of spurious/shallow intuition
        self.basis_logic = Q[:, 32:48]     # Dimensions of true multi-hop deduction
        
        # Ground truth coordinates
        self.true_logic_vec = self.basis_logic @ self.rng.normal(size=(16,))
        self.true_logic_vec /= np.linalg.norm(self.true_logic_vec)
        
        self.premise_vec = self.basis_premise @ self.rng.normal(size=(16,))
        self.premise_vec /= np.linalg.norm(self.premise_vec)
        
        self.heuristic_trap = self.basis_heuristic @ self.rng.normal(size=(16,))
        self.heuristic_trap /= np.linalg.norm(self.heuristic_trap)
        
        # Target representation
        self.target = self.premise_vec + 1.2 * self.true_logic_vec
        self.target /= np.linalg.norm(self.target)
        
        # Initial anchor e (has premise + strong bias toward heuristic trap)
        self.e = self.premise_vec + 0.8 * self.heuristic_trap + 0.1 * self.rng.normal(size=(dim,))
        self.e /= np.linalg.norm(self.e)
        
        # Transformation matrices for Layer A (Attention-like relational mixer)
        # and Layer B (FFN-like non-linear projector)
        self.W_attn = self.basis_logic @ (self.basis_premise.T * 0.4) # Bridges premise to logic
        self.W_ffn1 = self.rng.normal(scale=0.1, size=(dim, dim*2))
        self.W_ffn2 = self.rng.normal(scale=0.1, size=(dim*2, dim))

    def layer_A(self, x):
        """Relational attention-like layer: extracts logic from premises"""
        x_n = rms_norm(x)
        relational_step = self.W_attn @ x_n
        # Shallow trap pull (attention gets distracted by salient heuristic)
        trap_pull = 0.15 * (np.dot(x_n, self.heuristic_trap) * self.heuristic_trap)
        return relational_step + trap_pull

    def layer_B(self, x):
        """FFN-like layer: non-linear projection and constraint refinement"""
        x_n = rms_norm(x)
        # True logic attractor
        logic_proj = 0.3 * (self.basis_logic @ (self.basis_logic.T @ self.true_logic_vec))
        # Non-linear barrier
        nl = 0.1 * np.tanh(x_n @ self.W_ffn1) @ self.W_ffn2
        return logic_proj + nl

    def layer_F(self, x):
        """Combined standard layer (A + B)"""
        return self.layer_A(x) + self.layer_B(x)

    def evaluate(self, state):
        s_n = rms_norm(state)
        # Cosine similarity to true target
        target_sim = np.dot(s_n, self.target)
        # Trap score (how badly are we stuck in the spurious heuristic)
        trap_score = np.dot(s_n, self.heuristic_trap)
        # Premise retention
        premise_score = np.dot(s_n, self.premise_vec)
        return target_sim, trap_score, premise_score


def run_unconventional_vectors_experiment(n_trials=300):
    results = {
        "1. Vanilla T=1": {"target": [], "trap": [], "prem": []},
        "2. OpenMythos LTI (Euler step, T=4)": {"target": [], "trap": [], "prem": []},
        "3. Vector A: Heavy-Ball Momentum ODE (T=4)": {"target": [], "trap": [], "prem": []},
        "4. Vector B: Symplectic Predictor-Corrector (A->B, T=4)": {"target": [], "trap": [], "prem": []},
        "5. Vector C: Spectral Split (Low-freq lock, T=4)": {"target": [], "trap": [], "prem": []},
        "6. Vector D: Annealed Langevin Diffusion (T=4)": {"target": [], "trap": [], "prem": []},
        "7. Vector E: Resonant Momentum Dual-Track (Synthesis)": {"target": [], "trap": [], "prem": []}
    }
    
    for trial in range(n_trials):
        env = HardMultiHopEnvironment(dim=128, seed=trial)
        e = env.e
        
        # 1. Vanilla T=1
        h1 = e + env.layer_F(e)
        t, tr, p = env.evaluate(h1)
        results["1. Vanilla T=1"]["target"].append(t)
        results["1. Vanilla T=1"]["trap"].append(tr)
        results["1. Vanilla T=1"]["prem"].append(p)
        
        # 2. OpenMythos LTI (Euler step, T=4)
        h = e.copy()
        for _ in range(4):
            comb = rms_norm(h + e)
            h = 0.9 * h + 0.1 * e + 0.2 * env.layer_F(comb)
        t, tr, p = env.evaluate(h)
        results["2. OpenMythos LTI (Euler step, T=4)"]["target"].append(t)
        results["2. OpenMythos LTI (Euler step, T=4)"]["trap"].append(tr)
        results["2. OpenMythos LTI (Euler step, T=4)"]["prem"].append(p)
        
        # 3. Vector A: Momentum Neural ODE (Heavy-Ball Acceleration)
        # v_{t+1} = mu * v_t + F(RMSNorm(h_t + e))
        # h_{t+1} = h_t + lr * v_{t+1}
        h = e.copy()
        v = np.zeros_like(h)
        mu = 0.7  # Momentum factor
        lr = 0.2  # Learning rate / step size
        for _ in range(4):
            comb = rms_norm(h + e)
            grad = env.layer_F(comb)
            v = mu * v + (1.0 - mu) * grad
            h = 0.95 * h + 0.05 * e + lr * v
        t, tr, p = env.evaluate(h)
        results["3. Vector A: Heavy-Ball Momentum ODE (T=4)"]["target"].append(t)
        results["3. Vector A: Heavy-Ball Momentum ODE (T=4)"]["trap"].append(tr)
        results["3. Vector A: Heavy-Ball Momentum ODE (T=4)"]["prem"].append(p)
        
        # 4. Vector B: Symplectic Predictor-Corrector (Layer A position, Layer B momentum)
        h = e.copy()
        for _ in range(4):
            # Predictor step using Layer A (Attn)
            comb = rms_norm(h + e)
            h_pred = h + 0.15 * env.layer_A(comb)
            # Corrector step using Layer B (FFN)
            comb_pred = rms_norm(h_pred + e)
            h = 0.9 * h_pred + 0.1 * e + 0.15 * env.layer_B(comb_pred)
        t, tr, p = env.evaluate(h)
        results["4. Vector B: Symplectic Predictor-Corrector (A->B, T=4)"]["target"].append(t)
        results["4. Vector B: Symplectic Predictor-Corrector (A->B, T=4)"]["trap"].append(tr)
        results["4. Vector B: Symplectic Predictor-Corrector (A->B, T=4)"]["prem"].append(p)
        
        # 5. Vector C: Spectral Split (Preserve low-frequency DCT/FFT, recur on high-freq)
        # Low frequency = top components in anchor's eigen/energy spectrum
        h = e.copy()
        # Project e onto top-K singular vectors (here approximated by thresholding)
        threshold = np.percentile(np.abs(e), 75)
        mask_low = np.abs(e) > threshold # Top 25% energy features
        mask_high = ~mask_low
        for _ in range(4):
            comb = rms_norm(h + e)
            update = env.layer_F(comb)
            # Hard lock low frequencies to anchor e, allow high frequencies to recur
            h[mask_low] = e[mask_low]
            h[mask_high] = 0.85 * h[mask_high] + 0.15 * e[mask_high] + 0.25 * update[mask_high]
        t, tr, p = env.evaluate(h)
        results["5. Vector C: Spectral Split (Low-freq lock, T=4)"]["target"].append(t)
        results["5. Vector C: Spectral Split (Low-freq lock, T=4)"]["trap"].append(tr)
        results["5. Vector C: Spectral Split (Low-freq lock, T=4)"]["prem"].append(p)
        
        # 6. Vector D: Annealed Langevin Diffusion
        h = e.copy()
        for step in range(4):
            comb = rms_norm(h + e)
            update = env.layer_F(comb)
            sigma = 0.05 / (1.0 + step) # Decaying noise
            noise = env.rng.normal(scale=sigma, size=env.dim)
            h = 0.9 * h + 0.1 * e + 0.2 * update + noise
        t, tr, p = env.evaluate(h)
        results["6. Vector D: Annealed Langevin Diffusion (T=4)"]["target"].append(t)
        results["6. Vector D: Annealed Langevin Diffusion (T=4)"]["trap"].append(tr)
        results["6. Vector D: Annealed Langevin Diffusion (T=4)"]["prem"].append(p)
        
        # 7. Vector E: Resonant Momentum Dual-Track (Synthesis of Dual-Track + Momentum ODE + Spectral Lock)
        # State = e + delta
        # v_{t+1} = mu * v_t + (1 - mu) * [Mask_high * F(RMSNorm(e + delta_t))]
        # delta_{t+1} = delta_t + dt * v_{t+1}
        delta = np.zeros_like(e)
        v = np.zeros_like(e)
        mu = 0.65
        for _ in range(4):
            comb = rms_norm(e + delta)
            step = env.layer_F(comb)
            # Mask out anchor-critical channels so they never get perturbed
            filtered_step = step.copy()
            filtered_step[mask_low] = 0.0 # 100% immune to premise distortion
            v = mu * v + (1.0 - mu) * filtered_step
            delta = 0.90 * delta + 0.25 * v
        h_res = e + delta
        t, tr, p = env.evaluate(h_res)
        results["7. Vector E: Resonant Momentum Dual-Track (Synthesis)"]["target"].append(t)
        results["7. Vector E: Resonant Momentum Dual-Track (Synthesis)"]["trap"].append(tr)
        results["7. Vector E: Resonant Momentum Dual-Track (Synthesis)"]["prem"].append(p)

    print("="*105)
    print(f"UNCONVENTIONAL VECTORS EXPERIMENTAL REPORT ({n_trials} Monte Carlo Trials)")
    print("="*105)
    print(f"{'Paradigm / Hypothesis':52s} | {'Target Sim (^)':14s} | {'Trap Bias (v)':13s} | {'Premise (^)':13s}")
    print("-"*105)
    for k, v in results.items():
        t_m = np.mean(v["target"])
        tr_m = np.mean(v["trap"])
        p_m = np.mean(v["prem"])
        print(f"{k:52s} | {t_m:+.4f}         | {tr_m:+.4f}        | {p_m:+.4f}")
    print("="*105)

if __name__ == "__main__":
    run_unconventional_vectors_experiment(300)
