"""
Deep Falsification Suite 2:
Testing why ACT with naive delta failed, and testing:
1. Cosine delta vs Norm delta for ACT
2. Multi-channel selective decay (LTT2) vs Scalar LTI across different iteration depths
3. Entropy / Energy based Halting vs Fixed T
4. Dual-Track Hybrid State Space (Linear SSM recurrent state vs Anchor Guard)
5. Attractor Drift Analysis: Measuring why T=8 degrades compared to T=3
"""

import numpy as np

def rms_norm(x, eps=1e-6):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms

class MultiStepReasoningEnv:
    """
    Simulates a sequence of dependent reasoning steps (e.g. math word problem or logic chain).
    Each reasoning step requires resolving an intermediate premise without losing prior context.
    """
    def __init__(self, dim=128, steps_required=4, seed=42):
        self.rng = np.random.default_rng(seed)
        self.dim = dim
        self.steps_required = steps_required
        
        # Subspace for each reasoning step
        # Step 0: premise parsing
        # Step 1: constraint matching
        # Step 2: calculation / transformation
        # Step 3: target deduction
        self.stages = []
        for i in range(steps_required):
            Q, _ = np.linalg.qr(self.rng.normal(size=(dim, dim)))
            subspace = Q[:, :8] # 8-dim subspace per reasoning stage
            self.stages.append(subspace)
            
        # Target representation is alignment with stage 3 while maintaining consistency with 0..2
        self.target = np.sum([s @ self.rng.normal(size=(8,)) for s in self.stages], axis=0)
        self.target = self.target / np.linalg.norm(self.target)
        
        # Anchor e represents input with Stage 0 and noisy Stage 1
        self.e = self.stages[0] @ self.rng.normal(size=(8,)) + 0.3 * self.rng.normal(size=(dim,))
        self.e = self.e / np.linalg.norm(self.e)

    def transformer_core(self, x):
        """
        Simulates one pass through a pretrained hybrid transformer layer.
        Contains:
        1. Attention/SSM step: moves representation along the reasoning chain
        2. Non-linear FFN: projects onto learned features with finite basin of attraction
        """
        x_n = rms_norm(x)
        # Advance reasoning: project onto next compatible stage
        update = np.zeros(self.dim)
        for i in range(self.steps_required - 1):
            sim = np.linalg.norm(self.stages[i].T @ x_n)
            if sim > 0.4:
                # Transition towards stage i+1
                target_direction = self.stages[i+1] @ (self.stages[i+1].T @ self.target)
                update += 0.25 * target_direction
                
        # Non-linear distortion (simulating FFN saturation and noise)
        distortion = 0.08 * np.sin(x_n * 3.14)
        return update + distortion

    def evaluate(self, h):
        h_n = rms_norm(h)
        # 1. Target alignment (cosine similarity with true target)
        align = np.dot(h_n, self.target)
        # 2. Premise preservation (cosine similarity with anchor stage 0)
        premise_leak = np.linalg.norm(self.stages[0].T @ h_n)
        return align, premise_leak

def run_deep_experiments(n_trials=300):
    evals = {
        "Vanilla T=1": {"align": [], "leak": []},
        "Scalar LTI T=3": {"align": [], "leak": []},
        "Scalar LTI T=5": {"align": [], "leak": []},
        "Scalar LTI T=8": {"align": [], "leak": []},
        "LTT2 Selective Decay (Channel-wise) T=3": {"align": [], "leak": []},
        "LTT2 Selective Decay (Channel-wise) T=5": {"align": [], "leak": []},
        "LTT2 Selective Decay (Channel-wise) T=8": {"align": [], "leak": []},
        "Hybrid Dual-Track + Anchor Preserving": {"align": [], "leak": []},
        "ACT (Directional Cosine Convergence)": {"align": [], "leak": []}
    }
    
    act_steps = []
    
    for trial in range(n_trials):
        env = MultiStepReasoningEnv(dim=128, steps_required=4, seed=trial)
        e = env.e
        
        # 1. Vanilla T=1
        h1 = e + env.transformer_core(e)
        a, l = env.evaluate(h1)
        evals["Vanilla T=1"]["align"].append(a)
        evals["Vanilla T=1"]["leak"].append(l)
        
        # 2. Scalar LTI with fixed T = 3, 5, 8
        for T, name in [(3, "Scalar LTI T=3"), (5, "Scalar LTI T=5"), (8, "Scalar LTI T=8")]:
            h = e.copy()
            for _ in range(T):
                comb = rms_norm(h + e)
                out = env.transformer_core(comb)
                h = 0.9 * h + 0.1 * e + 0.2 * out
            a, l = env.evaluate(h)
            evals[name]["align"].append(a)
            evals[name]["leak"].append(l)
            
        # 3. LTT2 Channel-Selective Decay
        # Key concept: Channels aligned with anchor e have higher decay (resist corruption)
        # Channels where update is active have lower decay (adapt rapidly)
        # alpha_c = sigmoid(lambda * |e_c| / rms(e))
        e_norm = rms_norm(e)
        channel_importance = np.abs(e_norm)
        # Normalize between 0.70 and 0.95
        alpha = 0.70 + 0.25 / (1.0 + np.exp(- 3.0 * (channel_importance - np.mean(channel_importance))))
        beta = 1.0 - alpha
        
        for T, name in [(3, "LTT2 Selective Decay (Channel-wise) T=3"), 
                        (5, "LTT2 Selective Decay (Channel-wise) T=5"), 
                        (8, "LTT2 Selective Decay (Channel-wise) T=8")]:
            h = e.copy()
            for _ in range(T):
                comb = rms_norm(h + e)
                out = env.transformer_core(comb)
                h = alpha * h + beta * e + 0.2 * out
            a, l = env.evaluate(h)
            evals[name]["align"].append(a)
            evals[name]["leak"].append(l)
            
        # 4. Hybrid Dual-Track (Decoupled Anchor Projection + Dynamic State)
        # h_t = e + delta_t
        # delta_{t+1} = alpha * delta_t + gamma * Core(RMSNorm(e + delta_t))
        # This GUARANTEES that anchor e is never scaled down or forgotten,
        # only the incremental reasoning perturbation delta is transformed!
        delta = np.zeros_like(e)
        for t in range(5):
            comb = rms_norm(e + delta)
            out = env.transformer_core(comb)
            # update perturbation with selective decay
            delta = alpha * delta + 0.2 * out
        h_dual = e + delta
        a, l = env.evaluate(h_dual)
        evals["Hybrid Dual-Track + Anchor Preserving"]["align"].append(a)
        evals["Hybrid Dual-Track + Anchor Preserving"]["leak"].append(l)
        
        # 5. ACT Directional Cosine Halting
        # Halts when cosine(h_t, h_{t-1}) > 0.999 (angular velocity in latent space approaches 0)
        h_act = e.copy()
        s_count = 0
        for t in range(12):
            s_count += 1
            h_prev = h_act.copy()
            comb = rms_norm(h_act + e)
            out = env.transformer_core(comb)
            h_act = alpha * h_act + beta * e + 0.2 * out
            # Directional alignment check
            cos_change = np.dot(rms_norm(h_act), rms_norm(h_prev))
            if t >= 2 and cos_change > 0.998:
                break
        act_steps.append(s_count)
        a, l = env.evaluate(h_act)
        evals["ACT (Directional Cosine Convergence)"]["align"].append(a)
        evals["ACT (Directional Cosine Convergence)"]["leak"].append(l)
        
    print("="*85)
    print(f"DEEP EXPERIMENT 2 RESULTS ({n_trials} Monte Carlo Runs)")
    print("="*85)
    print(f"{'Method':45s} | {'Target Align (Higher)':20s} | {'Premise Retention':18s}")
    print("-"*85)
    for k, v in evals.items():
        mean_align = np.mean(v["align"])
        mean_leak = np.mean(v["leak"])
        print(f"{k:45s} | {mean_align:+.4f} (+/- {np.std(v['align']):.3f})   | {mean_leak:.4f} (+/- {np.std(v['leak']):.3f})")
    print("="*85)
    print(f"ACT Directional Halting Average Steps: {np.mean(act_steps):.2f} / 12 max")
    print("="*85)

if __name__ == "__main__":
    run_deep_experiments(300)
