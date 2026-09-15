"""
Hybrid Layer Recurrence Suite: Mamba-2 and DeltaNet-Gated Deep Dive

In modern hybrid architectures (Falcon-H1 Mamba-2, Qwen-3.5 DeltaNet / Gated Linear Attention):
Layers are NOT pure Attention + FFN. They contain:
1. Mamba-2: SSM State space h_t = A_t h_{t-1} + B_t x_t, y_t = C_t h_t + D x_t
2. DeltaNet (Qwen 3.5): Fast Associative Matrix Memory S_t = S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
   where (I - beta k k^T) is the Householder-like forgetting operator (Delta Rule)!

In naive OpenMythos:
- The recurrent core is treated as a black box F(h + e).
- BUT Mamba and DeltaNet ALREADY HAVE their own recurrent memory S_t!
- In falcon-h1.cpp and qwen35.cpp, build_recurrent_core runs the layer T times,
  which mutates ssm_states_all / conv_states_all T times for the same token!

Let's test 3 Paradigms for Hybrids:
1. Naive Hybrid Recurrence (current code: SSM state updated T times, overwriting native state)
2. Delta-Rule Latent Injection: Using the native DeltaNet / Mamba matrix memory as a temporary working scratchpad
3. Dual-Channel: DeltaNet handles intra-token fast reasoning while Attention maintains inter-token global context.
"""

import numpy as np

def rms_norm(x, eps=1e-6):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms

class DeltaNetSimulation:
    def __init__(self, d_model=64, d_state=16, seed=42):
        self.rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.d_state = d_state
        
        # Projections
        self.W_q = self.rng.normal(scale=0.1, size=(d_model, d_state))
        self.W_k = self.rng.normal(scale=0.1, size=(d_model, d_state))
        self.W_v = self.rng.normal(scale=0.1, size=(d_model, d_state))
        self.W_beta = self.rng.normal(scale=0.1, size=(d_model, 1))
        self.W_out = self.rng.normal(scale=0.1, size=(d_state, d_model))
        
        # State: S is (d_state, d_state)
        self.S_initial = self.rng.normal(scale=0.05, size=(d_state, d_state))

    def step(self, x, S_in, update_state=True):
        """One DeltaNet forward step"""
        x_n = rms_norm(x)
        q = x_n @ self.W_q
        k = x_n @ self.W_k
        v = x_n @ self.W_v
        
        # Normalize q, k
        q = q / (np.linalg.norm(q) + 1e-6)
        k = k / (np.linalg.norm(k) + 1e-6)
        
        beta = 1.0 / (1.0 + np.exp(- (x_n @ self.W_beta)[0])) # sigmoid
        
        # Output before state update or with current state:
        # y = S @ q
        y = S_in @ q
        
        # Delta Rule update:
        # S_next = S_in * (I - beta * k * k^T) + beta * v * k^T
        # = S_in - beta * (S_in @ k) @ k^T + beta * v @ k^T
        if update_state:
            v_err = v - (S_in @ k)
            S_next = S_in + beta * np.outer(v_err, k)
        else:
            S_next = S_in
            
        out = y @ self.W_out
        return out, S_next

def run_hybrid_experiments(n_trials=300):
    results = {
        "1. Vanilla T=1": [],
        "2. Naive SSM Mutate (T=4)": [],
        "3. Frozen SSM State (Read-Only during loops, T=4)": [],
        "4. DeltaNet Scratchpad (Rollback to S_0 after loop, update once)": []
    }
    
    for trial in range(n_trials):
        net = DeltaNetSimulation(d_model=64, d_state=16, seed=trial)
        
        # Input anchor
        e = net.rng.normal(size=(64,))
        e = e / np.linalg.norm(e)
        
        # Target: DeltaNet trained transformation of e
        # Ground truth state update from single clean step
        target_out, S_clean = net.step(e, net.S_initial, update_state=True)
        
        # 1. Vanilla T=1
        out1, S1 = net.step(e, net.S_initial, update_state=True)
        results["1. Vanilla T=1"].append(np.dot(rms_norm(out1), rms_norm(target_out)))
        
        # 2. Naive SSM Mutate T=4 (current implementation)
        # Updates S T times on the same token
        h = e.copy()
        S_cur = net.S_initial.copy()
        for t in range(4):
            comb = rms_norm(h + e)
            block_out, S_cur = net.step(comb, S_cur, update_state=True)
            h = 0.9 * h + 0.1 * e + 0.2 * block_out
        # Measure state distortion vs clean single-token state
        state_distortion = np.linalg.norm(S_cur - S_clean)
        results["2. Naive SSM Mutate (T=4)"].append(state_distortion)
        
        # 3. Frozen SSM State during loops (T=4)
        # S_in is NOT updated during intermediate loops
        h = e.copy()
        for t in range(4):
            comb = rms_norm(h + e)
            block_out, _ = net.step(comb, net.S_initial, update_state=False)
            h = 0.9 * h + 0.1 * e + 0.2 * block_out
        # Final update
        _, S_frozen = net.step(h, net.S_initial, update_state=True)
        results["3. Frozen SSM State (Read-Only during loops, T=4)"].append(np.linalg.norm(S_frozen - S_clean))
        
        # 4. DeltaNet Scratchpad: S evolves in temporary scratchpad during loop,
        # but final commit uses converged state
        h = e.copy()
        S_temp = net.S_initial.copy()
        for t in range(4):
            comb = rms_norm(h + e)
            block_out, S_temp = net.step(comb, S_temp, update_state=True)
            h = 0.9 * h + 0.1 * e + 0.2 * block_out
        # Commit clean step from h_final on S_initial
        _, S_commit = net.step(h, net.S_initial, update_state=True)
        results["4. DeltaNet Scratchpad (Rollback to S_0 after loop, update once)"].append(np.linalg.norm(S_commit - S_clean))

    print("="*90)
    print("HYBRID SSM / DELTANET STATE PRESERVATION TEST (State Distortion vs Clean Token)")
    print("="*90)
    print(f"2. Naive SSM Mutate (T=4, Current Code)          : Mean Distortion = {np.mean(results['2. Naive SSM Mutate (T=4)']):.4f}")
    print(f"3. Frozen SSM State (Read-Only during loops, T=4): Mean Distortion = {np.mean(results['3. Frozen SSM State (Read-Only during loops, T=4)']):.4f}")
    print(f"4. DeltaNet Scratchpad + Single Commit (T=4)    : Mean Distortion = {np.mean(results['4. DeltaNet Scratchpad (Rollback to S_0 after loop, update once)']):.4f}")
    print("="*90)

if __name__ == "__main__":
    run_hybrid_experiments(300)
