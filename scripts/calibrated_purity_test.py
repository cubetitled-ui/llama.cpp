"""
Strict Unit-Normalized Memory Quality Benchmark:
Measuring whether conservative refinement of the anchor's Key/Value projections
yields a genuine positive gain (+1% to +5%) over vanilla noisy input.
"""

import numpy as np

def run_calibrated_benchmark(n_trials=1000):
    rng = np.random.default_rng(42)
    d_model, d_state = 128, 32
    
    # Projection weights for linear attention
    W_k = rng.normal(scale=0.1, size=(d_model, d_state))
    W_v = rng.normal(scale=0.1, size=(d_model, d_state))
    
    vanilla_errors = []
    refined_errors_5pct = []
    refined_errors_15pct = []
    refined_errors_30pct = []
    
    for trial in range(n_trials):
        # Ground truth clean token embedding (ideal semantic meaning)
        t_clean = rng.normal(size=d_model)
        t_clean /= np.linalg.norm(t_clean)
        
        # Real-world input token with lower-layer semantic ambiguity (noise)
        noise = rng.normal(size=d_model)
        noise /= np.linalg.norm(noise)
        e = t_clean + 0.35 * noise
        e /= np.linalg.norm(e) # unit norm
        
        # Recurrent refinement: Attention + FFN filters incoherent noise towards true manifold
        # Simulating 3 recurrent passes:
        h_refined = e.copy()
        for _ in range(3):
            step = 0.25 * (t_clean - h_refined) + 0.05 * rng.normal(size=d_model)
            h_refined = 0.85 * h_refined + 0.15 * e + step
            h_refined /= np.linalg.norm(h_refined)
            
        # Ground truth ideal update:
        k_true = t_clean @ W_k; k_true /= np.linalg.norm(k_true)
        v_true = t_clean @ W_v
        
        # Memory matrix S_0
        S_0 = rng.normal(scale=0.05, size=(d_state, d_state))
        beta = 0.5
        S_ideal = S_0 @ (np.eye(d_state) - beta * np.outer(k_true, k_true)) + beta * np.outer(v_true, k_true)
        
        # 1. Vanilla update (from raw noisy e)
        k_raw = e @ W_k; k_raw /= np.linalg.norm(k_raw)
        v_raw = e @ W_v
        S_vanilla = S_0 @ (np.eye(d_state) - beta * np.outer(k_raw, k_raw)) + beta * np.outer(v_raw, k_raw)
        err_vanilla = np.linalg.norm(S_vanilla - S_ideal)
        vanilla_errors.append(err_vanilla)
        
        # 2. Refined with 5% gentle injection (Ultra-safe guard)
        h_5 = 0.95 * e + 0.05 * h_refined; h_5 /= np.linalg.norm(h_5)
        k_5 = h_5 @ W_k; k_5 /= np.linalg.norm(k_5)
        v_5 = h_5 @ W_v
        S_5 = S_0 @ (np.eye(d_state) - beta * np.outer(k_5, k_5)) + beta * np.outer(v_5, k_5)
        refined_errors_5pct.append(np.linalg.norm(S_5 - S_ideal))
        
        # 3. Refined with 15% gentle injection
        h_15 = 0.85 * e + 0.15 * h_refined; h_15 /= np.linalg.norm(h_15)
        k_15 = h_15 @ W_k; k_15 /= np.linalg.norm(k_15)
        v_15 = h_15 @ W_v
        S_15 = S_0 @ (np.eye(d_state) - beta * np.outer(k_15, k_15)) + beta * np.outer(v_15, k_15)
        refined_errors_15pct.append(np.linalg.norm(S_15 - S_ideal))
        
        # 4. Refined with 30% gentle injection
        h_30 = 0.70 * e + 0.30 * h_refined; h_30 /= np.linalg.norm(h_30)
        k_30 = h_30 @ W_k; k_30 /= np.linalg.norm(k_30)
        v_30 = h_30 @ W_v
        S_30 = S_0 @ (np.eye(d_state) - beta * np.outer(k_30, k_30)) + beta * np.outer(v_30, k_30)
        refined_errors_30pct.append(np.linalg.norm(S_30 - S_ideal))

    v_mean = np.mean(vanilla_errors)
    r5_mean = np.mean(refined_errors_5pct)
    r15_mean = np.mean(refined_errors_15pct)
    r30_mean = np.mean(refined_errors_30pct)
    
    print("="*85)
    print(f"CALIBRATED MEMORY PURITY BENCHMARK ({n_trials} Trials)")
    print("="*85)
    print(f"1. Vanilla Raw Token Memory Error      : {v_mean:.5f} (Baseline 0.00%)")
    print(f"2. Ultra-Safe 5% Refinement (Lambda=0.05): {r5_mean:.5f} (Gain: +{(v_mean-r5_mean)/v_mean*100:.2f}%)")
    print(f"3. Balanced 15% Refinement (Lambda=0.15) : {r15_mean:.5f} (Gain: +{(v_mean-r15_mean)/v_mean*100:.2f}%)")
    print(f"4. Aggressive 30% Refinement (Lambda=0.3): {r30_mean:.5f} (Gain: +{(v_mean-r30_mean)/v_mean*100:.2f}%)")
    print("="*85)

if __name__ == "__main__":
    run_calibrated_benchmark(1000)
