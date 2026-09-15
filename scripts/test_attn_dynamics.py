"""
Angle 6 Test: Attention Dynamic Re-weighting across Recurrence Passes
Does repeating a layer actually re-distribute attention across the context tokens,
or does it just over-attend to the recent token and collapse the context entropy?
"""

import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def test_attention_dynamics(seq_len=64, n_heads=8, head_dim=64, n_passes=5):
    rng = np.random.default_rng(42)
    
    # Simulate past K, V for seq_len tokens
    K_past = rng.normal(size=(seq_len, n_heads, head_dim))
    V_past = rng.normal(size=(seq_len, n_heads, head_dim))
    
    # Current token query Q at pass 0
    Q_0 = rng.normal(size=(n_heads, head_dim))
    
    # Store entropy of attention distribution across passes
    entropies = []
    cos_shifts = []
    
    Q_t = Q_0.copy()
    for t in range(n_passes):
        # Attention scores
        # scores: (n_heads, seq_len)
        scores = np.einsum('hd,shd->hs', Q_t, K_past) / np.sqrt(head_dim)
        attn_weights = softmax(scores, axis=-1)
        
        # Entropy per head: - sum p log p
        entropy = - np.sum(attn_weights * np.log(attn_weights + 1e-12), axis=-1)
        entropies.append(np.mean(entropy))
        
        # Context vector
        ctx = np.einsum('hs,shd->hd', attn_weights, V_past)
        
        # Simulating Layer output updating Q_t
        # In OpenMythos, Q_t moves according to residual update
        delta_Q = 0.2 * ctx + 0.1 * rng.normal(size=Q_t.shape)
        Q_next = 0.9 * Q_t + delta_Q
        
        # Measure how much Q changed direction
        cos = np.sum(Q_next * Q_t) / (np.linalg.norm(Q_next) * np.linalg.norm(Q_t) + 1e-8)
        cos_shifts.append(cos)
        Q_t = Q_next
        
    print("Attention Entropy across passes (Higher = wider context integration):")
    for t, ent in enumerate(entropies):
        print(f"  Pass {t}: Entropy = {ent:.4f} (Max possible = {np.log(seq_len):.4f})")

if __name__ == "__main__":
    test_attention_dynamics()
