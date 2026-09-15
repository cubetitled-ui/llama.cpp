# Academic Research Paper: Latent Recurrent Deliberation via ORSD

This directory contains the formal scientific research paper on the mathematical foundations and systems engineering of the Recurrent Inference Engine:

* 📄 **Compiled Paper (PDF)**: [`arxiv_preprint.pdf`](arxiv_preprint.pdf)
* 📝 **LaTeX Source (arXiv Ready)**: [`arxiv_preprint.tex`](arxiv_preprint.tex)
* 📖 **Full Article (Markdown)**: [`arxiv_preprint.md`](arxiv_preprint.md)

---

## Executive Summary for Systems Engineers

1. **The Problem**: Pre-trained transformer layers are ill-conditioned ($\kappa \approx 42{,}487$). When looped naively ($h_{t+1} = h_t + \gamma \Delta_t$), iterations execute **power iteration** toward the dominant eigenvector of $W_{\text{down}} W_{\text{gate}}$, collapsing sequence rank from 14.2 to 1.8 and causing activation blowup.
2. **The Fix (ORSD)**: Project candidate updates $\Delta_t$ onto the orthogonal complement of the historical thought subspace $\mathcal{U}_{t-1}$ via Modified Gram-Schmidt, and scale by the baseline RMS thought energy ($\text{RMS}(\Delta_0)$).
3. **KV Cache Isolation**: Hardware-level `store_kv = false` ensures deliberation passes attend to pristine context without corrupting future autoregressive token history ($O(1)$ memory overhead).
4. **Silicon Benchmark**: Tested on NVIDIA RTX 3050 Laptop GPU (GA107, 6GB VRAM), ORSD delivers $+11.1\%$ net accuracy gain on hardened algorithmic coding tests, and $100\%$ on probabilistic deductive logic, while unconstrained vanilla recurrence collapses down to $22.2\%$.
