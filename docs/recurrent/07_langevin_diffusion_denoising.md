# Architecture 07: Score-Based Langevin Diffusion Denoising (SBL-DD)

## 1. Mathematical Formulation & Core Mechanism
Recurrence is framed as sampling from a continuous multimodal probability density $p(h)$ of valid reasoning traces via an annealed Langevin diffusion process.

At recurrent pass $t \in \{1, \dots, T\}$, Gaussian noise is injected, and the decoder acts as a learned Score Function $s_\theta(h_t, \sigma_t) \approx \nabla_{h_t} \log p(h_t)$:
$$h_{t+1} = h_t + \frac{\gamma_t}{2} s_\theta(h_t, \sigma_t) + \sqrt{\gamma_t} \mathbf{z}_t, \quad \mathbf{z}_t \sim \mathcal{N}(0, I)$$
where:
$$s_\theta(h_t, \sigma_t) = \frac{\text{Decoder}(h_t) - h_t}{\sigma_t^2}$$
Noise schedule $\sigma_t = \sigma_{max} \left( \frac{\sigma_{min}}{\sigma_{max}} \right)^{t/T}$, and step size $\gamma_t = \epsilon \left( \frac{\sigma_t}{\sigma_{min}} \right)^2$.

At the final step $t=T$, noise is removed ($\mathbf{z}_T = 0$), projecting the state onto the deterministic mode of the manifold.

## 2. Why it is Radically Unique
Standard deterministic recurrence frequently gets trapped in sub-optimal local minima (hallucinations). Thermal noise injection enables stochastic tunneling through energy barriers, exploring diverse deduction hypotheses before cooling to the optimal logical solution.

## 3. Objective Cons & Limitations
1. **Inference Non-Determinism**: Output tokens become inherently stochastic even when setting sampling temperature to 0, complicating reproducible benchmarking.
2. **Extreme Sensitivity to Noise Schedules**: If $\sigma_{max}$ is slightly too high, syntactic coherence is destroyed; if too low, the diffusion collapses into standard greedy decoding.
3. **Variance Accumulation**: In multi-step sequences, unchecked variance can drift the hidden state into low-density latent voids where the LM head outputs gibberish.
