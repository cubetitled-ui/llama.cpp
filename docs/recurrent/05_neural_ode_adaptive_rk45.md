# Architecture 05: Continuous Neural ODE with Adaptive Runge-Kutta 45 (NODE-RK45)

## 1. Mathematical Formulation & Core Mechanism
Depth is treated as a continuous physical dimension  \in [0, 1]$. The transformation of hidden representations is modeled as an Ordinary Differential Equation (ODE):
330333rac{dh(s)}{ds} = f_	heta(h(s), s)330333
where 	heta$ is parameterized by the recurrent Transformer block with continuous depth embeddings.

Integration is computed via the embedded Dormand-Prince Runge-Kutta 4(5) pair:
330333h_{n+1} = h_n + \Delta s \sum_{i=1}^6 b_i k_i \quad (	ext{5th-order solution})330333
330333\hat{h}_{n+1} = h_n + \Delta s \sum_{i=1}^6 \hat{b}_i k_i \quad (	ext{4th-order solution})330333
Local truncation error estimate:
330333E_n = \|h_{n+1} - \hat{h}_{n+1}\|_{\infty}330333
Adaptive step-size controller:
330333\Delta s_{new} = \Delta s \cdot \min\left( 2.0, \max\left( 0.2, 0.9 \left( rac{	ext{Tol}}{E_n} ight)^{0.2} ight) ight)330333

## 2. Why it is Radically Unique
Provides continuous compute allocation. Trivial tokens take 2 large integration steps ($\Delta s pprox 0.5$); intricate logic puzzles trigger adaptive step subdivision, executing 15-20 micro-steps to resolve steep trajectory gradients.

## 3. Objective Cons & Limitations
1. **Dynamic Kernel Execution Breaks GPU Batching**: Because different tokens in a sequence require different numbers of ODE steps, static SIMD/SIMT execution suffers severe warp divergence.
2. **High Memory Overhead of Intermediate Stages**: RK45 requires evaluating and storing 6 stage evaluations ( \dots k_6$) per integration step.
3. **Non-Smooth Activations Cause Step Collapse**: Modern LLM activations (SwiGLU, LayerNorm) have non-smooth higher derivatives, causing step-size controllers to choke and over-subdivide steps unnecessarily.
