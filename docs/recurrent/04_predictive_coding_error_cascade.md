# Architecture 04: Predictive Coding Variational Error Cascade (PC-VEC)

## 1. Mathematical Formulation & Core Mechanism
Rooted in Karl Friston's Free Energy Principle, activations are not passed forward passively. Instead, each recurrent block generates top-down generative predictions $\hat{h}_l$, and bottom-up passes transmit strictly precision-weighted prediction errors $\epsilon_l$:

330333\epsilon_l^{(t)} = \Sigma_l^{-1} \left( h_l^{(t)} - g_	heta(h_{l+1}^{(t)}) ight)330333
where $\Sigma_l$ is a diagonal learned sensory precision matrix.

The internal state vector $ is iteratively relaxed via gradient descent on the Variational Free Energy $\mathcal{F}$:
330333h_l^{(t+1)} = h_l^{(t)} + \eta \left[ \epsilon_l^{(t)} - \left( rac{\partial g_	heta}{\partial h_l} ight)^T \epsilon_{l-1}^{(t)} ight]330333

## 2. Why it is Radically Unique
Completely inverts the forward-pass paradigm: representations are settled by hierarchical equilibrium rather than feedforward functional application. When a token is easily predictable, $\epsilon_l 	o 0$, and execution terminates immediately with zero extra compute.

## 3. Objective Cons & Limitations
1. **Compute Multiplier per Token**: Settling the free energy equilibrium requires multiple relaxation loops (typically 5-10 iterations) per token, drastically reducing throughput.
2. **Jacobian Transpose Overhead**: Evaluating $\left( rac{\partial g_	heta}{\partial h_l} ight)^T \epsilon_{l-1}$ requires explicit vector-Jacobian products (VJPs) during the inference forward pass.
3. **Instability & Limit Cycles**: For non-convex potentials, the gradient relaxation can become trapped in oscillatory limit cycles rather than converging to a fixed point.
