# Architecture 08: Orthogonal Gram-Schmidt Residual Subspace Decomposition (OGS-RSD)

## 1. Mathematical Formulation & Core Mechanism
To prevent redundant information recycling, each recurrent pass is strictly constrained to operate in an orthogonal subspace of all preceding thought vectors.

At step $t$, the decoder generates a raw thought candidate:
$$\Delta_t = \text{Decoder}(h_t) - h_t$$

We project $\Delta_t$ onto the orthogonal complement of the subspace spanned by historical update directions $\{u_1, u_2, \dots, u_{t-1}\}$ via modified Gram-Schmidt:
$$\Delta_t^{\perp} = \Delta_t - \sum_{j=1}^{t-1} \frac{\langle \Delta_t, u_j \rangle}{\|u_j\|^2} u_j$$
The new normalized basis direction is:
$$u_t = \frac{\Delta_t^{\perp}}{\|\Delta_t^{\perp}\| + \epsilon}$$
The state is updated along this novel orthogonal dimension with learned magnitude:
$$h_{t+1} = h_t + \lambda_t u_t$$

## 2. Why it is Radically Unique
Guarantees zero collinear overlap between passes ($\langle u_i, u_j \rangle = 0, \forall i \neq j$). Every recurrent pass is mathematically forced to extract completely orthogonal semantic features, maximizing representation entropy.

## 3. Objective Cons & Limitations
1. **History Vector Storage**: Requires caching all past update vectors $\{u_1, \dots, u_{t-1}\}$ in VRAM for the duration of the recurrent block.
2. **Dimensionality Saturation**: The maximum number of meaningful orthogonal passes is bounded by the hidden dimension $d_{model}$ ($T \le d$). In practice, after 6-8 passes, remaining orthogonal subspaces contain mostly noise.
3. **Gradient Shattering**: Backpropagating through the Gram-Schmidt normalization leads to ill-conditioned gradients when $\Delta_t$ is nearly collinear with existing basis vectors.
