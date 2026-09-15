# Architecture 06: Associative Memory Delta-Rule Matrix State (AM-DRM)

## 1. Mathematical Formulation & Core Mechanism
Rather than propagating a 1D state vector $h_t \in \mathbb{R}^d$, the recurrent memory is maintained as a fast-weight matrix associative memory $M_t \in \mathbb{R}^{d_k \times d_v}$. At each recurrent turn, representations are updated using the Hebbian error-correcting Delta Rule:

Given projected keys $k_t = W_k h_t$, values $v_t = W_v h_t$, queries $q_t = W_q h_t$, and dynamic write gate $\beta_t = \sigma(w_\beta^T h_t)$:
$$M_t = M_{t-1} + \beta_t (v_t - M_{t-1} k_t) \otimes k_t^T$$
where $(v_t - M_{t-1} k_t)$ represents the associative retrieval error (prediction residual).

The output features are recalled from associative memory and injected into the residual stream:
$$y_t = M_t q_t$$
$$h_{t+1} = \text{RMSNorm}(h_t + W_o y_t)$$

## 2. Why it is Radically Unique
Unlike fixed-dimension vector recurrence that suffers from severe memory bottlenecks (Schonhage-Strassen limits), the 2D matrix memory $M_t$ possesses quadratic associative capacity $\mathcal{O}(d_k \cdot d_v)$, allowing continuous error-corrected key-value binding and unbinding across multiple passes.

## 3. Objective Cons & Limitations
1. **$O(d^2)$ State Memory Consumption**: Maintaining a full matrix state $M_t$ per sequence token expands KV-cache requirements dramatically.
2. **Matrix-Vector GEMV Latency**: Each recurrent pass requires sequential outer-product updates and matrix-vector multiplications ($M_{t-1} k_t$), incurring high memory bandwidth costs.
3. **Eigenvalue Saturation**: Without careful normalization of $k_t$, repeated outer-product updates can cause the spectral radius $\rho(M_t)$ to explode or collapse to zero.
