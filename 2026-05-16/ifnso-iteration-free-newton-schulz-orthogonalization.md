---
title: "IFNSO: Iteration-Free Newton-Schulz Orthogonalization"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2602.02500"
---

## Executive Summary
IFNSO (Iteration-Free Newton-Schulz Orthogonalization) eliminates the iterative structure of traditional matrix orthogonalization methods, replacing repeated high-dimensional matrix multiplications with a single unified polynomial operation. It uses a constrained polynomial with exponentially growing term exponents to achieve stable convergence while reducing computational overhead. For practitioners building large-scale neural network training systems, this means up to 10% faster convergence in optimisers like Muon without sacrificing accuracy, directly lowering training costs in production environments.

## Why This Matters for Practitioners
If you're optimising large language models or other deep learning systems where orthogonalisation forms a bottleneck in training pipelines, IFNSO can reduce computation time by eliminating the N iterations required in traditional Newton-Schulz methods. For example, when processing a 128×512 matrix, IFNSO achieves an orthogonalisation error of 0.040 compared to 0.330 for the next best baseline (Cesista's NS), while maintaining comparable FLOPs (8.831 × 10^7 vs 6.332 × 10^7). This translates to 10% faster training convergence in practice, making it particularly valuable for teams working with limited GPU resources or tight latency budgets for model training.

## Problem Statement
Traditional Newton-Schulz iteration requires repeated matrix multiplications, like a slow, sequential production line where each worker must wait for the previous one to finish before starting, becoming increasingly costly as matrix dimensions grow. For example, orthogonalizing a 128×1024 matrix requires N iterations of matrix multiplication, with computational cost growing linearly with N. This iterative process creates a bottleneck in optimisers like Muon, where orthogonalisation constitutes 20-30% of total training time for large models, directly impacting scalability and cost-efficiency in production systems.

## Proposed Approach
IFNSO replaces the iterative Newton-Schulz process with a single unified polynomial operation that computes orthogonalisation in one pass. The core insight is to express the orthogonalisation process as a polynomial in the matrix, where coefficients are learnable but constrained to ensure convergence. The polynomial terms employ exponential growth in their exponents, allowing efficient coverage of the necessary mathematical space while reducing computational overhead. The system processes the input matrix through a series of projection operations and aggregates results using the learnable polynomial.

```python
def ifnso(M, W, L):
    # W = [w_1, w_2, ..., w_L] (learned coefficients)
    # L = depth of polynomial
    if M.shape[0] > M.shape[1]:
        A = M.T
    else:
        A = M
    T0 = A @ A.T
    o = np.linalg.norm(T0, 'fro')
    T1 = np.eye(*T0.shape) - T0 / o
    T = [T0, T1]
    for l in range(2, L+1):
        T.append(T[-1] @ T[-1])
    Y_prime = np.eye(*T[0].shape)
    for l in range(1, L+1):
        Y_prime += W[l-1] * T[l]
    Y = Y_prime @ A / np.sqrt(o)
    return Y
```

## Key Technical Contributions
The core innovation of IFNSO lies in its constrained polynomial formulation and exponential term selection strategy, which together enable stable single-pass orthogonalisation.

1. **Constrained polynomial formulation**: IFNSO's polynomial is designed as y = Σ(w_{2n+1} × (1 - x²)^n × x), ensuring y approaches 1 as x approaches 1. This constraint provides stable gradient flow during coefficient optimisation, preventing the significant error (3.846) seen in Muon's NS. The authors explicitly derive this constraint from the scalar map of the Newton-Schulz iteration, ensuring the polynomial remains within the valid orthogonalisation range without requiring additional hyperparameters.

2. **Exponential term selection**: Unlike previous approaches that use consecutive exponents (e.g., 0, 1, 2, 3...), IFNSO uses exponential growth (n_l = 2^{l-1} for l = 1,...,L) to select term exponents. This strategy enables broader coverage of the mathematical space with fewer terms, e.g., with L=14 terms, IFNSO achieves an error of 0.040 on 128×128 matrices versus 0.330 for Cesista's NS (15 terms), demonstrating that exponential growth provides more efficient coverage of the necessary mathematical domain.

3. **Optimised coefficient learning**: The coefficients are learned by minimising (f(x_j) - 1)² across 1,000 uniformly sampled points in (0,1). Crucially, the authors derive a simplified approximation for the final coefficient (w_L ≈ e^{1/2}(2^{L/2} - 1) - Σ_{l=1}^{L-1} w_l), eliminating the need for extensive backpropagation and reducing the hyperparameter tuning burden compared to prior approaches like CANS.

## Experimental Results
Experiments on matrices of sizes (128×128), (128×512), and (128×1024) show IFNSO achieves superior orthogonalisation error compared to all baselines (Table 1). For 128×128 matrices, IFNSO achieves 0.040 error versus 0.330 for Cesista's NS (the next best baseline), representing a 87.9% error reduction. FLOPs are comparable (6.314 × 10^7 for IFNSO vs 6.332 × 10^7 for Cesista's NS), and the error reduction is statistically significant given the consistent results across all matrix sizes.

In MNIST training with the Muon optimizer (Table 2), IFNSO achieves the lowest loss (4.25 vs 5.10 for Muon's NS) and highest accuracy (98.87% vs 98.83% for Muon's NS), with the training loss curve (Figure 7) showing the fastest convergence. The authors performed 20,000 epochs of training for coefficient learning, with Adam optimiser and a step scheduler decayed every 10,000 iterations.

## Related Work
IFNSO builds on the Newton-Schulz iteration, a standard technique for orthogonalisation in optimisers like Muon [17]. It improves upon prior works that introduced learnable coefficients (Cesista's NS [20]) and Chebyshev-based polynomials (CANS [21]) by eliminating the iterative structure entirely. Unlike Muon's NS (which uses fixed coefficients) or CANS (which requires learnable parameters γ, r, and u), IFNSO achieves stable convergence with a constrained polynomial that requires no additional hyperparameters beyond the polynomial depth L.

## Limitations
The authors acknowledge three key limitations: IFNSO converges to 1 more slowly than some methods, causing noticeable fluctuations when y first reaches 1; the method requires matrix multiplication involving X, leading to high computational cost when w ≫ h; and the experiments were limited to matrices with h ≤ w, with no tests on extremely large matrices (e.g., 1024×1024 or larger). In practice, these limitations mean IFNSO may not be suitable for all matrix shapes, particularly those where the width significantly exceeds the height, and engineers should consider these constraints when integrating into production pipelines.

## Appendix: Worked Example
Let's walk through the orthogonalisation of a 128×128 matrix using IFNSO with L=14.

1. Start with matrix A (128×128), where ||A||_F = 100
2. Compute T0 = A @ A^T (128×128 matrix), ||T0||_F = 10,000
3. Compute T1 = I - T0 / 10,000
4. Compute T2 = T1 @ T1
5. Compute T3 = T2 @ T2
6. Continue until T14 = T13 @ T13
7. Using learned coefficients W = [0.1, 0.05, 0.025, ..., 0.000007] (approximated from the paper's derivation)
8. Compute Y' = I + 0.1T1 + 0.05T2 + ... + 0.000007T14
9. Compute Y = Y' @ A / 10 (sqrt(10,000) = 100, so /100)

This process yields Y such that YY^T ≈ I with error 0.040, compared to 0.330 for Cesista's NS. The exponential term selection (T1, T2, T4, T8, T16...) allows the polynomial to cover the necessary mathematical space efficiently, with the L=14 terms achieving near-optimal coverage without requiring additional matrix multiplications.

## References

- **Code:** https://github.com/greekinRoma/Ieration_
- Chen Hu, Qianxi Zhao, Xiaochen Yuan, Hong Zhang, Ding Yuan, Yanbin Wu, Xiying Li, "IFNSO: Iteration-Free Newton-Schulz Orthogonalization", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2602.02500

Tags: #machine-learning #optimisation #matrix-algorithms #neural-networks #orthogonalisation
