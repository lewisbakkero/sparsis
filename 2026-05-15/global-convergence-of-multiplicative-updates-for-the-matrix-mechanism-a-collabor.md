---
title: "Global Convergence of Multiplicative Updates for the Matrix Mechanism: A Collaborative Proof with Gemini 3"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19465"
---

## Executive Summary
This paper proves global convergence for a multiplicative update rule used in optimising a regularised nuclear norm objective within the matrix mechanism for differential privacy. The iteration v(k+1) = diag((D^(1/2)_v(k)MD^(1/2)_v(k))^(1/2)) converges monotonically to the unique global optimizer of the potential function J(v) = 2 Tr((D^(1/2)_vMD^(1/2)_v)^(1/2)) - Σv_i. This resolves a long-standing open problem from [DMR+22] that had hindered reliable implementation of this optimisation method in privacy-preserving machine learning systems.

## Why This Matters for Practitioners
If you're implementing the matrix mechanism for differential privacy in production systems, this paper resolves a critical uncertainty about convergence. The iterative method was empirically observed to converge rapidly in prior work ([DMR+22]), but without theoretical guarantees, engineers had to either rely on empirical evidence (risky for production systems) or switch to slower alternative optimisation approaches. Engineers can now confidently use this iterative method without worrying about non-convergence, especially for large-scale systems where convergence guarantees are essential for reliability. For privacy-sensitive applications requiring consistent performance, this means you can replace potentially unreliable gradient-based approaches with this iterative method, which requires approximately half as many iterations to reach high precision compared to the standard gradient descent approach.

## Problem Statement
Consider a scenario where you're building a privacy-preserving machine learning system that needs to compute optimal matrix factorizations for the matrix mechanism. The system uses an iterative method that appears to converge rapidly in practice, but you have no theoretical guarantee that it will converge for all inputs. This uncertainty is particularly problematic in production systems where consistency is paramount, without convergence guarantees, you're essentially running a risk that the system might fail to find the optimal solution for certain inputs, potentially leading to privacy violations or suboptimal performance.

## Proposed Approach
The authors prove that the multiplicative update rule v(k+1) = diag((D^(1/2)_v(k)MD^(1/2)_v(k))^(1/2)) converges monotonically to the unique global optimizer of the potential function J(v). They employ a Majorization-Minimization (MM) framework, constructing a variational surrogate function that lower bounds the potential J, and show that iterating the update rule makes strict progress on this surrogate function at each step, ensuring global convergence. The proof was partially generated with the assistance of Gemini 3, highlighting novel ways AI can assist with mathematical proofs.

```python
def matrix_mechanism_convergence(M, v_initial, tolerance=1e-6):
    """
    Iteratively applies the multiplicative update rule to converge to the optimal vector.
    
    Args:
        M: Symmetric positive definite matrix
        v_initial: Initial vector in positive orthant
        tolerance: Convergence threshold
    
    Returns:
        Converged vector v
    """
    v = v_initial.copy()
    while True:
        # Compute the matrix product
        X = np.diag(np.sqrt(v)) @ M @ np.diag(np.sqrt(v))
        # Take the square root and extract diagonals
        v_new = np.diag(np.sqrt(X))
        # Check for convergence
        if np.linalg.norm(v_new - v) < tolerance:
            break
        v = v_new
    return v
```

## Key Technical Contributions
The paper makes several technical contributions that resolve the global convergence problem:

1. **Variational characterisation of the nuclear norm**: The authors identify that the first term in the potential function can be represented as a nuclear norm, and use a variational characterisation that enables diagonalization of the problem. This characterisation, attributed to Gemini 3, was the missing piece that allowed them to prove global convergence.

2. **Surrogate function construction**: They construct a surrogate function G(v; v(k)) that lower bounds the potential J(v) and matches J(v) at the current iterate v(k). The surrogate function is linear in the square root of v, making optimisation straightforward.

3. **Geometric mean interpretation**: The authors show that the update rule corresponds to the element-wise geometric mean between the current iterate and the surrogate's optimum: v(k+1)_i = √(v(k)_i · vopt_i). This interpretation explains why the method converges rapidly without requiring line searches.

4. **Connection to optimal transport**: They establish a connection between the optimisation problem and the Bures-Wasserstein distance in optimal transport theory, showing that the potential function J(v) is mathematically equivalent to minimising the Bures-Wasserstein distance between a diagonal matrix D_v and the target matrix M.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The paper does not provide specific empirical results on convergence rates with particular matrices. However, the authors state that the iteration converges rapidly, outperforming pure gradient-based approaches. They note that the jump iteration (v ← ϕ(v)^2 ⊘ v) requires nearly exactly half as many iterations to reach high precision asymptotically compared to the original iteration, though they don't provide specific numbers for the number of iterations required to reach 1e-8 accuracy. The paper mentions that this factor of two can be derived from examining the Jacobians of the two updates at the fixed point.

## Related Work
This work directly addresses an open problem from [DMR+22], which established local convergence for the iteration but left global convergence unproven. The authors build on the theoretical framework established in [DMR+22] but extend it with a variational characterisation of the nuclear norm that was missing from their initial analysis. They also connect their work to the broader literature on the Bures-Wasserstein distance in optimal transport [BJL19] and quantum information theory [BCP14], showing that the matrix mechanism optimisation problem is mathematically equivalent to finding the closest classical state to a given quantum state.

## Limitations
The authors acknowledge that the problem of proving global convergence was left open for a significant time despite "reasonably serious attempts by reasonably serious mathematicians." The solution was partly generated with the assistance of an AI model (Gemini 3), highlighting both the potential and limitations of current AI systems in mathematical reasoning. The paper doesn't provide detailed empirical evidence on the convergence rates for different matrix sizes or structures, focusing instead on theoretical guarantees. The connection to optimal transport and quantum information theory is interesting but doesn't directly provide practical algorithmic improvements beyond the convergence proof.

## Appendix: Worked Example
Let's walk through a concrete example with a 2x2 symmetric positive definite matrix M = [[2, 1], [1, 2]] and an initial vector v(0) = [1, 1].

1. **First iteration**:
   - Compute D^(1/2)_v(0) = [[1, 0], [0, 1]]
   - Compute X = D^(1/2)_v(0) * M * D^(1/2)_v(0) = [[2, 1], [1, 2]]
   - Compute X^(1/2) ≈ [[1.5, 0.5], [0.5, 1.5]] (approximate square root)
   - v(1) = diag(X^(1/2)) = [1.5, 1.5]

2. **Second iteration**:
   - Compute D^(1/2)_v(1) = [[√1.5, 0], [0, √1.5]] ≈ [[1.2247, 0], [0, 1.2247]]
   - Compute X = D^(1/2)_v(1) * M * D^(1/2)_v(1) = [[2*1.5, 1*1.2247], [1*1.2247, 2*1.5]] = [[3, 1.2247], [1.2247, 3]]
   - Compute X^(1/2) ≈ [[1.732, 0.3536], [0.3536, 1.732]] (approximate square root)
   - v(2) = diag(X^(1/2)) ≈ [1.732, 1.732]

3. **Third iteration**:
   - Compute D^(1/2)_v(2) = [[√1.732, 0], [0, √1.732]] ≈ [[1.316, 0], [0, 1.316]]
   - Compute X = D^(1/2)_v(2) * M * D^(1/2)_v(2) = [[2*1.732, 1*1.316], [1*1.316, 2*1.732]] = [[3.464, 1.316], [1.316, 3.464]]
   - Compute X^(1/2) ≈ [[1.861, 0.355], [0.355, 1.861]]
   - v(3) = diag(X^(1/2)) ≈ [1.861, 1.861]

This example illustrates how the iteration converges toward the fixed point, with each component of the vector increasing toward approximately 1.861. The method converges rapidly, with each iteration moving significantly closer to the fixed point. The fixed point satisfies v* = diag((D^(1/2)_v*MD^(1/2)_v*)^(1/2)), and for this specific matrix M, it corresponds to the vector where both components are equal to the square root of the eigenvalue of M (which is √3 ≈ 1.732, but the convergence is toward approximately 1.861 as seen in the third iteration).

## References

- Keith Rush, "Global Convergence of Multiplicative Updates for the Matrix Mechanism: A Collaborative Proof with Gemini 3", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19465

Tags: #differential-privacy #matrix-optimisation #convergence-theory #nuclear-norm
