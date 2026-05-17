---
title: "Heavy-Tailed and Long-Range Dependent Noise in Stochastic Approximation: A Finite-Time Analysis"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19648"
---

## Executive Summary
This paper establishes the first finite-time convergence guarantees for stochastic approximation under heavy-tailed and long-range dependent noise models, which commonly appear in real-world applications like finance and communications. By introducing a noise-averaging technique that regularises noise without modifying the iteration, the authors provide explicit error decay rates that quantify how heavy tails and temporal dependence slow convergence.

## Why This Matters for Practitioners
If you're implementing stochastic optimisation in production systems (like reinforcement learning or gradient-based training) with noisy data streams, this paper explains why your convergence might be slower than expected when dealing with heavy-tailed noise (e.g., financial market fluctuations) or long-range dependent noise (e.g., network traffic patterns). For instance, if your gradient noise follows a Pareto distribution with shape parameter α = 1.3 (heavier tails than α = 1.8), you should expect your optimisation error to decay as O(k^-0.3) rather than the O(k^-1) seen with Gaussian noise, meaning convergence will be significantly slower in practice. The authors' noise-averaging technique provides a theoretical foundation for understanding these slowdowns without requiring algorithm modifications, so you should focus on noise characterisation rather than immediately applying gradient clipping for heavy-tailed noise. When deploying in time-series applications with persistent correlations (e.g., network monitoring systems), be prepared to increase your iteration count by a factor of O(k^δ) where δ = 2-2H for fractional Gaussian noise with Hurst index H.

## Problem Statement
Traditional stochastic optimisation assumes "well-behaved" noise, like i.i.d. Gaussian gradients, that quickly averages out over iterations. But in reality, many production systems encounter "rough noise" that behaves more like a financial market's sudden crashes (heavy tails) or network traffic's persistent congestion patterns (long-range dependence), where large fluctuations don't average out quickly. Imagine trying to build a stable elevator system with random, unpredictable floor requests that occasionally include extreme values, your elevator would constantly overreact to these rare events, leading to inefficient operation and potentially dangerous behaviour.

## Proposed Approach
The authors develop a framework for stochastic approximation under non-classical noise models by introducing a noise-averaging technique that regularises the noise without modifying the underlying iteration. This transforms the problem into one where the randomness appears only through an averaged term, allowing them to establish finite-time moment bounds for both heavy-tailed and long-range dependent noise.

Here's the core algorithmic idea presented in the paper:

```python
def stochastic_approximation(F, x0, beta, eta, p, delta):
    """
    Implements stochastic approximation under heavy-tailed or LRD noise.
    
    Args:
        F: Strongly monotone operator
        x0: Initial iterate
        beta: Step size sequence (beta_k = beta/(k + K0))
        eta: Noise sequence (heavy-tailed or LRD)
        p: For heavy-tailed noise (1 < p < 2)
        delta: For LRD noise (0 < delta < 1)
    
    Returns:
        Convergence rates and error bounds
    """
    x = x0
    for k in range(max_iterations):
        # Standard SA iteration
        x = x - beta/(k + K0) * (F(x) + eta[k])
        
        # The paper shows that noise averaging regularises the error
        # without modifying the iteration (this is a proof technique)
        # The noise-averaged sequence Uk is introduced for analysis
        # but not used in the actual algorithm
    return convergence_rates
```

## Key Technical Contributions
The authors' core innovation lies in transforming the stochastic approximation iteration to separate noise effects through an auxiliary sequence and noise averaging:

1. **Noise-Averaged Auxiliary Iterates**: Instead of directly analysing the standard SA recursion, the authors introduce an auxiliary sequence zk = xk - Uk where Uk is a noise-averaged sequence defined as Uk+1 = (1 - βk)Uk + βkηk. This transforms the error analysis to focus on E[∥zk - x*∥^q] while bounding the noise contribution through E[∥Uk∥^q], which has improved moment properties even for heavy-tailed or LRD noise.

2. **Tail-Adaptive Moment Analysis**: For heavy-tailed noise with only finite p-th moments (1 < p < 2), the authors derive bounds on the p-th moment of the error, showing it decays as O(k^-(p-1)). This requires a tailored analysis of the averaged noise sequence Uk, where they prove E[∥Uk∥^p] ≤ 4ζ^pσ^pβ^(p-1)/k^(p-1), enabling them to establish the precise error decay rate without modifying the algorithm.

3. **LRD Noise Correlation Handling**: For long-range dependent noise with autocovariance decaying as O(h^-δ), the authors establish that the mean square error decays as O(k^-δ) by analysing the second moment of the averaged noise sequence Uk. They prove E[∥Uk∥^2] ≤ 6ζ^2σ^2β/(1-δ)k^δ, which directly translates to the error bound through their auxiliary iterate transformation.

4. **Non-Interventional Analysis**: Crucially, the noise-averaging technique is purely a proof device, no algorithm modification is required. The authors emphasise that "analysing this averaged noise sequence is just a proof technique and not a modification to the iteration," which is critical for production deployments where changing the core optimisation algorithm might be infeasible.

## Experimental Results
The paper provides numerical experiments that corroborate their theoretical guarantees. For heavy-tailed noise, they use centered Pareto noise with shape parameter α ∈ (1, 2) and scale parameter 1. The results show that smaller values of α (heavier tails) lead to slower convergence: for α = 1.3, the mean square error decays as O(k^-0.3), while for α = 1.8, it decays as O(k^-0.8). The authors plot the ℓ2 error for single runs and averaged results over 1000 independent runs (Figure 1).

For LRD noise, they use fractional Gaussian noise with Hurst parameter H ∈ (0.5, 1), where H = 0.5 corresponds to standard Gaussian white noise. Larger values of H (stronger temporal dependence) lead to slower convergence: for H = 0.9, the mean square error decays as O(k^-0.2) (since δ = 2-2H = 0.2), while for H = 0.5, it decays as O(k^-1). The authors plot these results in Figure 2, showing the error curves for different H values.

The experiments were conducted on a strongly convex function g(x) = ½∥Ax - b∥² + ∑ᵢ ϕ₁(xᵢ), where A ∈ ℝ⁶⁰ˣ³⁰ (random Gaussian matrix) and b ∈ ℝ³⁰ (random Gaussian vector), with stepsize βₖ = 1/(k+1). The paper doesn't specify statistical significance tests, but the results are averaged over 1000 independent runs with 10%-90% quantile bands to illustrate variability.

## Related Work
The authors position their work between classical SA theory (which assumes martingale difference or Markov noise with bounded moments) and recent work that has focused on asymptotic convergence under heavy-tailed or LRD noise. They explicitly note that "the only existing work on general SA under heavy-tailed or LRD noise focuses on asymptotic convergence guarantees" and that their work provides the first finite-time guarantees for both noise models. For SGD under heavy-tailed noise, they acknowledge that "there is a substantial body of work that analyses vanilla SGD under heavy-tailed gradient noise," but explain that their framework is more general as it applies to "strongly monotone operators" rather than specific gradient properties.

## Limitations
The paper focuses on theoretical analysis rather than practical implementation challenges. The authors don't test their framework on real-world production systems with actual heavy-tailed or LRD noise, experiments are conducted on synthetic noise models with specific parameters. Additionally, the noise-averaging technique is purely a proof device, so while it provides theoretical insights, it doesn't directly suggest algorithmic modifications for practitioners. The paper doesn't address how to characterise noise in production systems to determine whether it's heavy-tailed or LRD, which would be necessary for practical application.

## Appendix: Worked Example
Let's walk through the error decay for a specific case of heavy-tailed noise with α = 1.5 (so p = 1.5 in the analysis). The paper states that the p-th moment of error decays as O(k^-(p-1)) = O(k^-0.5).

Suppose we have an optimisation problem where the target error is 0.1. The paper shows that for α = 1.5 (p = 1.5), the error decays as E[∥xₖ - x*∥^1.5] ≤ C₆/(k + K₀)^0.5.

Starting from k = 100, suppose we have an initial error bound of C₆/(100 + K₀)^0.5 = 0.1. Setting C₆ = 1 for simplicity, we get:
1/(100 + K₀)^0.5 = 0.1 → (100 + K₀)^0.5 = 10 → 100 + K₀ = 100 → K₀ = 0

At k = 1000, the error bound becomes:
1/(1000 + 0)^0.5 = 1/31.62 ≈ 0.0316

So for α = 1.5, the error decays from 0.1 at k = 100 to approximately 0.03 at k = 1000, which is a 70% reduction over 900 iterations.

For comparison, with Gaussian noise (α = 2, p = 2), the error would decay as O(k^-1). At k = 100, E[∥xₖ - x*∥²] ≤ 0.1, so E[∥xₖ - x*∥] ≤ √0.1 ≈ 0.316. At k = 1000, E[∥xₖ - x*∥²] ≤ 0.01, so E[∥xₖ - x*∥] ≤ 0.1. This shows that the heavy-tailed noise with α = 1.5 leads to significantly slower convergence than the Gaussian case.

## References

- Siddharth Chandak, Anuj Yadav, Ayfer Ozgur, Nicholas Bambos, "Heavy-Tailed and Long-Range Dependent Noise in Stochastic Approximation: A Finite-Time Analysis", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19648

Tags: #optimisation #stochastic-algorithms #heavy-tailed-distributions #long-range-dependence #convergence-analysis
