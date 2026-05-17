---
title: "Scalable Learning of Multivariate Distributions via Coresets"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19792"
---

## Executive Summary
This paper introduces the first coreset construction framework for multivariate conditional transformation models (MCTMs), enabling scalable non-parametric density estimation for large-scale multivariate data. Their method ensures log-likelihood accuracy within (1 ± ε) multiplicative error while substantially reducing computational requirements. Practitioners handling multivariate distributional modelling at scale should care because this approach provides a theoretically grounded path to maintain statistical accuracy while cutting computation costs.

## Why This Matters for Practitioners
If you're currently using full-data MCTMs for multivariate density estimation in production systems processing >100K samples, this paper suggests you should immediately evaluate their coreset approach. The authors demonstrate that with just 30 points (a 0.3% subset of 10,000 samples), their method maintains model fit quality while reducing training time by up to 98% on large datasets. For teams running distributional regression on feature-rich datasets where the covariance structure is complex but not fully understood, this represents a practical way to scale without sacrificing statistical rigor. The key engineering action: replace uniform subsampling with their sensitivity-based sampling plus convex hull approximation to avoid catastrophic accuracy drops during data reduction.

## Problem Statement
Today's distributional models struggle with big data like an old-fashioned waterwheel trying to power a data centre: they simply can't process the volume efficiently while maintaining accuracy. Traditional approaches either require excessive computation on full datasets (e.g., 100,000+ samples taking hours to train) or rely on overly simplistic assumptions (like uniform subsampling) that distort the complex dependence structures inherent in real-world multivariate outcomes. This paper's authors observe that "uniform subsampling may lead to infeasible solutions" when dealing with the nuanced correlation patterns found in production data.

## Proposed Approach
The authors' solution involves a two-stage coreset construction: first using leverage score sampling for the quadratic terms of the log-likelihood, then applying a convex hull approximation to stabilize the logarithmic terms. This creates a small representative subset that maintains statistical accuracy within a (1 ± ε) multiplicative error bound. The method is compatible with existing MCTM implementations but requires minimal modification to scale up.

```python
def mctm_coreset_construction(data, epsilon):
    # Step 1: Preprocess data with basis functions
    A, A_prime = preprocess_with_basis_functions(data)
    
    # Step 2: Compute leverage scores for quadratic part
    leverage_scores = compute_leverage_scores(A)
    
    # Step 3: Sample points proportional to (leverage_scores + 1/n)
    coreset_indices = sample_with_sensitivity(leverage_scores, n=len(data))
    
    # Step 4: Add convex hull points for logarithmic stability
    convex_hull_points = compute_convex_hull(A_prime)
    coreset_indices.extend(convex_hull_points)
    
    # Step 5: Return weighted coreset
    return weighted_subset(data, coreset_indices, epsilon)
```

## Key Technical Contributions
The authors' approach introduces several novel mechanisms that distinguish it from prior work:

1. **Dual sampling strategy for mixed loss structure**: The log-likelihood function contains both quadratic (numerically stable) and logarithmic (numerically unstable) components. The paper solves this by separately sampling for each part: leverage scores for the quadratic terms and sensitivity-based sampling for the logarithmic terms, which is novel for MCTMs. This avoids the numerical instability that would arise from treating the entire loss uniformly.

2. **Convex hull approximation for derivative terms**: For the logarithmic terms involving derivatives of the basis functions, they compute the convex hull of the derivative points to avoid extreme values that would cause numerical instability. This geometric approach ensures the logarithmic terms remain stable during optimisation, as they note: "Numerical issues of logarithmic terms in the log-likelihood are eliminated based on prior work by a convex-hull approximation of the derivative a′(y) of transformed data."

3. **Theoretical guarantees with practical bounds**: They provide rigorous bounds on coreset size (O(J²d² ln³(cdJ)c⁶/ε²) while demonstrating practical applicability on real-world datasets. Crucially, they show that the convex hull approximation can be implemented with η-kernels of size O(1/η^(d-1)/2), which matches the theoretical lower bounds they establish.

## Experimental Results
The authors compared their method (ℓ2-hull) against uniform sampling and plain leverage score sampling (ℓ2-only) across 14 different data generation processes. On a 2D simulation with 10,000 samples (coreset size = 30), their method achieved log-likelihood accuracy within 1 ± 0.01 (ε = 0.01) for 12 out of 14 scenarios. For the two scenarios where it didn't significantly outperform (t-copula and skew-t distributions), they noted that "these have a dense convex hull and thus require the size of the convex hull approximation to be increased in order to compensate."

The paper demonstrated 98% reduction in training time on datasets of size 10,000+ while maintaining near-identical model fit (difference in log-likelihood < 0.02) compared to full-data training. The experiments were conducted on a 2021 MacBook Pro (Apple M1 Pro, 16GB RAM), showing that even on modest hardware, the method provides substantial computational gains.

## Related Work
The authors position their work as the first attempt to apply coresets to semi-parametric distributional models (MCTMs), filling a gap in the literature. Prior coreset work focused on parametric models like linear regression or generalised linear models. They note: "Very limited work has considered more complex and flexible non- or semi-parametric distribution models for multivariate outputs." Their contribution connects to transformation models (Hothorn et al., 2014) and normalizing flows (Kobyzev et al., 2021), but extends them to scale efficiently with large datasets while maintaining statistical rigor.

## Limitations
The authors acknowledge that their method shows limitations with heavy-tailed distributions (t-copula and skew-t) when the coreset size is fixed, requiring larger convex hull approximations to maintain accuracy. They don't explore how their method would perform with extremely high-dimensional data (beyond J = 10 dimensions), though they note MCTMs were originally tested on up to ten-dimensional outputs. The paper also doesn't address the impact of correlated features on the coreset selection process, which could be significant in real-world feature-rich datasets.

## Appendix: Worked Example
Consider a 2D dataset with n = 10,000 samples from a Gaussian copula model with J = 2 dimensions. The authors' method would:

1. Apply Bernstein polynomial basis functions to transform the raw data, creating matrices A and A' where A' contains the derivatives of the basis functions.

2. Compute leverage scores for the quadratic terms in the log-likelihood using matrix B (as defined in Section 2).

3. Sample 29 points proportionally to leverage scores plus uniform sampling (total 30 points), ensuring coverage of critical regions.

4. Compute the convex hull of all points in A' and add the extreme points to the coreset (typically 3-5 additional points for 2D data).

5. Train the MCTM on this coreset (30-35 points), weighting the sampled points according to their importance.

For example, if the raw data has 10,000 points representing weather patterns (temperature and humidity), the coreset would contain 30 representative points: 29 selected based on how much they contribute to the log-likelihood (with leverage scores), plus 1-2 extreme points (e.g., unusually high humidity with low temperature) added via convex hull approximation to prevent numerical instability during optimisation. This coreset maintains the essential statistical properties while reducing the training data footprint by 99.7%.

## References

- **Code:** https://github.com/zeyudsai/mctmcoreset
- Zeyu Ding, Katja Ickstadt, Nadja Klein, Alexander Munteanu, Simon Omlor, "Scalable Learning of Multivariate Distributions via Coresets", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19792

Tags: #applied-computing/statistics #computing-methodologies/machine-learning #scalable-ml #coresets #multivariate-distributions
