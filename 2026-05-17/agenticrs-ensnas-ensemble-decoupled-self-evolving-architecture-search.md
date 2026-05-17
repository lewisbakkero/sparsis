---
title: "AgenticRS-EnsNAS: Ensemble-Decoupled Self-Evolving Architecture Search"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.20014"
---

## Executive Summary
AgenticRS-EnsNAS introduces Ensemble-Decoupled Architecture Search, a framework that reduces neural architecture search validation costs from O(M) to O(1) per candidate by leveraging ensemble theory. This enables industrial-scale architecture iteration in production recommender systems where M=50-200 ensemble members are standard, transforming a resource-prohibitive process into a scalable optimisation loop with theoretical guarantees.

## Why This Matters for Practitioners
If you're maintaining production recommender systems with ensemble-based deployments (common in e-commerce and ad platforms), this paper directly solves your most painful bottleneck: the prohibitive cost of architecture iteration. Instead of waiting weeks to test a single candidate architecture (requiring 50-200 full model trainings), you can now validate candidates in constant time, enabling you to iterate hundreds of times more frequently within the same resource budget. Implement this framework to confidently experiment with architectural improvements to boost accuracy without requiring additional GPU resources or slowing down deployment cycles.

## Problem Statement
Imagine needing to evaluate a new car model before selling it to customers: You can't just test one prototype; you have to build 100 identical prototypes, drive them all on the same test track, and compare their performance before making a decision. This is essentially what industrial neural architecture search (NAS) currently requires for each candidate architecture, where M=50-200 ensemble members must be fully trained and evaluated before any candidate can be considered for deployment. This cost barrier makes regular architecture iteration impractical, trapping teams in suboptimal models.

## Proposed Approach
EnsNAS reframes NAS as a theoretically-grounded optimisation process by leveraging ensemble theory to predict system-level performance from single-learner evaluation. The framework establishes a monotonic improvement condition that guarantees reduced ensemble error without requiring full ensemble training during search. It unifies three solution strategies based on the continuity and tractability of the architecture pipeline: closed-form optimisation for continuous architectures, constrained differentiable optimisation for intractable continuous architectures, and LLM-driven search with iterative monotonic acceptance for discrete architectures.

```python
def ensemble_decoupled_search(current_ensemble, M, search_budget):
    πbest = current_ensemble.architecture
    stats = {
        'ρ': current_ensemble.correlation,
        'σ2': current_ensemble.variance,
        'E': current_ensemble.error
    }
    
    for _ in range(search_budget):
        π_candidate = llm_generate_diverse_architecture(πbest)
        model1, model2 = train_dual_proxies(π_candidate)
        E_candidate = average_error(model1, model2)
        σ2_candidate = variance(model1, model2)
        ρ_candidate = correlation(model1, model2)
        
        if ρ_candidate < stats['ρ'] - (M / (M - 1)) * (stats['E'] - E_candidate) / σ2_candidate:
            πbest = π_candidate
            stats = {
                'ρ': ρ_candidate,
                'σ2': σ2_candidate,
                'E': E_candidate
            }
    
    return train_full_ensemble(πbest, M)
```

## Key Technical Contributions
EnsNAS's key innovations lie in its theoretical foundation and practical implementation of the monotonic improvement condition. Here's how it works at the implementation level:

1. **The monotonic improvement condition**: This framework replaces the need for full ensemble training with three architecture-level properties (ΔE(π), ρ(π), σ²(π)) that can be estimated from training just two proxy models. The condition ρ(π) < ρ(πold) - (M/(M-1)) * (ΔE(π)/σ²(π)) provides a verifiable guarantee that a candidate architecture will improve system performance without requiring full ensemble training, reducing validation cost from O(M) to O(1).

2. **Dual gain decomposition**: The framework reveals two orthogonal improvement mechanisms through closed-form analysis: base diversity gain (inherent from ρ0 < 1) and accuracy gain (from architectural optimisation). This provides concrete engineering guidance, such as the optimal feature retention ratio in CTR prediction: α* = 1 + (σ²_base(M-1)k2)/(2k1M), where k1 and k2 are constants derived from empirical observations on industrial data.

3. **Iterative monotonic acceptance for LLM-driven search**: Unlike traditional LLM-NAS approaches that rely solely on validation accuracy, EnsNAS uses the monotonic condition to filter candidates before full deployment. Each accepted candidate becomes the new baseline, raising the bar for subsequent candidates through transitive improvement (Proposition 4.1), ensuring that every candidate in the search trajectory improves over the previous iteration.

4. **Cost-decoupling principle**: EnsNAS fundamentally decouples search cost from ensemble size M. While traditional NAS scales linearly with M, EnsNAS maintains constant cost per candidate (O(1)), with full ensemble training paid only once for the winning candidate. This enables 90x more candidate exploration within the same resource budget (e.g., 1,000 candidates instead of 10 candidates for M=100).

## Experimental Results
Preliminary results from internal pilot studies on Criteo data (M=50) show:
- Ensemble error follows a U-shaped curve in feature retention ratio α, with empirical optimum within 5% of theoretical α* from Eq. (15)
- Search cost scales linearly with candidate count (Ntrials) and independently of M, while traditional NAS scales with Ntrials × M
- The monotonic condition correctly accepts ~85% of improving candidates and rejects ~70% of degrading candidates in early trials

The authors plan comprehensive validation on Criteo and Avazu datasets with M ∈ {10, 50, 100}, expected to demonstrate ~M× speedup in search phase. The journal submission (target Q2 2026) will include full experimental results with statistical significance testing.

## Related Work
EnsNAS builds on ensemble theory foundations (Krogh & Vedelsby, 1995; Zhou, 2012) but fundamentally differs from existing NAS approaches:
- Zero-cost proxies (Abdelfattah et al., 2021) exhibit fidelity gaps in complex CTR tasks where gradient-based proxies fail to correlate with final ensemble performance
- Weight-sharing methods (ENAS, DARTS) introduce optimisation bias and restrict search space to subgraphs of a super-network
- LLM/RL approaches for NAS lack theoretical stopping criteria, leading to inefficient exploration

Unlike these, EnsNAS provides theoretical guarantees for validation cost reduction without proxy metrics or weight sharing.

## Limitations
The authors acknowledge that Theorem 3.1 assumes ensemble members are independent realizations of the same architecture, this holds exactly for feature bagging but only approximately for general architecture search. Future work will extend the theory to heterogeneous ensembles with bounded deviation analysis.

The framework also relies on zero-cost proxies for ΔE(π) estimation, which require calibration. The authors plan to derive confidence bounds and failure mode analysis for proxy-based decisions.

## Appendix: Worked Example
Let's walk through the feature bagging case study with concrete numbers. In a CTR prediction task, consider an existing ensemble with M = 100 members (πold) where:
- ρ(πold) = 0.8 (average correlation between members)
- σ²(πold) = 0.05 (variance of predictions)
- E(πold) = 0.20 (average error)

We're evaluating a candidate architecture π with:
- ρ(π) = 0.65 (lower correlation, indicating higher diversity)
- σ²(π) = 0.045 (slightly lower variance)
- E(π) = 0.18 (improved accuracy)

First, calculate ΔE(π) = E(π) - E(πold) = 0.18 - 0.20 = -0.02 (improvement of 0.02)

Then, check the monotonic improvement condition:
ρ(π) < ρ(πold) - (M/(M-1)) * (ΔE(π)/σ²(π))
0.65 < 0.8 - (100/99) * (-0.02/0.045)
0.65 < 0.8 - (1.01) * (-0.444)
0.65 < 0.8 + 0.449
0.65 < 1.249

The condition holds (0.65 < 1.249), so this candidate architecture π is guaranteed to reduce ensemble error compared to πold. This means we can deploy this architecture with confidence without needing to train all 100 ensemble members for validation, saving approximately 99% of the validation cost.

## References

- Yun Chen, Moyu Zhang, Jinxin Hu, Yu Zhang, Xiaoyi Zeng, "AgenticRS-EnsNAS: Ensemble-Decoupled Self-Evolving Architecture Search", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20014

Tags: #large-scale-ml #ai-applications
