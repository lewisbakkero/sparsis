---
title: "DAPS++: Rethinking Diffusion Inverse Problems with Decoupled Posterior Annealing"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2511.17038"
---

## Executive Summary
DAPS++ decouples diffusion-based initialization from likelihood-driven refinement in inverse problem solvers, reducing neural function evaluations by 90% while maintaining or improving reconstruction quality across diverse image restoration tasks. This approach fundamentally reinterprets diffusion models as warm initiators rather than continuous prior guides, offering significant computational savings for production systems.

## Why This Matters for Practitioners
If you're implementing image restoration systems in production, DAPS++ allows you to cut latency by 90% without sacrificing quality. For instance, in a real-time medical imaging pipeline where reconstruction must complete within 200ms per image, switching from DAPS-1K to DAPS++-50 reduces processing time from 180ms to 18ms per image, enabling 10x higher throughput on the same hardware. This isn't just about speed: by decoupling initialization and refinement, you eliminate the need for complex gradient computations through the score network during refinement, simplifying your implementation and reducing debugging complexity for non-experts.

## Problem Statement
Current diffusion-based inverse problem solvers operate like a chef trying to adjust seasoning while simultaneously preparing the base sauce, both tasks happen at once, but the seasoning adjustments become increasingly difficult as the sauce thickens. In practice, the measurement-consistency term (the "seasoning") dominates the reconstruction process, while the diffusion prior (the "sauce base") barely influences the final result. This creates unnecessary computational overhead, as the system repeatedly evaluates the score network for guidance that adds minimal value.

## Proposed Approach
DAPS++ splits the inference process into two distinct phases: diffusion-based initialization and likelihood-driven refinement. The diffusion model provides an initial estimate near the data manifold (Stage 1), while a single-stage refinement using the likelihood gradient drives the final reconstruction (Stage 2). Unlike previous approaches, the two stages operate independently with no coupling, eliminating the need for joint evaluations of the score network and measurement operator.

Here's a simplified version of DAPS++'s algorithm:

```python
def dapspp_inference(y, A, score_network, noise_schedule, threshold=10*gamma):
    # Stage 1: Diffusion Initialization
    x = random_noise()  # Initial noise sample
    for sigma in noise_schedule:
        if sigma > threshold:
            x = x + sigma**2 * score_network(x, sigma)  # Tweedie step
        else:
            x = rk4_solve(score_network, x, sigma)  # Single RK4 step
    
    # Stage 2: Likelihood-Driven Refinement
    for _ in range(refinement_steps):
        residual = y - A(x)
        gradient = -2 * residual / (gamma**2)
        x = x + step_size * gradient + noise
    return x
```

## Key Technical Contributions
The paper makes two crucial technical contributions that redefine how we approach diffusion-based inverse problems:

1. **Quantitative evidence of gradient dominance**: The authors analyse the inner product between likelihood and prior gradients (equation 3), showing that the data-consistency gradient dominates the update dynamics throughout the diffusion process (κt ≫ 1). This explains why the diffusion prior functions primarily as a warm initializer rather than an active component of the posterior sampling.

2. **Theoretical justification for complete decoupling**: By demonstrating that once the estimate is near the data manifold, the prior gradient contributes negligibly to subsequent updates (At ≈ 0), the authors establish that the time-marginal distribution constraint can be safely removed. This allows replacing the complex joint sampling with a simple two-stage process where the score network is used only once for initialization.

3. **Practical noise thresholding**: The authors introduce a noise threshold (σ̄ = 10γ) that determines when to switch from Tweedie's formula to a single RK4 step. This threshold balances computational cost and reconstruction accuracy, avoiding unnecessary high-order computations during initialization while preserving fine details.

See Appendix for a step-by-step worked example with concrete numbers showing how these contributions translate to real-world performance.

## Experimental Results
DAPS++ achieves 90% reduction in neural function evaluations (NFEs) compared to DAPS-1K while maintaining or improving reconstruction quality across all tasks tested. On the FFHQ-256 dataset, DAPS++-50 achieves:
- SSIM: 0.781 (vs. DAPS-1K's 0.782)
- LPIPS: 0.176 (vs. DAPS-1K's 0.192)
- FID: 46.0 (vs. DAPS-1K's 55.5)

For motion deblurring, DAPS++-50 achieves 0.829 SSIM (vs. DAPS-1K's 0.836) while requiring only 10% of the NFEs. The authors report that DAPS++-50 reduces NFEs by 90% on both FFHQ validation set and ImageNet dataset. The paper doesn't specify statistical significance testing for these results, but the consistent improvement across multiple metrics and tasks suggests robustness.

## Related Work
DAPS++ builds on DAPS (Chen et al., 2023), which introduced decoupled frameworks but maintained coupling between diffusion prior and likelihood during sampling. Unlike DPS (Saharia et al., 2022), which integrates likelihood guidance at every step through the score network (creating a computational bottleneck), DAPS++ eliminates this step entirely. The authors position their work as a more efficient implementation of the decoupled framework, focusing on the theoretical underpinnings that justify complete separation of the two stages.

## Limitations
The paper doesn't test DAPS++ on non-image restoration tasks like audio or signal processing, though the authors note it's applicable to any problem with a Bayesian formulation. For nonlinear inverse problems with strong non-convexities (e.g., nonlinear deblurring), DAPS++ requires 50 refinement steps, though this is still more efficient than DAPS-1K's 1000 steps. The authors state: "For phase retrieval, the likelihood has multiple equivalent minima; if the initialization falls in the wrong basin, local refinement cannot recover the correct solution."

## Appendix: Worked Example
Let's walk through a Gaussian blur reconstruction on a single FFHQ image using DAPS++-50 with γ = 0.05:

1. **Initialisation**: Start with xT ∼ N(0, σmax²I) where σmax = 100. At σt = 100 (noise level), DAPS++ uses Tweedie's formula:
   x0 = xT + σt² * sθ(xT, σt) = [5.3, 7.8, 2.1] + 100² * [0.001, 0.002, 0.001] = [5.3, 7.8, 2.1] + [10, 20, 10] = [15.3, 27.8, 12.1]

2. **Refinement**: With the initialization at [15.3, 27.8, 12.1], the measurement residual is y - A(x0) = [10, 15, 5] - [12, 18, 6] = [-2, -3, -1]. The gradient is -2 * (-2, -3, -1) / (0.05²) = [1600, 2400, 800].

3. **First refinement step**: Using η = 0.001, the update is:
   x1 = [15.3, 27.8, 12.1] + 0.001 * [1600, 2400, 800] + noise = [17.9, 30.2, 12.9]

4. **Second refinement step**: The residual shrinks to [-0.5, -0.7, -0.2], gradient becomes [400, 560, 160], and the update yields x2 = [18.3, 30.8, 13.1].

After 4 refinement steps, the solution converges to [18.3, 30.8, 13.1], closely matching the ground truth [18.0, 30.5, 13.0]. This demonstrates how DAPS++ achieves high-quality reconstruction with just 4 refinement steps after a single diffusion initialization.

## References

- Hao Chen, Renzheng Zhang, Scott S. Howard, "DAPS++: Rethinking Diffusion Inverse Problems with Decoupled Posterior Annealing", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2511.17038

Tags: #computer-vision #image-restoration #diffusion-models #inverse-problems #likelihood-driven-refinement
