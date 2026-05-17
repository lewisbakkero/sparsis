---
title: "Scale-Dependent Radial Geometry and Metric Mismatch in Wasserstein Propagation for Reverse Diffusion"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19670"
---

## Executive Summary
This paper introduces a radial stability certificate for reverse diffusion samplers that accounts for both radial width and height of expansive regions in the learned drift. The authors show how a "load-reserve" profile - quantifying width-weighted load before a contractive tail - determines propagation costs more accurately than traditional global stability summaries. For practitioners, this means avoiding the amplification of mismatch in diffusion model pipelines through more precise metric selection.

## Why This Matters for Practitioners
If you're implementing diffusion samplers in production systems with strict quality guarantees, this paper changes how you select error propagation metrics. Traditional approaches using global Lipschitz or dissipativity constants can miss critical radial geometry, causing mismatch amplification during reverse-time propagation. Instead, you should: (1) compute the load-reserve profile (AR, mR) for your model's drift using Gaussian-smoothed denoising geometry, (2) construct the affine-tail metric φR with slope aR, and (3) report quadratic Wasserstein error only at terminal time using the retained tail slope. This approach prevents unnecessary conservatism in error bounds while providing tighter guarantees for production deployment.

## Problem Statement
Current diffusion sampler analyses treat reverse-time propagation as a global stability problem, like measuring a mountain range's height without considering its width. Imagine two valleys: one narrow but deep (high height, small width), another wide but shallow (low height, large width). Traditional methods would rate both as "high barrier" based on height alone, but the wide valley actually requires less effort to cross. Similarly, diffusion samplers' error propagation can be misjudged when radial geometry is oversimplified to a single height metric, causing either overly conservative bounds or unexpected error amplification.

## Proposed Approach
The authors develop a profile-adapted propagation interface where: (1) score-modelling and solver residuals act as forcing inputs, (2) the certified radial profile provides stability, and (3) quadratic error is reported only at terminal time using a retained tail slope. Reflection coupling reduces the problem to radial geometry, where the load-reserve certificate (AR, mR) compiles into an affine-tail metric φR. The propagation bound integrates these elements without rebuilding geometry.

```python
def propagate_error(learned_drift, AR, mR, sigma_min, eta_t, zeta_t):
    """Profile-adapted error propagation for diffusion samplers"""
    phi_R, rho_R, a_R = compile_affine_tail(AR, mR, sigma_min)
    # Propagate using adapted metric
    W_phi = exp(-rho_R * T) * W_phi(epsilon_mu0, mu0) 
    + integrate(from 0 to T) [exp(-rho_R*(T-t)) * (eta_t + zeta_t)]
    # Terminal reporting using retained tail slope
    W2_squared = report_w2_from_affine_tail(W_phi, a_R, tail_info)
    return W2_squared
```

## Key Technical Contributions
The paper's core innovations refine how we measure diffusion model stability:

1. **Load-reserve radial certificate**: The authors define AR = ∫₀ᴿ [−κ(r)]⁺r dr (width-weighted load before reserve) and mR = ess infᵣ≥R κ(r) (contractive tail reserve), which together determine propagation costs. This captures how radial width impacts the slope budget in reflection coupling, unlike height-only summaries that miss this dimension.

2. **Affine-tail metric construction**: The metric φR is explicitly constructed to be affine on [R, ∞) with tail slope aR ≥ exp(−AR/(2σ²) − 1/2). This slope records available large-distance transportation cost after crossing the adverse core, directly linking the load-reserve profile to terminal reporting.

3. **Forcing-residual separation**: The framework cleanly separates score-modelling residuals ηt (from score estimation) and solver residuals ζt (from numerical discretization) as additive inputs to the propagation bound. This allows independent treatment of model and solver errors without conflating their effects.

4. **Radial geometry compilation**: The paper proves that Gaussian-smoothed denoising geometry supplies inverse-radius profiles (κ(r) ≥ α − β/r) for structured windows. This enables concrete computation of (AR, mR) from data geometry (e.g., α = g(s)²τₛ(1−ℓₛ)−f(s), β = g(s)²(τₛDₛ + 2Eₛ)).

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The paper provides theoretical guarantees rather than empirical benchmarks. It reports the propagation bound:
WφR(êμT, μT) ≤ e⁻ρRT WφR(êμ₀, μ₀) + ∫₀ᵀ e⁻ρR(T−t)(ηt + ζt) dt
where ρR ≥ (mR ∧ σ⁻²/R²) exp(−AR/(2σ⁻²)). The authors demonstrate with fixed-height examples that height-only summaries fail to capture the load dependence, but don't provide numerical comparisons against baselines. The absence of empirical results is noted as a limitation (see Section 7).

## Related Work
The paper positions itself against three main approaches: (1) global stability summaries (Lipschitz, Hessian, dissipativity constants), which compress radial geometry into single scalars; (2) reflection coupling methods (Eberle-type), which require fixed semigroups; and (3) modular diffusion analyses. The authors show their certificate is complementary to these, retaining radial geometry details that global summaries discard while avoiding the need for fixed semigroup geometry.

## Limitations
The theoretical results assume scalar-isotropic reverse SDEs and Gaussian smoothing, which may not capture all diffusion model variants. The paper acknowledges that terminal reporting requires tail, moment, or support information that may not be available in all production contexts. The authors don't address computational costs of computing AR or mR from learned drifts, though this would be critical for large-scale deployment.

## Appendix: Worked Example
Consider a Gaussian-smoothed denoiser with α = 0.05 (pull dominates pairwise expansion) and β = 0.02 at noise time s. For σ⁻ = 0.1 (diffusion scale), choosing R = 4β/α = 1.6 gives:

- AR ≤ β²/(2α) = (0.02)²/(2×0.05) = 0.004
- mR ≥ α − β/R = 0.05 − 0.02/1.6 = 0.04875

The propagation rate becomes:
ρR ≥ (mR ∧ σ⁻²/R²) exp(−AR/(2σ⁻²)) = (0.04875 ∧ (0.01/2.56)) exp(−0.004/(2×0.01)) = 0.0039 × 0.82 = 0.0032

The retained tail slope is:
aR ≥ exp(−AR/(2σ⁻²) − 1/2) = exp(−0.004/0.02 − 0.5) = exp(−0.7) ≈ 0.496

For a terminal reporting step using bounded support (diameter D = 10), the W₂² error bound is:
W₂² ≤ (D / aR) × ΓT = (10 / 0.496) × ΓT ≈ 20.16 × ΓT

This shows how the load-reserve profile directly determines the tightness of the final error bound through the retained slope aR.

## References

- Zicheng Lyu, Zengfeng Huang, "Scale-Dependent Radial Geometry and Metric Mismatch in Wasserstein Propagation for Reverse Diffusion", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19670

Tags: #diffusion-models #wasserstein-metric #radial-geometry #propagation-certificate #stability-analysis
