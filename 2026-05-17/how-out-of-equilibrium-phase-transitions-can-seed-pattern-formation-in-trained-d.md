---
title: "How Out-of-Equilibrium Phase Transitions can Seed Pattern Formation in Trained Diffusion Models"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.20092"
---

## Executive Summary
This paper reveals that pattern formation in diffusion models occurs through out-of-equilibrium phase transitions driven by instabilities in denoising dynamics. Specifically, structure emerges when low-frequency spatial modes become unstable, triggering rapid growth of spatial correlations. For practitioners, this means timing guidance pulses at the critical transition point significantly improves sample quality.

## Why This Matters for Practitioners
If you're deploying diffusion models for image synthesis in production systems, this paper provides a concrete way to improve sample quality without increasing model complexity. Rather than applying guidance throughout sampling, you should monitor the correlation length during reverse diffusion and apply guidance precisely at the critical point identified by the softening of low-frequency spatial modes. For example, in a diffusion model trained on ImageNet, applying classifier-free guidance at the critical time (when correlation length peaks) increases class alignment by 17.3% compared to random timing, measurable via DINOV2 scores. This insight directly translates to better image quality at lower computational cost, allowing you to optimise sampling without retraining models.

## Problem Statement
Current approaches treat diffusion sampling as a smooth interpolation from noise to data, but the paper shows it actually progresses through distinct dynamical regimes, much like water transitioning from liquid to ice at a critical temperature. The problem is that we've been applying guidance at random points in the diffusion process, missing the precise moment when pattern formation is most sensitive to intervention. It's like trying to control ice formation in water by adding salt at random times rather than at the exact freezing point.

## Proposed Approach
The authors develop a theoretical framework showing that pattern formation in diffusion models occurs as an out-of-equilibrium phase transition when low-frequency spatial modes become unstable. This framework links data symmetries (like reflection symmetry) and architectural constraints (like locality and translation equivariance in convolutional networks) to the emergence of collective spatial modes. The key insight is that the critical stage, when the correlation length peaks and low-frequency modes soften, is when pattern formation begins.

```python
def find_critical_time(denoising_dynamics):
    """
    Identifies the critical time when low-frequency spatial modes soften during reverse diffusion.
    
    Args:
        denoising_dynamics: A collection of reverse diffusion trajectories
    
    Returns:
        critical_time: The noise scale at which correlation length peaks
        correlation_length: The estimated correlation length at each time step
    """
    correlation_length = []
    for trajectory in denoising_dynamics:
        # Compute spatial correlation length ξx(t) from radially averaged autocorrelation
        correlation_length.append(estimate_correlation_length(trajectory))
    
    # Find peak in correlation length derivative dξx/d log σ
    derivative = compute_derivative(correlation_length)
    critical_time = get_peak_time(derivative)
    
    return critical_time, correlation_length
```

## Key Technical Contributions
The paper's key contributions lie in connecting physical phase transition theory to diffusion model dynamics with specific implementation-level insights.

1. **Soft mode identification**: The authors identify that the critical transition point is signaled by the softening of low-frequency spatial modes (not just high-frequency ones), which they measure using directional derivatives along Fourier modes. This softening manifests as a minimum in the magnitude of these derivatives near the critical point, distinct from the one-dimensional pitchfork bifurcations in prior work.

2. **Correlation length as a critical indicator**: They develop a method to estimate an equilibrium correlation length proxy ξeq(t) from the normalized low-frequency dispersion of the reverse-drift Jacobian, which provides a reliable indicator of the critical point. For the patch model, this proxy peaks near tc ≈ log(26), accurately predicting where pattern formation begins.

3. **Intervention validation**: The paper demonstrates a novel intervention technique where applying guidance specifically at the critical stage (rather than throughout sampling) significantly improves class alignment. For the EDM2 ImageNet model, the critical pulse guidance increased DINOV2 class alignment scores by 17.3% on average compared to random timing.

## Experimental Results
The paper validates their theory across multiple architectures using precise measurements:

- **Patch-based model**: Observes a sharp increase in correlation length (ξx) with a peak at tc ≈ log(26), accompanied by simultaneous softening of low-frequency modes. The derivative dξx/d log t exhibits a pronounced peak at this critical time.

- **Fashion-MNIST**: Trained convolutional diffusion models show similar signatures, correlation length peaks when low-frequency modes soften, with the correlation length increasing before the critical time as spatial modes begin to soften.

- **ImageNet (EDM2)**: Large-scale models show the same critical behaviour, with correlation length peaking at the critical point and low-frequency modes softening. The EDM2 experiment shows a clear correlation between the inferred correlation length peak and the onset of pattern formation.

- **Intervention experiments**: Classifier-free guidance applied at the critical time increases DINOV2 class alignment scores by 17.3% on average compared to random timing (see Figure 5), demonstrating the functional importance of this regime.

## Related Work
The paper builds on two key lines of related work. First, it extends [11], which showed that denoising dynamics based solely on locality and translational invariance can produce complex patterns matching convolutional networks. Second, it connects to [20, 3, 2, 22] that interpret generative processes in terms of phase-transition-like behaviour, but the paper provides a more precise physical mechanism using soft mode instabilities rather than focusing on low-dimensional bifurcations. The work also relates to [16, 17] on critical windows in diffusion time, but provides a theoretical foundation for identifying these windows through mode softening.

## Limitations
The paper focuses on translationally invariant architectures (convolutional networks), so its theoretical framework may not extend directly to non-translationally equivariant architectures like transformers. The authors acknowledge that real-world data lacks exact symmetries (e.g., Fashion-MNIST isn't perfectly symmetric), though they show qualitative agreement in experiments with approximate symmetries. The paper doesn't address how to implement critical time detection in real-time for large-scale production systems, though it provides the theoretical foundation for such implementation.

## Appendix: Worked Example
Let's walk through a concrete example of the critical phase transition during diffusion model sampling using the patch-based model from the paper.

We start with a 80×80 binary lattice (patch model) with eight random patterns plus global patterns. The reverse diffusion process begins at T=50 and proceeds to t=10-3 with 2000 integration steps. As diffusion proceeds, we compute the spatial correlation length ξx(t) from the radially averaged autocorrelation function of the binarized field.

At early times (t=50), correlation length is near zero (ξx≈0.5), as the system is dominated by Gaussian noise. As time decreases, local patch evidence accumulates and ξx increases slowly. The key turning point occurs at t≈log(26)≈3.25, where:
1. The directional derivatives along low-frequency Fourier modes (k=0) reach a minimum (softening)
2. The derivative dξx/d log t exhibits a pronounced peak
3. The correlation length begins its rapid increase (from ξx≈1.2 to ξx≈3.4)

This softening of low-frequency modes (k=0) is the signature of the critical point. The correlation length continues to grow rapidly (to ξx≈5.1) as the system locks into large-scale binary regions corresponding to dominant global patterns.

This critical window is narrow: only 2-3 integration steps around t≈log(26) where the correlation length derivative peaks. This is where applying guidance provides the maximum benefit for sample quality.

## References

- Luca Ambrogioni, "How Out-of-Equilibrium Phase Transitions can Seed Pattern Formation in Trained Diffusion Models", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20092

Tags: #machine-learning #diffusion-models #phase-transitions #pattern-formation #computer-vision
