---
title: "ProAR: Probabilistic Autoregressive Modeling for Molecular Dynamics"
category: "AI Applications"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36974"
---

## Executive Summary
ProAR introduces a probabilistic autoregressive framework for molecular dynamics trajectory generation that explicitly models each frame as a multivariate Gaussian distribution. It addresses the key limitation in existing methods that produce fixed-length trajectories through joint denoising of high-dimensional spatiotemporal representations, which conflicts with molecular dynamics' frame-by-frame integration process. For practitioners working on biomolecular simulation tools, ProAR offers flexible trajectory generation of arbitrary length with 7.5% lower reconstruction error and 25.8% higher conformation change accuracy compared to prior state-of-the-art methods.

## Why This Matters for Practitioners
If you're responsible for molecular dynamics simulation pipelines in drug discovery or protein engineering, ProAR directly solves two pain points in your current workflow. First, you've been constrained by fixed-length trajectory generation that requires generating entire sequences at once, leading to wasted computational resources when shorter or longer trajectories are needed, ProAR's autoregressive design lets you generate exactly the trajectory length you require with no extra computation. Second, existing methods fail to capture the time-dependent conformational diversity that's critical for understanding protein function; ProAR's probabilistic modelling of each frame as a multivariate Gaussian distribution provides more accurate representations of protein motion patterns, as evidenced by its 25.8% improvement in conformation change accuracy. You can immediately integrate ProAR into your simulation pipeline to replace fixed-length trajectory generators, using the ATLAS dataset (1300 proteins with three 100ns trajectories each) for benchmarking without needing to retrain from scratch.

## Problem Statement
Imagine trying to reconstruct a film sequence by randomly selecting frames from a large database and pasting them together, this is how current deep learning approaches generate molecular dynamics trajectories. They jointly denoise all frames simultaneously, ignoring the natural progression of time in biological processes. Just as a film editor needs to sequence frames in chronological order to convey motion, molecular dynamics simulations must integrate step-by-step. Existing methods produce "stitched-together" trajectories that fail to capture time-dependent conformational changes, like trying to understand a dancer's movement by randomly assembling frames from different dance sequences instead of watching the full performance.

## Proposed Approach
ProAR uses a dual-network system that models each trajectory frame as a multivariate Gaussian distribution, allowing probabilistic sampling that captures conformational uncertainty. The framework combines an interpolator that predicts intermediate states between observed frames and a forecaster that refines future conformations, alternating between them through an anti-drifting sampling strategy to prevent error accumulation during long trajectory generation. This approach enables flexible, variable-length trajectory generation that aligns with molecular dynamics' sequential nature.

```python
def generate_trajectory(initial_frame, target_length):
    trajectory = [initial_frame]
    # Start with forecaster prediction
    for i in range(0, target_length - 1, 2):
        # Forecaster predicts the next frame
        next_frame = forecaster(trajectory[-1], trajectory[-1], 0)
        trajectory.append(next_frame)
        # Interpolator predicts intermediate frames
        interpolated_frame = interpolator(trajectory[-2], trajectory[-1], 1)
        trajectory.append(interpolated_frame)
    # Ensure final trajectory length matches target
    if len(trajectory) > target_length:
        trajectory = trajectory[:target_length]
    return trajectory
```

## Key Technical Contributions
ProAR introduces novel mechanisms that address the limitations of previous approaches through precise implementation choices:

1. **Multivariate Gaussian frame modelling** - Instead of predicting a single deterministic frame, ProAR predicts both the mean and covariance of a multivariate Gaussian distribution for each frame, enabling probabilistic sampling that captures conformational uncertainty. The covariance matrix is parameterized through its Cholesky factor to maintain positive definiteness and reduce computational complexity, with sparsity imposed to reflect localized correlations in protein dynamics.

2. **Dual-network anti-drifting sampling** - The interpolator and forecaster alternate during sampling: the forecaster first predicts a future frame, then the interpolator refines intermediate frames between the initial frame and the forecasted frame. This alternation prevents error accumulation during long trajectory generation, as the forecaster's predictions improve iteratively as the context moves closer in time to the target frame.

3. **Structured noise modelling** - The model explicitly incorporates structured noise through the covariance prediction, rather than relying on random noise injection. The interpolator's objective combines a deterministic structural loss (adapted from AlphaFold2) with the negative log-likelihood of the predicted distribution, ensuring physically plausible structures while modelling variability beyond the ensemble mean.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
In trajectory generation tasks on ATLAS (1300 proteins, three 100ns trajectories each), ProAR achieved a 7.5% reduction in reconstruction RMSE (R250: 3.529 vs MDGEN's 3.813) and an average 25.8% improvement in conformation change accuracy compared to MDGEN, the state-of-the-art baselines. For conformation sampling, ProAR performed comparably to specialized time-independent models like AlphaFlow (Jing, Berger, and Jaakkola 2024) and CONFDIFF (Wang et al. 2024), achieving the best results on 5 out of 7 metrics. The paper does not report statistical significance testing for the performance improvements, but the specific numbers are provided in Tables 1 and 2 for direct comparison.

## Related Work
ProAR positions itself as a direct evolution of trajectory generation methods that previously relied on joint denoising of high-dimensional spatiotemporal representations. It differs from MDGEN (Jing et al. 2024) and AlphaFolding (Cheng et al. 2025), which use non-autoregressive, fixed-length approaches, by embracing the autoregressive nature of molecular dynamics. The method also advances beyond deterministic autoregressive frameworks like GST (Li et al. 2025) by introducing probabilistic modelling of each frame as a multivariate Gaussian distribution, capturing the inherent stochasticity of molecular dynamics simulations.

## Limitations
The paper does not explicitly discuss limitations beyond the excluded AlphaFolding comparison due to training challenges on larger proteins (>256 residues). The evaluation focuses solely on the ATLAS dataset, which may not generalise to more complex biological systems. The authors acknowledge that the anti-drifting sampling strategy requires careful tuning of the horizon length (h=6 in their experiments), and the paper doesn't explore the sensitivity of this parameter across diverse protein sizes. The probabilistic modelling approach may introduce additional computational overhead compared to deterministic methods, though the paper doesn't quantify this trade-off.

## Appendix: Worked Example
Let's walk through a concrete example of how ProAR generates a 4-frame trajectory using the anti-drifting sampling strategy. We begin with an initial frame at timestep 0 (x0), with the goal of generating frames up to timestep 3 (x3).

1. **Initial frame**: x0 = [N, Cα, C] backbone coordinates for a protein with 50 residues (size: 50 × 3 × 3 = 450 dimensions)
2. **First forecaster step**: The forecaster Fθ predicts x3 from x0 using the model's initial parameters: Fθ(x0, x0, 0) = x̂3 (size: 450 dimensions)
3. **First interpolator step**: The interpolator Iϕ predicts the intermediate frame x1 from x0 and x̂3: Iϕ(x0, x̂3, 1) = x̂1 (size: 450 dimensions), which is modelled as a multivariate Gaussian N(μ̂1, Σ̂1)
4. **Second forecaster step**: The forecaster refines x3 using the interpolated frame x̂1: Fθ(x̂1, x0, 1) = x̂3 (size: 450 dimensions), which is closer to the ground truth x3
5. **Second interpolator step**: The interpolator predicts the second intermediate frame x2 using x̂1 and x̂3: Iϕ(x̂1, x̂3, 1) = x̂2 (size: 450 dimensions)

The final trajectory consists of the frames: [x0, x̂1, x̂2, x̂3]. Each interpolated frame (x̂1, x̂2) is sampled from the predicted multivariate Gaussian distribution, capturing conformational uncertainty. The mean and covariance for x̂1 are predicted by the interpolator's Iμ,ϕ and IΣ,ϕ branches, while the forecaster's corruption, refinement process ensures structural fidelity. The trajectory generation process continues until the desired length is achieved, with the anti-drifting sampling strategy preventing error accumulation from one frame to the next.


## References

- Kaiwen Cheng, Yutian Liu, Zhiwei Nie, Mujie Lin, Yanzhen Hou, Yiheng Tao, "ProAR: Probabilistic Autoregressive Modeling for Molecular Dynamics", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36974
