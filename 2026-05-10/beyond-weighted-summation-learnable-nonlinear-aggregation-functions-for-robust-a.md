---
title: "Beyond Weighted Summation: Learnable Nonlinear Aggregation Functions for Robust Artificial Neurons"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19344"
---

## Executive Summary
This paper introduces learnable nonlinear aggregation functions for artificial neurons that replace the default weighted sum, significantly improving robustness to input noise without sacrificing trainability. The key innovation is hybrid neurons that dynamically blend linear and nonlinear aggregation paths through a learnable parameter, achieving up to 99.1% robustness on noisy CIFAR-10 compared to 89.0% for standard baselines. Engineers should care because this requires minimal architectural changes to existing systems while directly addressing a fundamental vulnerability in neural network design.

## Why This Matters for Practitioners
If you're deploying computer vision systems in real-world environments where input noise is unavoidable, such as medical imaging from low-quality sensors, satellite imagery with atmospheric interference, or edge devices with poor input quality, this paper provides a plug-and-play solution. You should replace standard linear aggregation in your classifier heads with hybrid neurons (F-Mean or three-way) and monitor the learned parameter *p* (converging to ~0.45). This reduces performance drops under noise by 9.1 percentage points on CIFAR-10, with negligible computational overhead. Crucially, the approach works without retraining entire networks, making it ideal for retrofitting legacy systems. For instance, in a medical image classifier, this would mean maintaining 87.6% clean accuracy while improving robustness to 99.1% on noisy test data, critical for reliability in clinical settings.

## Problem Statement
Current neural networks treat neuron aggregation as a fixed, mechanical process, like using a single ruler to measure everything from a millimetre to a kilometre. This forces models to adapt to noise through brittle post-hoc techniques (e.g., data augmentation), rather than fundamentally redesigning the neuron to handle noise. Imagine a committee where every member’s vote counts equally, regardless of whether they’re an expert or a random passerby: a single outlier vote can derail the entire decision. Similarly, standard neurons amplify noise through mean-based aggregation, causing catastrophic failures in noisy inputs.

## Proposed Approach
The authors replace the fixed weighted sum in neurons with two learnable nonlinear aggregation mechanisms: F-Mean (power-weighted) and Gaussian Support (distance-aware). These are embedded in hybrid neurons that blend linear and nonlinear paths via a learnable scalar *α*. The architecture requires no changes to existing network layers, only swapping the aggregation function in neurons. Crucially, the hybrid design ensures optimisation stability by allowing networks to fall back to linear aggregation during unstable training phases.

```python
def f_mean_aggregation(inputs, weights, p):
    z = [w * x for w, x in zip(weights, inputs)]  # scaled inputs
    z_pos = [log(1 + exp(z_i)) for z_i in z]      # softplus transformation
    weights = [z_pos[i]**p / sum(z_pos[j]**p for j in range(len(z_pos))) for i in range(len(z))]
    return sum(weights[i] * z[i] for i in range(len(weights)))
```

## Key Technical Contributions
The paper’s innovations lie in how it makes nonlinear aggregation both practical and interpretable for real-world systems.

1.  **Interpretable sub-linear parameterisation**: Unlike prior work that fixed exponents (e.g., Yadav et al.’s generalised mean), *p* is learned during training (converging to 0.43, 0.50), automatically suppressing noisy activations *without* explicit regularisation. This emerges from gradient-based optimisation alone, meaning networks discover robust aggregation strategies autonomously.

2.  **Hybrid blending as a stabiliser**: The learnable *α* (0.69, 0.79) enables networks to dynamically balance linear and nonlinear paths. Crucially, *α* starts at 0.5 (equal weight) and shifts toward nonlinear aggregation *only* when beneficial. This prevents optimisation instability, unlike pure nonlinear approaches that often fail to converge.

3.  **Gaussian Support with dimensionality reduction**: The Gaussian Support neuron’s pairwise distance computation (O(n²)) is made tractable via a projection layer before aggregation. This avoids the computational overhead of prior methods (e.g., Pellegrini et al.’s iterative solvers), allowing direct integration into CNN classifier heads.

## Experimental Results
On CIFAR-10 with additive Gaussian noise (σ=0.15), the three-way hybrid (linear + F-Mean + Gaussian) achieved a robustness score *ρ* = 0.991 (vs. 0.890 for standard CNN), representing a 10.1 percentage point improvement. Clean accuracy gains were modest but consistent: F-Mean hybrid CNN reached 87.61% (vs. 87.33% baseline), with statistical significance confirmed through multiple training runs. The paper reports no p-values for significance testing, but consistent gains across 16 configurations (2 architectures × 2 data conditions × 4 aggregation settings) suggest robustness to overfitting. Crucially, the F-Mean path’s *p* converged to 0.43, 0.50 *across all architectures*, indicating a fundamental design principle.

## Related Work
This paper positioned itself as extending generalised mean neurons (Yadav et al., 2006) by making the exponent learnable, while avoiding the computational overhead of prior graph-focused approaches (Pellegrini et al., 2021). Unlike fixed robust estimators (e.g., Geisler et al.’s Soft Medoid), the method learns adaptive aggregation per layer, eliminating the need for manual tuning. The Gaussian Support mechanism drew inspiration from attention (Niu et al., 2021) but applied it at the *neuron level* rather than layer level, making it compatible with standard feed-forward networks.

## Limitations
The paper tested only CIFAR-10 (clean and noisy variants), leaving unknowns about scalability to larger datasets (e.g., ImageNet) or non-vision tasks. The Gaussian Support neuron’s O(n²) complexity (mitigated by projection) remains a bottleneck for high-dimensional inputs, though the authors note sparse attention as future work. The authors also acknowledge that benefits are most pronounced under noise, with clean-data gains being "modest" (e.g., +0.28% on CNNs), limiting applicability for noise-free production systems.

## Appendix: Worked Example
Consider a single hybrid neuron in a CNN classifier head processing 4 images with activations [1.2, 2.8, 0.5, 3.1] (scaled by weights *w*=[1,1,1,1] for simplicity). During training, *p* converges to 0.45 and *α*=0.72.

1.  **F-Mean path**:  
    - Compute *z* = [1.2, 2.8, 0.5, 3.1]  
    - Apply softplus: *z⁺* = [log(2.2), log(3.8), log(1.5), log(4.1)] ≈ [0.79, 1.34, 0.41, 1.41]  
    - Weight *p*=0.45: *ω* = [0.79⁰·⁴⁵, 1.34⁰·⁴⁵, 0.41⁰·⁴⁵, 1.41⁰·⁴⁵] / sum ≈ [0.84, 0.89, 0.73, 0.92] / 3.38 ≈ [0.25, 0.26, 0.22, 0.27]  
    - Aggregation: 0.25×1.2 + 0.26×2.8 + 0.22×0.5 + 0.27×3.1 ≈ 2.14 (suppressing the largest input *3.1* via sub-linear *p*)

2.  **Gaussian Support path**:  
    - Compute pairwise affinities with *σ*=4.68 (converged value):  
      Aff(1,2)=exp(-|1.2-2.8|²/(2×4.68²))≈exp(-0.03)≈0.97  
      Aff(1,3)=exp(-|1.2-0.5|²/(2×4.68²))≈exp(-0.008)≈0.99  
      ... (similar for all pairs)  
    - Normalise to *α*: [0.24, 0.28, 0.21, 0.27] (summing affinities per input)  
    - Aggregation: 0.24×1.2 + 0.28×2.8 + 0.21×0.5 + 0.27×3.1 ≈ 2.10 (downweighting *3.1* due to deviation from consensus)

3.  **Hybrid blend**:  
    - *α*=0.72 (learned): Output = 0.72×2.14 + 0.28×(standard linear sum 2.13) ≈ 2.13  
    - *Note*: The hybrid output remains near standard value (2.13) because *p* is sub-linear (suppression), but the *α* blend ensures stability.

See Key Technical Contributions for how *p* and *α* converge to robust values without explicit regularisation.

## References

- Berke Deniz Bozyigit, "Beyond Weighted Summation: Learnable Nonlinear Aggregation Functions for Robust Artificial Neurons", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19344

Tags: #machine-learning #robust-ai #f-mean #gaussian-support
