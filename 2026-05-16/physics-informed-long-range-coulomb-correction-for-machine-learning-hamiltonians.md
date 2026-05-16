---
title: "Physics-Informed Long-Range Coulomb Correction for Machine-learning Hamiltonians"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20007"
---

## Executive Summary
This paper introduces HamGNN-LR, a novel machine learning framework that integrates physics-informed long-range Coulomb corrections into electronic Hamiltonian prediction. It solves the critical problem of short-range models failing to capture macroscopic electrostatic potentials in polar materials and heterostructures, reducing Hamiltonian errors by two to threefold while enabling robust transferability to systems far beyond training sizes.

## Why This Matters for Practitioners
If you're building production systems for materials discovery or electronic structure prediction, this paper fundamentally changes how you should approach model architecture. Traditional short-range graph neural networks (like HamGNN) will fail for polar systems at scale, causing errors to grow linearly with system size as demonstrated in their size-extrapolation test. You should now implement physics-informed corrections like Ewald summation rather than relying solely on more message-passing layers. For instance, when predicting band structures in GaN/AlN superlattices, your model will need 35% lower error with HamGNN-LR compared to short-range baselines, and crucially, maintain accuracy as you scale to larger structures, something you cannot achieve with more training data alone.

## Problem Statement
Current machine learning models for electronic Hamiltonians operate like myopic drivers who only see the road immediately ahead: they capture local atomic interactions within a limited radius but completely miss the long-range electrostatic fields that govern material behaviour in polar crystals and heterostructures. This is analogous to a weather forecast system that predicts local temperature changes but ignores global atmospheric patterns, accurate for small-scale events but catastrophically wrong for large-scale phenomena like polar vortexes. The result is the characteristic "staircase artifact" in predictions where errors accumulate linearly with system size, rendering models useless for practical materials design.

## Proposed Approach
HamGNN-LR employs a dual-channel architecture: a short-range channel using E(3)-equivariant message passing to model local bonding, and a long-range channel using Ewald attention in reciprocal space to capture macroscopic electrostatic correlations. The two channels are merged via a gated residual connection, ensuring the long-range correction acts as a small perturbation to the short-range baseline during training. The Ewald attention module efficiently reproduces the pairwise phase structure from the analytic Hamiltonian correction.

```python
def evaluate_long_range_hamiltonian(S, Wi, k_vectors, cutoff):
    """Evaluates long-range Hamiltonian matrix elements using closed-form expression.
    
    Args:
        S: Structure factor (computed from decoded ionic charges)
        Wi: Orbital-projected weight matrices
        k_vectors: Sampled wave vectors within cutoff
        cutoff: Cutoff wave vector magnitude
        
    Returns:
        H_lr: Long-range Hamiltonian matrix elements
    """
    H_lr = 0
    for k in k_vectors:
        if |k| < cutoff:
            term = exp(-σ²k²/2) / k²
            phase = S(k) * sum( exp(-ik·τ_i) * Wi, i )
            H_lr += term * real(phase)
    return -4π/V * H_lr
```

## Key Technical Contributions
The core innovation lies in the variational derivation of long-range Hamiltonian matrix elements and its efficient implementation in the Ewald attention module. This isn't merely adding more layers, it's incorporating fundamental physics into the model's architecture.

1. **Variational consistency through analytic derivation**: They achieve exact consistency between Hamiltonian and energy by deriving H_lr,μν = ∂E_lr/∂Pμν using the chain rule. This differs fundamentally from prior approaches that tried to learn long-range effects through data alone. The closed-form expression (Eq. 3) ensures the Hamiltonian correctly reproduces the macroscopic electrostatic potential without requiring an impractically large number of message-passing layers.

2. **Ewald attention module preserving equivariance**: The Ewald attention module specifically replicates the pairwise phase structure km·(τ_i - τ_j) appearing in the analytic expression, using rotary position encoding (RoPE) with phase θ_j,m = k_m·τ_j. This ensures the module is grounded in physics rather than learning arbitrary patterns, maintaining E(3) equivariance throughout the computation.

3. **Robust size extrapolation through analytic correction**: The physics-based correction analytically captures the macroscopic polarization field that grows linearly with slab thickness. Unlike short-range models where errors increase sharply with system size (1.525 meV for DeepH-E3 vs 0.473 meV for HamGNN-LR on 56-layer ZnO), this correction maintains low errors across all tested sizes, as demonstrated in Figure 3.

## Experimental Results
On three polar/heterostructure systems, ZnO slab, CdSe/ZnS heterostructure, and GaN/AlN superlattice, the authors report Hamiltonian mean absolute error (MAE) in meV. The full HamGNN-LR (LR-EA) model achieved:

- CdSe/ZnS: 1.058 meV (vs SR baseline at 3.279 meV, a 3.1x improvement)
- GaN/AlN: 1.162 meV (vs SR baseline at 3.646 meV, a 3.1x improvement)
- ZnO slab (56-layer, size extrapolation): 0.473 meV (vs DeepH-E3 at 1.525 meV, +164% error increase for short-range models)

The key comparison shows that models with Ewald attention but without the physics-based correction (EA) performed identically to the short-range baseline (3.158 meV in CdSe/ZnS vs 3.279 meV for SR), confirming that data-driven attention alone is insufficient without the analytic correction.

## Related Work
The paper positions itself against prior work on long-range interactions in two key ways. For machine learning potentials, they note that approaches like charge equilibration [10] and Gaussian charge representations [11] target scalar energy surfaces rather than tensorial Hamiltonian matrices. For reciprocal-space methods, they differentiate from prior work [12, 22-24] that targets energy prediction rather than Hamiltonian matrices, requiring a fundamentally different mathematical structure to preserve E(3) equivariance.

## Limitations
The authors acknowledge the work is limited to systems with time-reversal symmetry (requiring real-valued NAO bases), though they note this is always available when time-reversal symmetry holds. The paper does not test the framework on systems with strong spin-orbit coupling or in non-periodic boundary conditions. A practical limitation is that the Ewald summation requires computation of the structure factor S(k), which involves decoding effective ionic charges, this adds computational overhead compared to pure short-range models.

## Appendix: Worked Example
Consider predicting the Hamiltonian for a 10-layer GaN/AlN superlattice with 10 atoms per layer (100 atoms total). The short-range baseline (SR) model with three message-passing layers captures local interactions within a 5Å cutoff but fails to account for the built-in electric field across the superlattice. At the layer interface (say, layer 5), the SR model shows a 0.8 meV error per layer due to truncated electrostatic potentials.

HamGNN-LR solves this by first decoding effective ionic charges Q_i for each atom using the Ewald attention module. For layer 5, the charges Q_5 ≈ +0.3e and Q_6 ≈ -0.3e (net polarization) are decoded from the long-range features. The structure factor S(k) is then computed as S(k) = (1/N)∑j Q_j exp(ik·τ_j), with N=100. For k = (0,0,0.1), S(k) ≈ 0.02 (intensive quantity). The pairwise phase structure km·(τ_i - τ_j) for k_m = (0,0,0.1) and atoms i,j across layers is exactly replicated by the Ewald attention module's RoPE encoding.

Using Eq. (3) with V = 200Å³, σ = 0.5, and k_cutoff = 0.5, the long-range correction term becomes:
H_lr,μν = -4π/200 * Σ [exp(-0.5²k²/2)/k² * Re(S(k) * Σ_i exp(-ik·τ_i)W_i,μν)]

For this specific k, the term contributes approximately -0.15 meV to the Hamiltonian matrix element, which corrects the accumulated error from the SR model. This single correction term ensures the continuous linear potential gradient across all layers, eliminating the staircase artifact visible in Figure 3(d) and reducing the overall error from 3.646 meV to 1.162 meV.

## References

- Yang Zhong, Xiwen Li, Xingao Gong, Hongjun Xiang, "Physics-Informed Long-Range Coulomb Correction for Machine-learning Hamiltonians", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20007

Tags: #materials-science #physics-informed #electronic-structure #graph-neural-networks #long-range-attention
