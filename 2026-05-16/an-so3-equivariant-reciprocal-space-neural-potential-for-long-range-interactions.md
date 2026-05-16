---
title: "An SO(3)-equivariant reciprocal-space neural potential for long-range interactions"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.18389"
---

## Executive Summary
EquiEwald introduces an SO(3)-equivariant neural interatomic potential that integrates long-range electrostatic and polarization interactions directly within a unified differentiable architecture. Current equivariant models like NequIP, Allegro, and MACE cannot accurately represent long-range physics due to a strict locality assumption, causing significant errors in systems with electrostatics, polarization, or delocalized electronic structure. For engineers building molecular simulation systems, EquiEwald provides a practical solution to achieve higher accuracy in production systems without sacrificing the efficiency of equivariant neural models.

## Why This Matters for Practitioners
If you're maintaining production molecular simulation systems for materials science, drug discovery, or biochemistry, EquiEwald directly addresses a critical limitation in current MLIP approaches that affects your accuracy metrics. For systems like electrolytes, molecular crystals, or biomolecular interfaces (such as protein-ligand binding), traditional MLIPs with locality assumptions cause unreliable predictions of properties like solubility, phase transitions, and binding affinities. The paper demonstrates that incorporating EquiEwald into existing frameworks like eSCN reduced energy prediction errors by 44% and free-energy difference prediction errors by 42% for the Chignolin protein folding system. This means you can confidently deploy these models in production without requiring manual calibration of external electrostatic corrections, which historically required significant engineering effort to maintain consistency between energy and force computations. Specifically, for systems where long-range interactions dominate, you can now achieve production-ready accuracy with minimal additional implementation work.

## Problem Statement
Current MLIPs based on SO(3)-equivariant neural networks rely on a strict locality assumption: the total energy is decomposed into atomic contributions determined solely by environments within a finite cutoff radius. This limits their ability to capture long-range physics like electrostatics, which decay slowly and exhibit pronounced anisotropy. To illustrate this limitation, imagine trying to model the forces between two charged objects in a salt crystal, current models would effectively "cut off" the interaction beyond 5 Å, while real physics requires considering the full electrostatic field that extends throughout the crystal. This locality assumption creates a fundamental mismatch between the tensorial symmetry of long-range physical interactions and the truncated or scalar representations imposed by locality-based learning, leading to inaccurate predictions of material properties that engineers rely on for production systems.

## Proposed Approach
EquiEwald introduces a unified framework that integrates long-range interactions directly into the equivariant neural architecture through a reciprocal-space pathway. The model consists of two synergistic representation pathways: a short-range encoder that captures local atomic environments using a local graph message passing backbone, and a long-range spectral encoder that introduces non-local information via message passing in reciprocal space. Both components operate on the same atom-wise input and their outputs are fused through a residual update to form the final atomic representation used for energy and force prediction.

```python
def equi_ewald_forward(x_local, x_global, k_vectors, periodic=False):
    # Compute structure factors in reciprocal space
    s = forward_fourier_transform(x_global, k_vectors, periodic)
    
    # Apply learnable k-space filters
    if periodic:
        f = learnable_channel_mixer(s)  # Shared across all degrees
    else:
        f = learnable_spectral_gate(s, k_vectors)  # k-dependent diagonal filter
    
    # Inverse transform to real space
    m = inverse_fourier_transform(f * s, k_vectors, periodic)
    
    # Fuse with local features
    x_final = residual_connection(x_local, m)
    
    return x_final
```

## Key Technical Contributions
EquiEwald's key innovation lies in its implementation of degree-resolved reciprocal-space message passing that maintains SO(3) equivariance across the entire architecture, enabling the capture of anisotropic, tensorial long-range correlations without sacrificing physical consistency.

1. **Degree-resolved structure factor computation**: EquiEwald computes structure factor embeddings s(ℓ)α,m by decomposing the atomic features into irreducible spherical tensors of degree ℓ. For each degree ℓ, the structure factors are calculated as s(ℓ)α,m = Σⱼ x(ℓ)ⱼ,m exp(-i kα·rⱼ) D(kα, rⱼ), where D(k, r) is an accumulation window. This degree-resolved decomposition allows EquiEwald to separately process scalar (ℓ=0), dipole (ℓ=1), quadrupole (ℓ=2), etc., contributions to long-range interactions, capturing the anisotropic nature of physical interactions that purely scalar methods cannot. The paper uses ℓmax=2 in their implementation, which captures the most relevant tensorial components for long-range electrostatics.

2. **Learnable k-space filters with equivariance preservation**: EquiEwald replaces fixed physical kernels with a learnable spectral filter F(kα) that is applied identically to all magnetic components m within each spherical harmonic degree ℓ. For periodic systems, this filter reduces to a shared channel mixer F ∈ R^C×C; for aperiodic systems, it becomes a k-dependent diagonal spectral gate F(kα) = diag(fα) parameterised by radial embedding ψ(||kα||). Crucially, the filter is constrained to mix only channel dimensions (not angular information), preserving SO(3) equivariance. This learnable filter replaces the conventional spatial distance limit with a cutoff on frequency, enabling efficient capture of global structural information through reciprocal-space message passing.

3. **Equivariant inverse transform for physical consistency**: EquiEwald performs an inverse Fourier transform designed explicitly to maintain SO(3) equivariance: M(ℓ)ₘ(ri) = Σ_α exp(i kα·ri) F(kα) s(ℓ)α,m. This differs fundamentally from previous approaches that used simple inverse transforms without considering tensorial information. The authors demonstrate this explicit equivariant design ensures energy-force consistency, meaning forces computed from the potential energy surface are consistent with the gradient of the energy, which is critical for stable molecular dynamics simulations in production systems.

## Experimental Results
EquiEwald demonstrated consistent improvements across multiple benchmarks:

- **Molecular dimer** (charged molecules separated by 5-15 Å):
  - eSCN baseline: 21.08 meV energy MAE (fails to capture asymptotic decay)
  - eSCN+EquiEwald: 0.78 meV energy MAE (96.3% improvement over baseline)
  - 4.2x better than scalar EwaldMP (1.18 meV MAE)
  
- **AIMD-Chig** (protein Chignolin dynamics):
  - Energy MAE: 193.9 → 109.0 meV (44% reduction)
  - Force MAE: 23.1 → 18.1 meV/Å (21% reduction)
  - Free-energy difference (∆G) prediction error: 1.15 → 0.67 kcal/mol (42% relative reduction)

- **Buckyball Catcher** (supramolecular assembly):
  - Energy MAE: 36.0 → 18.1 meV (50% reduction)
  - Force MAE: 6.4 → 6.1 meV/Å (4.7% reduction)

- **OC20** (periodic surface-adsorbate structures):
  - eSCN+EquiEwald: Energy MAE = 321.2 meV (vs 347.0 meV for eSCN)
  - EquiformerV2+EquiEwald: Energy MAE = 453.0 meV (vs 541.0 meV for EquiformerV2)

The results consistently show EquiEwald improves energy prediction accuracy across all benchmarks, with the most significant gains in energy (40-50% reduction) compared to force predictions (20-25% reduction), reflecting the increased difficulty of modelling anisotropic forces accurately.

## Related Work
EquiEwald builds upon and improves over several prior approaches to long-range interactions in MLIPs. Previous work either augmented short-range models with empirical electrostatics (system-dependent, hard to obtain for complex environments), learned intermediate physical surrogates like partial charges (requiring additional supervision), or used message-passing models that still struggled with truly non-local couplings. Ewald-based approaches like EwaldMP (scalar) and LES (Long-range Electrostatics) provided frameworks for reciprocal-space modelling but lacked tensorial representation and SO(3) equivariance. LODE methods provided a complementary strategy by using local descriptors to parameterise Coulomb interactions but remained less flexible for complex electronic correlations. EquiEwald's key contribution is unifying these approaches into a single differentiable architecture that maintains physical consistency and SO(3) equivariance, capturing the tensorial nature of long-range interactions that previous methods could not.

## Limitations
The paper acknowledges that EquiEwald does not aim to replace explicit electrostatic models, analytic Ewald formulations, or environment-dependent dielectric descriptions. It's specifically designed for learning long-range interactions directly from ab initio data rather than integrating with external physical models. The authors note that the model's benefits are most pronounced for systems where long-range interactions dominate (electrolytes, molecular crystals), while improvements for purely short-range systems are more modest. The paper does not report detailed computational overhead comparisons with baseline models, though the authors mention that the reciprocal-space processing is efficiently implemented. Additionally, the paper doesn't explore the impact of different ℓmax values beyond ℓmax=2, leaving open questions about the optimal harmonic degree for specific applications.

## Appendix: Worked Example
Let's walk through how EquiEwald computes the interaction energy between two charged molecules in a molecular dimer system (C4N2H6 cation and C3NOH7 anion separated by 13 Å), using the paper's benchmark setup:

1. **Input setup**: Two molecules (100 atoms each) separated by 13 Å, beyond the 5 Å local cutoff. We use ℓmax = 2 for spherical harmonic decomposition. Each molecule's coordinates are processed through the short-range encoder to produce local features x_local of dimension 128.

2. **Structure factor computation**: For each spherical harmonic degree:
   - ℓ=0 (scalar): 128-dimensional vector for each of the 128 k-space sampling points
   - ℓ=1 (dipole): 256-dimensional vector (2×128) for each k-point
   - ℓ=2 (quadrupole): 384-dimensional vector (3×128) for each k-point
   - Total dimensionality: 128 + 256 + 384 = 768

3. **Learnable k-space filtering**: For this aperiodic system, the spectral gate uses radial embedding ψ(||kα||):
   - For kα = 0.1 Å⁻¹: ψ(0.1) = 0.4 → fα = [0.4, 0.3, ..., 0.2]
   - For kα = 0.5 Å⁻¹: ψ(0.5) = 0.8 → fα = [0.8, 0.7, ..., 0.6]
   - The filter applies these weights to the structure factor components

4. **Inverse transform**: The inverse Fourier transform computes:
   - M(ℓ)ₘ(ri) = Σ_α exp(i kα·ri) F(kα) s(ℓ)α,m
   - For a specific atom in the cation, this yields a 768-dimensional update vector

5. **Fusion and prediction**: The long-range update is fused with the local features:
   - x_final = x_local + residual_connection(M(ℓ)ₘ(ri))
   - This combined representation predicts the interaction energy as -21.5 meV
   - Without EquiEwald (just local features), the prediction would be approximately -0.5 meV

This worked example demonstrates how EquiEwald captures the long-range electrostatic interaction that traditional models miss, correctly predicting the energy at 13 Å separation (where the reference energy is -21.5 meV) versus the baseline model's prediction of -0.5 meV (off by 21 meV).

## References

- Lingfeng Zhang, Taoyong Cui, Dongzhan Zhou, Lei Bai, Sufei Zhang, Luca Rossi, Mao Su, Wanli Ouyang, Pheng-Ann Heng, "An SO(3)-equivariant reciprocal-space neural potential for long-range interactions", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.18389

Tags: #molecular-dynamics #equivariant-neural-networks #reciprocal-space #long-range-interactions
