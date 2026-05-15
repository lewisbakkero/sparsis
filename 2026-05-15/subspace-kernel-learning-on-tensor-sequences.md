---
title: "Subspace Kernel Learning on Tensor Sequences"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19546"
---

## Executive Summary
Uncertainty-driven Kernel Tensor Learning (UKTL) introduces a novel kernel framework for M-mode tensors that compares mode-wise subspaces derived from tensor unfoldings, enabling expressive and robust similarity measures. It addresses the computational inefficiency of traditional kernel methods on tensor data while incorporating uncertainty-aware subspace weighting to improve robustness and interpretability. For engineers working with structured multi-way data like video, sensor streams, or biomedical signals, UKTL offers a more efficient and interpretable alternative to flattening tensors or using expensive direct tensor comparisons.

## Why This Matters for Practitioners
If you're currently processing high-dimensional multi-way data (such as video sequences, sensor streams, or biomedical signals) by flattening tensors into vectors before applying standard ML techniques, UKTL offers a more efficient and interpretable approach. The authors demonstrate that UKTL outperforms state-of-the-art graph, hypergraph, and transformer models on action recognition benchmarks while maintaining a simpler architecture.

You should consider implementing UKTL for your tensor-based applications when:
1. You're working with structured multi-way data (e.g., 3D skeletons, video frames, multi-sensor readings)
2. You need to capture complex interactions across tensor modes without destroying the tensor structure
3. You require interpretability about which tensor modes contribute most to the model's decisions
4. You're experiencing performance issues with traditional kernel methods due to computational complexity

For immediate action, experiment with UKTL's tensor kernel linearization approach to replace your current vectorized kernel methods, especially if you're dealing with large-scale tensor data where computational efficiency is critical. The framework is fully end-to-end trainable and can be integrated into existing tensor processing pipelines with minimal architectural changes.

## Problem Statement
Today's approaches to learning from structured multi-way data often face a fundamental trade-off: either they flatten tensors into vectors (destroying the inherent structure, leading to inefficient models), or they use expensive direct tensor comparisons that don't scale well. This is analogous to trying to navigate a city by flattening all its buildings onto a single plane - you lose all the 3D relationships that make the city navigable and end up with a confusing, inefficient map.

## Proposed Approach
UKTL introduces a three-component framework for tensor-based learning:
1. A product subspace kernel that compares mode-wise subspaces derived from tensor unfoldings
2. A Nyström-based kernel linearization with dynamically selected pivot tensors via soft k-means clustering
3. An uncertainty-aware regularisation framework that adaptively weights subspaces based on their reliability

The end-to-end pipeline processes tensor sequences through a tensor encoder (MLP + Higher-order Transformer), extracts mode-wise subspaces via Tucker decomposition, applies uncertainty weighting, and uses Nyström approximation to create a scalable kernel embedding for classification.

```python
def UKTL_pipeline(tensor_sequence):
    # 1. Encode tensor sequence
    encoded_tensor = tensor_encoder(tensor_sequence)  # MLP + HoT
    
    # 2. Extract mode-wise subspaces via Tucker decomposition
    subspaces = []
    for mode in range(M):
        mode_mats = matricize(encoded_tensor, mode)
        u, _, _ = svd(mode_mats)
        subspaces.append(u)
    
    # 3. Apply uncertainty weighting via MSN
    uncertainty_vectors = MSN(subspaces)
    weighted_subspaces = [subspace / np.sqrt(uncertainty) for subspace, uncertainty in zip(subspaces, uncertainty_vectors)]
    
    # 4. Compute kernel similarity using weighted subspaces
    kernel_similarity = product_subspace_kernel(weighted_subspaces)
    
    # 5. Apply Nyström approximation for scalability
    pivots = soft_kmeans_clustering(kernel_similarity, C)
    kernel_embedding = nystrom_approximation(kernel_similarity, pivots)
    
    # 6. Classify
    return classifier(kernel_embedding)
```

## Key Technical Contributions
The core innovations of UKTL are:
1. **Mode-wise subspace comparison via Grassmann kernel**: Unlike traditional approaches that flatten tensors or compare raw tensors directly, UKTL compares tensor sequences by their mode-wise subspaces on Grassmann manifolds. For each tensor mode, UKTL computes a projection matrix from the top singular vectors of the mode-m matricization, then measures similarity between these projection matrices using a Grassmann-based kernel. This captures multi-way structure while being computationally efficient.

2. **Dynamic pivot selection via soft k-means clustering**: UKTL's Nyström approximation doesn't use random or static pivots. Instead, it dynamically selects pivot tensors through differentiable soft k-means clustering, ensuring the pivots evolve during training to remain relevant to the underlying data distribution. This is different from most existing kernel approximation techniques, which rely on static, randomly selected dictionary elements.

3. **Uncertainty-aware regularisation via Multi-mode SigmaNet**: The paper introduces a novel mechanism for modelling uncertainty at the mode-level rather than globally. The Multi-mode SigmaNet (MSN) processes each mode's subspace to produce uncertainty vectors that weight the subspace components, effectively down-weighting unreliable directions. This is grounded in maximum likelihood estimation, making it data-driven and interpretable.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
On standard action recognition benchmarks, UKTL consistently outperforms state-of-the-art methods:
- On NTU-60 (X-Sub): UKTL achieves 93.1% top-1 accuracy compared to the best baseline (DSDC-GCN) at 93.0%
- On NTU-120 (X-Setup): UKTL achieves 91.4% top-1 accuracy compared to the best baseline (DSDC-GCN) at 90.6%
- On Kinetics-Skeleton: UKTL achieves 39.2% top-1 accuracy compared to the best baseline (DSDC-GCN) at 38.6%

The paper shows UKTL improves by 0.6% top-1 accuracy over DSDC-GCN on NTU-60 (X-Sub), 0.8% on NTU-120 (X-Setup), and 0.6% on Kinetics-Skeleton. These improvements are statistically significant as measured by standard error margins in the paper's experiments.

## Related Work
UKTL builds upon tensor decomposition methods like Tucker (Kolda & Bader, 2009) and tensor subspace learning approaches, but addresses their limitations by introducing a non-linear, kernel-based formulation on mode-specific subspaces. Unlike traditional tensor subspace techniques that apply uniform regularisation across all modes, UKTL models mode-wise uncertainty using maximum likelihood estimation.

The work also extends kernel methods for tensor data by addressing their computational inefficiency through dynamic pivot selection with soft k-means clustering, rather than relying on static kernel definitions or random pivot selection as in most existing approaches.

## Limitations
The paper acknowledges that UKTL was evaluated primarily on action recognition benchmarks, and future work could explore applying it to other tensor-based tasks like video understanding or medical imaging. The authors also note that the current implementation uses a fixed number of pivots (C), which might need to be adjusted for different datasets or applications.

From an engineering perspective, UKTL's computational complexity might still be higher than simpler vector-based approaches for very small datasets, though it scales better for large-scale tensor data. The implementation also requires careful tuning of the uncertainty regularisation hyperparameter β.

## Appendix: Worked Example
Let's walk through UKTL's core mechanism step by step with a concrete example. Consider a skeleton sequence with 3 temporal blocks (τ=3), 20 body joints (J=20), and 3D coordinates (d=3).

1. **Tensor encoding**: The sequence is processed by an MLP into a 64×1140×3 tensor (d'=64 features, Nξ=1140 hyper-edges, τ=3 temporal blocks).

2. **Mode-wise matricization**: The tensor undergoes mode-m matricization:
   - X(1) ∈R64×3420 (mode 1: features × hyper-edges×time)
   - X(2) ∈RNξ×192 (mode 2: hyper-edges × features×time)
   - X(3) ∈R3×7680 (mode 3: time × features×hyper-edges)

3. **Subspace extraction**: For each mode, perform SVD to extract the top-10 left singular vectors (p=10):
   - Mode 1: U1 ∈R64×10
   - Mode 2: U2 ∈RNξ×10
   - Mode 3: U3 ∈R3×10

4. **Uncertainty weighting**: The Multi-mode SigmaNet (MSN) processes each subspace to produce uncertainty vectors:
   - Mode 1: σ1 ∈R10, where σ1,k = 0.45 + 0.05*random noise
   - Mode 2: σ2 ∈R10, where σ2,k = 0.35 + 0.05*random noise
   - Mode 3: σ3 ∈R10, where σ3,k = 0.25 + 0.05*random noise

5. **Weighted subspaces**: The subspaces are normalized by their uncertainty vectors:
   - Mode 1: U1' = U1 / sqrt(σ1)
   - Mode 2: U2' = U2 / sqrt(σ2)
   - Mode 3: U3' = U3 / sqrt(σ3)

6. **Kernel computation**: The product subspace kernel computes similarity between two sequences:
   k = exp(-||U1'1U1'1⊤ - U1'2U1'2⊤||²/2σ²) × exp(-||U2'1U2'1⊤ - U2'2U2'2⊤||²/2σ²) × exp(-||U3'1U3'1⊤ - U3'2U3'2⊤||²/2σ²)

7. **Nyström approximation**: For scalability, UKTL uses soft k-means clustering to find C=20 pivot tensors. The kernel matrix is approximated using these pivots, reducing computational complexity from O(N²) to O(NC).

## References

- Lei Wang, Xi Ding, Yongsheng Gao, Piotr Koniusz, "Subspace Kernel Learning on Tensor Sequences", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19546

Tags: #tensor-kernel #uncertainty-aware #action-recognition #multiway-data #kernel-methods
