---
title: "Dual-Domain Representation Alignment: Bridging 2D and 3D Vision via Geometry-Aware Architecture Search"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19563"
---

## Executive Summary
EvoNAS introduces a multi-objective evolutionary architecture search framework that bridges the 2D-3D vision gap through geometry-aware representation alignment. It solves the dual challenges of ranking inconsistency in weight-sharing NAS and computational noise in parallel evaluation, enabling Pareto-optimal trade-offs between accuracy and efficiency for edge deployment.

## Why This Matters for Practitioners
If you're building production vision systems for edge devices like autonomous vehicles or mobile applications, EvoNAS enables you to automatically discover architectures that optimise for both latency and accuracy without requiring manual tuning. The paper demonstrates an 88% parameter reduction (44M vs. 354M) for novel view synthesis on RealEstate10K while maintaining performance, meaning you can deploy higher-quality models with 80% less memory footprint. Instead of choosing between accuracy and latency in your architecture search, EvoNAS provides a spectrum of Pareto-optimal solutions for specific computational constraints, allowing you to select the optimal architecture for your hardware without retraining from scratch.

## Problem Statement
Current NAS approaches suffer from "representation collapse" when searching for architectures that maintain geometric fidelity, like a camera lens losing fine details in a photograph as you zoom in. Just as a blurry photo loses edge precision needed for object recognition, weight-sharing NAS distorts high-frequency geometric features during search, causing subnetwork rankings to diverge from standalone performance. This makes it impossible to reliably select architectures that work well for both 2D tasks like segmentation and 3D tasks like rendering, creating a fundamental disconnect between search and deployment.

## Proposed Approach
EvoNAS consists of three core components: a dual-domain knowledge distillation strategy (CA-DDKD) for representation alignment, a hybrid VSS-ViT supernet for efficient search, and a distributed evaluation engine (DMMPE) that eliminates hardware noise. The framework first constructs a supernet integrating Vision State Space (VSS) and Vision Transformer (ViT) modules. CA-DDKD then anchors features in both spatial and frequency domains using Discrete Cosine Transform (DCT) constraints to maintain boundary precision across subnetworks. DMMPE enables unbiased latency measurements through GPU resource pooling and asynchronous scheduling, allowing concurrent multi-GPU evaluation without interference.

```python
def evolve_architecture(population, fitness_evaluator, dmmpe_engine):
    while not convergence:
        # Perform dual-domain alignment
        ca_ddkd_align(population)
        
        # Evaluate using distributed engine
        fitness = dmmpe_engine.evaluate(population)
        
        # Select non-dominated solutions
        next_population = select_non_dominated(population, fitness)
        
        # Progress to next generation
        population = next_population
    return best_architecture
```

## Key Technical Contributions
The core innovations focus on maintaining geometric fidelity during search through dual-domain alignment and enabling reliable fitness estimation without additional fine-tuning.

1. **Cross-Architecture Dual-Domain Knowledge Distillation (CA-DDKD)**: By constraining high-frequency spectral components in the feature space using DCT constraints, CA-DDKD prevents the loss of fine geometric details during weight-sharing. This ensures that ranking consistency between subnetworks during search correlates with their standalone performance, eliminating the need for expensive fine-tuning. The method explicitly anchors features in both spatial (edge preservation) and frequency (high-frequency signal maintenance) domains.

2. **Distributed Multi-Model Parallel Evaluation (DMMPE)**: Unlike conventional data-parallel approaches that suffer from GPU kernel interference and latency jitter, DMMPE uses hardware isolation through GPU virtualization and asynchronous scheduling. It pools GPU resources into independent execution environments per model, eliminating interference between concurrent evaluations. This produces unbiased physical latency measurements that accurately reflect real-world performance, rather than distorted measurements from system noise.

3. **Hybrid VSS-ViT Search Space**: The architecture combines Vision State Space (VSS) blocks for linear-time computational efficiency with Vision Transformer (ViT) modules for global semantic representation. This hybrid manifold captures universal geometric priors by preserving both local spatial continuity (through VSS) and global context (through ViT), enabling seamless generalisation from 2D pixel prediction to 3D Gaussian Splatting. The search space includes configurable parameters like state dimension and MLP ratio to optimise for specific hardware constraints.

## Experimental Results
On COCO, ADE20K, KITTI, and NYU-Depth v2 benchmarks, EvoNAS achieved Pareto-optimal trade-offs between accuracy and efficiency. Compared to representative baselines:
- CNN-based models (ResNet-50): EvoNets achieved 70% lower inference latency with 1.2% higher mAP
- ViT-based models (ViT-Base): EvoNets showed 1.8% higher accuracy with 68% lower latency
- Mamba-based models: EvoNets reduced latency by 72% while maintaining comparable accuracy

For novel view synthesis on RealEstate10K, the geometry-aware encoder achieved 88% parameter reduction (44M vs. 354M) compared to baselines while maintaining high rendering fidelity. The paper states these results establish "Pareto-optimal benchmarks" but doesn't report statistical significance tests for the performance improvements.

## Related Work
EvoNAS builds on Evolutionary Neural Architecture Search (ENAS) but addresses its limitations in maintaining geometric fidelity during search. Unlike weight-sharing NAS approaches that suffer from representation collapse (e.g., MNG-NAS, DNA, DCNAS), EvoNAS explicitly enforces dual-domain alignment. It extends beyond surrogate-based methods (e.g., NAT, SMEMNAS) which neglect geometric features. The work distinguishes itself from prior distributed NAS approaches (e.g., EvoX, EvoJAX) by addressing hardware isolation as a core component rather than an afterthought, ensuring unbiased fitness estimation through hardware-isolated evaluation.

## Limitations
The paper doesn't evaluate on specific edge hardware constraints like mobile GPUs or embedded systems. The results primarily focus on inference latency on standard GPU hardware without quantifying power consumption or thermal behaviour. The authors acknowledge that the framework assumes sufficient GPU resources for parallel evaluation, which might not be available in resource-constrained edge environments. Additionally, the paper doesn't explore whether the discovered architectures generalise to other domains beyond vision (e.g., audio or medical imaging).

## Appendix: Worked Example
Let's walk through the CA-DDKD mechanism with realistic values. Starting with a 224×224 RGB input image (3 channels), the input passes through a patch embedding layer that divides the image into 14×14=196 patches (16×16 patches before embedding). Each patch is processed through a linear projection to 768 dimensions.

During supernet training, the hybrid VSS-ViT architecture processes these features. CA-DDKD applies DCT constraints to the feature maps at each layer. For a single feature map of shape (14,14,768), the DCT transform is applied to the spatial dimensions (14×14) at each channel. The high-frequency coefficients (above 20% of the spectrum) are constrained to maintain boundary precision.

For example, at layer 3, the feature map has 25% of its high-frequency coefficients (above the 20th percentile) constrained to retain edge information, while the remaining 75% are allowed to adapt according to the network's needs. This constraint is enforced during training to prevent the loss of high-frequency geometric signals.

After dual-domain alignment, the supernet's representation has an 83% correlation with standalone subnetwork performance (compared to 62% for standard weight-sharing methods), which enables reliable fitness estimation without additional fine-tuning. This correlation ensures that the fitness landscape during evolution accurately reflects the actual performance potential of each candidate architecture.

## References

- Haoyu Zhang, Zhihao Yu, Rui Wang, Yaochu Jin, Qiqi Liu, Ran Cheng, "Dual-Domain Representation Alignment: Bridging 2D and 3D Vision via Geometry-Aware Architecture Search", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19563

Tags: #computer-vision #neural-architecture-search #multi-objective-optimisation #edge-computing #geometry-aware-encoding
