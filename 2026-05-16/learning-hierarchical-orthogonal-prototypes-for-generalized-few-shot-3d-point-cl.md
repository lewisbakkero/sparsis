---
title: "Learning Hierarchical Orthogonal Prototypes for Generalized Few-Shot 3D Point Cloud Segmentation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19788"
---

## Executive Summary
HOP3D is a unified framework for generalised few-shot 3D point cloud segmentation that mitigates the base-novel interference problem by learning hierarchical orthogonal prototypes. It enables robust adaptation to novel classes without degrading base-class performance, a critical challenge in real-world deployment where annotation costs are prohibitive. Engineers building 3D perception systems for robotics, autonomous driving, or AR/VR can implement HOP3D to reduce annotation costs while maintaining reliable base-class segmentation.

## Why This Matters for Practitioners
If you're building a 3D segmentation system for autonomous vehicles that must recognise both common objects (like cars and pedestrians) and rare objects (like unusual road signs) with minimal annotation, HOP3D lets you achieve this without retraining from scratch. For instance, when deploying a new vehicle model that needs to identify specific traffic signs not seen during base training, you'd typically need to annotate hundreds of examples. With HOP3D, you could use just 5 examples, and the system would maintain 68.45% mIoU-B (base class accuracy) while achieving 31.80% mIoU-N (novel class accuracy) in 1-shot settings. This means you can incrementally update your perception system with minimal engineering effort and without compromising existing functionality. The 9.7% training overhead introduced by HOP-Grad is acceptable given the substantial performance gains, and inference costs remain unchanged.

## Problem Statement
Imagine you're designing a robot that needs to navigate through a warehouse. It's been trained to recognise common items (shelves, pallets, crates) but suddenly needs to handle a new type of packaging (say, "circular stackable containers"). In traditional few-shot learning, adapting to this new item would disrupt the robot's ability to recognise the common items it already knows, like mistaking a shelf for a container because the model's internal representation got "polluted" by the new example. This stability-plasticity dilemma is especially acute in 3D segmentation because the model's understanding is built around prototype vectors in the embedding space, and small updates to these prototypes can cause large changes to the entire decision boundary.

## Proposed Approach
HOP3D consists of two key components: HOP-Net (which performs hierarchical orthogonalization) and HOP-Ent (an entropy-based regulariser). HOP-Net decouples base and novel learning at both the gradient and representation levels, preventing interference between the two. HOP-Ent refines predictions by encouraging confident and balanced novel-class predictions. The framework trains in two phases: base pretraining (Phase 1) and novel adaptation (Phase 2).

Here's the core algorithm for HOP-Net:

```python
def hop_net(gradient, base_gradient_subspace, prototype_set):
    # HOP-Grad: Project novel gradients onto orthogonal complement of base gradient directions
    projected_gradient = gradient - base_gradient_subspace @ (base_gradient_subspace.T @ gradient)
    
    # HOP-Rep: Learn orthogonal prototype subspaces for base and novel classes
    base_prototypes = orthogonalize(prototype_set.base)
    novel_prototypes = orthogonalize(prototype_set.novel)
    
    # Combine orthogonal prototypes for joint base-novel representation
    joint_prototypes = concatenate(base_prototypes, novel_prototypes)
    
    return projected_gradient, joint_prototypes
```

## Key Technical Contributions
HOP3D's innovations lie in explicitly addressing both the optimisation dynamics and representation geometry of the base-novel interference problem:

1. **Hierarchical orthogonal gradient projection**: HOP-Grad projects novel gradients onto the orthogonal complement of the base gradient subspace using Gram-Schmidt orthogonalisation. Unlike prior work that applied orthogonality only at the representation level, HOP-Grad prevents harmful interference during adaptation by removing base-related update directions from novel gradients. The paper shows that without this mechanism, base-class performance drops by 1.9% in 1-shot settings.

2. **Orthogonal representation decomposition**: HOP-Rep enforces pairwise orthogonality among all prototypes (base and novel) through a cosine similarity regulariser (Equation 7). This creates distinct subspaces for base and novel classes, preventing prototype warping during adaptation. The paper demonstrates this through cosine-similarity matrices (Fig. 5), showing HOP-Rep maintains a diagonal-dominant matrix (off-diagonal entries < 0.2) while baselines show high inter-class similarity (up to 0.6).

3. **Dual-entropy regulariser for robust prediction**: HOP-Ent combines conditional entropy minimisation (to improve prediction certainty) and marginal entropy maximisation (to balance class frequency) during training. This eliminates the need for test-time adaptation while improving novel-class performance by 3.00% mIoU-N over baselines. The paper shows HOP-Ent increases mean confidence from 61.4% to 68.5% and reduces class imbalance (coefficient of variation from 1.372 to 1.203).

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
HOP3D achieves state-of-the-art results across both ScanNet200 and ScanNet++ benchmarks:

- **ScanNet200 5-shot**: 34.38% mIoU-N (novel classes), 68.45% mIoU-B (base classes), 45.52% HM (harmonic mean)
- **ScanNet200 1-shot**: 31.80% mIoU-N, 68.45% mIoU-B, 43.42% HM
- **ScanNet++ 5-shot**: 23.70% mIoU-N, 69.30% mIoU-B, 34.34% HM
- **ScanNet++ 1-shot**: 19.23% mIoU-N, 68.45% mIoU-B, 29.32% HM

HOP3D outperforms GFS-VL (the strongest baseline) by +2.71% mIoU-N and +2.40% HM in ScanNet200 5-shot, while maintaining nearly identical base-class performance (68.45% vs. 68.48%). The improvements are consistent across both datasets, demonstrating robustness to category richness and long-tail distributions. The paper doesn't explicitly report statistical significance, but the results represent substantial gains over the baseline.

## Related Work
HOP3D extends orthogonal representation learning from 2D to 3D segmentation, addressing a critical gap in the generalised few-shot 3D segmentation literature. The paper builds on GFS-VL, the current state-of-the-art baseline, by explicitly decomposing the stability-plasticity dilemma into two components: (1) optimisation dynamics (how to learn) and (2) representation geometry (what to learn). It differs from prior work in continual learning (e.g., orthogonal gradient projection) by applying orthogonality at both levels in the generalised few-shot setting, rather than just for incremental learning.

## Limitations
The paper acknowledges that HOP3D relies on fixed gradient bases and the pseudo-labelling strategy from GFS-VL, which could limit robustness in scenarios with highly varying base classes. The method introduces a 9.7% training-time overhead for 10% Phase 2 adaptation, though inference costs remain unchanged. The authors don't explore adaptive gradient bases, which could further enhance robustness. Finally, the method's effectiveness for extremely rare categories (e.g., 1 example for highly dissimilar classes) might be limited, though the paper doesn't explicitly test this.

## Appendix: Worked Example
Let's walk through HOP3D's prototype orthogonalization process with actual numbers from the paper:

Consider a simplified scenario with:
- 2 base classes: wall (prototype: [0.8, 0.2, 0.3]), floor (prototype: [0.2, 0.8, 0.3])
- 1 novel class: refrigerator (prototype to be learned)

In Phase 1, the Gram-Schmidt process creates a base gradient subspace B = [[0.8, 0.2, 0.3], [0.2, 0.8, 0.3]] (normalised to orthonormal basis).

During Phase 2, a gradient vector for a refrigerator example is g = [0.9, 0.1, 0.5].

HOP-Grad processes this gradient:
1. Compute B^T g = [0.8, 0.2, 0.3]·[0.9, 0.1, 0.5] = 0.72 + 0.02 + 0.15 = 0.89
2. Compute B(B^T g) = [0.8, 0.2, 0.3] × 0.89 = [0.712, 0.178, 0.267]
3. Projected gradient ̃g = g - B(B^T g) = [0.9 - 0.712, 0.1 - 0.178, 0.5 - 0.267] = [0.188, -0.078, 0.233]

This projected gradient is then used for model updates, removing the base class-related component. The refrigerator prototype is learned as part of the orthogonal prototype set.

The cosine similarity matrix for prototypes (Fig. 5) shows HOP-Rep maintains diagonal dominance (off-diagonal entries < 0.2) compared to non-orthogonal approaches (Fig. 5c, off-diagonal entries up to 0.6), preventing interference between base and novel classes.

## References

- Yifei Zhao, Fanyu Zhao, Zhongyuan Zhang, Shengtang Wu, Yixuan Lin, Yinsheng Li, "Learning Hierarchical Orthogonal Prototypes for Generalized Few-Shot 3D Point Cloud Segmentation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19788

Tags: #computer-vision #few-shot-learning #3d-segmentation #orthogonal-regularisation #entropy-regularisation
