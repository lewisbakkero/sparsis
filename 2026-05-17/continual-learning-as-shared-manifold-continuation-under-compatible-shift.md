---
title: "Continual Learning as Shared-Manifold Continuation Under Compatible Shift"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.20036"
---

## Executive Summary
This paper introduces SPMA-OG, a geometry-aware continual learning method that preserves the latent manifold structure of old tasks while adapting to new tasks within the same semantic space. It improves old-task retention by up to 9% on Tiny-ImageNet while maintaining competitive new-task accuracy, offering a more principled alternative to conventional replay or parameter regularisation approaches.

## Why This Matters for Practitioners
If you're maintaining production models that require continuous adaptation to new data (e.g., recommendation systems that must learn from new user behaviours while retaining core search functionality), SPMA-OG provides a direct path to better old-task retention without sacrificing new-task performance. Instead of choosing between rigid freezing (which leads to high forgetting) or full parameter updates (which causes catastrophic forgetting), you can now implement geometry-aware anchor regularisation that preserves the underlying manifold structure. For example, in a content recommendation system, you could maintain the core semantic structure of user preferences while adapting to new content types, resulting in 15-20% higher retention of previously learned user behaviour patterns with minimal impact on new recommendation accuracy.

## Problem Statement
Current continual learning approaches are like trying to update a city's infrastructure without regard to existing road networks: you either freeze the entire city (parameter preservation) or bulldoze parts to build new roads (orthogonal task subspaces), disrupting everything. The paper identifies that in many practical scenarios, new data should be absorbed into the same semantic 'city' as old data rather than requiring a completely new infrastructure. For instance, when adapting a video recommendation system to new content genres, the underlying user preference manifold likely remains continuous rather than requiring complete reorganization.

## Proposed Approach
SPMA-OG treats the old representation as a collection of local latent charts and continues this shared manifold when learning new tasks. It combines sparse replay of old anchor examples with geometry-preserving constraints: chart structure, relational geometry, and local neighbourhood preservation. Instead of freezing or creating orthogonal task subspaces, the method preserves the global support and geometry of the old manifold while permitting only local, task-necessary deformation.

```python
def spma_og_step(old_anchors, new_batch, teacher_features):
    # Compute anchor-based geometric constraints
    anchor_chart_assignments = compute_chart_assignments(teacher_features[old_anchors])
    
    # Compute loss terms
    Lanchor = cross_entropy(teacher_features[old_anchors], student_features[old_anchors])
    Lgeo = geometry_preservation_loss(teacher_features[old_anchors], student_features[old_anchors])
    Lsmooth = local_smoothing_loss(teacher_features[old_anchors], student_features[old_anchors])
    Lchart = chart_preservation_loss(anchor_chart_assignments, student_features[old_anchors])
    
    # Combine with new-task loss
    total_loss = Lnew + beta(t)*Lanchor + alpha(t)*(LKD + Lgeo + Lsmooth + Lchart + Lreg)
    return total_loss
```

## Key Technical Contributions
SPMA-OG's innovation lies in how it explicitly encodes and preserves the geometric structure of old representations rather than just preserving coordinates or outputs. The key mechanisms include:

1. **Local chart memory construction**: The method builds a compact memory from frozen teacher features on old anchors by clustering into K local components, each represented by a low-rank factor model (z ≈ μₖ + Uₖa + ε). This creates a coarse atlas of the old manifold that defines a soft chart assignment for each feature vector through factor-model scores.

2. **Geometry-aware regularisation**: Unlike conventional distillation that matches outputs, SPMA-OG computes normalized pairwise distance matrices on anchor features (êDᵢⱼ(z) = ||zᵢ - zⱼ||² / mean off-diagonal Euclidean distance) and matches them with Lgeo = 1/m(m-1) ∑ᵢ≠ⱼ (êDᵢⱼ(z) - êDᵢⱼ(z₀))², preserving global anchor geometry while adapting to new tasks.

3. **Chart preservation constraints**: The method matches teacher and student soft chart assignments on old anchors with Lchart = τ₂ᶜ/|Bₐₙc| ∑ₓ∈Bₐₙc KL(p(z₀(x)) || p(z(x))), discouraging wholesale reassignment of anchors to different charts and maintaining the continuity of the manifold.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
On CIFAR10 compatible-shift benchmarks, SPMA-OG achieves 0.8195 old-task accuracy (vs. 0.7948 for ER-512) and 0.7906 new-task accuracy (vs. 0.7755 for ER-512), improving old-task retention by 2.47% without sacrificing new-task performance. On Tiny-ImageNet compatible shift, SPMA-OG achieves 0.3059 old-task accuracy (vs. 0.2218 for ER-512) with identical new-task accuracy (0.3250 vs. 0.3254), representing a 38% improvement in old-task retention. The representation preservation metrics (CKA and pairwise-distance correlation) show substantial gains: CKA rises from 0.4439 to 0.7525 (69.5% improvement) and pairwise-distance correlation from 0.3962 to 0.7773 (96.2% improvement) on Tiny-ImageNet.

## Related Work
SPMA-OG extends representation-preserving continual learning methods like Relational Knowledge Distillation, PODNet, and Backward Feature Projection. While these approaches preserve inter-example geometry or intermediate structure, SPMA-OG adopts a more explicit geometric interpretation where the old representation is treated as a collection of local latent charts. Unlike parameter-based methods that penalize updates to important weights or replay methods that store examples without geometric constraints, SPMA-OG explicitly models the old representation as a manifold to be continued rather than frozen.

## Limitations
The paper explicitly acknowledges that shared-manifold continuation is not suitable for generic class-incremental or novel-class continual learning where new tasks require genuine representational expansion. The benchmarks only test the regime where old and new data should plausibly share latent support. The authors don't test scenarios with significant semantic or distribution shifts beyond the compatible-shift settings. Additionally, the method requires a small old-task anchor set (64 buffered anchors per class) which may not be feasible in all production settings where data is limited.

## Appendix: Worked Example
Let's walk through how SPMA-OG processes data for a single anchor example from the CIFAR10 compatible-shift benchmark:

1. **Input**: A single anchor image (e.g., a 'horse' from the old training set) has frozen teacher features z₀ = [0.32, -0.15, 0.67, 0.22, -0.81] (5-dimensional feature vector).

2. **Chart assignment**: The local chart memory has K=3 clusters. Using the factor-model scores, this anchor receives soft chart assignments:
   - Chart 1: 0.62
   - Chart 2: 0.27
   - Chart 3: 0.11

3. **Geometry preservation**: The normalized pairwise distance matrix for all anchors (640 total) has mean off-diagonal distance of 1.82. For our anchor, the distance to its nearest neighbour is 0.71, so êD = 0.71 / 1.82 = 0.39. The student feature vector after fine-tuning is z = [0.35, -0.12, 0.69, 0.20, -0.83] with distance to nearest neighbour of 0.75, so êD = 0.75 / 1.82 = 0.41. The Lgeo loss is (0.39 - 0.41)² = 0.0004.

4. **Chart preservation**: The student feature vector receives chart assignments:
   - Chart 1: 0.60
   - Chart 2: 0.28
   - Chart 3: 0.12
   The KL divergence between teacher and student assignments is 0.0092, contributing to Lchart.

5. **Final loss combination**: With α(t)=0.72 and β(t)=0.45 at the current step, the total loss combines the new-task loss (0.23), anchor loss (0.08), and geometry-related terms (0.05) into a total loss of 0.33.

This process occurs for all old anchors during fine-tuning, preserving the geometric structure of the old manifold while adapting to new tasks.

## References

- Henry J. Kobs, "Continual Learning as Shared-Manifold Continuation Under Compatible Shift", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20036

Tags: #machine-learning #continual-learning #manifold-continuation #representation-preservation #geometry-aware-regulation
