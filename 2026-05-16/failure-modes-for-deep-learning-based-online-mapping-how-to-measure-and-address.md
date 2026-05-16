---
title: "Failure Modes for Deep Learning-Based Online Mapping: How to Measure and Address Them"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19852"
---

## Executive Summary
This paper introduces a systematic framework for measuring and addressing failure modes in deep learning-based online mapping systems for autonomous vehicles. The authors disentangle two key failure modes, localization overfitting (memorization of input features) and geometric overfitting (overfitting to map geometries), and propose evaluation metrics based on Fréchet distance that capture geometric reconstruction quality without threshold tuning. Practitioners should adopt these measures to develop more trustworthy generalisation assessments and design map geometry-aware training datasets.

## Why This Matters for Practitioners
If you're building or deploying real-time mapping systems for autonomous vehicles, this paper directly challenges your current validation practices. Most teams evaluate models on geographically overlapping splits (like the standard nuScenes split), which inflate performance by 25-79% compared to geographically disjoint splits, meaning your model might perform 25-79% worse in real-world scenarios than your test results suggest. The authors demonstrate that simply using geographically disjoint splits (as proposed by [24, 42]) reduces performance drops by 12-50% on standard models. To implement this, immediately replace your current validation splits with geometrically dissimilar splits (available for nuScenes and Argoverse 2), and adopt the Fréchet distance-based reconstruction score (M ± IQR) as a primary metric alongside standard mAP. This change alone will prevent your team from shipping systems that fail dramatically when encountering novel road geometries.

## Problem Statement
Imagine a navigation system that works flawlessly for every route you've taken before, but fails catastrophically when you encounter a new intersection layout. Current online mapping models suffer from exactly this: they memorize the specific geographical features of their training environments rather than learning to generalise. This isn't just about "better data", it's a fundamental flaw in how we measure model performance. When models trained on nuScenes show 79.47% mAP on the original split but only 24.96% on a geographically disjoint split (as shown in Table 1), it reveals that what we've been calling "generalisation" is actually location-specific memorization. The problem isn't just that models fail on new environments, it's that our entire benchmarking process has been systematically measuring the wrong thing.

## Proposed Approach
The authors propose a two-pronged framework: first, disentangling localization overfitting from geometric overfitting through carefully constructed evaluation sets; second, introducing metrics to quantify these failure modes. They derive validation subsets based on geographical distance (d(v)) and geometric similarity (s(v)), then use Fréchet distance-based reconstruction scores to measure performance. The key innovation is a geometric similarity measure (sim(v, t)) that evaluates map element matching by accounting for point order and global arrangement, unlike traditional Chamfer distance. For dataset bias, they introduce minimum spanning tree (MST) based diversity measures to quantify geometric diversity and coverage.

```python
def compute_geometry_similarity(v, t):
    """Compute geometric similarity between validation sample v and training sample t.
    
    Args:
        v: Validation sample map elements
        t: Training sample map elements
    
    Returns:
        similarity_score: Normalized cost (low = similar, high = dissimilar)
    """
    # Form assignment-cost matrix for each map element class
    Avt = form_assignment_cost_matrix(v, t)  
    # Get matched pair costs via minimum-cost bipartite matching
    amatched = min_cost_bipartite_matching(Avt)  
    # Calculate similarity score with penalty for unmatched elements
    similarity_score = (amatched + nunmatched * DELTA) / (nmatched + nunmatched)
    return similarity_score
```

## Key Technical Contributions
The paper makes three specific contributions that solve the measurement problem in online mapping:

1. **Geometric similarity measure based on discrete Fréchet distance**: Unlike Chamfer distance (which is permutation-invariant), this measure accounts for point order within map elements. It computes the minimum discrete Fréchet distance over all point orderings (including reversed order for polylines/polygons), then applies bipartite matching to form a cost matrix. This captures shape fidelity without threshold tuning, as shown in Figure 3 where Chamfer distance fails to detect geometric deviations that Fréchet distance captures.

2. **Failure mode scores for disentangled evaluation**: The authors define two novel metrics: a localization overfitting score (Oloc = (Mfar* - Mclose*)/Mclose*) that quantifies performance drop when geographical cues disappear, and a geometric overfitting score (Ogeom) that measures degradation as scenes become geometrically novel. For example, MapTRv2 showed Oloc = 24.73 on nuScenes original split, indicating strong localization overfitting.

3. **MST-based dataset sparsification strategy**: They show that training set diversity (measured as geomdiv(T) in Table 1) directly correlates with model performance. Their MST-based strategy reduces redundancy by pruning samples with redundant map geometries, improving geometric balancing and performance while shrinking training size. On nuScenes, geometric splits (geomdiv(T) = 91.3 km) achieved 9.75% higher mAP than geographically disjoint splits (geomdiv(T) = 80.6 km).

## Experimental Results
Experiments on nuScenes and Argoverse 2 across multiple state-of-the-art models (MapTRv2, MapTR, MapQR, MGMap) revealed consistent failure modes. On nuScenes original split, models showed 79.47% mAP but plummeted to 24.96% on a geographically disjoint split (Table 1), indicating severe geographic memorization. The Fréchet distance-based reconstruction score M (median) showed similar trends (60.95 vs 4.07), with IQR confirming the significance of the difference (1.94±3.05 vs 4.07±6.14).

The geometric overfitting score Ogeom (Table 1) revealed that all models suffer from geometric overfitting (values >9.75), with MapTRv2 showing the lowest scores (10.49 on nuScenes geometric split vs 21.22 on original split). Crucially, models trained on geometric splits (with higher geomdiv(T) = 91.3 km) achieved 28.37% mAP on validation compared to 24.96% on geographically disjoint splits (Table 1), demonstrating that geometric diversity directly improves generalisation.

## Related Work
The paper builds on Lilja et al.'s work highlighting geographic memorization in mapping models [24], but extends it by introducing geometric similarity as a critical factor beyond mere geographical overlap. While prior work focused on geographical splits [24, 31, 33, 42], this paper introduces a geometrically aware metric to quantify split quality and proposes a new dataset sparsification strategy based on MST diversity. It also extends traditional overfitting measures (like the Overfitting Index [1]) to the specific context of online mapping.

## Limitations
The study focuses on static map elements rather than dynamic scenes, which may not fully capture real-time navigation challenges. The geometric similarity measure assumes static map structures and doesn't account for temporal dynamics. The paper doesn't explore how their framework applies to edge cases like construction zones or temporary road modifications. The authors acknowledge that the MST sparsification strategy requires additional validation for datasets with complex temporal sequences.

## Appendix: Worked Example
Let's walk through the geometric similarity computation (sim(v, t)) with actual numbers from the paper. Consider two road map samples from nuScenes: a validation sample v (a residential street with 12 lane elements) and its closest training sample t (a similar residential street).

1. For each map element class (e.g., lane markings), form an assignment-cost matrix Avt between v's 12 elements and t's 10 elements (12x10 matrix).
2. For each element pair, compute discrete Fréchet distance over all point orderings (e.g., for polylines: 2 orientations × 2 directions = 4 possibilities).
3. Apply minimum-cost bipartite matching to get the optimal assignment.
4. For v's elements, the matched cost sum (amatched) = 4.8 (for 9 matched elements), with 3 unmatched elements incurring penalty δ = 2 (paper specifies this fixed penalty).
5. Compute geometric similarity cost: sim(v, t) = (4.8 + 3×2)/(9+3) = (4.8+6)/12 = 1.08.

This similarity score (1.08) is then used to stratify the validation set into bins based on geometric similarity (Bi), and the geometry overfitting score Ogeom is calculated by regressing Mfar,i (Fréchet median score) against the mean similarity in each bin.

## References

- Michael Hubbertz, Qi Han, Tobias Meisen, "Failure Modes for Deep Learning-Based Online Mapping: How to Measure and Address Them", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19852

Tags: #autonomous-vehicles #map-reconstruction #generalisation-robustness #dataset-bias #geometric-similarity
