---
title: "Any-Subgroup Equivariant Networks via Symmetry Breaking"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19486"
---

## Executive Summary
ASEN is a single neural network architecture that can simultaneously exhibit equivariance to multiple symmetry groups by modulating an auxiliary input feature. This solves the key problem of inflexible equivariant models that require separate architectures for different symmetries, enabling a unified approach for building multi-modal foundation models. Practitioners should care because this reduces engineering overhead for symmetry-aware models and enables better transfer learning across domains with different symmetries.

## Why This Matters for Practitioners
If you're building production systems that process diverse data types with different geometric symmetries (like graphs, images, and sequences), this paper suggests you can replace multiple custom equivariant models with a single flexible ASEN architecture. For instance, in a multi-modal recommendation system handling both user interaction graphs (with permutation symmetry) and image content (with translation symmetry), you'd no longer need separate GNNs and CNNs for each modality. Instead, you could use a single ASEN-based model with different symmetry-breaking inputs for each data type, reducing deployment complexity while improving model transferability between tasks. The paper's experiments show this approach outperforms both separate equivariant models and single non-equivariant models, meaning you could get better results without increasing engineering complexity.

## Problem Statement
Current equivariant models are like specialized chefs: each can perfectly handle one type of cuisine (like French or Italian), but can't adapt to other styles without learning a new recipe. Similarly, graph neural networks (GNNs) designed for permutation symmetry can't process image data with translation symmetry, and vice-versa. This creates an engineering bottleneck: every time you encounter a new data type with different symmetries, you need to rebuild your model from scratch. The paper identifies this as a key barrier to developing flexible, multi-modal foundation models that can process diverse data types with appropriate symmetry constraints.

## Proposed Approach
ASEN starts with a fully permutation-equivariant base model (like a standard GNN) and then introduces a symmetry-breaking input feature whose automorphism group matches the target subgroup. By modulating this feature, a single network can be made equivariant to different subgroups. This avoids the need to design and implement separate equivariant layers for each symmetry.

For practical implementation, the authors use graph neural networks as the base model, and compute symmetry-breaking inputs as edge features. The algorithm computes edge features such that the automorphism group of the edge features matches the 2-closure of the target subgroup, which is computationally efficient.

```python
def compute_symmetry_breaking_features(group_generators):
    # Lift group generators to act on pairs of nodes
    lifted_generators = [lift_to_pairs(g) for g in group_generators]
    
    # Form the diagonal subgroup in the symmetric group on pairs
    diagonal_subgroup = PermutationGroup(lifted_generators)
    
    # Compute edge orbits to define edge features
    edge_orbits = diagonal_subgroup.orbits()
    
    # Assign edge features based on orbit membership
    edge_features = assign_features_by_orbit(edge_orbits)
    
    return edge_features
```

## Key Technical Contributions
ASEN's core innovation lies in how it handles symmetry breaking without requiring a separate model for each symmetry group. Specifically:

1. They leverage the 2-closure of a group to approximate the desired automorphism group efficiently, avoiding the computationally hard problem of finding exact symmetry-breaking inputs. For a permutation group G, the 2-closure G(2) is the largest group that acts the same as G on pairs of elements. When G is totally 2-closed (which includes many common groups), G(2) = G, making the approximation exact.

2. They prove that ASEN can simulate equivariant MLPs, which are a common type of equivariant model, to arbitrary accuracy. This means ASEN can replace specialized equivariant architectures with a single flexible model.

3. They demonstrate that the universality of ASEN follows from the universality of its base model. If the base model can approximate any G-equivariant function, then ASEN can too. This is crucial because it means the flexibility of ASEN doesn't come at the cost of reduced expressivity.

4. For practical implementation, they show that using graph neural networks as the base model with edge features computed to approximate the 2-closure of the target subgroup enables efficient symmetry breaking. This avoids the need for combinatorially large feature spaces required by other universal architectures.

## Experimental Results
The paper validates ASEN on multiple tasks:

1. For human pose estimation on Human3.6M dataset, using weakly sparse graph edge features (combining fully-connected and skeleton graph features) with S2 symmetry (left-right reflection) achieves 38.80 P-MPJPE error, outperforming other symmetry groups and matching results reported by Huang et al. (2023) who required multiple distinct equivariant MLPs.

2. For traffic flow prediction on METR-LA dataset, using sparse graph features with Sn1 × Sn2 symmetry (two clusters representing highway branches) achieves 2.69 MAE, outperforming both fully-connected features (2.72 MAE) and a non-equivariant baseline (DCRNN with Sn symmetry at 2.77 MAE).

3. For image tasks using Pathfinder-64, using local symmetry (sharing position vectors within p×p patches for p=2,3,4) improves accuracy from 0.656 (1D-PE, no symmetry) to 0.818 (2D-PE, no symmetry) while slightly reducing model parameters.

The paper doesn't explicitly report statistical significance for these results, but the comparisons are made against established baselines.

## Related Work
ASEN builds on existing work on subgroup equivariance by Blum-Smith et al. (2025), Ashman et al. (2024), and Lim et al. (2024), which also use symmetry-breaking inputs but are limited to single-task settings. The authors extend this to multitask and transfer learning scenarios. Unlike previous work, ASEN uses the 2-closure to approximate symmetry breaking efficiently and proves theoretical guarantees about expressivity and universality.

The paper also relates to symmetry breaking via node identification, which has been used to enhance GNN expressivity, but ASEN breaks symmetry uniformly for all inputs rather than input-dependently. Existing works on approximate equivariance focus on single groups, while ASEN enables a single model to handle multiple groups.

## Limitations
The authors acknowledge that constructing symmetry-breaking objects with exact automorphism groups can require high-order hypergraphs (up to K ≤ n), which may be inefficient. However, they demonstrate that using K = 2 (edge features) is sufficient for many practical cases where the target group G is totally 2-closed.

The paper doesn't explore the performance of ASEN on very large graphs or in distributed training settings, which could be a limitation for industrial-scale applications. Additionally, the method requires the generating elements of the symmetry group G, which might not always be readily available for all applications.

## Appendix: Worked Example
Consider a graph with 4 nodes representing a sequence of 4 elements: [A, B, C, D]. The target symmetry group is G = S2 × S2 × S2, meaning we want the model to be invariant to swapping the first two elements and the last two elements independently (i.e., the group of permutations that map {1,2} to {1,2} and {3,4} to {3,4}).

The 2-closure of G is G(2) = G itself, since G is totally 2-closed (as it's a direct product of symmetric groups). To compute the edge features A(2) with Aut(A(2)) = G(2), we:

1. Compute the orbits of the group G acting on pairs of nodes:
   - (1,2) ~ (2,1)
   - (3,4) ~ (4,3)
   - (1,3) ~ (2,4) ~ (1,4) ~ (2,3)

2. Assign edge features based on orbit membership:
   - For edge (1,2): feature = 0
   - For edge (3,4): feature = 1
   - For edges (1,3), (1,4), (2,3), (2,4): feature = 2

3. The resulting edge features matrix A(2) is:
   ```
   [[0, 0, 2, 2],
    [0, 0, 2, 2],
    [2, 2, 1, 1],
    [2, 2, 1, 1]]
   ```

4. This edge feature matrix has the desired automorphism group Aut(A(2)) = G = S2 × S2 × S2.

The base GNN then processes the input graph with these edge features, resulting in a model that is equivariant to G but not to the full symmetric group S4. This allows the model to capture the specific symmetries of the sequence while ignoring irrelevant permutations.

## References

- **Code:** https://github.com/amgoel21/perm_equivariance_graph_
- Abhinav Goel, Derek Lim, Hannah Lawrence, Stefanie Jegelka, Ningyuan Huang, "Any-Subgroup Equivariant Networks via Symmetry Breaking", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19486

Tags: #graph-neural-networks #equivariance #symmetry-breaking #multi-task-learning #foundation-models
