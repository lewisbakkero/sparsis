---
title: "Multimodal Fused Learning for Solving the Generalized Traveling Salesman Problem in Robotic Task Planning"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2506.16931"
---

## Executive Summary

MMFL (Multimodal Fused Learning) introduces a novel framework for solving the Generalised Traveling Salesman Problem (GTSP) in robotic task planning by combining graph-based topology with image-based spatial representations. The approach consistently achieves optimal or near-optimal paths while maintaining real-time computational efficiency, outperforming state-of-the-art methods across various problem scales and spatial configurations.

## Why This Matters for Practitioners

If you're building warehouse automation systems where robots must collect items from multiple locations (e.g., selecting one shelf per SKU from distributed options), MMFL directly addresses the path planning bottleneck. Current approaches require trade-offs: exact algorithms like LKH (1.98 objective value for n=50) are too slow for real-time use (6.90 seconds per instance), while neural approaches like POMO (2.72 objective value) lack spatial awareness that leads to longer paths (20.89% gap from optimal). For production systems, MMFL delivers 0% gap on standard datasets (2.25 objective value for n=100) with only 0.13 seconds inference time, comparable to POMO but significantly better than LKH (2.73 objective, 6.90 seconds) and MA (2.71 objective, 106.80 seconds). Engineers should replace existing GTSP solvers with MMFL's path planning component, particularly in environments with complex spatial distributions where pure graph-based methods underperform.

## Problem Statement

Imagine a warehouse robot that must collect items from shelves scattered across the facility: for each SKU category, multiple shelves contain that item, and the robot must choose one shelf per category. Traditional graph-based approaches treat this as a pure connectivity problem, ignoring spatial relationships between shelves. For example, if all "SKU A" shelves cluster near the entrance while "SKU B" shelves cluster in the back, a graph-only path might zig-zag inefficiently between these clusters. Current solutions either use slow exact algorithms (practically unusable for real-time systems) or require extensive tuning of metaheuristics like genetic algorithms, making them brittle across different warehouse layouts.

## Proposed Approach

MMFL represents GTSP instances through dual modalities: graph structure (capturing connectivity relationships) and spatial image (encoding geometric relationships). The framework converts node coordinates into a spatial image, processes both modalities through specialized encoders, fuses them with a dedicated bottleneck mechanism, and generates paths through a multi-start decoder. Its Adaptive Resolution Scaling strategy ensures consistent spatial information density regardless of problem size, while the multimodal fusion mechanism intelligently integrates topological and geometric information for better path planning decisions.

```python
def mmfl_path_planning(g tsp_instance):
    # Convert GTSP instance to spatial image
    image = coordinate_to_image(g, adaptive_resolution(g))
    
    # Process through graph and image encoders
    graph_repr = graph_encoder(g)
    image_repr = image_encoder(image)
    
    # Fuse representations with bottleneck tokens
    fused_repr = multimodal_fusion(graph_repr, image_repr)
    
    # Generate path with multi-start exploration
    path = multi_start_decoder(fused_repr)
    
    return path
```

## Key Technical Contributions

MMFL introduces several innovations that directly solve the problem of spatial-aware path planning:

1. **Coordinate-to-image builder with Adaptive Resolution Scaling**: Unlike previous methods that treat GTSP as purely topological, MMFL converts node coordinates into spatial images where pixel values represent cluster memberships. The Adaptive Resolution Scaling formula (W = H = floor(α√n) × w) dynamically increases resolution with problem size (α=1.5, w=16), maintaining consistent node density. This ensures the model doesn't lose spatial relationships as problem size increases, critical for warehouse environments where node distributions vary significantly.

2. **Bidirectional cross-attention fusion with bottleneck tokens**: MMFL's fusion mechanism uses 10 learnable bottleneck tokens per modality (graph and image) that act as information conduits. Instead of simple concatenation or attention, these tokens allow the model to selectively focus on relevant information from each modality. The fusion layers iteratively refine representations through cross-attention (Gout = MHA(Gin, Iin, Iin), Iout = MHA(Iin, Gin, Gin)), enabling more context-aware decisions than previous approaches.

3. **Multi-start decoder for parallel exploration**: The decoder begins at the depot node, then explores k = int(n/4) nearest neighbours (e.g., 25 neighbours for n=100) in parallel. This overcomes the local optima problem of single-start decoders. The policy computation (αi = C · tanh(qT WKhi/√d)) dynamically weights node compatibility based on the fused representation, masking visited nodes and clusters to enforce GTSP constraints.

## Experimental Results

MMFL consistently outperforms all baselines across every tested metric:

- **Optimality gap**: For n=100, m=20, MMFL achieves 0% gap (optimal solution) compared to POMO's 20.89%, LKH's 21.33%, and ALNS's 17.33% (Table 1). In Large Groups (10-12 groups, 8-10 nodes each), MMFL improves by 35.83% over POMO and 45.00% over LKH (Table 3).

- **Computational efficiency**: MMFL's inference time (0.13 seconds) is comparable to POMO (0.12 seconds) but significantly faster than LKH (6.90 seconds) and MA (106.80 seconds) for n=100. This efficiency enables real-time execution on robotic platforms.

- **Generalisation**: MMFL achieves best performance in 3 of 4 group distributions (Table 2) and consistently outperforms baselines across varying group sizes (Table 3), demonstrating strong capability to handle diverse spatial arrangements.

Physical robot tests confirmed these results, MMFL successfully navigated multi-zone exploration tasks as shown in Figures 5-6, with white paths representing the solution.

## Related Work

MMFL bridges a critical gap between graph-based learning approaches (like POMO) and spatial reasoning methods. Previous neural approaches for GTSP either focused solely on graph inputs (ignoring spatial relationships) or required high-quality datasets that are difficult to obtain. Hybrid approaches like [27] combine neural networks with exact algorithms but require substantial computation. MMFL uniquely integrates both topological and spatial information through a dedicated fusion mechanism, avoiding the need for extensive dataset collection.

## Limitations

The authors acknowledge three limitations: (1) Limited generalisation to novel distributions (e.g., asymmetric layouts), (2) Computational scalability challenges for very large instances (thousands of nodes), and (3) Assumption of a static environment with known node positions, which doesn't account for dynamic obstacles or localization uncertainty. For production systems, these limitations suggest MMFL should be combined with dynamic planning modules for real-world environments.

## Appendix: Worked Example

Let's walk through a specific GTSP instance with n=100 nodes (5 nodes per cluster across m=20 clusters) to see how MMFL processes spatial information:

1. **Input representation**: Coordinates are normalized to [0,1]². Using ARS, image resolution is W = H = floor(1.5×√100)×16 = 240 pixels (α=1.5, w=16).

2. **Image construction**: A node at (0.75, 0.45) in cluster 3 is mapped to pixel (180, 108) with value 4 in the 240×240 image. Each cluster gets a distinct colour.

3. **Image processing**: The image encoder splits the 240×240 image into 15×15 = 225 patches (16×16). Each patch is embedded into 128 dimensions through a ViT transformer stack.

4. **Graph processing**: Each node's embedding is [x, y, cluster_id] → 128 dimensions via linear projection, processed through 3 graph encoder layers.

5. **Fusion**: The multimodal fusion module uses 10 bottleneck tokens per modality. Cross-attention processes graph features (hgraph) and image features (himage) through 3 fusion layers. The final fused representation combines graph and image features with α=0.5.

6. **Path generation**: The multi-start decoder begins at depot (node 0), then explores 25 nearest nodes (k = int(100/4)). At each step, compatibility scores αi are computed using the fused representation, with visited nodes masked. The optimal path (2.25 total length) is generated.

This workflow demonstrates how MMFL leverages spatial relationships (e.g., clustering of shelves) that graph-only approaches miss, resulting in more efficient paths.

## References

- **Code:** https://github.com/Carveller/MMFL-for-GTSP
- Jiaqi Cheng, Mingfeng Fan, Xuefeng Zhang, Jingsong Liang, Yuhong Cao, Guohua Wu, Guillaume Adrien Sartoretti, "Multimodal Fused Learning for Solving the Generalized Traveling Salesman Problem in Robotic Task Planning", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2506.16931

Tags: #robotics #task-planning #multimodal-learning #graph-neural-networks #optimisation
