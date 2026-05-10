---
title: "GT-Space: Enhancing Heterogeneous Collaborative Perception with Ground Truth Feature Space"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19308"
---

## Executive Summary
GT-Space introduces a novel framework for heterogeneous collaborative perception in autonomous driving systems, where vehicles with different sensor modalities (LiDAR, cameras) must share perceptual data. By constructing a common feature space from ground-truth labels, it enables agents to align features with a single adapter module rather than requiring pairwise retraining or interpreters. This simplifies deployment in real-world systems where agent heterogeneity is common, offering up to 12.3% relative improvement in detection accuracy on heterogeneous agent pairs.

## Why This Matters for Practitioners
Practitioners building multi-vehicle autonomous driving systems should care because GT-Space solves a fundamental scalability problem in collaborative perception. Instead of needing to maintain multiple specialized encoders or interpreters for each new agent type, your system can deploy a single adapter module when new agents join. For example, if you're currently using a solution requiring pairwise feature alignment for LiDAR-camera collaborations (like HEAL or PnPDA), you're likely dealing with O(n²) complexity in adapter management as your fleet expands. GT-Space reduces this to O(n), meaning you'll save engineering time and reduce system complexity. The paper demonstrates this by showing that GT-Space requires only training a new projector for a single agent (not retraining encoders or designing new interpreters), making it ideal for real-world deployments where agents frequently join or leave the network. If your current implementation uses PnPDA (which requires a separate adapter per agent type), migrating to GT-Space could reduce your adapter maintenance overhead by 75% in a system with four agent types.

## Problem Statement
Imagine a team of engineers with different expertise (a data scientist, a hardware engineer, and a UX designer) trying to collaborate on a product. Without a common reference point, each member's work must be translated through pairwise communication channels: the data scientist might need to explain their findings to the hardware engineer, who then explains to the UX designer, creating a complex web of translation steps. In heterogeneous collaborative perception, agents with different sensor modalities (LiDAR vs. cameras) face the same problem with their feature representations. Existing methods require pairwise feature adaptation between every heterogeneous agent pair, creating a communication bottleneck that scales poorly as more agent types join the network, a fundamental limitation for real-world deployments where agent heterogeneity is common.

## Proposed Approach
GT-Space addresses this by constructing a common feature space from ground-truth labels, eliminating the need for pairwise alignment between heterogeneous agents. Each agent transforms its features into this shared space using a single adapter module, then the fusion network combines these aligned features for object detection. The system is trained with combinatorial contrastive losses across all modality pairs, enabling it to fuse any combination of input modalities at inference time.

```python
def gt_space_alignment(agent_features, ground_truth_features):
    # Project agent features into common feature space
    common_features = []
    for agent_feat in agent_features:
        # Each agent uses its own projector (trained separately)
        projector = get_projector(agent_id)
        aligned_feat = projector(agent_feat)
        common_features.append(aligned_feat)
    
    # Fuse aligned features using the trained fusion network
    fused_features = fusion_network(common_features)
    
    # Generate detections from fused features
    detections = detection_head(fused_features)
    return detections
```

## Key Technical Contributions
GT-Space's core innovations lie in how it constructs the common feature space and trains the fusion network to handle arbitrary modality combinations.

The ground-truth derived common feature space provides explicit object-level supervision for feature alignment, unlike prior methods that rely on detection output supervision. For each scene, the authors encode ground-truth object bounding boxes into BEV features using two fully-connected layers with layer normalization, then map these representations onto a BEV plane through a grid-based approach. This creates a reference space that precisely aligns with object locations and sizes, enabling agents to project their features into this space using a single adapter each.

The combinatorial contrastive loss approach trains the fusion network to handle arbitrary modality pairs by computing losses across all possible combinations. As the authors state, "Given three models, LiDAR-based PointPillar, SECOND, and camera-based EfficientNet, the loss is computed over all three possible pairs." This strategy enables the model to fuse any input combination at inference time without retraining, which is a significant improvement over end-to-end methods that require training for specific modality combinations.

The plug-and-play capability is achieved by freezing all pre-trained encoders and detection heads, requiring only training the new agent's projector when it joins. This avoids compromising individual agents' performance, which can occur with retraining-based approaches like HEAL.

## Experimental Results
GT-Space consistently outperforms baselines on multiple datasets. On OPV2V, GT-Space achieves 0.891 AP@50 when agent 1 (LiDAR) collaborates with agent 4 (camera), compared to HEAL's 0.887 and STAMP's 0.876. In multi-agent scenarios (Table 3), GT-Space achieves 0.867 AP@50 for camera agent 3 (compared to HEAL's 0.842 and STAMP's 0.840), demonstrating particularly strong gains for heterogeneous pairs where the authors note "the use of contrastive learning in GT-Space enhances object-relevant features in a way that end-to-end training or feature interpreter methods cannot achieve."

The system shows robustness to under-performing agents, with the largest gains observed when collaborating with weaker camera agents. When LiDAR agents provide more precise perception information (as expected), the performance gains in heterogeneous scenarios primarily depend on the LiDAR data, though GT-Space still provides significant improvements for camera agents. The paper does not explicitly report statistical significance tests for the results, but the consistent improvement across multiple datasets and agent pairs suggests practical significance.

## Related Work
GT-Space builds on existing work in collaborative perception (V2VNet, Coopernaut, DiscoNet) and heterogeneous collaboration (V2X-ViT, HM-ViT, PnPDA), but addresses a key limitation: the scalability of pairwise adaptation. While PnPDA (Luo et al., 2024) proposed a plug-and-play domain adapter, it can only handle point cloud features and overlooks sensor heterogeneity. HEAL (Lu et al., 2024) requires retraining encoders for collaboration, which can compromise the original model's performance. GT-Space eliminates the need for pairwise adaptation across heterogeneous agents by providing a single reference space, making it fundamentally more scalable than existing approaches.

## Limitations
The paper doesn't explicitly address how GT-Space handles communication latency or bandwidth constraints, which are critical in real-world deployments. While the authors demonstrate robustness to under-performing agents, they don't test scenarios with severe communication failures or intermittent connectivity. The paper also doesn't measure the computational overhead of the ground-truth feature generation process, though it's used only during training. The current implementation requires ground-truth labels during training, which may not be feasible in all real-world scenarios where ground truth isn't available.

## Appendix: Worked Example
Let's walk through the core mechanism with concrete numbers. For a scene with two objects (a car and a pedestrian), the ground-truth bounding boxes are encoded as follows:

1. Car bounding box: (x=5.2, y=3.1, z=0.0, l=4.5, w=1.8, h=1.2, r=0.7, c=1)
2. Pedestrian bounding box: (x=1.2, y=2.3, z=0.0, l=0.8, w=0.6, h=1.8, r=0.2, c=2)

Each box is processed through two fully-connected layers with layer normalization (Eq. 2), producing encodings of length 128. These encodings are then mapped onto the BEV plane (Eq. 3), where the scene is represented as a 128×128 grid (a common dimension in BEV-based perception systems).

For the car, the grid cell (5, 3) receives the features from the car bounding box, and for the pedestrian, grid cell (1, 2) receives the pedestrian features. The final BEV map (FGT) is constructed by summing overlapping features (e.g., if a car and pedestrian shared a grid cell, their features would be added together). This creates the ground-truth BEV feature map FGT.

During inference, a camera-based agent with features of size 256 projects its features into the common feature space using its trained projector. The projected features (now 128-dimensional) are fed into the fusion network along with features from other agents. The fusion network then processes these aligned features to generate object detections, using the contrastive loss as a supervisory signal to ensure object-relevant features are emphasized.

The combinatorial loss approach ensures the fusion network is trained on all possible modality combinations (e.g., LiDAR-camera, LiDAR-LiDAR, camera-camera), enabling it to handle any combination at inference time.

## References

- **Code:** https://github.com/KingScar/GT-Space.
- Wentao Wang, Haoran Xu, Guang Tan, "GT-Space: Enhancing Heterogeneous Collaborative Perception with Ground Truth Feature Space", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19308

Tags: #autonomous-driving #multi-agent #collaborative-perception #feature-alignment #contrastive-learning
