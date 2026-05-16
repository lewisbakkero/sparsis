---
title: "HALO: Hierarchical Reinforcement Learning for Large-Scale Adaptive Traffic Signal Control"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2506.14391"
---

# Technical Article

## Executive Summary
HALO is a hierarchical reinforcement learning framework that solves the scalability-coordination tradeoff in large-scale adaptive traffic signal control. By separating global planning (high-level guidance) from local execution (intersection-level control) and using adversarial goal-setting to align objectives, HALO delivers up to 6.8% lower average travel time and 5.0% lower average delay than state-of-the-art approaches across both standard benchmarks and a large-scale Manhattan-like network with 2,668 intersections.

## Why This Matters for Practitioners
If you're responsible for urban traffic management systems in cities with over 1,000 intersections, HALO provides a practical blueprint for scaling beyond the limitations of centralized approaches without sacrificing coordination. Instead of implementing complex centralized systems that become computationally intractable at city scale, you can deploy a hierarchical structure where the global guidance policy operates on a manageable subregion abstraction (typically 20-30 subregions for city-scale networks) while local policies handle real-time decisions at each intersection. The adversarial goal-setting mechanism means your system will learn to coordinate effectively without requiring excessive communication between agents, reducing latency by up to 50% compared to approaches using neighbour-only messaging. For production systems, this translates to reduced infrastructure costs (no need for high-bandwidth communication between all intersections), improved robustness to real-world conditions like peak hours and adverse weather, and a clear path to city-wide deployment without the maintenance overhead of centralized control. The lightweight global guidance signals (only 128-dimensional embeddings) can be broadcast every 5-10 seconds with minimal network overhead.

## Problem Statement
Imagine trying to coordinate a symphony orchestra with 1,000 musicians where each musician only hears the music from their immediate neighbours. This is the current challenge in large-scale traffic signal control: decentralized systems (each intersection acting as a musician) lack citywide coherence, while centralized systems (one conductor for the whole orchestra) become impossible to manage as the number of intersections grows. Traffic flow behaves like a complex wave: a small timing change at one intersection can trigger queue spillback several corridors away, fracture green waves, or reshape platoons far from the origin, a problem traditional approaches can't handle at city scale due to the non-local, long-range dependencies that emerge from tightly coupled cause-and-effect interactions.

## Proposed Approach
HALO decouples decision-making into two levels: a high-level global guidance policy that models network-wide traffic patterns and broadcasts compact guidance signals, and low-level local intersection policies that execute control conditioned on both local observations and global context. The global policy uses Transformer-LSTM encoders to model spatio-temporal dependencies across the network, while the local policies make real-time decisions at each intersection. Crucially, they introduce an adversarial goal-setting mechanism where the global policy proposes challenging-but-feasible targets that local policies are trained to surpass, fostering robust coordination.

Here's the core algorithm in pseudocode:

```python
def HALO_training_step(global_policy, local_policies, network_state):
    # Global policy generates guidance signals
    Fg, Gt = global_policy(network_state)
    
    # Local policies make decisions using global guidance
    for intersection in all_intersections:
        observation = combine_local_state(intersection, Fg)
        action = local_policies[intersection](observation)
    
    # Execute actions in the environment
    new_network_state = environment_step(action)
    
    # Compute rewards for global and local objectives
    global_reward = compute_global_reward(new_network_state, Gt)
    local_rewards = [compute_local_reward(intersection) for intersection in all_intersections]
    
    # Update policies using adversarial loss
    global_policy.update(global_reward, Gt)
    for intersection in all_intersections:
        local_policies[intersection].update(local_rewards[intersection], Gt)
```

## Key Technical Contributions
HALO introduces several key innovations that address the scalability-coordination tradeoff:

1. **Hierarchical Guidance with Subregion Aggregation**: HALO partitions the road network into M subregions (typically 20-30 for city-scale networks) rather than attempting to model the entire city at once. Each subregion generates a regional feature vector summarising local traffic status (e.g., number of stopped vehicles, cumulative waiting time, vehicle counts), which is processed by a Transformer with a learnable global token to capture spatial relationships. This reduces computational complexity from O(N) to O(M), where M is the number of subregions. The key implementation choice is using a learnable global token within the Transformer to summarise spatial context across all subregions, enabling the model to capture citywide patterns without requiring full connectivity between all intersections.

2. **Adversarial Goal-Setting Mechanism**: The global policy is rewarded for proposing challenging-but-feasible targets (Gt), while local policies are trained to surpass those targets. This transforms training into a minimax game that avoids trivial solutions. The specific implementation uses a margin-based hinge loss that penalizes when network-level metrics fall short of proposed targets, creating a natural incentive for local policies to improve performance beyond the global policy's expectations. For example, if the global policy proposes a target waiting time of 3,500 vehicle-seconds (Gt_w), the local policies learn to push the actual total waiting time below this threshold.

3. **Dynamic Neighbour Aggregation with Directional Attention**: For local intersection policies, HALO uses a lightweight attention mechanism over the four nearest neighboring intersections. The attention weights are learned dynamically using LeakyReLU activation followed by softmax, allowing the model to capture directional traffic flows (e.g., dominant upstream inflow during rush hours) rather than averaging neighbour messages. This preserves directional sensitivity and mitigates over-smoothing, a critical improvement over previous approaches that used simple averaging of neighbour messages, which would smooth out important directional signals during peak hours.

## Experimental Results
HALO was evaluated on five small-scale benchmarks (Cologne8, Grid4×4, Arterial4×4, Ingolstadt21, Grid5×5) and a large-scale Manhattan-like network with 2,668 intersections under three real-world traffic patterns (peak transitions, adverse weather, and holiday surges). The Manhattan-like network was designed to mimic real-world traffic flow patterns and included actual traffic data from Manhattan.

Results showed HALO consistently outperformed state-of-the-art baselines (MPLight, DenseLight, GPLight) across all scenarios, with improvements becoming more pronounced as network complexity increased. On the Manhattan-like network, HALO delivered 6.8% lower average travel time and 5.0% lower average delay compared to the best baseline (MPLight). The paper reports these improvements as statistically significant (though specific statistical tests aren't detailed in the provided text), with the largest gains observed during peak transition hours and adverse weather conditions. On the smaller benchmarks (up to 25 intersections), HALO showed competitive performance but became increasingly dominant as network size grew beyond 21 intersections.

## Related Work
HALO builds on existing hierarchical RL approaches but addresses key limitations of prior work. Unlike centralized approaches that become intractable at city scale, HALO maintains scalability through subregion aggregation. Unlike previous decentralized approaches that lack global context, HALO injects citywide awareness through the global guidance policy. The adversarial goal-setting mechanism improves upon previous coordination techniques (e.g., graph-attention messaging, pressure-based parameter sharing) by turning coordination into a minimax game that avoids trivial solutions and fosters more robust alignment between local execution and global objectives. HALO distinguishes itself from the meta-learning approaches (X-Light, FedLight, MetaLight) by focusing on a single city's traffic patterns rather than cross-scenario transfer, and from dense communication approaches (DenseLight) by using lightweight global guidance signals rather than requiring communication to all intersections.

## Limitations
The paper acknowledges that HALO's performance may be sensitive to the choice of subregion aggregation method, which requires careful design for different city layouts and may not generalise across diverse urban environments without reconfiguration. The adversarial training process may also be sensitive to the weighting parameters (βw, βq) in the hinge loss, though the paper doesn't explore this sensitivity in depth. Additionally, the evaluation was limited to specific traffic patterns (peak transitions, adverse weather, holiday surges), and the paper doesn't address how HALO would perform under more extreme conditions like catastrophic events (e.g., major accidents, natural disasters) or widespread infrastructure failures. The paper also doesn't provide a detailed analysis of computational overhead for the global policy, making it difficult to assess the tradeoffs for real-time implementation.

## Appendix: Worked Example
Let's walk through a concrete example of HALO in action using the Manhattan-like network with 2,668 intersections. 

Start with a city divided into 20 subregions (M=20) for global modelling. Each subregion contains approximately 133 intersections (2,668 ÷ 20 = 133.4). At time t, each subregion produces a regional feature vector summarising its traffic status (e.g., number of stopped vehicles, cumulative waiting time, vehicle counts).

For simplicity, let's say Subregion 1 (containing intersections 1-133) has a regional feature vector [120, 1500, 2500] representing stopped vehicles, total waiting time, and total vehicle count respectively. Similarly, Subregion 2 has [85, 1200, 2000], Subregion 3 has [210, 2800, 3500], and so on for all 20 subregions.

The Transformer processes these feature vectors with a learnable global token. After processing, the global token's embedding (E^T_G) becomes the global guidance embedding Fg, which is broadcast to all intersections. For this example, Fg = [0.75, 0.82, 0.68, ..., 0.91] (a 128-dimensional vector).

The LSTM-based sub-goal generation takes the aggregated subregion embeddings over a temporal window of length T=5 and produces a network-level target Gt = (Gt_w, Gt_q) = (3500, 2200) representing the desired total waiting time and queue length for the entire network.

At intersection 50 (within Subregion 1), the local observation combines:
- Its own state: [10, 120, 180] (stopped vehicles, waiting time, queue length)
- Dynamic neighbour features from its four nearest neighbours (using attention weights of [0.3, 0.25, 0.2, 0.25] for simplicity)
- The global guidance embedding Fg

The final observation for intersection 50 is formed by concatenating these features, resulting in a vector like [10, 120, 180, 8, 90, 150, 12, 110, 190, 15, 130, 210, ...] (with the exact dimensions depending on feature size).

The local policy then selects the optimal signal phase based on this observation, aiming to exceed the global target Gt = (3500, 2200) for waiting time and queue length. The global policy updates its strategy based on whether the network achieved the target (using the margin-based hinge loss), while local policies adjust to better surpass the targets in future steps.

## References

- Yaqiao Zhu, Hongkai Wen, Geyong Min, Man Luo, "HALO: Hierarchical Reinforcement Learning for Large-Scale Adaptive Traffic Signal Control", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2506.14391

Tags: #urban-transportation #multi-agent #reinforcement-learning #traffic-management #hierarchical-control
