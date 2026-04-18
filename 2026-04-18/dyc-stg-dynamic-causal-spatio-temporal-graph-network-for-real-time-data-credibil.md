---
title: "DyC-STG: Dynamic Causal Spatio-Temporal Graph Network for Real-time Data Credibility Analysis in IoT"
category: "AI Applications"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36973"
---

## Executive Summary

DyC-STG introduces a novel framework for real-time data credibility analysis in IoT sensor networks, specifically targeting smart home environments where data reliability directly impacts critical services. It addresses two fundamental limitations of existing spatio-temporal graph models: static graph topologies that fail to capture physical dynamics and confusion between spurious correlations and true causality. Practitioners should care because unreliable IoT data underpins smart home systems that manage security, energy, and user experience - and this framework provides a method to distinguish genuine sensor readings from spurious correlations without costly manual verification.

## Why This Matters for Practitioners

If you're building a smart home analytics system that relies on sensor data for critical decisions (like security alerts or energy management), this paper suggests you should re-evaluate how you model sensor relationships. Current approaches often treat sensor connections as fixed, which leads to failures when physical events alter sensor correlations (e.g., when a door opens, the relationship between indoor/outdoor temperature sensors changes). Instead, you should implement event-driven graph updates that directly reflect physical state changes - for example, having door sensors trigger topology adjustments in your graph representation.

The paper also highlights that simply capturing correlations between sensors (like noticing coffee machines and toasters often activate together) isn't enough. Your system should enforce temporal causality, meaning it shouldn't assume that if two events co-occur, one causes the other. For instance, if your system learns that coffee machine usage always precedes toaster use, it should be able to correctly handle scenarios where only the coffee machine is used (without flagging it as an anomaly). This means incorporating causal masking in your temporal models to prevent the system from learning spurious correlations.

## Problem Statement

Imagine a smart home where sensors incorrectly report high energy consumption when a window is opened. Existing models treat sensor relationships as fixed, like a static map of a building that never changes. But in reality, opening a window creates a new physical relationship between indoor and outdoor temperature sensors - the correlation changes dramatically. This is like a navigation app that won't update its routes when a road is closed, leading you to take a detour that doesn't exist. The paper identifies that current IoT models can't capture these physical dynamics, leading to unreliable data credibility assessment.

## Proposed Approach

DyC-STG combines two core ideas: an event-driven dynamic graph module that adapts the graph topology in real-time to reflect physical state changes, and a causal reasoning module that ensures temporal precedence. The architecture processes sensor data through a dynamic graph that evolves based on physical events (like door openings), followed by spatial and temporal modelling with causal constraints.

```python
def dy_c_stg(sensors, control_nodes, sensor_data):
    # Dynamic Graph Construction: Modulates base graph based on physical states
    dynamic_graph = modulate_graph(base_graph, control_nodes, sensor_data)
    
    # Spatial Dependency Modelling: Uses GAT layers on dynamic graph
    spatial_features = spatial_gat_layers(sensor_data, dynamic_graph)
    
    # Temporal Feature Extraction: Bidirectional Transformer on spatial features
    temporal_features = bidirectional_transformer(spatial_features)
    
    # Causal Context Refinement: Enforces temporal causality with masking
    causal_features = causal_masked_self_attention(temporal_features)
    
    # Gated Fusion: Combines causal and bidirectional features
    fused_features = gated_fusion(temporal_features, causal_features)
    
    # Output: Predicts data credibility score
    credibility = prediction_head(fused_features)
    return credibility
```

## Key Technical Contributions

DyC-STG's core innovations lie in how it implements physical grounding and causal reasoning within the architecture.

1. **Event-driven dynamic graph construction**: Instead of learning static or slowly evolving graph structures, the model uses physical control nodes (like doors) to directly modulate edge weights. When a door opens (state = 1), the adjacency matrix for relevant sensor pairs is multiplied by the control node state (At_ij = fmod(st_c) · ABase(i, j)). This means the graph topology explicitly responds to physical events rather than learning correlations abstractly from data. The paper specifies they use an identity map (fmod(s) = s) for binary controls, avoiding complex learned transformations.

2. **Causal reasoning with strict temporal masking**: The framework redefines the temporal receptive field to respect temporal precedence. Instead of using standard bidirectional self-attention that accesses future data, it applies a causal mask that confines each time step's receptive field to only historical context (steps 1 to t). This is implemented through a masked self-attention mechanism that sets attention scores for future time steps to -∞, forcing the model to distinguish cause-and-effect relationships from spurious correlations.

3. **Gated fusion of causal and spatial representations**: The model intelligently combines the bidirectional temporal features (Hst) with the causally constrained features (HCausal) through a gate vector generated by concatenating both representations and applying a sigmoid activation. The gate values dynamically determine how much weight to assign to causal versus non-causal relationships at each sensor and time step.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results

DyC-STG achieves an F1-Score of 0.9297 and an AUC of 0.9886 on the SHSD92 dataset, establishing a new state-of-the-art. This represents absolute improvements of 1.44 percentage points (F1-Score) and 0.51 percentage points (AUC) over the strongest baselines. The paper compares against 12 state-of-the-art baselines across three categories: graph neural network-based models (DCRNN, STGCN, GWNet, MTGNN, STFGNN, STGNCDE), self-attention-based models (STTN, GMAN, ASTGCN, STJGCN), and dynamic graph models (AGCRN, PDFormer).

The results demonstrate that dynamic graph models substantially outperform static methods, validating the necessity of modelling dynamic dependencies. Crucially, DyC-STG shows superior robustness on the more complex SHSD104 dataset, where its F1-Score drops by only 1.2% compared to a 5.4% decline for classic models like STGCN. The paper does not explicitly state whether the improvements are statistically significant, but it notes they used a validation set to calibrate the threshold ζ and employed a Focal Loss function to address class imbalance (α = 0.75, γ = 2.0).

## Related Work

The paper positions itself as addressing a critical gap between two separate lines of recent work: dynamic graph models that capture spatial dependencies but often learn abstract correlations ungrounded in physical events, and causal discovery approaches that attempt to learn causal graphs but often decouple this from the end-to-end learning process. While some prior work has explored event-driven graph adaptation (Geng et al. 2024; Liu and Zhang 2024), these models learn graph structures that evolve too slowly to capture abrupt topological changes from discrete events. Similarly, causal discovery methods (Fu, Pan, and Zhang 2024; Gong et al. 2024) often rely on strong statistical assumptions and don't integrate causal learning with representation learning.

DyC-STG bridges this gap by directly grounding the dynamic graph in physical events through control nodes, and by embedding causal reasoning within the architecture through strict temporal masking, rather than learning causal graphs separately.

## Limitations

The paper doesn't explicitly state limitations beyond the scope of its focus on smart home environments with 31 heterogeneous sensors. The authors acknowledge they designed SHSD104 to present more complex scenarios than SHSD92, suggesting the framework's performance may vary with different sensor configurations or environments with more complex dynamics.

A key limitation for practitioners is that the method requires identifying and logging control nodes (like doors and windows) for physical state changes. This means the approach may not generalise to IoT environments where such physical control nodes aren't well-defined or easily measurable. Additionally, while the paper demonstrates the method's effectiveness on credibility analysis, it doesn't address how the framework would scale to millions of sensors in industrial IoT deployments, which would require significant computational optimisation.

## Appendix: Worked Example

Let's walk through a single sensor reading's journey through the DyC-STG framework with concrete values. We'll focus on a temperature sensor in the kitchen (sensor node 1) during a scenario where a door (control node 5) opens at time t=3.

**Input Data (at time t=3):**
- Sensor readings: [22.1°C (kitchen), 20.3°C (living room), 1.2kW (coffee machine), 3.2kW (toaster)]
- Control node state: Door closed → st_c = 0 (at t=2), Door open → st_c = 1 (at t=3)
- Base graph (ABase): Adjacency matrix showing all possible physical connections (e.g., kitchen to living room, kitchen to coffee machine)

**Dynamic Graph Construction:**
- At t=2 (door closed): At=2 = fmod(st_c=0) · ABase = 0 · ABase = 0 (no connections through door)
- At t=3 (door open): At=3 = fmod(st_c=1) · ABase = 1 · ABase = ABase (connections restored)
- For sensor node 1 (kitchen temperature), the relevant connections (to living room temperature and coffee machine) are activated at t=3 but not at t=2.

**Spatial Dependency Modelling:**
- GAT processes sensor data through the dynamic graph at t=3:
  - For sensor node 1 (kitchen), it aggregates information from neighbours (living room temperature, coffee machine) using the dynamic graph topology.
  - The attention weights are masked to only allow connections present in At=3 (so the living room connection is active, but connections through the closed door are inactive).
  - This creates a spatially-aware representation Hspatial for sensor node 1 that reflects its current physical context.

**Temporal Feature Extraction:**
- The bidirectional Transformer processes the sequence of spatially-aware features across the 150-step window (5 minutes).
- For time step t=3, it captures relationships beyond just the immediate context (e.g., how temperature changes at t=2 relate to current state).

**Causal Context Refinement:**
- The causal mask restricts attention at t=3 to only steps 1-2 (not steps 4-150).
- This means the model cannot use future data to predict the current state, forcing it to learn true cause-and-effect relationships.
- For example, it learns that kitchen temperature (t=2) may influence the current reading (t=3), but cannot learn that coffee machine usage (t=3) causes the temperature reading (t=3) - a spurious correlation.

**Gated Fusion:**
- The gate vector G combines the bidirectional representation (Hst) with the causal representation (HCausal).
- For sensor node 1 at t=3, the gate might assign 60% weight to the causal representation (which learned the true temperature relationship) and 40% to the bidirectional representation (which might contain spurious correlations).
- The final fused representation Hfused integrates this weighted combination.

**Output:**
- The prediction head converts Hfused into a credibility score, which is thresholded at ζ (calibrated on validation data) to determine if the reading is trustworthy.
- In our scenario, the credibility score would be high because the model correctly learned that the door opening (physical event) changed sensor relationships, and filtered out spurious correlations.


## References

- Guanjie Cheng, Boyi Li, Peihan Wu, Feiyi Chen, Xinkui Zhao, Mengying Zhu, "DyC-STG: Dynamic Causal Spatio-Temporal Graph Network for Real-time Data Credibility Analysis in IoT", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36973
