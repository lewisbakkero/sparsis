---
title: "Sensing Without Colocation: Operator-Based Virtual Instrumentation for Domains Beyond Physical Reach"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2510.18041"
---

## Executive Summary
STONe establishes a new sensing principle that replaces physical sensors with learned operators mapping sparse ground-based measurements to inaccessible domains. It reconstructs global radiation dose fields at 10,000m altitude from 12 ground-based neutron monitors with sub-millisecond inference, bypassing the need for physical sensors in unattainable locations. This transforms how we approach sensing in domains where physical measurement is impossible.

## Why This Matters for Practitioners
If you're building systems for environmental monitoring in inaccessible locations, like aviation safety, nuclear reactor monitoring, or subsurface infrastructure, this paper demonstrates that you can replace physical sensor deployment with operator-based virtual sensing. Instead of investing in expensive, hard-to-maintain sensor networks that can't reach target domains, you can deploy embedded AI on low-power hardware that maps existing accessible measurements to the inaccessible field. For example, if you're responsible for monitoring radiation exposure in commercial aircraft, you no longer need to wait for sensor development or physical deployment at altitude; you can leverage existing ground station data through a model like STONe. The key engineering decision: prioritise operator-theoretic sensing design over sensor deployment where physical access is impossible, reducing both cost and complexity while enabling real-time monitoring.

## Problem Statement
Imagine trying to measure the temperature inside a burning house from outside, traditional sensors can't survive the environment, and you can't position them where the data is needed. This is exactly the problem with cosmic radiation monitoring at aviation altitudes: the target domain (10,000m) is inaccessible to sensors, while the only available measurements (ground-based neutron monitors) are physically disjoint from the desired field. Classical sensing rests on the assumption that sensors must be colocated with the target quantity, but this barrier fails completely for applications where the target domain is physically unreachable, making conventional approaches impossible.

## Proposed Approach
STONe is a non-autoregressive neural operator framework that directly maps sparse, indirect measurements from ground-based neutron monitors to the complete global radiation dose field at 10,000m altitude in a single forward pass. Its core insight is that when the sensor manifold and target field manifold are physically disjoint, a learned operator bridging them constitutes the instrument itself. The system combines a recurrent or attention-based branch network to process historical sensor data with a coordinate-conditioned trunk network to generate spatiotemporal basis functions, enabling stable long-horizon predictions without iterative error accumulation.

```python
def ston_extrapolate(sensor_history, spatial_coords, K_fut):
    """STONe's non-autoregressive operator for virtual field reconstruction.
    
    Args:
        sensor_history: (T, N) tensor of ground sensor time series (T=180 steps, N=12 sensors)
        spatial_coords: (P, 2) tensor of latitude-longitude grid points
        K_fut: number of future time steps to predict
        
    Returns:
        dose_field: (P, K_fut) tensor of reconstructed radiation dose fields
    """
    branch_output = branch_network(sensor_history)  # Temporal coefficients
    trunk_output = trunk_network(spatial_coords, K_fut)  # Spatiotemporal basis
    dose_field = einsum("q, qpk -> pk", branch_output, trunk_output) + bias
    return dose_field
```

## Key Technical Contributions
STONe advances virtual sensing through three novel mechanisms that fundamentally differ from prior approaches:

1. **Operator as instrumentation principle**: Unlike prior neural operators that serve as computational accelerators for existing physics-based simulators, STONe formalises the learned operator itself as the measurement apparatus. This reframes the problem from "how to accelerate an existing simulation" to "how to create a virtual sensor that replaces unavailable physical measurements." The operator G: Hₛ → Hₜ mapping ground sensor inputs to high-altitude dose fields constitutes the instrument, with the physical hardware (like the Jetson Orin Nano) acting as the deployment platform rather than the sensor.

2. **Non-autoregressive cross-domain operator decomposition**: STONe's core innovation is its non-autoregressive formulation that bypasses the error accumulation inherent in autoregressive models. By learning the full sequence-to-sequence mapping H(Y, r) → X₁:ₖ in a single pass (rather than step-by-step prediction), it avoids error propagation over long horizons. The architecture achieves this through a coordinate-conditioned trunk network that directly outputs basis functions for all future time steps, eliminating the need for iterative state propagation.

3. **Operational-scale embedding in physical constraints**: STONe's deployment on a Jetson Orin Nano at 7.3W average power with 143.3MB GPU memory footprint demonstrates that operator-theoretic sensing can operate within the physical constraints of field-deployable hardware. This represents a significant departure from prior neural operator applications that required high-performance GPUs for inference, making virtual sensing practical for remote monitoring systems.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
STONe achieved 0.93 relative L2 error on the test set (vs 0.99 for the best baseline, DeepONet), demonstrating superior reconstruction capability for the radiation dose field. The model achieved sub-millisecond inference time (43.5ms per complete 180-day global rollout), which is orders of magnitude faster than Monte Carlo transport solvers requiring hours. The authors trained on 23 years of neutron monitor data (2001-2023) at daily temporal resolution with 12 ground stations, processing 180-day time windows for both input and output. The Jetson Orin Nano deployment achieved 7.3W average power consumption and 143.3MB GPU memory footprint without modification from the training environment.

## Related Work
STONe builds on neural operator frameworks like DeepONet and FNO but fundamentally reframes their purpose. Unlike prior work that uses neural operators as surrogates for known simulators, STONe establishes a new instrumentation principle where the operator itself constitutes digital instrumentation. The paper positions its work against sequence-to-sequence models (e.g., FourCastNet) and physics-informed models (e.g., TI(L)-DeepONet), showing that these approaches fail to address the fundamental domain mismatch between ground-based observations and high-altitude fields. The authors explicitly cite the structural gap in existing frameworks that cannot stably bridge disjoint physical regimes at global scale over extended horizons.

## Limitations
The authors acknowledge that STONe's applicability depends on the physical stability of the atmospheric transfer function over the sensing horizon, which may not hold in domains with rapidly changing physics. The approach requires long-term temporal observability of the underlying drivers (solar modulation cycles), which may not be available for all environmental monitoring applications. The paper focused on radiation dose at 10,000m altitude but didn't test the framework's performance across different altitude ranges or atmospheric conditions. Additionally, the model was trained on historical data from neutron monitors but wasn't validated on real-time sensor networks with varying deployment patterns.

## Appendix: Worked Example
Here's how STONe processes a single input to generate a radiation dose field at 10,000m altitude:

1. **Input preparation**: The system receives 180 days of neutron count data from 12 ground stations (shape: 180 × 12), normalized to a [0,1] scale. For example, the neutron counts for station 1 over 180 days might be [1.2, 1.5, 0.8, ..., 2.1] (values scaled to the 12-station average).

2. **Branch network processing**: The branch network (a Transformer with 8 attention heads) encodes this input into a temporal coefficient vector of dimension 128. For instance, the network might output [0.32, -0.15, 0.78, ..., 0.05] (128 elements), representing the encoded temporal dynamics of the radiation field.

3. **Trunk network processing**: The trunk network receives spatial coordinates for 360 × 180 (global grid points) and the future horizon (180 days), processing them into a matrix of dimension 128 × 1 × 180. For the first latitude-longitude point (0.1, 0.2), the trunk network might output basis function values like [0.83, -0.45, 0.21, ..., 0.09] for each of the 128 latent dimensions.

4. **Field reconstruction**: The system combines the branch output and trunk output using tensor contraction (as shown in the pseudocode). For the first output point and time step, the reconstruction would be the dot product: (0.32 × 0.83) + (-0.15 × -0.45) + (0.78 × 0.21) + ... + (0.05 × 0.09) = 0.43. This value represents the radiation dose at the specified location and time.

5. **Full field generation**: This process repeats for all 360 × 180 grid points and 180 future time steps, producing a complete radiation dose field reconstruction in a single forward pass.

## References

- Jay Phil Yoo, Kazuma Kobayashi, Souvik Chakraborty, Syed Bahauddin Alam, "Sensing Without Colocation: Operator-Based Virtual Instrumentation for Domains Beyond Physical Reach", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2510.18041

Tags: #environmental-monitoring #radiation-safety #neural-operators #embedded-ai #sensing-instrumentation
