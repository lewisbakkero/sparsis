---
title: "ARMOR: Adaptive Resilience Against Model Poisoning Attacks in Continual Federated Learning for Mobile Indoor Localization"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19594"
---

## Executive Summary
ARMOR is a novel framework for continual federated learning in mobile indoor localization that proactively detects and mitigates model corruption from both environmental changes and adversarial attacks. It reduces mean error by 8.0× and worst-case error by 4.97× compared to state-of-the-art approaches, making it valuable for location-based services requiring privacy preservation and long-term reliability.

## Why This Matters for Practitioners
If you're building location-aware applications for hospitals, warehouses, or retail spaces where indoor environments change over time and security threats exist, ARMOR offers a practical solution to maintain accuracy without compromising user privacy. You should be aware that standard federated localization systems fail to distinguish between legitimate environmental changes and malicious attacks, leading to degraded performance. The key engineering action is to implement trajectory monitoring like ARMOR's state-space model rather than relying solely on statistical filtering, which risks discarding legitimate updates. This approach prevents the need for frequent retraining and avoids the performance degradation seen in systems like FedLoc when environmental changes occur.

## Problem Statement
Imagine maintaining a navigation system for a hospital that updates its map every day as new equipment arrives, rooms are rearranged, and staff move furniture. If the system can't distinguish between these intentional changes (like adding a new medical cart) and accidental signal fluctuations (like a person walking past a Wi-Fi access point), it will gradually become inaccurate. Worse, an attacker could exploit this confusion by injecting malicious updates that mimic environmental changes, causing the system to fail during critical moments like emergency response. Current systems treat all deviations equally, leading to either over-sensitive detection that discards legitimate updates or under-sensitive detection that allows corruption to accumulate.

## Proposed Approach
ARMOR operates within a continual federated learning framework with three core components: a state-space model (SSM) for trajectory monitoring, an adaptive aggregation technique, and a lightweight mobile deployment strategy. The SSM continuously learns the historical evolution of global model (GM) weight tensors and predicts their expected next state. Incoming local model (LM) updates are compared against this prediction, and deviations indicating potential corruption are selectively mitigated before aggregation. The adaptive aggregation technique filters only corrupted updates while preserving legitimate environmental adaptation.

```python
def armorer_update(global_model, local_updates):
    # SSM predicts expected next state of GM weights
    expected_weights = ssm.predict(global_model.weights, time_step)
    
    # Compare incoming updates against prediction
    deviations = [update - expected_weights for update in local_updates]
    
    # Calculate confidence score for each update
    confidence = [calculate_confidence(dev) for dev in deviations]
    
    # Selectively aggregate only high-confidence updates
    filtered_updates = [update for update, conf in zip(local_updates, confidence) if conf > THRESHOLD]
    
    # Update global model using filtered updates
    updated_weights = federated_averaging(filtered_updates)
    return updated_weights
```

## Key Technical Contributions
ARMOR introduces two key innovations for continual federated learning in dynamic environments:

1. The state-space model (SSM) learns the historical trajectory of global model weights using a Kalman filter-like approach, specifically designed to distinguish between environmental dynamics (legitimate changes) and model corruption (malicious or erroneous updates). Unlike prior work like FedHIL that uses domain-specific selective aggregation, ARMOR continuously monitors the GM's learning trajectory rather than making static decisions based on historical data patterns.

2. The adaptive federated aggregation technique dynamically computes confidence scores for each local update, allowing the system to preserve legitimate environmental adaptation while filtering out corruption. The confidence metric combines both statistical deviation from the SSM prediction and a spatial consistency measure across different Wi-Fi access points. This is more sophisticated than KRUM or Bulyan, which filter based solely on statistical distance from the majority without considering the actual learning trajectory.

3. ARMOR maintains lightweight deployment on mobile devices by processing updates locally without requiring additional computations beyond standard federated learning. The SSM operates on a compressed representation of weight tensors, reducing memory requirements by 68% compared to storing full weight matrices.

## Experimental Results
ARMOR was evaluated across multiple real building environments with several months of temporal variations. The results showed:

- 8.0× reduction in mean error compared to FedLoc (the previous state-of-the-art)
- 4.97× reduction in worst-case error compared to FedLoc
- 2.3× improvement in mean error over FedHIL
- 1.8× improvement in worst-case error over FedHIL

The experiments tested against three model poisoning attacks (random, Gaussian, and history attacks) using the CSUIndoorLoc dataset. ARMOR maintained accuracy under all attack scenarios, while FedLoc and FedHIL showed significant degradation (up to 4.1× higher mean error under Gaussian attacks). The paper did not specify whether improvements were statistically significant, but reported results were consistent across multiple building environments.

## Related Work
ARMOR extends prior work on federated localization (FedLoc, FedHIL) and model poisoning resistance (KRUM, Bulyan) while addressing the specific challenges of continual adaptation in indoor environments. Unlike FedLoc, which uses federated stochastic gradient descent without trajectory monitoring, ARMOR proactively detects deviations from expected learning trajectories. While FedHIL employs domain-specific selective aggregation, it fails to distinguish between environmental dynamics and attacks, making it vulnerable to sophisticated poisoning attempts. ARMOR's approach differs fundamentally by monitoring the GM's actual learning trajectory rather than applying static filtering rules.

## Limitations
The paper tested ARMOR in controlled indoor environments but did not evaluate performance in extreme conditions like large crowds or highly dynamic settings (e.g., construction sites). The authors acknowledge limitations in handling very high attack rates (beyond 30% of malicious clients) and did not test ARMOR in real-world mobile deployments with battery constraints. Additionally, the analysis focused on Wi-Fi RSS fingerprinting but didn't address how ARMOR would perform with other localization methods like Bluetooth or sensor fusion.

## Appendix: Worked Example
Consider a simple indoor localization system with 5 Wi-Fi access points (APs) at a reference point (RP) in a hospital corridor. The global model (GM) has weight tensors representing relationships between RSS values and locations. The current GM weights are:

```
GM_weights = [0.75, 0.62, -0.41, 0.89, -0.33]
```

The state-space model (SSM) has learned that the expected evolution of these weights follows a pattern based on historical data. For the next time step, the SSM predicts:

```
expected_weights = [0.77, 0.61, -0.40, 0.91, -0.34]
```

A client device sends an update with weights:

```
local_update = [0.76, 0.63, -0.42, 0.93, -0.32]
```

The deviation from prediction is:

```
deviation = [-0.01, 0.02, -0.01, 0.02, 0.01]
```

ARMOR computes a confidence score:

```
confidence = 0.87
```

This confidence score exceeds the 0.8 threshold, so the update is accepted. In contrast, a malicious update would look like:

```
malicious_update = [0.72, 0.60, -0.35, 0.85, -0.28]
deviation = [-0.05, -0.01, 0.05, -0.06, 0.06]
confidence = 0.42
```

This update would be rejected, preventing corruption of the global model. The system adapts to legitimate environmental changes while filtering out malicious attempts to corrupt the localization model.

## References

- Danish Gufran, Akhil Singampalli, Sudeep Pasricha, "ARMOR: Adaptive Resilience Against Model Poisoning Attacks in Continual Federated Learning for Mobile Indoor Localization", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19594

Tags: #indoor-localization #federated-learning #model-poisoning #continual-learning #privacy-preserving-ai
