---
title: "Pseudo-Simulation for Autonomous Driving"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2506.04218"
---

## Executive Summary
Pseudo-simulation introduces a novel evaluation paradigm for autonomous vehicles that combines the scalability of open-loop evaluation with the error-recovery assessment capabilities of closed-loop testing. By pre-generating synthetic observations using 3D Gaussian Splatting and applying proximity-based weighting, it achieves strong correlation with closed-loop results (R² = 0.8) while requiring only 1/6 of the computational resources.

## Why This Matters for Practitioners
If you're building autonomous systems that must navigate unpredictable traffic scenarios, this paper directly addresses a critical gap in your evaluation pipeline. Traditional open-loop evaluation (like NAVSIM v1) fails to test error recovery, leading to deployments that work under expert-aligned conditions but fail when the system deviates from the expected path. Pseudo-simulation reveals these hidden failure modes, such as PDM-Closed's poor performance on comfort metrics (HC: 47.9% in Stage 2) that were previously undetected. For your production systems, this means: **Stop relying solely on displacement error metrics. Implement a two-stage evaluation framework that tests robustness to minor deviations, using pre-generated synthetic observations to avoid the computational overhead of sequential simulation.** If you're using existing benchmarks, your models might be passing metrics while failing in real-world scenarios where small path deviations trigger compounding errors.

## Problem Statement
Current evaluation approaches for autonomous vehicles face a fundamental trade-off: closed-loop testing is like testing a car's brakes by crashing it on a controlled track (accurate but expensive and dangerous), while open-loop testing is like judging braking performance by measuring how close the car stays to the centre line on a perfectly straight highway (incomplete and misleading). Neither approach adequately tests what happens when the vehicle's path deviates slightly from the intended route, like when it drifts to avoid an obstacle but then struggles to recover, creating a cascade of errors. This is the critical gap that pseudo-simulation solves.

## Proposed Approach
Pseudo-simulation evaluates autonomous vehicles across two stages. Stage 1 assesses performance on the real-world observation from the dataset. Stage 2 evaluates performance on synthetic observations generated from these original frames using 3D Gaussian Splatting, then weights these results based on proximity to the trajectory endpoint predicted in Stage 1. This non-interactive method enables scalable, parallel evaluation that captures error recovery and causal confusion without sequential simulation.

```python
def compute_pseudo_simulation_score(real_obs, synthetic_observations, ego_trajectory):
    # Stage 1: Evaluate on real observation
    stage1_score = evaluate_on_observation(real_obs)
    
    # Stage 2: Evaluate on synthetic observations
    stage2_scores = []
    for obs in synthetic_observations:
        stage2_scores.append(evaluate_on_observation(obs))
    
    # Weight synthetic scores based on proximity to Stage 1 endpoint
    weights = compute_gaussian_weights(synthetic_observations, ego_trajectory[-1])
    
    # Aggregate scores
    stage2_score = sum(w * s for w, s in zip(weights, stage2_scores))
    combined_score = stage1_score * stage2_score
    return combined_score
```

## Key Technical Contributions
The paper's core innovations address specific limitations of prior evaluation approaches:

1. **Pre-generated synthetic observation pipeline**: Unlike traditional simulation where observations are generated online, the authors use 3D Gaussian Splatting to reconstruct driving scenes and generate diverse observations (varying position, heading, and speed) before evaluation. This eliminates the need for sequential simulation and enables parallel processing of all synthetic observations. The pipeline samples points around the expert's 4-second endpoint (with lateral sampling every 0.5m up to 2m on each side, longitudinal sampling every 5m across the physical range of possible positions).

2. **Proximity-based weighting scheme**: The authors introduce a Gaussian-weighted average to aggregate Stage 2 scores, where weights are calculated based on the Euclidean distance between each synthetic observation's start point and the endpoint of the trajectory predicted in Stage 1. This prioritises "more likely futures" and prevents undue penalties for failures in improbable scenarios (e.g., a synthetic observation 5m from the endpoint gets 70% of the weight of one 1m away).

3. **Extended Predictive Driver Model Score (EPDMS) with context-aware filtering**: The paper refines the existing scoring metric by introducing filterm, which ignores penalties for rule violations also committed by the human expert driver in the same scene. This prevents penalising contextually justified maneuvers (like briefly entering the opposite lane to bypass an obstacle), making the evaluation more realistic and less prone to false negatives.

## Experimental Results
The authors evaluated 83 diverse planners (10 rule-based, 15 IDM, 15 PDM-Closed, 22 PlanCNN, 24 Urban Driver) across 244 initial observations (Stage 1) and 4164 synthetic observations (Stage 2). Pseudo-simulation achieved Pearson correlation r = 0.89 (R² = 0.8) with closed-loop simulation, outperforming the best open-loop approach (r = 0.83, R² = 0.7). Crucially, using only 25% of synthetic observations (3 per scene) maintained correlation above 0.85, meaning the method remains reliable even with reduced observation coverage. The NAVSIM v2 leaderboard revealed specific failure modes, such as PDM-Closed's poor performance on comfort metrics (HC: 47.9% in Stage 2 compared to 97.1% in Stage 1), indicating a trade-off between safety and comfort that previous benchmarks missed.

## Related Work
Pseudo-simulation builds on counterfactual data augmentation (used primarily for training) but adapts it for evaluation. It differs from closed-loop simulation (which relies on interactive environments and requires 80 planner inferences per scenario for 8-second rollouts) by using pre-rendered synthetic observations. Unlike NAVSIM v1, which remains limited to open-loop evaluation from expert-aligned observations, pseudo-simulation accounts for compounding errors and causal confusion through its two-stage approach.

## Limitations
The paper acknowledges that 3D Gaussian Splatting might struggle with extreme weather conditions or scenes with significant sensor failures (e.g., water droplets or flares), though the authors filtered these during reconstruction. The method also assumes that the synthetic observations generated from the original dataset will sufficiently capture the range of possible future states, a limitation that could manifest in rare scenarios not represented in the original dataset. The authors note that while the approach reveals new failure modes, it doesn't explicitly address how to handle edge cases like extreme weather conditions, which remain a challenge for all autonomous driving evaluation frameworks.

## Appendix: Worked Example
Let's walk through a single example scenario in detail to understand how pseudo-simulation works. Starting with a real-world observation (a) from the dataset where the ego vehicle is navigating an intersection:

1. **Stage 1 (real observation)**: The system predicts a 4-second trajectory ending at (x=5.0, y=3.2). The EPDMS score for this trajectory is 0.775 (77.5% for Ego Progress).

2. **Synthetic observation generation**: The authors sample start points around the expert's 4-second endpoint (x=4.8, y=3.0), with lateral sampling every 0.5m up to 2.0m on each side (creating points at ±0.5m, ±1.0m, ±1.5m, ±2.0m) and longitudinal sampling every 5m across the range of possible positions (creating points at x=0.0, 5.0, 10.0, 15.0). After filtering for velocity, acceleration, and heading consistency, they generate 12 synthetic observations (b, c, d, ..., l).

3. **Stage 2 (synthetic evaluation)**: For each synthetic observation, they compute EPDMS scores (e.g., synthetic observation (b) scores 0.713 for Ego Progress, (c) scores 0.771, etc.).

4. **Weighting and aggregation**: Using σ² = 0.1, they calculate weights based on distance from the Stage 1 endpoint (5.0, 3.2) to each synthetic start point. Observation (b) at (4.9, 3.1) gets weight 0.73, while observation (l) at (1.0, 3.0) gets weight 0.02. The weighted average of the Stage 2 scores is 0.713.

5. **Combined score**: The final pseudo-simulation score is 0.775 × 0.713 = 0.552. (See Key Technical Contributions for how this reflects error recovery capabilities.)

## References

- **Code:** https://github.com/autonomousvision/navsim.
- Wei Cao, Marcel Hallgarten, Tianyu Li, Daniel Dauner, Xunjiang Gu, Caojun Wang, Yakov Miron, Marco Aiello, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, Kashyap Chitta, "Pseudo-Simulation for Autonomous Driving", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2506.04218

Tags: #autonomous-vehicles #simulation #benchmarking #closed-loop-evaluation #3d-gaussian-splatting
