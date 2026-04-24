---
title: "Measuring What Matters: Scenario-Driven Evaluation for Trajectory Predictors in Autonomous Driving"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36978"
---

## Executive Summary

The authors propose ED-Eva, a scenario-driven evaluation framework that dynamically balances prediction accuracy and diversity based on traffic scenario complexity. This addresses a critical gap where traditional error-based metrics (like ADE, FDE) fail to correlate with real-world driving performance, potentially leading to unsafe decisions in complex scenarios like intersections.

## Why This Matters for Practitioners

If your autonomous driving system evaluates trajectory predictors using only ADE or FDE, you risk selecting models that perform well on paper but lead to unsafe planning in complex environments. The paper demonstrates a weak correlation between ADE and actual driving performance (Figure 3), meaning a predictor with low ADE might still cause dangerous driving decisions in intersections.

Practitioners should immediately implement scenario-aware evaluation in their pipelines: when evaluating trajectory predictors, measure both accuracy and diversity, then dynamically weight these based on scenario complexity. For example, in highway scenarios (low criticality), prioritize accuracy; in intersections (high criticality), prioritize diversity. This ensures your selected predictor actually supports safe, robust decision-making in production.

## Problem Statement

Current evaluation practices for trajectory predictors rely solely on error-based metrics like ADE (Average Displacement Error) and FDE (Final Displacement Error), which measure geometric closeness between predicted and ground-truth trajectories. These metrics are fundamentally flawed because they ignore how prediction quality translates to real-world driving performance.

Imagine evaluating a football coach's performance based only on how closely their training drills match the official game playbook, ignoring whether players actually score goals or avoid dangerous tackles. Similarly, measuring trajectory prediction by ADE alone misses whether the output actually leads to safe driving decisions, especially in complex scenarios where multiple possible agent movements must be considered.

## Proposed Approach

ED-Eva is a closed-loop evaluation framework that dynamically combines prediction accuracy and diversity based on scenario criticality. The framework has three main components:

1. ScenarioNN: A graph-based neural network that classifies scenarios as 'critical' (requiring diversity) or 'simple' (requiring accuracy)
2. GAD: A metric measuring prediction diversity using Gaussian Mixture Models
3. Eerror: A standard error-based metric (like ADE) measuring prediction accuracy

These components integrate into a single score that reflects the predictor's impact on real driving performance.

```python
def ed_eva_score(predictor, scenario):
    # ScenarioNN determines criticality probability Pc
    Pc = scenario_nn.predict(scenario)
    
    # GAD measures prediction diversity (higher is better)
    gad = gad_metric(predictor.predictions)
    
    # Eerror measures prediction accuracy (lower is better)
    eerror = error_metric(predictor.predictions)
    
    # Combine adaptively: higher Pc means more weight on diversity
    return Pc * gad + (1 - Pc) * -eerror  # Negative to align direction
```

## Key Technical Contributions

The paper introduces three key technical contributions that address the limitations of current evaluation practices:

1. **Scenario Classification with ScenarioNN**: The authors build a spatial-temporal graph network that determines scenario criticality by analysing the ego vehicle's interaction with nearby agents. The network processes a 15-step history of spatial-temporal features through two graph convolutional layers (each with ReLU activation), then uses an LSTM (hidden size 32) to capture temporal patterns. The final score is a sigmoid output from a linear head. This architecture specifically identifies critical scenarios (like intersections) where prediction diversity is more important, using an adjacency matrix with dth = 5m to determine vehicle interactions.

2. **GMM-Area Diversity (GAD) Metric**: Unlike previous diversity metrics that only measure spread in one direction, GAD uses a two-dimensional Gaussian Mixture Model (GMM) to capture the spread of predictions in all directions. For each prediction index and time step, it fits a GMM to the endpoints of predicted trajectories, computes the determinant of the covariance matrix, and aggregates these values. This produces a bidirectional measure of diversity (higher values indicate more diverse predictions), robust to outliers and computationally efficient with only a negligible cost for eigen-decomposition of 2x2 matrices.

3. **Adaptive Scoring System**: The paper's key innovation is dynamically weighting GAD and Eerror based on scenario criticality (Pc). In critical scenarios (high Pc), diversity receives more weight (Pc × GAD), while in simple scenarios (low Pc), accuracy receives more weight ((1 - Pc) × Eerror). This adaptive combination directly aligns evaluation with the downstream impact on driving performance, as demonstrated by the stronger correlation with real-world driving outcomes.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results

The authors evaluated four trajectory predictors (CV, AutoBot, MTR, Wayformer) across 9059 scenarios from the nuScenes dataset. They compared their ED-Eva framework against standard error-based metrics (ADE, FDE, minADE, minFDE, aveADE, aveFDE) using Pearson correlation coefficients and AUROC.

Key results:
- Table 1 shows ED-Eva's evaluation score correlates more strongly with driving performance (safety, comfort, efficiency) than any error-based metric. For example, ED-Eva (GAD, ADE) achieved a correlation of +0.2200 with Overall performance (higher is better), while the best error-based metric (minFDE) achieved only +0.1843.
- Figure 5 shows ED-Eva consistently outperforms other metrics in AUROC across all predictor configurations, indicating better ranking of predictors by their real-world impact.
- The authors note they tested 9059 scenarios but don't specify the distribution of critical vs. simple scenarios.

The paper doesn't report statistical significance for these results, but the consistent improvement across all metrics and predictors suggests the framework's effectiveness is substantial.

## Related Work

The paper positions itself as a critical advancement over existing work in three areas:

1. **Error-based metrics**: While metrics like ADE and FDE are widely used, the paper demonstrates they fail to correlate with downstream driving performance (Figure 3). The authors cite recent work by Weng et al. (2023) and Phong et al. (2023) showing this limitation.

2. **Diversity metrics**: Previous works introduced diversity-aware metrics like AMV and energy scores. However, these are typically computed in an open-loop setting that doesn't account for how prediction diversity affects planning outcomes. The authors point out that closed-loop evaluation frameworks exist but don't adapt to scenario context.

3. **Scenario criticality**: Early works used LSTM or graph-based methods for scene understanding, but the paper's novel contribution is using this understanding to dynamically weight diversity versus accuracy in evaluation, rather than relying on predefined scenario categorizations.

## Limitations

The authors acknowledge they tested 9059 scenarios from nuScenes, which may not represent all possible driving environments. They don't address whether their ScenarioNN generalizes to different regions with different driving styles or infrastructure.

Additionally, the authors don't explicitly state how the framework would scale to real-time production systems, though the computational cost of GAD is minimal (only requiring eigen-decomposition of 2x2 matrices).

The paper also doesn't address how their framework would integrate with existing evaluation pipelines in major autonomous driving systems.

## Appendix: Worked Example

Let's walk through a concrete example of how ED-Eva evaluates a trajectory predictor for a single scenario:

1. **Scenario Input**: A highway scenario with 3 vehicles (ego + 2 neighbours) over a 15-step history (T=15).
2. **ScenarioNN Processing**: The spatial-temporal graph features (including relative positions, velocities, and 5-dimensional neighbour motion features) are processed through two graph convolutional layers (H(1) = ReLU(AH(0)W(0)), H(2) = ReLU(AH(1)W(1))). The graph adjacency matrix A uses dth = 5m to determine connections between vehicles.
3. **Scenario Classification**: The LSTM processes the graph features into a hidden state hT, which is passed through a linear head to produce logit ℓ = w^T hT + b. For this highway scenario, Pc = σ(ℓ) = 0.15 (low criticality).
4. **Predictor Output**: The trajectory predictor (e.g., Wayformer) generates N=6 future paths for each of the 2 neighboring vehicles over Tp=15 time steps.
5. **GAD Calculation**: For each time step t (1-15) and each of the 6 prediction paths (n=1-6), they fit a 2D GMM to the endpoints of the predicted trajectories. The covariance matrix Σ(n)_t is computed, then the determinant is used to calculate the diversity score at each point.
   - For example, at time t=5, for prediction path n=3, det(Σ(3)_5) = 0.0245 → sqrt(0.0245) = 0.1565
   - Aggregating over all N=6 paths and Tp=15 time steps: GAD = (1/(6*15)) * Σ Σ sqrt(det(Σ(n)_t)) = 0.142
6. **Eerror Calculation**: Using ADE as the error metric, they calculate the average displacement error across all predictions (N=6 paths) and time steps (Tp=15): Eerror = 0.87
7. **Final Score**: ED-Eva score = Pc * GAD + (1 - Pc) * (-Eerror) = 0.15 * 0.142 + 0.85 * (-0.87) = -0.73

This example shows how the framework dynamically weights diversity (low weight of 0.15) and accuracy (high weight of 0.85) for a highway scenario, resulting in a lower score due to the higher error (0.87).

## References

- Longchao Da, David Isele, Hua Wei, Manish Saroya, "Measuring What Matters: Scenario-Driven Evaluation for Trajectory Predictors in Autonomous Driving", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36978

Tags: #autonomous-driving #trajectory-prediction #evaluation-metrics #multi-agent-systems #scenario-aware-evaluation
