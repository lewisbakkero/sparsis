---
title: "Optimizing Resource-Constrained Non-Pharmaceutical Interventions for Multi-Cluster Outbreak Control Using Hierarchical Reinforcement Learning"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19397"
---

## Executive Summary
This paper presents a hierarchical reinforcement learning framework for optimising resource-constrained NPIs in multi-cluster outbreak control, where limited testing resources must be allocated across asynchronous disease clusters. The method introduces a generalised local policy that adapts to changing resource constraints without retraining, while a global controller dynamically adjusts the perceived cost of testing to balance resource usage. The approach improves outbreak control effectiveness by 20%-30% over heuristic baselines and scales to 40+ concurrent clusters.

## Why This Matters for Practitioners
If you're building production systems that must allocate limited resources across multiple asynchronous processes (such as cloud resource managers, emergency response platforms, or healthcare systems), this paper demonstrates a practical way to separate constraint management from decision-making. Specifically, for engineering teams at health tech companies, this means you can deploy a single generalised model that adapts to changing resource budgets by simply adjusting a cost parameter instead of retraining multiple models for different budget scenarios. The key implementation insight is to design your system with a "cost multiplier" abstraction layer between global constraints and local policies, which avoids the combinatorial complexity of fully centralized control while maintaining strict adherence to resource constraints. This is particularly relevant for any system where resource constraints can shift rapidly during operation.

## Problem Statement
Imagine managing a healthcare response system during a pandemic where new infection clusters emerge at different times across a city, each with varying sizes and risk levels. Current approaches either treat clusters independently (ignoring budget constraints) or attempt global optimisation (becoming computationally intractable as clusters grow). This creates a tension between the need for local decision-making (which cluster to test first) and the need for global coordination (how to allocate limited tests across all active clusters), making traditional methods ineffective for real-world outbreak scenarios where clusters arrive asynchronously.

## Proposed Approach
The authors propose a hierarchical framework where global resource allocation is handled by a controller that adjusts a cost multiplier, while local policies determine individual testing priorities. The local policies use a generalised DQN that conditions on the perceived test cost, and a global Q-ranking policy selects the highest-value tests within the budget. This separates global constraint management from local decision-making without requiring retraining.

```python
def hierarchical_allocation(global_budget, active_clusters, global_controller, local_policies):
    # Global controller adjusts test cost multiplier
    cost_multiplier = global_controller.step(
        system_state=compute_system_state(active_clusters)
    )
    
    # Local policies evaluate marginal test values
    marginal_values = {}
    for cluster in active_clusters:
        for individual in cluster:
            marginal_values[(cluster, individual)] = (
                local_policies[cluster].evaluate(
                    individual_features=individual.features,
                    cost_multiplier=cost_multiplier
                )
            )
    
    # Global Q-ranking selects tests within budget
    ranked_tests = sorted(
        marginal_values.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    selected_tests = []
    for (cluster, individual), value in ranked_tests:
        if value > 0 and len(selected_tests) < global_budget:
            selected_tests.append((cluster, individual))
        else:
            break
    
    return selected_tests
```

## Key Technical Contributions
The paper's key technical contributions go beyond the hierarchical framework to address specific implementation challenges in resource-constrained coordination:

1. **Generalised Local DQN with Monotonicity Regularisation**: The authors train a single DQN that conditions on a cost parameter (α₃) rather than separate models for each cost regime. Crucially, they introduce a gradient-based regularisation term that forces the policy to behave monotonically: higher perceived costs should lead to reduced testing recommendations. This prevents counterintuitive behaviours (like recommending more tests when costs increase) that would undermine trust in healthcare applications. The regularisation term explicitly enforces this requirement through the gradient penalty: Lgrad = E[ max(0, ∂Q/∂α₃ - gtarget)² ].

2. **Parameterized Global Cost Multiplier via PPO**: The global controller uses Proximal Policy Optimisation (PPO) to learn a continuous cost multiplier (mₜ) that modulates the perceived test cost across all clusters. This avoids the computational overhead of solving the constrained optimisation problem directly, while enabling adaptive resource allocation as budget constraints evolve. The multiplier directly parameterises the active test cost coefficient: α₃^active = mₜ · α₃^true, where mₜ ≥ 1.

3. **Deterministic Global Q-Ranking Execution Layer**: When local policies generate testing demand exceeding the budget, the system ranks all candidate tests by their marginal value (ΔQ) and selects the highest-value tests within the limit. This guarantees hard budget adherence while preserving local priorities, avoiding the instability of direct learning of discrete constrained allocations. The ranking procedure is deterministic and interpretable: tests are allocated to individuals with the highest estimated marginal benefit under the current cost regime.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The framework was evaluated in a realistic agent-based SARS-CoV-2 transmission simulator with the following results:

- Against heuristic baselines (Symptom-AvgRand, Thres-AvgRand, Thres-SizeRand), the method improved outbreak control effectiveness by 20%-30%.
- Against RMAB-inspired baselines (Fixed-M-QR, Bin-M-QR), it improved effectiveness by 5%-12%.
- The framework scaled to 40+ concurrently active clusters with approximately 5× speedup in decision-making compared to RMAB-inspired methods.
- All methods strictly enforced the hard global testing budget at every time step, with results averaged over five random seeds.

The paper doesn't specify statistical significance testing for these results, though it mentions results were averaged over five random seeds.

## Related Work
The authors position their work at the intersection of three research areas:
- Restless Multi-Armed Bandit (RMAB) approaches for resource allocation in healthcare, which they improve upon by handling asynchronous cluster arrivals and variable state dimensions.
- Hierarchical reinforcement learning for coordination problems, which they adapt to resource-constrained settings by decoupling global constraint management from local decision-making.
- Price-based coordination mechanisms, which they extend to handle variable-sized, asynchronously arriving clusters with strict budget constraints.

They specifically note that prior work (like Peng et al. 2023) focused on single-cluster decision-making, while their work addresses the more complex multi-cluster scenario with asynchronous cluster arrivals.

## Limitations
The paper acknowledges several limitations:
- The work focuses on testing allocation, but resource constraints could include other interventions like quarantine or contact tracing.
- The experiments are conducted in a SARS-CoV-2 simulator, so results may not generalise to other disease types or outbreak scenarios.
- The method assumes perfect knowledge of infection status (via the simulator), though in reality contact tracing would have limited accuracy.

From a practitioner perspective, the biggest limitation is the reliance on a simulator for evaluation, real-world healthcare systems would need extensive validation before deployment, particularly regarding the accuracy of cluster identification and infection status estimation.

## Appendix: Worked Example
Let's walk through a concrete scenario with 3 active clusters and a global budget of 5 tests:

**Initial Conditions:**
- Cluster 1: 10 individuals with marginal test value 0.3
- Cluster 2: 5 individuals with marginal test value 0.6
- Cluster 3: 7 individuals with marginal test value 0.4
- Global budget: 5 tests
- True per-test cost (α₃^true): 0.05

**Step 1: Global controller observes system state**
The controller computes system features: current timestep (day 14), number of active clusters (3), budget utilisation ratio (22/5 = 4.4), demand tightness (high), and previous multiplier (mₜ₋₁ = 1.0).

**Step 2: Controller adjusts cost multiplier**
Since demand (22 potential tests) exceeds budget (5), the controller increases the multiplier to mₜ = 1.5 (1.0 → 1.5) to suppress testing demand.

**Step 3: Local policies evaluate tests with adjusted cost**
The local policies now use α₃^active = mₜ · α₃^true = 1.5 · 0.05 = 0.075. Each individual's marginal test value (ΔQ) is recalculated using this adjusted cost:
- Cluster 1: 10 individuals at ΔQ = 0.3
- Cluster 2: 5 individuals at ΔQ = 0.6
- Cluster 3: 7 individuals at ΔQ = 0.4

**Step 4: Global Q-ranking selects tests**
All candidate tests are pooled and ranked by ΔQ:
1. Cluster 2 (5 tests at ΔQ = 0.6)
2. Cluster 3 (7 tests at ΔQ = 0.4)
3. Cluster 1 (10 tests at ΔQ = 0.3)

The top 5 tests are selected from Cluster 2 (all 5 tests have the highest ΔQ).

**Step 5: Execution**
The system executes the 5 tests from Cluster 2, leaving Cluster 1 and Cluster 3 with no tests executed (despite having positive ΔQ, they're below the threshold due to the adjusted cost).

This demonstrates how the global controller's cost multiplier effectively suppresses testing demand from lower-value clusters without requiring retraining of local policies.

## References

- Xueqiao Peng, Andrew Perrault, "Optimizing Resource-Constrained Non-Pharmaceutical Interventions for Multi-Cluster Outbreak Control Using Hierarchical Reinforcement Learning", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19397

Tags: #biomedicine #outbreak-control #resource-allocation #hierarchical-rl #multi-agent
