---
title: "PA2D-MORL: Pareto Ascent Directional Decomposition based Multi-Objective Reinforcement Learning"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19579"
---

## Executive Summary
PA2D-MORL introduces a novel approach to multi-objective reinforcement learning that efficiently approximates the Pareto frontier by leveraging Pareto ascent directions for policy optimisation, avoiding reliance on prediction models. It outperforms state-of-the-art methods in both quality and stability across seven MuJoCo robot control environments, generating denser and higher-quality policy sets for conflicting objectives like speed and energy efficiency.

## Why This Matters for Practitioners
If you're building production systems that require balancing multiple conflicting objectives (e.g., autonomous driving systems optimising speed versus passenger comfort, or robotic control systems balancing energy efficiency versus task completion), this paper provides a robust framework that eliminates the need for manual preference specification. Instead of retraining models for new trade-offs, PA2D-MORL generates a comprehensive policy set that enables immediate selection of optimal policies for varying user preferences. For instance, in robot control systems, this means you can deploy a single model that automatically adapts to different energy constraints or performance requirements without retraining.

## Problem Statement
Current multi-objective reinforcement learning methods face a fundamental tension between flexibility and stability. Imagine trying to tune a car's suspension for both highway comfort and racing performance, traditional approaches require manually adjusting settings for each scenario, like changing spring rates for different tracks. Similarly, existing MORL methods either rely on prediction models (which risk performance degradation when predictions fail) or use fixed scalarization weights (which require retraining for new trade-offs). This creates a bottleneck where systems can't dynamically adapt to evolving user preferences without costly recomputation.

## Proposed Approach
PA2D-MORL constructs an efficient scheme for multi-objective problem decomposition and policy improvement through three key components: Pareto Ascent Directional Decomposition, Partitioned Greedy Randomized (PGR) policy selection, and Pareto Adaptive Fine-tuning (PA-FT). The method maintains a population of policies throughout training, with each policy evolving towards different points on the Pareto frontier. The core innovation lies in how it automatically determines optimisation directions without requiring prior knowledge of objective preferences.

```python
# Algorithm 1: PA2D-MORL
Input: total generations: M; PA-FT generations: Mft; iterations per generation: m; warmup iterations: mw; initialized non-dominated policy set Pn.
1: Warmup:
2: Generate p randomly initialized policies and p evenly distributed weights ω.
3: Update each policy for mw iterations according to (6) to generate the first generation of policy population P
4: Evolution:
5: for generation = 1 to M do
6:   if generation < Mft then
7:     pa = p, pb = 0.
8:   else
9:     pa = pb = p//2.
10:  end if
11:  Select pa policies from P by the PGR approach.
12:  for k = 1 to pa do
13:    Calculate the policy gradient of πk according to (5) and solve (7) to obtain α*.
14:    Update πk for m iterations by (6), where ω = α*.
15:  end for
16:  if generation ≥ Mft then
17:    Select and update pb policies from Pn by the PA-FT approach.
18:  end if
19:  Update Pn and P with new policies.
20: end for
Output: Pareto policy set approximation Pn.
```

## Key Technical Contributions
PA2D-MORL overcomes critical limitations of existing MORL methods through three novel mechanisms:

1. **Pareto Ascent Directional Decomposition** determines the optimisation direction for non-Pareto policies by solving a minimum norm optimisation problem, rather than relying on manually designed scalarization weights or prediction models. This mathematically guarantees simultaneous improvement across all objectives, avoiding the need for human-designed scalarization functions that may not align with actual policy performance.

2. **Partitioned Greedy Randomized (PGR) policy selection** divides the objective space into angular partitions and identifies the top policies within each partition based on distance from a dominated reference point. By combining greedy selection with randomization, it ensures policies move toward a wider performance space while avoiding getting stuck in local minima, unlike PGMORL's prediction model which can lead to suboptimal exploration.

3. **Pareto Adaptive Fine-tuning (PA-FT)** identifies gaps in the current Pareto frontier approximation and selectively fine-tunes policies surrounding those gaps. Unlike other methods that uniformly sample the objective space, PA-FT actively targets regions where the frontier approximation is sparse, using nearest neighbour searches to identify the most valuable policies for fine-tuning.

## Experimental Results
PA2D-MORL achieved superior results across all seven MuJoCo environments compared to state-of-the-art baselines (PGMORL, MOEA/D, and PFA), as measured by the hypervolume (HV) and sparsity (SP) metrics. In the Walker2d environment, PA2D-MORL achieved HV = 5.743 × 10⁶ (vs. PGMORL's 4.849 × 10⁶) with SP = 0.014 × 10⁴ (vs. PGMORL's 0.021 × 10⁴), demonstrating both higher-quality policy sets and denser frontier approximation. On Humanoid, PA2D-MORL achieved HV = 51.23 × 10⁶ (vs. PGMORL's 44.75 × 10⁶) and SP = 0.133 × 10⁴ (vs. PGMORL's 0.255 × 10⁴). The results remained consistently better across all environments, with lower standard deviations indicating greater stability, particularly notable in Humanoid where PA2D-MORL's stability improved by 20% over PGMORL.

## Related Work
PA2D-MORL positions itself as a significant advancement over existing multi-policy MORL methods. It directly addresses the limitations of PGMORL (the current state-of-the-art) by eliminating the need for a prediction model that can introduce errors and instability. While methods like MOEA/D and PFA have proven effective in evolutionary optimisation, they were adapted for DRL domains and lack the mathematical foundation for automatic direction determination that PA2D-MORL provides. Unlike single-policy approaches (e.g., PPA), PA2D-MORL maintains a comprehensive policy set without requiring retraining for new preferences.

## Limitations
The paper acknowledges that PA2D-MORL has been validated only on environments with two conflicting objectives (e.g., speed and energy efficiency), with Hopper-3 being the only three-objective environment tested. The authors note that extending to more objectives may require additional exploration strategies. Additionally, the paper doesn't extensively discuss computational overhead compared to baselines, though it implies that the method's stability and performance improvements justify any minor overhead.

## Appendix: Worked Example
Consider the Walker2d environment where the objectives are forward speed (objective 1) and energy efficiency (objective 2). During training, the algorithm maintains a population of 8 policies. At generation 1, it computes the Pareto ascent direction for each policy by solving the optimisation problem in Equation (7). For a policy with gradients ∇θJπθ₁ = [0.2, 0.3] and ∇θJπθ₂ = [0.1, 0.2], the minimum norm solution gives α* = [0.7, 0.3], meaning the scalarization weight is ω = [0.7, 0.3] for this policy.

The PGR approach divides the objective space into 4 angular partitions. For a policy with returns Jπ = [1.2, 0.8], the distance from the reference point (dominated by all policies) is D = √((1.2-0)² + (0.8-0)²) = 1.44. In partition 1 (where policies have better speed), this policy ranks 3rd among the 8 policies, but because of the random selection component, it's selected with 25% probability.

After training through the warmup phase and initial generations, the algorithm identifies a large gap between the current Pareto frontier points at Jπ = [0.9, 1.2] and Jπ = [1.3, 0.7]. The nearest neighbour search identifies these two policies, which are then fine-tuned in opposite directions using PA-FT. The policy corresponding to [0.9, 1.2] is updated using the direction ∇θJπ₁ (for speed), while the policy at [1.3, 0.7] is updated using ∇θJπ₂ (for energy efficiency), effectively filling the gap in the frontier.

## References

- Tianmeng Hu, Biao Luo, "PA2D-MORL: Pareto Ascent Directional Decomposition based Multi-Objective Reinforcement Learning", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19579

Tags: #multi-objective-reinforcement-learning #pareto-frontier #policy-gradient #robot-control #evolutionary-algorithms
