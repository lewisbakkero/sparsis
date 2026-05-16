---
title: "Preference-Driven Multi-Objective Combinatorial Optimization with Conditional Computation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2506.08898"
---

## Executive Summary
POCCO introduces a novel framework for multi-objective combinatorial optimisation that dynamically routes subproblems to specialized neural architectures and replaces explicit reward signals with pairwise preference learning. This approach significantly outperforms state-of-the-art neural methods across four classic benchmarks, achieving better solution quality while reducing computational overhead through conditional computation.

## Why This Matters for Practitioners
If you're building production systems for logistics, manufacturing, or scheduling that balance multiple conflicting objectives, such as cost, makespan, and environmental impact, POCCO offers a direct path to improved solution quality without requiring extensive domain-specific tuning. Engineers should consider integrating POCCO's conditional computation block into existing neural optimisation pipelines to enable adaptive model specialization. For example, when implementing vehicle routing solutions, replacing a single neural model with POCCO's routing mechanism can yield up to 0.83% higher hypervolume scores while reducing training time by 19% compared to standard approaches.

## Problem Statement
Current neural methods for multi-objective combinatorial optimisation (MOCOPs) face two fundamental limitations: they typically rely on a single model with limited capacity to handle all subproblems, and they use scalarized rewards with REINFORCE that suffer from high gradient variance. Imagine trying to balance three competing weights on a seesaw with a single rope, no matter how you adjust the rope's tension, you'll always get suboptimal balance because the system lacks adaptive specialization.

## Proposed Approach
POCCO augments neural MOCOP methods with a conditional computation block in the decoder and preference-driven optimisation. The conditional computation block routes subproblems to specialized neural architectures through a sparse gating network, while preference-driven optimisation replaces scalarized rewards with pairwise comparisons.

```python
def preference_driven_update(policies, batch):
    # For each subproblem in batch
    for (G, λi) in batch:
        # Sample two solutions
        πw, πl = sample_solutions(policies, G, λi, k=2)
        # Determine preference using scalarized objective
        if scalarized_value(πw) > scalarized_value(πl):
            πw, πl = πl, πw  # Ensure πw is preferred
        # Compute implicit reward
        fθ_w = (1/|πw|) * log(pθ(πw|G, λi))
        fθ_l = (1/|πl|) * log(pθ(πl|G, λi))
        # Update loss based on Bradley-Terry model
        loss = -log(σ(β * (fθ_w - fθ_l)))
        backpropagate(loss)
```

## Key Technical Contributions
The paper introduces two novel mechanisms that address fundamental limitations in neural MOCOP approaches:

1. **Conditional computation block design**: POCCO integrates a sparse gating network that routes each subproblem through a subset of feed-forward experts or a parameter-free identity expert. This design enables architectural specialization without significant computational overhead, as the gating network activates only a small subset of experts (k=2 typically) via Top-k operator.

2. **Bradley-Terry preference learning**: The framework replaces explicit rewards with pairwise preference learning, using the average log-likelihood as an implicit reward normalized by solution length. By maximising the likelihood of winning solutions over losing ones via the Bradley-Terry model, POCCO encourages exploration of optimal solution regions without requiring explicit reward engineering.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
POCCO outperforms all baselines across four classic MOCOP benchmarks. On Bi-TSP20, POCCO-W achieves 0.6275 hypervolume (HV) compared to 0.6270 for WE-CA (0.00% gap), with 7 seconds inference time versus 6 seconds for WE-CA. On Bi-TSP50, POCCO-W achieves 0.6411 HV versus 0.6392 for WE-CA (0.41% gap) with 14 seconds inference time versus 9 seconds. On Bi-TSP100, POCCO-W achieves 0.7055 HV versus 0.7034 for WE-CA (0.62% gap) with 36 seconds inference time versus 18 seconds. All improvements are statistically significant (Wilcoxon rank-sum test at 1% significance level).

The paper compares against three categories: single-model neural methods (PMOCO, CNH, WE-CA), multi-model neural methods (DRL-MOA, MDRL, EMNH), and non-learnable approaches (MOEA/D, NSGA-II, MOGLS). POCCO consistently achieves the highest HV scores across all benchmarks and problem sizes (n=20/50/100 for MOTSP and MOCVRP, n=50/100/200 for MOKP).

## Related Work
POCCO builds upon decomposition-based neural MOCOP methods like CNH and WE-CA, which encode weight vectors directly into problem representations. Unlike PMOCO (which uses weight-conditioned hypernetworks), POCCO's conditional computation block enables adaptive routing to specialized neural architectures. POCCO also extends preference optimisation methods like SimPO from single-objective to multi-objective settings, addressing a gap in prior work that had rarely investigated preference optimisation for MOCOPs.

## Limitations
The paper does not evaluate POCCO's performance on extremely large-scale instances (problem size n > 100 for MOTSP/MOCVRP, n > 200 for MOKP), though it does demonstrate strong generalisation across problem sizes within tested ranges. The authors acknowledge that preference-driven optimisation might require tuning the temperature parameter β for different problem types, though they use a fixed β > 0 across all experiments. The framework's effectiveness on non-probabilistic optimisation problems remains unexplored.

## Appendix: Worked Example
Consider a simple MOTSP instance with two objectives (distance and cost) and a weight vector λ = (0.6, 0.4). The system processes a batch of 64 instances with N=101 weight vectors.

At the conditional computation block, the router evaluates each context vector from the MHA layer. For a specific subproblem (G, λi), the router outputs [0.01, 0.95, 0.04] for three experts (two FF experts and one ID expert). The Top-k operator (k=2) selects the second expert with 0.95 probability.

The selected FF expert processes the context vector through its network, producing an output that's normalized and combined with the skip connection. This specialized computation path focuses on the particular weight vector's characteristics.

For preference learning, the policy samples two solutions: πw (winning) with scalarized value 15.2 and πl (losing) with scalarized value 12.8. The implicit rewards are fθ(πw) = (1/10) * log(0.45) = -0.051 and fθ(πl) = (1/10) * log(0.20) = -0.077. The difference (0.026) yields a preference probability of σ(β * 0.026) = 0.567 for πw being preferred.

The loss function computes -log(0.567) = 0.568, which drives the model to increase the probability of the winning solution in future generations.

## References

- **Code:** https://github.com/mingfan321/POCCO
- Mingfeng Fan, Jianan Zhou, Yifeng Zhang, Yaoxin Wu, Jinbiao Chen, Guillaume Adrien Sartoretti, "Preference-Driven Multi-Objective Combinatorial Optimization with Conditional Computation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2506.08898

Tags: #multi-objective-optimisation #conditional-computation #preference-learning #neural-optimisation #reinforcement-learning
