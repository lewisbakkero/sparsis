---
title: "On the Structural Non-Preservation of Epistemic Behaviour under Policy Transformation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2602.21424"
---

## Executive Summary
This paper formalises how reinforcement learning (RL) agents lose their ability to distinguish between latent contexts under common policy transformations like aggregation and optimisation. It proves that even when agents maintain high reward under dominant training distributions, their information-conditioned behaviour can degrade, leading to brittle performance under distributional shift. Practitioners should care because standard RL evaluation through reward metrics alone fails to capture this structural degradation, potentially causing production systems to fail in real-world scenarios with rare or shifted contexts.

## Why This Matters for Practitioners
When deploying RL systems in production (e.g., recommendation engines, adaptive user interfaces), the standard practice of optimising for aggregate reward on common data distributions creates a dangerous blind spot: agents can maintain high reward while losing their ability to distinguish between latent user contexts. For example, a recommendation system might achieve stable high click-through rates on popular products (dominant training distribution) but fail to recommend niche items when user demographics shift (rare mode), despite no visible drop in aggregate metrics. 

This is not a theoretical concern - the paper shows that in controlled experiments, agents with zero behavioural distance (the structural equivalent of a "short-circuit" in information processing) achieve 86% reward under biased training but collapse to 6% performance under prior shift. To prevent this, engineers must instrument their systems to measure probe-conditioned response profiles at evaluation time - not just reward metrics. Implement this by adding structured probes that induce distinct latent contexts during evaluation, then compute the within-policy behavioural distance as defined in the paper. This simple change will reveal degradation patterns that standard metrics hide.

## Problem Statement
Imagine a medical diagnostic system that can accurately identify diseases when testing with familiar symptoms (common scenarios) but fails when confronted with atypical symptom combinations (rare scenarios). The system's ability to adapt based on subtle internal cues (like symptom relationships) is crucial for consistent accuracy, but this skill might degrade when the system is trained on a narrow set of common cases. Similarly, RL agents trained on limited observation distributions might maintain high reward on common scenarios but lose the ability to distinguish between different latent contexts (like different symptom patterns) when encountering rare or shifted scenarios. This degradation isn't evident from standard reward metrics, creating a critical blind spot in production systems.

## Proposed Approach
The paper formalises "epistemic behaviour" as "behavioural dependency" - systematic variation in action selection with respect to internal information under fixed observations. It introduces "probe-relative behavioural equivalence" by defining equivalence classes based on how policies respond to specific probes that induce distinct latent interaction contexts. The authors quantify this with a within-policy behavioural distance metric that measures probe-conditioned response profiles, without requiring access to internal representations.

This approach allows evaluating policy transformations at the level of observable behaviour rather than internal structure, which is essential for production systems where internal states might not be accessible. The core insight is that reward stability does not guarantee preservation of information-conditioned interaction patterns.

```python
def compute_behavioural_distance(policy, probe_set, evaluation_observation):
    """Compute within-policy behavioural distance using the paper's definition.
    
    Args:
        policy: The RL policy to evaluate
        probe_set: Set of probes inducing distinct latent contexts
        evaluation_observation: Fixed observation where evaluation occurs
        
    Returns:
        d: Behavoural distance (L1 difference between probe responses)
    """
    responses = []
    for p in probe_set:
        response = policy.response_profile(p, evaluation_observation)
        responses.append(response)
    
    # Compute L1 distance between the first and last probe responses
    d = np.linalg.norm(responses[0] - responses[-1], ord=1)
    return d
```

## Key Technical Contributions
The paper introduces specific formalisations of how RL agents interact with the environment, with implementation-level insights:

1. **Probe-relative behavioural equivalence**: The paper defines equivalence classes based on observable action distributions under specific probes rather than architectural assumptions. This allows evaluating policies at the level of behaviour rather than internal structure. Unlike prior work that focuses on state abstractions (e.g., bisimulation) or multi-agent comparisons, this approach defines equivalence over probe-induced response profiles within a single policy without assuming identifiable latent states.

2. **Within-policy behavioural distance**: The authors define a specific metric - the L1 difference between response profiles under two probes at a fixed observation - that quantifies how much a policy's behaviour changes with respect to internal information. This distance metric can be computed from observable behaviour without needing access to internal states, making it feasible to instrument in production systems. For the gridworld experiment, the paper reports behavioural distances of d=2 for the probing policy, d=0 for the shortcut policy, and d=1 for the aggregated policy.

3. **Structural results under policy transformations**: The paper proves that policies exhibiting non-trivial behavioural dependency are not closed under convex aggregation (Proposition 1), showing why simply averaging policies from different training runs won't maintain the ability to distinguish between latent contexts. It also proves a sufficient local condition (Theorem 1) under which gradient-based optimisation contracts behavioural distance under skewed latent priors. This explains why standard optimisation processes might degrade information-conditioned behaviour.

## Experimental Results
The paper provides controlled experiments in a partially observable gridworld (Figure 1) with three policies evaluated under biased and reversed priors:

| Policy           | Biased Prior (P(m=0)=0.9) | Reversed Prior (P(m=0)=0.1) |
|------------------|---------------------------|-----------------------------|
| Probing          | 0.810 ± 0.000             | 0.810 ± 0.000               |
| Shortcut         | 0.863 ± 0.017             | 0.060 ± 0.019               |
| Aggregated       | 0.710 ± 0.023             | 0.000 ± 0.020               |

For the gradient-based optimisation experiment (Figure 3), the paper shows that under a heavily biased prior (P(m=0)=0.98), behavioural distance d(π) decreases during training while return under dominant prior remains stable. The return under reversed prior degrades significantly even as dominant-prior reward stays high. The paper reports: "Behavioural distance decreases during biased optimisation and stabilises at a reduced plateau" but doesn't provide specific d(π) values.

The results demonstrate that reward statistics under a dominant prior do not reflect structural robustness to latent distribution shift - robustness scales approximately with behavioural distance rather than biased-prior reward.

## Related Work
The paper positions itself by acknowledging previous work on information-conditioned behaviour in RL, like meta-RL (Duan et al., 2016) and world models (Ha & Schmidhuber, 2018), but distinguishes itself by characterising information-conditioned interaction patterns directly through probe-induced response profiles rather than architectural assumptions. It contrasts with trajectory-level comparisons (Ferns et al., 2011; Zang et al., 2023) that focus on state abstractions and value consistency, and with multi-agent policy-distance metrics (Bettini et al., 2025; Hu et al., 2024) that compare action distributions across agents. The authors' formulation defines equivalence over probe-induced response profiles within a single policy, without assuming identifiable latent states.

## Limitations
The paper acknowledges that the experiments are minimal and controlled, not intended as benchmark evaluations but as diagnostic settings. It doesn't address how to automatically design probes for specific application domains, a critical gap for practical implementation. The authors also don't provide guidance on scaling the approach to high-dimensional environments where appropriate probe selection becomes challenging. In practice, these limitations mean the paper provides a theoretical foundation but requires significant engineering effort to implement in production systems.

## Appendix: Worked Example
Let's walk through the gridworld experiment from the paper, using the specific numbers provided:

1. **Setup**: A partially observable gridworld where an agent must navigate to a goal that depends on a latent mode m ∈ {0, 1}. The correct action at observation o* depends on the latent mode m.

2. **Probing policy** (d=2):
   - For mode m=0: Action a0 (correct goal) 90% probability, a1 10%
   - For mode m=1: Action a1 (correct goal) 90% probability, a0 10%
   - Behavioural distance d = ||[0.9, 0.1] - [0.1, 0.9]||₁ = 1.6

3. **Shortcut policy** (d=0):
   - For both modes m=0 and m=1: Action a0 90% probability, a1 10%
   - Behavioural distance d = ||[0.9, 0.1] - [0.9, 0.1]||₁ = 0.0

4. **Aggregated policy** (α=0.5):
   - For mode m=0: Action a0 90% probability (from probing policy), action a0 90% probability (from shortcut policy) → averaged to 90% a0, 10% a1
   - For mode m=1: Action a1 90% probability (from probing policy), action a0 90% probability (from shortcut policy) → averaged to 50% a0, 50% a1
   - Behavioural distance d = ||[0.9, 0.1] - [0.5, 0.5]||₁ = 0.8

5. **Evaluation under bias**:
   - Biased prior P(m=0)=0.9:
     - Probing policy: 0.810 return (stable across modes)
     - Shortcut policy: 0.863 return (high due to bias)
     - Aggregated policy: 0.710 return
   - Reversed prior P(m=0)=0.1:
     - Probing policy: 0.810 return (stable)
     - Shortcut policy: 0.060 return (fails under reversed mode)
     - Aggregated policy: 0.000 return

This example shows that the probing policy maintains high performance across distributions due to high behavioural distance, while the shortcut policy maintains high reward under the biased prior but fails under prior shift. The aggregated policy demonstrates intermediate behaviour, with robustness correlating with behavioural distance rather than biased-prior reward.

## References

- Alexander Galozy, "On the Structural Non-Preservation of Epistemic Behaviour under Policy Transformation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2602.21424

Tags: #reinforcement-learning #behavioural-equivalence #policy-transformation #partial-observability #information-conditioned-behaviour
