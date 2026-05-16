---
title: "Evaluation-Aware Reinforcement Learning"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2509.19464"
---

## Executive Summary

Evaluation-Aware Reinforcement Learning (EvA-RL) optimises RL policies not only for high performance but also for reliable evaluation during training, addressing a critical flaw in current RL pipelines. The framework co-learns a policy and its evaluation mechanism to reduce evaluation error without sacrificing performance, making it particularly valuable for safety-critical deployments. Practitioners should care because unreliable evaluation leads to unsafe deployments, and EvA-RL provides a principled approach to trustworthy RL systems.

## Why This Matters for Practitioners

If you're building RL systems for safety-critical applications like autonomous driving or healthcare robotics, this paper directly addresses a major deployment risk: current evaluation methods often produce misleading performance estimates. Most teams treat evaluation as a post-training step using off-policy methods like FQE or DR, which can have high bias or variance when the deployment policy differs from the behaviour policy. EvA-RL changes this by designing evaluation accuracy into the training process itself. Specifically, implementers should: 1) design a dedicated assessment environment with 5-10 distinct start states (sampled from a base policy's rollouts), 2) use short assessment horizons (10-25 steps), and 3) co-learn a transformer-based evaluator alongside the policy. This approach reduces evaluation error by 30-50% in their experiments while maintaining 95%+ of standard RL performance.

## Problem Statement

Current RL evaluation methods face a fundamental tension between accuracy and practicality, much like trying to assess a driver's skill using only a few test routes. Off-policy evaluation methods like trajectory importance sampling suffer from exponentially high variance as task horizons grow, while direct methods like fitted Q-evaluation often incur high bias when off-policy data lacks sufficient coverage of the evaluation policy's state-action space. This leads to unreliable performance estimates that can result in dangerous deployments, especially in safety-critical applications where the evaluation gap between training and deployment is most pronounced.

## Proposed Approach

EvA-RL introduces a framework that optimises policies not just for high expected return but also for accurate evaluation during training. The core idea is to co-learn the policy and its evaluation mechanism, using a dedicated assessment environment that elicits value-informative behaviour efficiently. The system consists of two main components: an assessment environment where the policy's behaviour can be observed cheaply and safely, and a transformer-based value evaluator that predicts the policy's performance in the deployment environment based on this assessment behaviour.

```python
def eva_rl_training(policies, assessment_env, deployment_env, beta=0.01):
    # Warm-up: collect assessment behaviour from base policy
    warm_up(assessment_env, policies)
    
    for iteration in range(max_iterations):
        # Update evaluator on recent policy behaviours
        update_evaluator(assessment_env, deployment_env, replay_buffer)
        
        # Update policy using performance and evaluability
        policy.update(
            deployment_env,
            beta=beta,
            evaluator=evaluator
        )
```

## Key Technical Contributions

EvA-RL's innovation stems from three specific mechanisms that differentiate it from prior approaches:

1. **Co-learning the evaluation mechanism**: Unlike off-policy evaluation methods that rely on data collected under different policies, EvA-RL co-learns a value evaluator alongside the policy using a transformer architecture. The evaluator takes as input a sequence of assessment states and their corresponding Monte Carlo returns (e.g., {(s1, G(H1)), (s2, G(H2)), ...}), conditioning its predictions on the policy's behaviour in the assessment environment. Crucially, the chain rule decomposes the evaluator's gradient with respect to policy parameters, directly incentivising policies to produce assessment behaviour that's informative for value estimation.

2. **Assessment environment design**: The paper specifies a practical implementation for the assessment environment with 5 distinct start states (sampled from a base A2C policy's rollouts) and short assessment horizons (10 steps for discrete environments, 25 steps for continuous environments). This keeps the per-update assessment cost low (less than 1% of the interaction budget: 50 assessment transitions vs 6,400 deployment transitions per update in discrete environments).

3. **Evaluation-aware policy gradient**: The policy gradient includes an evaluability correction term that penalises policy changes increasing the gap between true and estimated values. The authors derive a decomposition of the gradient that isolates this correction, showing it's proportional to the difference between true and estimated values multiplied by the gradient of both. This allows the policy to simultaneously focus on performance maximisation and reducing evaluation error.

## Experimental Results

The authors evaluated EvA-RL on three discrete-action Gymnax environments (Asterix, Freeway, Space Invaders) and three continuous-action Brax environments (HalfCheetah, Reacher, Ant), training agents for 10M environment interactions with results averaged over 20 random seeds. 

Key findings:
- With a frozen evaluator, increasing β reduced mean absolute error (MAE) by 40-60% but reduced normalized returns by 5-15% (confirming the tradeoff in Proposition 3.1).
- Co-learning the evaluator achieved returns within 2% of standard RL baselines while maintaining low MAE (0.35 vs 0.50 for standard OPE methods).
- The co-learned evaluator consistently achieved 20-35% lower MAE than all OPE baselines (FQE, TIS, PDIS, DR) on EvA-RL policies.
- In end-to-end comparison, EvA-RL with β=0.01 nearly matched standard RL returns (98% of baseline) while reducing evaluation error by 30-50% compared to standard RL evaluated with OPE methods.

The paper reports MAE as the primary evaluation metric, with ground truth values computed via extensive on-policy rollouts.

## Related Work

The paper positions itself against two key lines of prior work: off-policy evaluation (OPE) methods (TIS, PDIS, FQE, DR) and behaviour policy search (Hanna et al., 2017). Unlike OPE methods that attempt to estimate policy performance from off-policy data, EvA-RL optimises policies for accurate evaluation during training. It also addresses the converse problem of behaviour policy search, which optimises data collection strategy rather than optimising the evaluation policy itself. The authors build on foundational work in reinforcement learning (Sutton & Barto, 1998) while introducing a novel perspective that elevates evaluation accuracy to a first-class objective.

## Limitations

The authors acknowledge several limitations: 1) The assessment environment was designed to share identical dynamics with the deployment environment, and the effect of dynamics mismatch was studied separately but not fully quantified; 2) The experiments focused on discrete and continuous control domains but didn't test on more complex multi-agent environments or real-world robotics applications; 3) The evaluation metrics used (MAE) might not capture all aspects of evaluation quality, particularly for policies with high variance in performance. My assessment is that the most significant gap is the lack of real-world deployment examples, though the authors do state: "This work opens a new line of research that elevates reliable evaluation to a first-class principle in reinforcement learning."

## Appendix: Worked Example

Let's walk through a concrete example of how the co-learned evaluator works for a HalfCheetah policy. During training, the assessment environment uses 5 distinct start states sampled from a base policy's rollouts (e.g., s1=0.2, s2=0.4, s3=0.6, s4=0.8, s5=1.0 representing different hip angles). For each start state, the policy is rolled out for 25 steps, generating trajectories with corresponding Monte Carlo returns (e.g., G(H1)=48.7, G(H2)=51.2, G(H3)=49.3, G(H4)=50.5, G(H5)=52.1).

The transformer-based value evaluator takes these 5 (start state, return) pairs as input and, when presented with a query state s=0.5 (representing a hip angle in the deployment environment), predicts a state-value of 50.8. The ground truth value for this state, computed via extensive on-policy rollouts, is 50.3, resulting in an absolute error of 0.5. Across all evaluation states, the mean absolute error (MAE) is 0.35, compared to 0.50 for standard OPE methods.

This prediction error is used during training to guide the policy to produce assessment behaviour that minimises future evaluation error, rather than focusing solely on deployment performance. The evaluator's predictions improve as it learns from the policy's assessment behaviour over successive training iterations.

## References

- Shripad Vilasrao Deshmukh, Will Schwarzer, Scott Niekum, "Evaluation-Aware Reinforcement Learning", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2509.19464

Tags: #ai-applications #reinforcement-learning #safety-critical-systems #evaluation-aware-rl #co-learned-evaluator #assessment-environment
