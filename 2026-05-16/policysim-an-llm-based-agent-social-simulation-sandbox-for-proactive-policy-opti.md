---
title: "PolicySim: An LLM-Based Agent Social Simulation Sandbox for Proactive Policy Optimization"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19649"
---

# Technical Article

## Executive Summary
PolicySim is an LLM-based social simulation sandbox that proactively assesses and optimises platform intervention policies such as recommendation systems before deployment. It models bidirectional dynamics between user behaviour and platform interventions through a user agent module refined via supervised fine-tuning and direct preference optimisation, and an adaptive intervention module that employs contextual bandits with message passing. This enables engineers to identify potential negative societal impacts before they affect real users.

## Why This Matters for Practitioners
This paper is crucial for engineers building recommendation systems at scale. If you're responsible for platform interventions like content filtering or recommendation algorithms, PolicySim provides a way to test potential societal impacts before deployment, avoiding costly reactive fixes after user polarization or echo chamber effects manifest. For example, you can now simulate how a new recommendation algorithm might increase polarization across different user groups before rolling it out, potentially avoiding regulatory scrutiny and user churn. The concrete action to take: integrate PolicySim-like simulation into your pre-deployment testing pipeline to evaluate intervention policies against metrics like cross-viewpoint interactions and misinformation spread.

## Problem Statement
Today's social platforms operate in a reactive mode: they deploy intervention policies (like recommendation systems), wait for users to react, and then measure outcomes through A/B testing. This is like changing a car's tire while driving - you only notice problems after they've caused accidents. The problem isn't just about speed - it's about risk. By the time A/B testing identifies that a recommendation algorithm is amplifying polarization, it's already caused measurable harm to users and the platform's reputation. PolicySim aims to create a "test drive" for social interventions before they hit real users, allowing engineers to identify negative impacts before they happen.

## Proposed Approach
PolicySim consists of two main modules: a user agent module that simulates realistic user behaviours, and an intervention policy module that evaluates and optimises platform interventions. The user agent module is trained to capture platform-specific behaviours through a combination of supervised fine-tuning (SFT) and direct preference optimisation (DPO), while the intervention policy module uses a contextual bandit framework with message passing to dynamically optimise interventions based on feedback from the simulation.

```python
def adaptive_intervention_policy(context, arms):
    # Context: user and post embeddings
    # Arms: candidate intervention policies
    # Compute context-aware rewards for each arm
    rewards = []
    for arm in arms:
        reward = compute_reward(context, arm)
        rewards.append(reward)
    
    # Balance exploration and exploitation
    exploitation_score = neural_network_predict(context, arms)
    exploration_score = estimate_exploration_gain(context, arms)
    
    # Select action with highest total score
    action_index = argmax(exploitation_score + exploration_score)
    return arms[action_index]
```

## Key Technical Contributions
PolicySim makes several key contributions that address limitations in current social simulation approaches.

1. **Unified training paradigm for social agents**: The paper introduces a novel training approach that combines supervised fine-tuning (SFT) with direct preference optimisation (DPO) to create agents that better model platform-specific user behaviours. Unlike prior approaches that rely on prompt engineering, PolicySim's agents are trained directly on platform data, ensuring behavioural alignment with real user data while capturing diverse user intents. The key implementation detail is how they construct the preference dataset for DPO by generating multiple candidate actions for each (event, user) pair and selecting those that differ in action choice or have low semantic similarity to the preferred response.

2. **Contextual bandit with message passing for adaptive interventions**: The paper's intervention policy module uses a contextual bandit framework augmented with message passing to capture dynamic network structures and information flows. This is different from prior work because it explicitly models the evolving social network topology (via follow/unfollow actions) and propagates context embeddings across the network using a label propagation-inspired mechanism (equation 8 in the paper). This allows the system to consider social influence when making intervention decisions, rather than treating users as isolated nodes.

3. **Real-time reward assessment for multiple intervention goals**: PolicySim implements specific reward functions for different intervention objectives, such as promoting cross-viewpoint interactions or mitigating misinformation. For cross-viewpoint interactions, the reward balances stance divergence between sender and receiver, penalises toxic interactions using Perspective API, and weights by engagement. This is more precise than general "improvement" claims because the paper explicitly details the calculation (equation 9 in the paper).

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The paper shows that PolicySim can accurately simulate platform ecosystems at both micro and macro levels. The authors conducted experiments across multiple datasets, though specific numbers are limited in the provided text. They compared against existing frameworks like Oasis, Agent4rec, HiSim, and Stopia, showing PolicySim outperforms these in modelling platform interventions and capturing network dynamics. The paper states that PolicySim "enables scalable and proactive assessment of intervention policies," though specific quantitative results (like percentage improvements over baselines) are not fully detailed in the provided text.

## Related Work
PolicySim builds on prior work in LLM-based agent simulations (e.g., HiSim, Oasis) but addresses key limitations: it explicitly models platform intervention policies (which most prior work doesn't), trains agents directly on platform data rather than relying on prompt engineering, and incorporates feedback from the simulation to optimise real-world policies. Unlike traditional agent-based models that don't capture the bidirectional dynamics between user behaviour and platform interventions, PolicySim explicitly models this interplay.

## Limitations
The paper acknowledges that PolicySim's accuracy depends on the quality of the training data used for agent training. It also notes that the simulation environment might not fully capture all the nuances of real-world social dynamics. The authors don't test PolicySim across a wide variety of platforms beyond X and Weibo, though they mention deploying interventions on these platforms.

## Appendix: Worked Example
Let's walk through a concrete example of how PolicySim's intervention policy module works, based on the paper's description.

In a simulation environment with 100 users (50 with positive stances on climate change, 50 with negative stances), the system aims to promote cross-viewpoint interactions. For a user with a neutral stance (s = 0) receiving a post from a user with a positive stance (s = 1), the system evaluates three candidate interventions:

1. **Recommendation A**: Suggests content from users with opposing stances (positive)
2. **Recommendation B**: Suggests content from users with similar stances (neutral)
3. **Recommendation C**: Suggests trending content (non-personalised)

The system computes rewards using equation (9):

- **Reward for Recommendation A**: |0 - 1|/2 * h(reaction) * max(0, 1 - τ(reaction)) = 0.5 * 0.8 * 0.9 = 0.36
- **Reward for Recommendation B**: |0 - 0|/2 * h(reaction) * max(0, 1 - τ(reaction)) = 0
- **Reward for Recommendation C**: |0 - 0.5|/2 * h(reaction) * max(0, 1 - τ(reaction)) = 0.25 * 0.8 * 0.9 = 0.18

Assuming engagement_weight = 0.8 (h(reaction)) and toxicity_score = 0.1 (τ(reaction)), the reward for Recommendation A becomes 0.36. The system selects Recommendation A, which promotes cross-viewpoint interaction with a higher reward.

## References

- Renhong Huang, Ning Tang, Jiarong Xu, Yuxuan Cao, Qingqian Tu, Sheng Guo, Bo Zheng, Huiyuan Liu, Yang Yang, "PolicySim: An LLM-Based Agent Social Simulation Sandbox for Proactive Policy Optimization", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19649

Tags: #social-computing #multi-agent-systems #llm-simulation #intervention-policy #contextual-bandit
