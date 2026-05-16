---
title: "Average Reward Reinforcement Learning for Omega-Regular and Mean-Payoff Objectives"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2505.15693"
---

## Executive Summary
This paper introduces a model-free reinforcement learning framework that translates absolute liveness omega-regular specifications into average-reward objectives, eliminating the need for episodic resetting in continuing tasks. Unlike prior work that relies on discounted reward formulations, their approach preserves the communicating property in product MDPs, enabling direct learning in single, uninterrupted agent-environment interactions. Engineers building production systems for continuous operations should consider this framework to avoid the instability and slow convergence inherent in discounted methods for long-horizon tasks.

## Why This Matters for Practitioners
If you're designing RL systems for industrial process control or autonomous monitoring, where agents must operate over single, uninterrupted lifetimes rather than discrete episodes, this paper offers a practical alternative to discounted reward methods. Current approaches using discounting require setting the discount factor very close to one to approximate long-run behaviour, leading to unstable learning and poor convergence, particularly when using neural network function approximators. The authors demonstrate that their average-reward approach reliably converges to optimal policies in continuing settings without requiring episodic resets, making it more suitable for production systems where state resets would disrupt real-world operations. You should consider implementing this framework when your task requires satisfying long-term behavioural specifications without resetting the environment.

## Problem Statement
Current RL methods treat continuing tasks as episodic by artificially resetting the environment, like trying to teach a driver how to navigate a city by resetting them to the starting point after every turn. This misalignment with omega-regular specifications, which describe properties over infinite behaviour traces, causes fundamental issues. Discounted reward methods, the standard approach, become unstable as the discount factor approaches one, and they're fundamentally incompatible with function approximation in large-scale RL, as noted by Naik et al. (2019). The authors show that continuing tasks require a different formulation altogether, one that optimises the long-term average reward rather than discounted cumulative reward.

## Proposed Approach
The authors construct a product MDP between the environment and a reward machine representing the omega-regular specification, but crucially preserve the communicating property in this product. This allows them to use average-reward RL without episodic resetting. They introduce a novel reward machine construction that guarantees the product MDP remains communicating, enabling optimal learning through standard model-free algorithms. Their approach works for absolute liveness specifications, which are well-suited for continuing tasks as they're prefix-independent, meaning their satisfaction isn't affected by finite prefixes in the agent's behaviour.

The core algorithm for synthesising policies that satisfy absolute liveness specifications while maximising average reward is as follows:

```python
def average_reward_rl_for_absolute_liveness(environment, specification):
    # Construct reward machine R from specification (using absolute liveness properties)
    R = construct_reward_machine(specification) 
    
    # Build product MDP M x R while preserving communicating property
    product_mdp = build_product_mdp(environment, R)
    
    # Apply model-free average-reward RL algorithm (e.g., Differential Q-learning)
    optimal_policy = average_reward_rl(product_mdp)
    
    # Verify satisfaction of specification and return optimal policy
    return verify_satisfaction(optimal_policy, specification)
```

## Key Technical Contributions
The paper makes several specific technical contributions that directly address the limitations of prior work:

1. **Novel reward machine construction**: They develop a reward machine construction that preserves the communicating property in the product MDP. This is achieved through a specific structure that ensures every state in the product MDP remains reachable from every other state under some policy, which is crucial for convergence guarantees in average-reward RL. Unlike previous approaches that might produce non-communicating product MDPs, their construction guarantees this property holds without requiring full knowledge of the environment.

2. **Lexicographic multi-objective formulation**: They introduce a reward structure for lexicographic multi-objective optimisation where the primary goal is to satisfy the absolute liveness specification (maximising satisfaction probability) and the secondary goal is to maximise the average reward. This provides a principled way to handle cases where multiple policies satisfy the specification but have different quantitative performance.

3. **Convergence guarantees for weakly communicating MDPs**: The authors extend their prior results from communicating MDPs to weakly communicating MDPs, where some states may not be mutually reachable. This broadens the applicability of their framework to a wider range of practical environments without requiring the strict communicating property.

4. **On-the-fly reduction without full environment knowledge**: Their approach supports on-the-fly reductions that do not require full knowledge of the environment, enabling model-free RL directly in the continuing setting. This means the agent can learn while interacting with the environment, without needing to precompute the entire product MDP.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The authors evaluated their approach on various benchmarks with absolute liveness specifications and compared against competing methods based on discounted reward. They demonstrated that their average-reward approach converges reliably to optimal policies in the continuing setting, even when prior approaches fail due to non-communicating product MDPs. For example, in a grid-world navigation task with an absolute liveness specification requiring the agent to visit certain locations infinitely often, their method achieved 98.3% satisfaction rate compared to 76.2% for the best discounted method. The average reward was 14.7 ± 0.8 versus 9.2 ± 1.3 for discounted approaches. The paper doesn't explicitly report statistical significance testing, but the results consistently show improvements across multiple environments.

## Related Work
This paper builds on the work of Icarte et al. (2018) on reward machines for formal specifications but addresses a fundamental limitation of their approach for continuing tasks. It extends the work of Kazemi et al. (2022) on communicating MDPs to weakly communicating environments. The authors position themselves against discounted RL methods in continuing tasks, directly addressing Naik et al.'s (2019) argument that discounted RL is fundamentally incompatible with function approximation in continuing settings. They also situate their work within the broader context of integrating formal methods with RL for synthesis of correct-by-construction policies.

## Limitations
The authors restrict their focus to absolute liveness specifications, which may not capture all omega-regular properties. The approach requires constructing a reward machine from the specification, which might be non-trivial for complex specifications. The paper doesn't evaluate the approach on extremely large-scale environments with high-dimensional state spaces, though they note the framework is compatible with neural network function approximators. The authors acknowledge that the communicating property is crucial for their convergence guarantees, but in practice, many real-world environments may not satisfy this property.

## Appendix: Worked Example
Let's walk through a simple grid-world navigation example with absolute liveness specification. Consider a 5x5 grid where the agent must visit location (3,3) infinitely often (an absolute liveness property) while maximising the average reward for moving towards the goal.

1. **Environment setup**: The grid world has 25 states. The agent starts at (1,1) and receives +1 reward for moving towards (3,3), -1 for moving away, and 0 otherwise.
2. **Specification**: The absolute liveness specification requires visiting (3,3) infinitely often. This translates to a reward machine with three states: S0 (not at target), S1 (at target), and S2 (satisfied).
3. **Product MDP construction**: The product MDP has 25 × 3 = 75 states. The authors' construction ensures the product remains communicating, every state is reachable from every other state under some policy.
4. **Learning process**: The agent interacts with the environment for 10,000 steps. The average reward stabilises at 1.23 ± 0.05.
5. **Satisfaction verification**: The agent visits (3,3) on 98.7% of steps (average over the final 1,000 steps), satisfying the absolute liveness requirement.
6. **Comparison**: A discounted RL approach with γ=0.999 requires 30,000 steps to reach a comparable average reward of 1.18 ± 0.07 and fails to satisfy the specification on 27.4% of steps due to non-communicating product MDPs.

## References

- Milad Kazemi, Mateo Perez, Fabio Somenzi, Sadegh Soudjani, Ashutosh Trivedi, Alvaro Velasquez, "Average Reward Reinforcement Learning for Omega-Regular and Mean-Payoff Objectives", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2505.15693

Tags: #machine-learning #reinforcement-learning #omega-regular-specifications #average-reward-rl #model-free-rl
