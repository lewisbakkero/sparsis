---
title: "Breaking the Capability Ceiling of LLM Post-Training by Reintroducing Markov States"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19987"
---

## Executive Summary
This paper introduces a fundamental rethinking of LLM post-training by reintroducing explicit Markov states, which breaks through the persistent "capability ceiling" observed in standard RL-based training. The authors demonstrate that models using structured Markov states consistently outperform history-dependent approaches on complex logic puzzles, achieving up to 76% Pass@128 on Sokoban while standard methods plateau around 2%. For production engineers building reasoning-capable LLMs, this offers a clear path to unlock genuinely new problem-solving capabilities beyond simple pattern refinement.

## Why This Matters for Practitioners
If you're currently using RL-based methods (PPO or GRPO) to improve your LLM's reasoning capabilities, this paper suggests your model may be hitting an artificial ceiling due to the "history-as-state" formulation. The authors' experiments show that action-sequence models (the standard approach) plateau on complex logic puzzles like Sokoban (2.3% Pass@128 for Qwen3-4B) while Markov state-based models consistently surpass those boundaries (76.1% Pass@128). 

Practically, this means if you're building production systems requiring complex reasoning (e.g., code generation, mathematical problem-solving, or multi-step planning), you should consider implementing a separate state transition model that explicitly maintains and uses Markov states. You don't need to rebuild your entire training pipeline, simply add a state prediction model (like the Qwen2.5-3B-Instruct model they used) that predicts the next state from current state and action, then condition your policy on this explicit representation. For example, in code debugging, the state would represent a snapshot of the current codebase rather than your history of proposed changes.

## Problem Statement
Current RL post-training for LLMs suffers from a fundamental mismatch: it's like trying to navigate a city using only a detailed map of every street you've ever walked, rather than a current location marker. In classical RL systems like AlphaGo, agents use compact, informative state representations (the current board configuration) to make decisions. But LLMs currently treat the entire sequence of tokens as the state, making it exponentially harder to discover new paths. This "capability ceiling" appears because the state representation is inherently noisy and inefficient for decision-making, like trying to solve a maze while carrying the entire map of your past route instead of just knowing your current position.

## Proposed Approach
The authors propose a paradigm shift in LLM post-training by explicitly incorporating Markov states into the learning process. Instead of conditioning the next action on the full history of previous actions, models condition on a compact, informative state representation that contains all necessary information for decision-making. This requires adding a state transition model that predicts the next state from the current state and action, which can be implemented as a separate neural network or rule-based system. The policy then conditions on this explicit state rather than the history of actions.

The key insight is that this simple change aligns LLM post-training with classical RL principles, enabling genuinely new reasoning capabilities rather than merely refining existing patterns.

```python
# Markovian LLM Post-Training Approach
def train_markov_model(model, state_predictor, env, epochs):
    # Initialize with a starting state
    s_current = env.reset()
    
    for epoch in range(epochs):
        # Generate next action based on current state
        a_current = model(s_current)
        
        # Predict next state using state transition model
        s_next = state_predictor(s_current, a_current)
        
        # Calculate reward (1 for success, 0 otherwise)
        reward = env.get_reward(s_next)
        
        # Update policy based on state-action-reward
        model.update(s_current, a_current, reward)
        
        # Move to next state
        s_current = s_next
```

## Key Technical Contributions
The core innovations lie in how Markov states are integrated into the training pipeline with specific implementation choices:

1. **Explicit state transition modelling:** The authors train a separate state prediction model (bP) via supervised fine-tuning on Qwen2.5-3B-Instruct to predict the next board configuration from current state and action. This external model replaces the environment at test time, allowing deployment without environment access. Crucially, this avoids the base model implicitly behaving like a transition model during generation, which would undermine the state decomposition.

2. **State-conditioned policy training:** The policy network receives the Markov state (e.g., current board configuration) as input rather than the history of actions. This enables precise reward attribution to state-action pairs rather than entangled trajectories, as shown by their improved training reward convergence (Figure 5). The authors deliberately constrain the model to output only the next action (without chain-of-thought), preventing it from behaving like an implicit transition model that would bypass the state decomposition.

3. **Theoretical sample complexity guarantees:** The authors provide rigorous theoretical guarantees demonstrating that Markovian learning achieves significantly lower sample complexity compared to standard approaches. They prove that under the KL-regularised RL objective, the computational complexity is lower-bounded by min{Ccov(π⋆β), exp(Rmax/β)}, and that Markov states reduce the necessary coverage coefficient Ccov(π⋆β), allowing the model to avoid the exponential regime.

4. **Practical implementation for LLMs:** The authors address a critical implementation detail: they intentionally require models to output only the next action without chain-of-thought, as allowing reasoning would make the model behave like an implicit transition model. This constraint (enforced during training) ensures that the policy conditions on the explicit Markov state rather than inferring it during generation.

## Experimental Results
The authors conducted experiments on three synthetic logical reasoning tasks: Sudoku, Sokoban, and Futoshiki. They compared three approaches:
1. Action-sequence models (standard RL post-training)
2. Markov models (their proposed approach)
3. State-action-sequence models (intermediate baseline conditioning on full state history)

For Qwen3-4B on Futoshiki OOD benchmarks:
- Action-sequence: 0.1% Pass@128
- Markov: 75.0% Pass@128
- State-action-sequence: 44.4% Pass@128

For Qwen3-4B on Sokoban OOD:
- Action-sequence: 2.3% Pass@128
- Markov: 76.1% Pass@128
- State-action-sequence: 57.4% Pass@128

For Qwen2.5-3B-Instruct on Sudoku OOD:
- Action-sequence: 6.6% Pass@128
- Markov: 79.8% Pass@128
- State-action-sequence: 67.4% Pass@128

Training convergence analysis (Figure 5) shows Markov models reach higher rewards in fewer training steps, providing empirical evidence of lower sample complexity. The paper doesn't explicitly report statistical significance testing, though the consistent performance gaps across multiple tasks suggest strong statistical significance.

## Related Work
This paper positions itself as a fundamental rethinking of RL post-training for LLMs, building on classical RL principles while addressing a limitation specific to the LLM-RL intersection. The authors acknowledge that RL has been successful in classical domains like robotics (AlphaGo, MuZero) but identify a critical disconnect in LLM applications where the "history-as-state" formulation has become standard. They contrast their approach with prior work that claimed RL could induce novel capabilities (Zhang et al., 2025; Sun et al., 2025; Yuan et al., 2025a), noting that such gains were typically limited to "modest extensions" of the pre-training boundary or required "dense reward shaping or specialized domain-specific designs." The authors cite Foster et al. (2025) as providing theoretical evidence for the "discovery" bottleneck in RL, which their work aims to overcome.

## Limitations
The paper acknowledges that their approach requires Markov states to be accessible during training, which they demonstrated for synthetic logic puzzles but may not be straightforward for all real-world tasks. They also note that for complex state transitions, the accuracy of the state prediction model could become critical.

From a practical perspective, the paper doesn't test the approach on real-world production applications like code generation or complex planning, only on synthetic logic puzzles. The performance gains on these puzzles are significant, but it's unclear how well the approach generalizes to more open-ended tasks where the Markov state isn't explicitly defined. Additionally, the paper doesn't address how to handle tasks without clear Markov state representations.

## Appendix: Worked Example
Let's walk through the Markov state approach for Sudoku using Qwen3-4B:

Consider a partially filled Sudoku puzzle with a known solution state. The current Markov state s_current is the 9x9 grid configuration with some numbers filled in.

1. The policy model (Qwen3-4B) takes s_current as input and predicts the next action a_current (e.g., "place 5 in cell (2,3)").
2. The state transition model (trained on Qwen2.5-3B-Instruct) receives s_current and a_current, and predicts the updated grid configuration s_next (with 5 placed in cell (2,3)).
3. The environment updates to s_next, and the reward is calculated (0 until the puzzle is solved).
4. The policy model is updated based on the state-action-reward triple.

At test time, the state transition model replaces the environment. For example, after training, when presented with a new Sudoku puzzle, the system:
- Starts with the initial state (partially filled grid)
- Uses the policy to predict the next action (e.g., "place 5 in cell (2,3)")
- Uses the state transition model to predict the next state (the grid with 5 in cell (2,3))
- Repeats until the puzzle is solved or a maximum number of steps is reached

This process enables the model to reason over the current board state rather than its entire history of proposed changes, with the state transition model handling the complex dynamics of the puzzle.

## References

- Yurun Yuan, Tengyang Xie, "Breaking the Capability Ceiling of LLM Post-Training by Reintroducing Markov States", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19987

Tags: #ai-reasoning #markov-states #state-conditioned-policies #rl-post-training #sample-efficiency
