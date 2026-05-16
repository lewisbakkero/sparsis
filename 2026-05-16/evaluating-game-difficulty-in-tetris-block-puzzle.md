---
title: "Evaluating Game Difficulty in Tetris Block Puzzle"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.18994"
---

## Executive Summary
This paper introduces a method for objectively assessing game difficulty in Tetris Block Puzzle variants using Stochastic Gumbel AlphaZero (SGAZ), a budget-aware planning agent. By evaluating how rule changes affect training rewards and convergence iterations, the authors provide actionable insights for game designers seeking to balance challenge and accessibility in puzzle games.

## Why This Matters for Practitioners
When designing game mechanics for production systems, engineering teams often rely on subjective playtesting rather than data-driven difficulty assessment. This paper demonstrates that adding holding blocks (h) and preview blocks (p) systematically reduces difficulty, making games more accessible without compromising engagement. For instance, increasing h from 1 to 3 (from the classic Tetris Block Puzzle's 3) raised rewards from 39 to 6,544 while reducing convergence iterations from 'not converged' to 61. If you're building a game with a similar puzzle mechanic, implement a baseline configuration with h=3 and p=0, then systematically test increasing h to find the optimal difficulty point for your target audience. For games needing more challenge, avoid adding T-pentomino blocks (which increase convergence iterations by 165% compared to standard blocks) and instead consider adding less complex variants like U-pentomino.

## Problem Statement
Game designers today face a paradox: they need to balance accessibility for new players with challenge for veterans. Traditional approaches rely on subjective playtesting, which is time-consuming and inconsistent. Imagine trying to calibrate the difficulty of a puzzle game without any objective metrics, every change feels like guessing in the dark. This paper addresses that by providing a repeatable, data-driven method to assess how specific rule changes affect game difficulty, turning subjective intuition into measurable engineering decisions.

## Proposed Approach
The authors extend Gumbel AlphaZero to handle stochastic environments using afterstates and a learned model, creating Stochastic Gumbel AlphaZero (SGAZ). This agent evaluates game difficulty through two metrics: training reward (average total reward over last 50 iterations) and convergence iterations (iterations needed to reach maximum reward consistently). The key insight is that a strong agent's learning speed and performance directly reflect the game's underlying difficulty.

```
def evaluate_game_difficulty(game_rules):
    # SGAZ agent training with given rules
    agent = StochasticGumbelAlphaZero(
        game_environment=GameEnvironment(game_rules),
        simulation_budget=10_000
    )
    
    # Train until convergence
    agent.train(max_iterations=500)
    
    # Return objective metrics
    return {
        "training_reward": agent.average_reward(last_n=50),
        "convergence_iterations": agent.convergence_steps
    }
```

## Key Technical Contributions
The paper advances the field by demonstrating how a single agent framework can measure game difficulty across variants with consistent metrics.

1. **Afterstate decomposition for stochastic environments**: SGAZ separates agent actions from environmental randomness by mapping states to afterstates. For Tetris, placing a block (action) creates a deterministic afterstate, while the grid's randomness (e.g., next block) determines successor states. This enables accurate difficulty assessment without modelling every stochastic outcome.

2. **Gumbel-Top-k for budget-aware planning**: Instead of random sampling, SGAZ uses Gumbel-Top-k selection at the root to allocate simulations strategically. This guarantees policy improvement under small budgets (e.g., 10,000 simulations), making difficulty assessment feasible without excessive compute.

3. **Quantitative difficulty metrics**: The authors establish two objective metrics, training reward and convergence iterations, that correlate with perceived difficulty. For instance, the T-pentomino variant increased convergence iterations by 165% (from 164 to 429) under h=2, p=0, providing a measurable benchmark for difficulty.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The paper provides clear metrics showing how rule changes affect difficulty:

- **Holding blocks (h)**: Increasing h from 1 to 3 raised average rewards from 39.0 to 6,544.0 while reducing convergence iterations from 'did not converge' to 61. This demonstrates a strong, non-linear relationship between h and difficulty.

- **Preview blocks (p)**: For h=1, increasing p from 0 to 6 raised rewards from 39.0 to 4,965.1 (127x improvement), but the impact was less pronounced than changing h (e.g., h=2, p=2 achieved 5,527.2 rewards while h=3, p=0 reached 6,544.0).

- **Pentomino variants**: Adding the T-pentomino under h=2, p=0 increased convergence iterations by 165% (from 164 to 429), while other variants (U, V, X) showed smaller effects (increases of 41-72%). This establishes the T-pentomino as the most significant difficulty factor among variants.

The paper doesn't provide statistical significance tests for these results, though they report consistent patterns across multiple experiments.

## Related Work
The authors build on AlphaZero's success in deterministic environments by extending it for stochastic games through Stochastic AlphaZero. They connect to recent work using AlphaZero for chess variant analysis (Tomašev et al.) but extend this to puzzle games. Unlike dynamic difficulty adjustment methods (AlphaDDA, ROSAS), which adapt in real-time to player skill, this work provides a static framework to assess how rule sets affect difficulty, a complementary approach for game design rather than runtime adaptation.

## Limitations
The paper evaluates difficulty only through AI agent performance, not human player experience. The authors acknowledge this limitation: "We plan to conduct studies to evaluate how these variations affect human players' perceived challenge in the future." The experiments also lack statistical significance testing for the observed differences. Additionally, the work focuses on Tetris Block Puzzle variants but doesn't generalise to other puzzle genres.

## Appendix: Worked Example
Let's walk through the T-pentomino impact on difficulty under h=2, p=0 using concrete numbers from Table 2 and Figure 10:

1. **Baseline environment**: Standard Tetris Block Puzzle with h=2, p=0 (no additional blocks)
   - Convergence iterations: 164
   - Average reward: 4,126.1

2. **Adding T-pentomino variant**:
   - Convergence iterations increase to 429
   - Average reward decreases to 696.9

3. **Data flow through SGAZ**:
   - The agent starts with a random block (e.g., L-shape) and two holding blocks
   - The T-pentomino presents a unique challenge: it has an asymmetric shape requiring precise placement to complete lines
   - SGAZ must explore more paths to find configurations where T-pentomino completes lines
   - This increases the number of simulations needed per iteration (convergence iterations: 164 → 429)
   - The reward decreases because the agent can't complete as many lines as efficiently (4,126.1 → 696.9)

4. **Why this matters for engineers**:
   - The 165% increase in convergence iterations directly correlates to 165% more compute time needed for difficulty assessment
   - For a production game with 100,000 users, this means 165% more server resources for difficulty calibration
   - Engineers should avoid the T-pentomino unless deliberately designing for expert players

## References

- Chun-Jui Wang, Jian-Ting Guo, Hung Guei, Chung-Chin Shih, Ti-Rong Wu, I-Chen Wu, "Evaluating Game Difficulty in Tetris Block Puzzle", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.18994

Tags: #game-design #difficulty-assessment #reinforcement-learning #stochastic-games #alpha-zero
