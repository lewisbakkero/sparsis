---
title: "The Art of Efficient Reasoning: Data, Reward, and Optimization"
venue: "Reward"
paper_url: "https://arxiv.org/abs/2602.20945"
---

## Executive Summary
This paper systematically investigates efficient reasoning for LLMs through a unified RL framework, revealing that training on easier prompts provides denser positive reward signals and avoiding the 'short-is-correct' trap. Their approach reduces CoT lengths by 15-47% while preserving or improving accuracy across Qwen3 models from 0.6B to 30B parameters.

## Why This Matters for Practitioners
If you're deploying reasoning-intensive LLM applications in production, this paper provides actionable guidelines to reduce inference costs without sacrificing accuracy. Specifically, for any system requiring constrained token budgets (e.g., real-time chat applications), you should: (1) train with easier prompts to ensure sufficient positive reward density, (2) sample more rollouts (N=24) if computational resources allow, and (3) avoid penalising overlong correct responses, this approach achieves up to 47.3% shorter outputs while maintaining or improving Mean@8 scores. For instance, on Qwen3-4B-Instruct, this reduces response length from 9.1k to 4.8k tokens while increasing Mean@8 from 45.42 to 46.67.

## Problem Statement
Today's LLM reasoning systems suffer from an overthinking trap: they generate unnecessarily long Chain-of-Thought (CoT) trajectories that increase latency without improving accuracy. Imagine a doctor who spends 30 minutes listing every possible symptom before diagnosing a patient with a common cold, this cognitive overhead is what current LLM reasoning systems exhibit, consuming excessive computational resources for marginal accuracy gains.

## Proposed Approach
The authors propose a unified RL framework for efficient reasoning, optimising for both short and accurate outputs. The pipeline consists of three main components: training prompts, reward shaping, and optimisation. Crucially, they argue that reward engineering alone is insufficient, data composition and optimisation strategy are equally important.

```python
def efficient_reasoning_training(
    model, 
    prompts, 
    target_length=4000,
    rollout_size=24,
    max_length=16000
):
    """Optimises LLM for short yet accurate reasoning trajectories."""
    # Data composition: use easier prompts for denser positive rewards
    easy_prompts = filter_prompts(prompts, pass_rate > 0.5)
    
    # Reward shaping: don't penalise overlong correct responses
    def reward(rollout):
        if is_correct(rollout):
            if len(rollout) > target_length:
                return 1.0  # Avoid penalty for overlong correct outputs
            else:
                return 1.0
        else:
            return 0.0  # Incorrect responses get negative reward
    
    # Optimisation: use off-policy with appropriate staleness
    for step in range(optimisation_steps):
        rollouts = sample(model, easy_prompts, size=rollout_size, max_length=max_length)
        rewards = [reward(r) for r in rollouts]
        update_policy(model, rollouts, rewards, staleness=4)
```

## Key Technical Contributions
The paper provides novel insights into the mechanics of efficient reasoning training, going beyond prior reward-focused approaches.

1. **Two-stage training paradigm**: They identify that training follows a predictable sequence: first rapidly adapting to token constraints (length adaptation), then refining reasoning within that constraint (reasoning refinement). This explains why simply adjusting reward functions often fails, the training dynamics follow a natural progression that must be respected.

2. **Positive reward density**: They demonstrate that training on easier prompts creates denser positive reward signals, which is essential for stable training. When the authors trained exclusively on hard prompts, policy entropy spiked, rollout length collapsed prematurely, and performance degraded significantly. In contrast, training on easier prompts maintained low, stable entropy while adapting smoothly to target length.

3. **Strategic reward masking**: Contrary to common practice of penalising all non-optimal outputs, they show that masking overlong but correct rollouts (instead of penalising them) leads to better performance. Their experiments revealed that the strategy "-L&C" (masking overlong correct rollouts) outperformed vanilla approaches, showing that overlong correct responses contain valuable reasoning information that shouldn't be discarded.

4. **Optimised off-policy training**: They demonstrate that moderate off-policy updates (staleness=4) accelerate training without sacrificing accuracy. This differs from prior work that often reports catastrophic failure with off-policy training, as their approach maintains sufficient positive reward density through the use of easier prompts and larger rollout sizes.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The paper conducted extensive experiments across the Qwen3 model family, from 0.6B to 30B parameters, on AIME'25 and other reasoning benchmarks. Key results:

- **Length reduction**: For Qwen3-4B-Instruct, the average output length decreased from 9.1k to 4.8k tokens (a 47.3% reduction) while Mean@8 improved from 45.42 to 46.67.
- **Broad applicability**: The approach reduced output length by 15% to 47.3% across all models, with Qwen3-0.6B showing a 36.7% length reduction (from 14.9k to 8.9k tokens) and Mean@8 improvement from 13.33 to 24.58.
- **Budget awareness**: Performance varied significantly across token budgets. On a strict 2k budget, aggressive length penalties (Kimi) excelled, but on generous 32k budgets, these strategies suffered from "reasoning collapse," while Laser showed a U-shaped trajectory (initial drop followed by recovery).

The paper reports Pass@8 and Mean@8 metrics across multiple benchmarks, with statistically significant improvements (the paper doesn't explicitly state statistical tests, but the consistent improvements across models and benchmarks suggest robustness).

## Related Work
The paper distinguishes itself from prior work by moving beyond isolated reward function evaluation to systematically investigate the full training recipe. While many prior approaches focused solely on reward shaping (e.g., Kimi, Laser), this work demonstrates that data composition (prompt difficulty) and optimisation strategy (rollout number, off-policy updates) are equally important. The authors position their work as the first to conduct a unified experimental protocol across these dimensions, revealing the two-stage training paradigm and the critical role of positive reward density.

## Limitations
The authors acknowledge several limitations: (1) their approach was validated primarily on mathematical and coding tasks (not creative writing), (2) they used fixed rather than adaptive token lengths, and (3) they did not evaluate on extremely large models like Qwen3-235B. A realistic limitation for practitioners is that while the approach improves efficiency, it requires additional compute during training (0.2 million GPU hours for their experiments), which may not be feasible for all teams.

## Appendix: Worked Example
Let's walk through a specific scenario: training a 4B model for 640 steps on AIME'25 with a target length of 4k tokens. The training process begins with 24 rollouts (N=24) per prompt, using DeepScaleR-Easy prompts (pass rate > 0.5).

*Step 1-400 (Length Adaptation)*: The model rapidly adjusts output distribution. At step 200, the average rollout length is 6.3k tokens; by step 400, it's reduced to 4.2k tokens. The policy entropy remains low (0.45) throughout, indicating stable training. The first 50% of training steps focus almost exclusively on meeting the length constraint.

*Step 400-640 (Reasoning Refinement)*: Once the length constraint is met, the model focuses on accuracy. At step 450, Mean@8 is 42.1; by step 640, it's improved to 46.67. The average rollout length stabilises at 4.8k tokens (slightly above the target), but crucially, the model has learned to make each token count by removing redundant reasoning steps.

This is the mechanism behind the 15-47.3% length reduction: the model learns to maintain high information density per token rather than generating longer, redundant reasoning paths. The "short-is-correct" trap was avoided because the training data contained dense positive rewards (easier prompts with high pass rates), creating a natural incentive to be concise without sacrificing accuracy.

## References

- Taiqiang Wu, Zenan Xu, Bo Zhou, Ngai Wong, "The Art of Efficient Reasoning: Data, Reward, and Optimization", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2602.20945

Tags: #ai-applications #reinforcement-learning #efficient-inference #chain-of-thought #token-optimisation
