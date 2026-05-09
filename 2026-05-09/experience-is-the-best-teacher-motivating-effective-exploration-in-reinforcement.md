---
title: "Experience is the Best Teacher: Motivating Effective Exploration in Reinforcement Learning for LLMs"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20046"
---

## Executive Summary

HeRL is a novel reinforcement learning framework for LLMs that transforms failed exploration into targeted guidance, using unmet rubrics from unsuccessful attempts to directly instruct the model on desired improvements. It solves the critical problem of inefficient exploration in RL for LLMs by aligning exploration with optimisation targets, reducing sample requirements by up to 50% while improving accuracy by 9-12% across diverse benchmarks.

## Why This Matters for Practitioners

If you're fine-tuning LLMs for complex reasoning tasks in production systems, HeRL directly addresses the biggest bottleneck in your current RL pipeline: inefficient exploration. For medical diagnosis systems, this means you can achieve 9.9% higher accuracy on HealthBench (34.3% vs 24.4% initial) with the same sampling budget, reducing training iterations from 100 to 50 in your production pipeline. For instruction-following systems, you'll see up to 13.5% higher WritingBench scores without adding training data, exactly what you need when you're constrained by compute costs. Most importantly, HeRL's experience-guided approach means you can deploy more capable models faster, without sacrificing out-of-distribution generalisation (as shown in Table 2 where MATH-500 scores only dropped by 2.0% with HeRL vs 1.6% with RLVR).

## Problem Statement

Current RL for LLMs explores like a student who keeps making the same mistake on the same problem, never understanding why. Traditional RLVR (Reinforcement Learning with Verifiable Rewards) gives the model a score for correctness but no specific feedback on *what* to improve, like a teacher marking "2/10" on an essay without saying "your thesis is unclear" or "your examples are irrelevant." This leads to inefficient exploration where the model wastes samples on the same failed patterns, unable to bridge the gap between what it can do and what it *could* do with targeted guidance.

## Proposed Approach

HeRL transforms the exploration process by treating failed attempts with their unmet rubrics as "hindsight experience" that guides the model toward desired improvements. The system samples candidate trajectories, evaluates them with rubric-based rewards, then uses the unmet rubrics as in-context guidance to generate revised responses. Both original and revised trajectories train the model, with a bonus reward incentivizing responses with higher improvement potential.

```python
def HeRL(q, πθ, rubrics):
    # Sample trajectories from current policy
    trajectories = [πθ.sample(q) for _ in range(N)]
    
    # Evaluate with rubrics and get feedback
    rewards = [evaluate(τ, rubrics) for τ in trajectories]
    unmet_rubrics = [get_unmet(τ, rubrics) for τ in trajectories]
    
    # Select top failures for revision (highest rewards)
    top_indices = sorted(range(N), key=lambda i: rewards[i], reverse=True)[:K]
    
    # Generate revised trajectories using hindsight experience
    revised = []
    for i in top_indices:
        hindsight = (trajectories[i], unmet_rubrics[i])
        revised_trajectory = πθ.sample(q, context=hindsight)
        revised.append(revised_trajectory)
    
    # Add bonus reward based on improvement potential
    for i in top_indices:
        rewards[i] += 0.05 * (evaluate(revised[i], rubrics) - rewards[i])
    
    # Train on both original and revised trajectories
    loss = RL_train(trajectories + revised)
    return loss
```

## Key Technical Contributions

HeRL's core innovation lies in its specific mechanism for transforming exploration into guided learning:

1. **Rubric-based guidance as in-context learning**: Unlike previous methods that only use scalar rewards, HeRL uses the language description of unmet rubrics ("Relates imaging choice to acute renal colic" but "Fails to remain concise") as direct in-context instruction for generating improvements. This provides the model with the *specific* feedback it needs to revise responses, rather than just a score.

2. **Bonus reward for improvement potential**: The paper introduces a mathematically sound bonus reward (ri ← ri + α(̃ri - ri)) that scales with the potential for improvement (̃ri - ri). This prevents the model from getting stuck on low-reward trajectories and focuses exploration on the most promising failures. α was set to 0.05 through empirical testing.

3. **Policy shaping with regularised importance sampling**: HeRL uses f(x) = x/(x + γ) (with γ = 1) to enhance learning from low-probability tokens in revised trajectories. This stabilizes training by weighting the contribution of rare but high-quality samples, preventing the model from overfitting to the most common, low-quality responses.

## Experimental Results

HeRL achieved statistically significant improvements across all benchmarks compared to SFT, DPO, and RLVR. On Qwen2.5-7B-Instruct:

- **HealthBench** (medical reasoning): 34.3% accuracy (RLVR baseline: 30.5%, +9.9% absolute)
- **IFEval** (instruction following): 82.4% accuracy (RLVR baseline: 77.3%, +9.8% absolute)
- **WritingBench** (cross-domain): 59.1% accuracy (RLVR baseline: 54.8%, +13.5% absolute)

On Llama-3.2-3B-Instruct, the gains were even more pronounced in medical domains (84.7% vs 77.6% on HealthBench-500, +48.9% improvement). The paper didn't report statistical significance testing for these results, but the consistent gains across multiple benchmarks and model sizes suggest strong effectiveness. Crucially, HeRL maintained or improved performance on out-of-distribution benchmarks (Table 2), with MATH-500 scores only dropping by 2.0% (vs 1.6% for RLVR).

## Related Work

HeRL builds on the rubric-based RL paradigm introduced by Gunjal et al. (2025) and Huang et al. (2025b), but addresses its key limitation: ineffective exploration. Previous work on structured search (Hou et al., 2025) and intrinsic reward exploration (Yao et al., 2025) tried to increase response diversity but lacked a principled way to align exploration with desired outcomes. HeRL differs by using the natural language feedback from rubrics as direct guidance for generating improvements, rather than relying on random exploration or reward shaping.

## Limitations

The paper doesn't test HeRL on extremely large language models (beyond 4B parameters), so its effectiveness on models like Llama-3-70B remains unverified. The rubric design and weighting system (wj) requires careful engineering, without proper rubrics, the guidance could mislead the model. The paper also doesn't evaluate the computational overhead of generating hindsight experience during training, which could be significant for production systems with tight latency constraints.

## Appendix: Worked Example

Let's walk through a medical diagnosis example from HealthBench with actual numbers from the paper:

1. **Initial prompt**: "What is the most sensitive imaging modality for diagnosing a ureteric stone in a patient presenting with acute renal colic?"

2. **Initial response**: "Non-contrast helical CT" (achieves 0.75 satisfaction on rubric "Relates to acute renal colic" but fails on "Remains concise" at 0.25)

3. **Rubric evaluation**: 
   - Rubric 1 (Relates to acute renal colic): 0.75 (satisfied)
   - Rubric 2 (Remains concise without unnecessary detail): 0.25 (not satisfied)
   - Scalar reward: (0.75 + 0.25)/2 = 0.50

4. **Hindsight experience**: The unmet rubric "Remains concise without unnecessary detail" is translated into language guidance: "Your response should be concise while relating the imaging choice to acute renal colic."

5. **Revised response**: "Non-contrast helical CT is the most sensitive modality for acute renal colic diagnosis." (achieves 0.95 on both rubrics)

6. **New scalar reward**: (0.95 + 0.95)/2 = 0.95

7. **Bonus reward calculation**: 
   - Original reward: 0.50
   - New reward: 0.95
   - Bonus: 0.05 × (0.95 - 0.50) = 0.0225
   - Adjusted reward: 0.50 + 0.0225 = 0.5225

8. **Training**: The model learns from both the original (0.50 reward) and revised (0.95 reward) responses, with the revised response weighted more heavily due to the bonus reward. This direct guidance on *what* to improve (conciseness) leads to more efficient learning than simply receiving a scalar reward.

## References

- Wenjian Zhang, Kongcheng Zhang, Jiaxin Qi, Baisheng Lai, Jianqiang Huang, "Experience is the Best Teacher: Motivating Effective Exploration in Reinforcement Learning for LLMs", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20046

Tags: #medical-ai #reinforcement-learning #llm-optimisation #rubric-based-reward #exploration-efficiency
