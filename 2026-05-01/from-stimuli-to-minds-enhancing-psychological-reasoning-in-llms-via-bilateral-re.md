---
title: "From Stimuli to Minds: Enhancing Psychological Reasoning in LLMs via Bilateral Reinforcement Learning"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36988"
---

## Executive Summary
The authors introduce Psy-Interpreter, a reinforcement learning framework that enhances psychological reasoning in compact language models through expert-labelled data and bilateral reward design. Their approach enables small models (0.5B-3B parameters) to achieve expert-level performance on psychological reasoning benchmarks, rivaling much larger commercial models without requiring additional training data.

## Why This Matters for Practitioners
If you're building production systems that require nuanced understanding of human emotion and social context, such as customer support chatbots, mental health applications, or content moderation tools, this paper demonstrates a practical path to improve psychological reasoning without massive model sizes. Your team should prioritize building domain-specific psychological datasets with expert annotation rather than relying on LLM-generated training data, and implement bilateral reward functions that balance format compliance, reasoning depth, and token accuracy. The results show that compact models trained with these techniques can outperform larger systems on psychological reasoning tasks, directly reducing inference costs while improving accuracy on emotionally sensitive scenarios.

## Problem Statement
Current language models struggle with psychological reasoning like humans do, because they treat emotional cues like surface-level text patterns rather than internal mental states. Imagine trying to understand a friend's emotional state from a brief text message without knowing their full life context, cultural background, or current stressors: you'd likely miss key nuances. Similarly, LLMs often rely on superficial patterns rather than internalizing the complex mental models required for accurate psychological interpretation.

## Proposed Approach
The authors propose a three-component system: StimuliQA (a dataset of psychologically annotated narratives), Psy-Interpreter (a reinforcement learning framework with bilateral reasoning), and Continual Learning (for self-improvement). The system works by training small language models on narratives annotated with psychological variables, using a reward function that balances accuracy, format compliance, reasoning depth, and repetition control. This enables compact models to internalize psychological reasoning patterns without requiring massive parameter counts.

```python
def bilateral_reward(predicted_answer, ground_truth, output_length, batch_avg_length, batch_avg_f1):
    # Token-level F1 score between predicted and ground truth
    f1 = calculate_token_f1(predicted_answer, ground_truth)
    
    # Format compliance check (ensuring proper XML tags)
    format_correct = check_format_compliance(predicted_answer)
    
    # Bilateral reasoning term: reward appropriate reasoning length
    if (output_length / batch_avg_length) < 0.8 and f1 > batch_avg_f1:
        reasoning_length_reward = 0.5
    elif (output_length / batch_avg_length) > 1.2 and f1 > batch_avg_f1:
        reasoning_length_reward = 0.5
    else:
        reasoning_length_reward = 0.0
    
    # Repetition penalty based on 4-gram overlap
    repetition_penalty = -min(0.3, calculate_repeat_ratio(predicted_answer))
    
    # Combined reward (weights normalized to sum to 1)
    total_reward = (
        0.3 * f1 + 
        0.2 * format_correct + 
        0.3 * reasoning_length_reward + 
        0.2 * repetition_penalty
    )
    return total_reward
```

## Key Technical Contributions
The core innovation lies in how the reward function and trajectory cache work together to foster expert-like psychological reasoning. The authors don't just add more data, they engineer the learning process to align with psychological theories.

1. **Bilateral Reward Design**: The reward function combines four specific components (token accuracy, format compliance, reasoning depth, and repetition control) with weights that reflect psychological reasoning complexity. The system calculates a batch-level average of reasoning lengths and F1 scores to ensure the model receives appropriate rewards for reasoning depth, not just for longer responses. This prevents models from generating verbose responses without genuine understanding.

2. **Trajectory Cache for Stable Training**: The authors implement a trajectory cache that tracks recent rollout performance to dynamically adjust rewards based on trends rather than fixed values. With B batches and cache size C, the batch cache Bc = B × C summarizes training dynamics, stabilizing estimation across different model scales and tasks. This avoids the suboptimal fixed rewards that often plague reinforcement learning approaches.

3. **Psychology-Driven Dataset Construction**: The StimuliQA dataset contains 3,280 real-life stimuli with 58 psychological variables across three dimensions: Emotional Reactions (29 variables), Narrative Transformation (12 variables), and Collective Psychology (17 variables). The variables are grounded in psychological theories (Lazarus's appraisal theory, McAdams's narrative theory, and Ryff & Keyes's psychological well-being model), ensuring the dataset reflects actual psychological processes rather than superficial text patterns.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The authors conducted comprehensive experiments across multiple benchmarks. Models trained on StimuliQA consistently outperformed those trained on synthetic data (LlamaQA/MistralQA) across all model sizes and benchmarks. For instance, on SimpleToM, Qwen2.5-3B achieved 37.62 F1 and 56.44% accuracy, significantly exceeding Llama 3.3 (18.48/26.16%) and Mistral 8×7B (35.02/33.33%).

The Bilateral Reward (BR) consistently achieved the highest F1 scores across all benchmarks and model sizes. On the Qwen2.5-3B model, BR outperformed the baseline Basic R1 by +5.34 F1 (from 35.04 to 40.38), with particularly strong gains on National Theme (41.79→44.44) and Self Identification (37.98→44.44).

Most impressively, compact models trained with Psy-Interpreter (0.5B-3B) rival or outperform larger commercial models: Psy-Interpreter-SFT (3B) achieved 82.82 F1 on SocialIQa, surpassing GPT-4 nano (57.03) and Claude 3 Haiku (15.94). The paper doesn't specify statistical significance tests, but reports consistent gains across all benchmarks.

## Related Work
The paper positions itself within several research areas: psychological reasoning (ToMbench, Psychobench, CogBench), Chain-of-Thought reasoning, and reinforcement learning for language models. It specifically acknowledges that while previous work has focused on moral reasoning or social inference, they address the gap in psychological reasoning that requires inference of implicit mental states in ambiguous, contextually rich scenarios. Their contribution extends beyond existing benchmarks by introducing a dataset grounded in professional psychological theory rather than LLM-generated content, and by developing a reward function specifically calibrated for psychological reasoning rather than generic language tasks.

## Limitations
The paper acknowledges several limitations: the dataset contains only English narratives (limiting cross-lingual application), the psychological variables are limited to specific theory-driven dimensions (leaving out other psychological models), and the experiments focus on question-answering rather than more complex conversational reasoning. The authors don't explore whether the framework generalizes to other types of psychological tasks beyond the specific benchmarks used. Additionally, the paper doesn't provide detailed analysis of computational overhead or inference latency, which would be important for production deployment of such systems.

## Appendix: Worked Example
Let's walk through a single example from the StimuliQA dataset using the Bilateral Reward framework. The input narrative is: "Mrs. Li's son, Xiao Qiang, now a college student, tells her he has been chosen as the captain of the school's popular football team." The ground truth answer for the psychological variable "Emotional Reaction: P_p/n" (How do you feel about this personal event?) is "Proud."

During training, the model generates: "I guess it started in middle school. Mrs. Li would likely be very proud of her son." The tokenized output contains 15 words, while the batch average output length is 12 words (from the training data). The model achieves a token-level F1 score of 0.85 (based on matching tokens after normalization) against the ground truth.

The Bilateral Reward calculation:
1. Token accuracy (rF1): 0.85
2. Format compliance (rfmt): 1.0 (response is properly structured in XML tags)
3. Reasoning length: 15/12 = 1.25, which exceeds the upper threshold (τ+ = 1.2) and the F1 (0.85) exceeds the batch average (0.78), so rBR = 0.5
4. Repetition penalty: The response has low semantic repetition (4-gram overlap ratio = 0.08), resulting in rrep = 0.0 (no penalty)

Weighted reward calculation (normalized to sum to 1):
- Token accuracy: 0.3 × 0.85 = 0.255
- Format compliance: 0.2 × 1.0 = 0.200
- Bilateral reasoning: 0.3 × 0.5 = 0.150
- Repetition penalty: 0.2 × 0.0 = 0.000
- Total reward: 0.605

This specific reward calculation directly encourages the model to produce concise, accurate responses that align with psychological reasoning patterns, rather than simply generating longer strings or repeating phrases.

## References

- **Code:** https://github.com/Githubuseryf/Stimuli2Minds
- Yichao Feng, Haoran Luo, Lang Feng, Shuai Zhao, Anh Tuan Luu, "From Stimuli to Minds: Enhancing Psychological Reasoning in LLMs via Bilateral Reinforcement Learning", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36988

Tags: #psychology #language-models #reinforcement-learning #social-cognition #mental-state-inference
