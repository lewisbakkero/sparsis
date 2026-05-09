---
title: "LARFT: Closing the Cognition-Action Gap for Length Instruction Following in Large Language Models"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19255"
---

## Executive Summary
LARFT (Length-Aware Reinforcement Fine-Tuning) closes the "cognition-action gap" in LLM length instruction following by training models to internally represent output length, rather than just externally enforcing constraints. It achieves an average 20.92-point improvement on length benchmarks with only a 1.45-point decline on general capability benchmarks. Practitioners building LLM applications requiring precise output control will benefit from this technique without significant performance trade-offs.

## Why This Matters for Practitioners
If you're building LLM-powered tools for creative writing, report generation, or content summarisation in production, this paper directly addresses a critical pain point: your models currently struggle with consistent output length, causing user frustration when they request "a 300-word summary" but get outputs that are 200 or 500 words. LARFT solves this without requiring external token insertion or complex prompt engineering. Specifically, for content generation applications, you can now integrate LARFT training into your existing fine-tuning pipeline with minimal overhead (only a 0.01 hyperparameter adjustment) to achieve 20-25% more precise length control. This means you can confidently implement length constraints in your user-facing APIs without worrying about degraded performance on other tasks - a crucial factor for maintaining user satisfaction while optimising for resource efficiency.

## Problem Statement
Current LLMs struggle with length control like a human trying to dance to a specific beat without ever having heard music before - they can produce output of various lengths, but can't consistently match a target because they lack an internal "rhythm sense" for length. This is particularly problematic when models expand their context windows (as they do with each new release), yet their ability to control output length doesn't scale proportionally. For example, when generating a 100-word marketing description, a model might consistently produce 75 words (too short) or 150 words (too long), leading to inconsistent user experiences in applications like content management systems or conversational agents where length constraints are part of the design.

## Proposed Approach
LARFT unifies three components: Length-Oriented Reinforcement Learning (to guide generation toward length targets), Hindsight Length Awareness (to train the model to count its own output length), and a Unified Optimisation Mechanism (to dynamically balance these objectives). The core idea is to create a feedback loop between understanding length (cognition) and generating precisely length-constrained text (action).

Here is the key algorithm in pseudocode:

```python
def larft_training_step(prompt, target_length, model):
    # Step 1: Generate candidate responses (with length constraints)
    responses = model.generate(prompt, max_tokens=target_length)
    
    # Step 2: Transform responses into hindsight awareness tasks
    awareness_tasks = []
    for response in responses:
        awareness_prompt = f"{response}\nCount the words in the text above?"
        awareness_tasks.append(awareness_prompt)
    
    # Step 3: Compute length reward for standard RL
    length_rewards = [compute_length_reward(response, target_length) for response in responses]
    
    # Step 4: Compute awareness loss (train to count own output length)
    awareness_losses = [compute_awareness_loss(task) for task in awareness_tasks]
    
    # Step 5: Unified optimisation with dynamic weighting
    loss = rl_loss(length_rewards) + lambda_t * awareness_loss(awareness_losses)
    
    # Step 6: Update model
    model.update(loss)
    
    return model
```

## Key Technical Contributions
LARFT's core innovations enable precise length control by directly addressing the missing cognitive component. Each contribution explains how the authors specifically solved the length cognition problem:

1. **Hindsight Length Awareness mechanism**: Rather than trying to teach length through external signals, LARFT repurposes the model's own outputs as training data. For each generated response, it creates a "Count the words in the text above?" prompt and uses this to train the model to self-evaluate its output length. This effectively transforms on-policy samples (which would normally be discarded as length-mismatched) into valuable training examples for length cognition, creating a closed-loop system where the model learns to count its own outputs.

2. **Dynamic weighting with Cosine Annealing**: The authors implement a learning curriculum that prioritises establishing length cognition early (using λ_max = 0.01) before shifting focus to generation control. This is achieved through a Cosine Annealing schedule that gradually reduces the weight on the awareness loss over training steps. This avoids the entanglement problem where length constraints become mixed with semantic features during training.

3. **Verified Length Reward formulation**: Instead of using a binary pass/fail criterion, LARFT uses a piecewise linear reward function based on normalized absolute deviation from target length. This provides a stable, dense reward signal for reinforcement learning, enabling more precise length control compared to methods that treat length as a binary constraint.

## Experimental Results
LARFT was evaluated across four base models (Qwen2.5-3B, Qwen2.5-7B, Llama-3.2-3B, and Llama-3.1-8B) on three length instruction following benchmarks (LIFEBench, LongBench, and Lenctrl-Bench) and four general capability benchmarks (MMLU, GSM8k, IFEval, GPQA).

On length instruction following benchmarks, LARFT achieved an average improvement of 20.92 points (compared to base models) and outperformed the strongest RL baseline by 4.59 points. Specifically, on LIFEBench it achieved the lowest Length Deviation (LD) and highest Length Score (LS) across all four models. On LongBench, LARFT improved Length Score (Sl) by 1.11-4.36 points over the pure RL baseline. For short-length constraints (Lenctrl-Bench), LARFT achieved the lowest Mean Absolute Error (MAE) on three out of four models.

Crucially, LARFT maintained general capabilities with only a marginal performance decline: the maximum drop on general benchmarks was 2.02 points (on GPQA with Qwen2.5-3B), while showing a slight improvement (+2.22 points) on generation quality metrics.

## Related Work
LARFT positions itself against three categories of prior work on length control: (1) External length marker incorporation (using length-specific tokens or prompts, like PositionID and Ruler), (2) Length-constrained policy optimisation (using RL approaches), and (3) Inference-time interventions. The authors demonstrate that external length markers often incur inference overhead and struggle with long-form generation, while RL approaches fail to decouple length from semantics, leading to poor sample efficiency. LARFT improves upon these by addressing the root cause: the lack of internal length cognition, rather than trying to enforce length from the outside.

## Limitations
The authors acknowledge that LARFT is limited to length constraints under 4,000 words due to base model output length limitations. The paper doesn't explicitly evaluate LARFT on extremely short lengths (e.g., 10-20 words), though it does show good performance on Lenctrl-Bench which covers these ranges. The authors also don't explore the potential impact of different language models' tokenisation strategies on LARFT's effectiveness, which could affect results across different models. From an engineering perspective, the primary limitation for practitioners is that LARFT requires training time (3 epochs of RL training), which might be prohibitive for rapidly iterating on models in production environments.

## Appendix: Worked Example
Let's walk through a concrete example of LARFT in action with a specific prompt and target length. Suppose we want to generate a 100-word explanation of LLMs for a non-technical audience.

1. **Initial Prompt**: "Explain what LLMs are in about 100 words for a non-technical audience"
2. **Base Model Output**: "Large Language Models (LLMs) are artificial intelligence systems trained on vast amounts of text data to understand and generate human-like language. They can perform tasks like answering questions, writing stories, and summarising documents. LLMs like GPT-3 or BERT represent a significant advancement in natural language processing, enabling more sophisticated interactions between humans and computers. They are widely used in applications from chatbots to content creation."

   This output contains 108 words (8 words over target).
3. **Hindsight Awareness Task**: The model is presented with this output followed by "Count the words in the text above?" and the expected length (100 words) is used to train it to self-evaluate. The awareness loss is computed based on the difference between the actual output length (108) and the target (100).
4. **Length Reward**: The length reward is calculated as max(0, 1 - |108-100|/100) = max(0, 1 - 0.08) = 0.92.
5. **Training Update**: The model is updated with a combined loss of RL loss (based on 0.92) and awareness loss (based on the self-evaluation error).
6. **Iterative Refinement**: After multiple training steps, the model learns to generate responses closer to the target length. In subsequent generations, it might produce 99 words, which would yield a length reward of 0.99, and the awareness loss would be smaller.

This process creates a feedback loop where the model improves its length cognition (how it internally represents length) and its generation (how it produces length-constrained output) together.

## References

- Wei Zhang, Lintong Du, Yuanhe Zhang, Zhenhong Zhou, Kun Wang, Li Sun, Sen Su, "LARFT: Closing the Cognition-Action Gap for Length Instruction Following in Large Language Models", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19255

Tags: #language-models #length-constraint #reinforcement-learning #fine-tuning #llm-optimisation
