---
title: "Maximizing mutual information between user-contexts and responses improve LLM personalization with no additional data"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19294"
---

## Executive Summary
MIPO (Mutual Information Preference Optimisation) is a self-improvement framework for LLMs that personalizes responses without additional data or human supervision by maximising mutual information between user contexts and responses. It constructs contrastive preference pairs using the base model's outputs and trains via Direct Preference Optimisation, achieving 3, 40% relative gains on personalized instruction-following and 1, 18% gains on reasoning tasks. Practitioners can adopt this technique to enhance user personalization in production systems without costly data collection pipelines.

## Why This Matters for Practitioners
If you're building LLM-powered applications that require user-specific responses (e.g., customer support chatbots, educational platforms, or personal assistants), MIPO provides a practical way to improve personalization without additional data collection pipelines. For instance, in a customer support system, you can train your LLM to adjust response complexity based on user profiles (technical vs. non-technical users) using only the model's own outputs, eliminating the need for costly user preference datasets. This approach can be implemented as a post-training step before deployment, requiring no changes to your existing inference pipeline. Engineers should consider adding MIPO as a standard post-training step for any LLM that interacts with diverse user groups, particularly where manual personalization data collection is infeasible.

## Problem Statement
Current personalization techniques for LLMs rely on human-labelled preference data or external verifiers, creating a bottleneck similar to a traffic jam at a toll booth. Just as toll booths slow down highway traffic when they're the only access point, the reliance on human-labelled data slows down the development of personalized LLMs. The authors describe this as "data is the fossil fuel of AI", a resource that's becoming increasingly scarce and expensive to extract. Without a better approach, personalization efforts will remain limited to projects with dedicated data collection budgets, leaving most applications with generic, non-adaptive responses.

## Proposed Approach
MIPO is a two-part framework for enhancing LLM personalization through contrastive data augmentation. First, it constructs preference pairs by generating a positive response conditioned on the correct prompt, and a negative response by conditioning on a random, unrelated prompt. Second, it applies Direct Preference Optimisation (DPO) to learn from these paired data, maximising the pointwise conditional mutual information between prompts and responses under the base LLM.

The core algorithm involves:
1. For each prompt, generate a positive response (correct context)
2. Generate a negative response using a random, unrelated prompt
3. Train using DPO on these paired responses

Here's the specific pseudocode for MIPO for personalization:

```python
def mipo_personalization(prompts, contexts, reference_model, policy):
    # Initialize policy with reference model
    policy = reference_model.copy()
    
    # Generate preference pairs
    preference_pairs = []
    for prompt in prompts:
        # Generate positive response (correct context)
        positive_response = reference_model.generate(prompt, context)
        
        # Generate negative response (random context)
        random_context = random.choice(contexts)
        negative_response = reference_model.generate(prompt, random_context)
        
        # Add to preference pairs
        preference_pairs.append((prompt, positive_response, negative_response))
    
    # Train using DPO
    loss = dpo_loss(policy, preference_pairs)
    optimize_policy(policy, loss)
    
    return policy
```

## Key Technical Contributions
MIPO's novelty lies in its ability to leverage intrinsic signals from the model itself to improve personalization and general reasoning without external supervision. This section details the specific mechanisms that make MIPO effective:

1. **Contrastive Data Augmentation via Prompt Mismatching**: MIPO creates negative examples by conditioning responses on random, unrelated prompts rather than relying on larger models or human feedback. This approach is theoretically grounded in InfoNCE loss, where the negative samples are drawn from the marginal distribution πref(y) rather than the conditional distribution. The key insight is that a response conditioned on a random prompt will be a poor match for the original prompt, creating a clear contrast for the model to learn from.

2. **Pointwise Mutual Information as Implicit Reward**: By connecting DPO to InfoNCE, MIPO identifies the pointwise mutual information log(πref(y|x)/πref(y)) as the implicit reward signal. This insight allows MIPO to maximise mutual information between prompts and responses without needing explicit reward functions or verifiable rewards. Unlike RLHF approaches that require human labels, MIPO's reward is derived purely from the base model's behaviour.

3. **Personalization via Conditional Mutual Information**: MIPO extends to personalization by maximising the conditional mutual information I(Y;C|X) = log(πref(y|x,c)/πref(y|x)), which encourages the model to use user-specific context rather than relying on general prompt information. This is implemented by generating negative responses with random user contexts rather than random prompts, creating a more targeted contrast for personalization.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
MIPO was evaluated on personalized instruction-following tasks using three datasets: Multi-Bench (Lee et al., 2024b), PRISM (Kirk et al., 2024a), and Community Alignment (Zhang et al., 2025c). On these tasks, models trained with MIPO achieved 3, 40% relative improvements over strong personalized prompting baselines, depending on the model size and dataset. For example, on the Community Alignment dataset, Qwen-7B-Instruct showed a 30% relative improvement over the best prompting baseline.

For reasoning tasks, MIPO was evaluated on GSM8K (Cobbe et al., 2021), the AI2 Reasoning Challenge (Clark et al., 2018), and other multiple-choice question datasets. On these benchmarks, MIPO increased instruction-tuned model performance by 1, 4% on average and up to 18% for smaller models (e.g., Llama-3-8B). Notably, these gains were achieved without any additional data or human supervision, and MIPO's results often matched or exceeded those achieved by RLVR using ground truth rewards.

The paper does not explicitly report statistical significance tests for these results, though it's likely they used standard methods given the scale of the experiments.

## Related Work
MIPO builds upon and improves over existing work in post-training for LLMs, particularly in the realm of self-improvement without human supervision. It extends the concept of using contrastive data augmentation, which has been explored in computer vision (e.g., rotation and translation augmentations), to natural language processing. Unlike prior work that required larger models (Xu et al., 2024; Geng et al., 2025) or human instructions for revision (D'Oosterlinck et al., 2024), MIPO only needs the base model's outputs to generate contrastive pairs.

MIPO is related to the SAMI method (Fränken et al., 2024), which also uses InfoNCE for constitution following, but MIPO differs by being built directly on DPO rather than introducing a new loss function. This makes MIPO more practical for implementation, as it requires only a simple data augmentation step before standard DPO training.

## Limitations
The authors acknowledge that MIPO's performance may degrade in scenarios where the base model's responses are poor quality, as the contrastive pairs would be based on incorrect information. They also note that the method has not been tested on extremely large models (over 70B parameters), though they expect it to generalise based on their results with various-sized Llama- and Qwen-Instruct models.

The paper doesn't test MIPO on tasks with long-term user context or multi-turn conversations, focusing instead on single-turn interactions. Additionally, while MIPO improves personalization, it doesn't address the challenge of maintaining user privacy when using personal context, which is a critical consideration for production systems.

## Appendix: Worked Example
Let's walk through how MIPO personalizes a response for a user with a specific context. Consider a scenario where we have:
- A user query: "Explain quantum entanglement"
- User context: "I have a physics PhD" (context A)
- Random context: "I'm a high school student" (context B)

The base model (reference model) first generates a positive response using context A:
```
Positive response (context A):
"Quantum entanglement is a phenomenon where two or more particles become interconnected, such that the state of one particle instantaneously influences the state of the other, regardless of the distance between them. This correlation persists even when the particles are separated by large distances, and it's a key feature of quantum mechanics that has been experimentally verified through Bell's theorem and numerous experiments like those by Aspect."
```

Then the reference model generates a negative response using a random context (context B):
```
Negative response (context B):
"Quantum entanglement is a phenomenon where particles interact with each other in a way that their states become correlated, like two people who are always in the same room. This correlation is important in quantum mechanics and is used in the study of quantum computers."
```

The model then learns to differentiate between these two responses using DPO. The loss function emphasizes responses that are more likely under the reference model when conditioned on the correct context (context A) compared to the marginal probability across all contexts.

The key insight is that the positive response (context A) should be preferred as it's more specific to the user's expertise level, while the negative response (context B) is a more generic explanation suitable for a high school student. MIPO helps the model learn this distinction by maximising the conditional mutual information I(Y;C|X), which encourages responses to adapt to the specific user context rather than using a general explanation.

## References

- Hyunji Nam, Haoran Li, Natasha Jaques, "Maximizing mutual information between user-contexts and responses improve LLM personalization with no additional data", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19294

Tags: #natural-language-processing #personalization #contrastive-learning #mutual-information
