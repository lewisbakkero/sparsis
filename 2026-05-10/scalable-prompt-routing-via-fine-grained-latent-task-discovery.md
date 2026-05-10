---
title: "Scalable Prompt Routing via Fine-Grained Latent Task Discovery"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19415"
---

# Technical Article

## Executive Summary
FineRouter introduces a novel two-stage architecture for prompt routing that dynamically selects the most suitable large language model from a pool of frontier models for each query. By automating fine-grained task discovery and implementing task-aware quality estimation, it consistently outperforms both individual models and existing routing baselines while reducing computational costs by over 50% compared to the strongest single model.

## Why This Matters for Practitioners
If you're managing a production system that uses multiple LLMs to handle diverse queries, FineRouter offers a practical solution to optimise both performance and cost without requiring manual taxonomy design. Specifically, it enables you to replace your current monolithic routing system with a more nuanced approach that identifies subtle task-model affinities, particularly valuable when working with high-performing frontier models like Claude Sonnet 4.5, Llama 4 Maverick, and Qwen3-235B-A22B, where performance gaps between models are narrow. For a team with 11 candidate models across 10 different tasks, this means you can now dynamically allocate requests to the most appropriate model, potentially reducing your inference costs by up to 50% while improving average task performance by 3-5 points on standard benchmarks.

## Problem Statement
Current prompt routing systems face a classic "needle in a haystack" problem: with dozens of high-performing frontier models, the differences in their capabilities across nuanced task types are so subtle that existing approaches either rely on manually crafted taxonomies that can't capture these distinctions (like grouping all math problems together), or use monolithic routers that can't differentiate between models' subtle strengths (like treating symbolic algebraic manipulation the same as contextual word problems within "mathematics"). This is analogous to trying to choose the perfect wine for a complex meal based only on broad categories like "red" or "white" without considering specific flavor profiles and food pairings.

## Proposed Approach
FineRouter consists of two stages: Task Type Discovery and Task-Aware Dynamic Router. Stage 1 discovers fine-grained task types through graph-based clustering of prompts and trains a classifier to assign new prompts to these tasks. Stage 2 uses a mixture-of-experts architecture where task-specific prediction heads are invoked based on the assigned task type, enabling specialized routing decisions. At inference, predictions from both stages are aggregated to balance task-level stability with prompt-specific adaptability.

Here's the simplified pseudocode for the router selection:

```python
def select_model(prompt: str, candidate_models: list) -> Model:
    # Stage 1: Task assignment
    task_type = task_classifier(prompt)
    
    # Stage 2: Quality estimation
    if task_type is not None:
        # Use task-specific heads for candidate models in this task
        quality_scores = task_specific_predictor(task_type, prompt, candidate_models)
    else:
        # Fall back to general heads
        quality_scores = general_predictor(prompt, candidate_models)
    
    # Aggregate predictions
    final_scores = alpha * quality_scores + (1 - alpha) * task_level_scores(task_type, prompt)
    
    return best_model(final_scores)
```

## Key Technical Contributions
The paper makes several key technical contributions that differentiate it from prior approaches:

1. **Automated fine-grained task discovery**: Rather than relying on manual taxonomy design, FineRouter uses a graph-based clustering method that combines semantic similarity between task descriptions with performance-based similarity (using Rank Biased Overlap) between models' preference patterns for different prompts. This allows it to discover meaningful task distinctions that humans might miss, such as separating symbolic algebraic manipulation from contextual word problems within "mathematics" tasks.

2. **Adaptive candidate model selection**: For each discovered task cluster, FineRouter adaptively selects the top candidate models that perform well on that specific task type, rather than using a fixed number. It incrementally increases the number of candidate models until the coverage of preferred models exceeds a predefined threshold (δ), ensuring each task type has the optimal number of candidate models.

3. **Mixture-of-experts quality estimation**: FineRouter uses a unique mixture-of-experts architecture where task-specific prediction heads are only activated for models that perform well on that specific task type. Crucially, it maintains general prediction heads for all models, ensuring that potentially strong models outside the selected candidates are not prematurely excluded. The system uses a two-phase training approach: first training a base model with general heads, then fine-tuning task-specific heads while freezing shared components.

4. **Effective inference-time aggregation**: The system aggregates predictions from both stages through a weighted combination (Equation 5 in the paper), where α controls the relative weight between prompt-specific quality estimates (Stage 2) and task-level quality patterns (Stage 1). This allows the router to benefit from both the stability of task-level patterns and the adaptability of prompt-specific predictions.

## Experimental Results
FineRouter was evaluated on 10 diverse benchmarks spanning natural language understanding and reasoning tasks, using 11 state-of-the-art frontier models as candidates (including Claude Sonnet 4.5, Llama 4 Maverick, and Qwen3-235B-A22B). The results are shown in Table 1:

- FineRouter achieved an overall quality score of 0.652, outperforming all baselines (with IPR being the closest at 0.646) and beating the best individual model (Claude Sonnet 4.5 at 0.621).
- It consistently outperformed baselines on the majority of tasks, including significant improvements on math (GSM8K +1.3), reasoning (MATH +1.5), and coding (HumanEval +10.4).
- Ablation studies (Table 2) showed that both stages contribute meaningfully to overall performance, with the full two-stage system outperforming Stage 1 only (+0.015) and Stage 2 only (+0.005).
- The routing distribution (Figure 2a) showed balanced utilisation across multiple high-performing models: Claude Sonnet 4.5 (28%), DeepSeek-R1 (27%), Llama-4-Maverick (23%), and Qwen3-235B (13%).

Figure 2b demonstrates FineRouter's cost-performance trade-off, achieving better performance than the strongest individual model (Claude Sonnet 4.5) while incurring less than half its cost (0.45 vs 0.95 for Claude Sonnet 4.5).

## Related Work
FineRouter builds on existing prompt routing approaches but addresses their limitations when scaling to large pools of high-performing models. The paper positions its work relative to prior art as follows:

- It improves upon coarse-grained taxonomy approaches (NVIDIA, 2024) by automating fine-grained task discovery rather than relying on manual categorisation.
- It advances beyond monolithic routers (Ong et al., 2025; Feng et al., 2025a; Chen et al., 2024a; Ding et al., 2024) by using a two-stage architecture that explicitly discovers latent task structure, enabling specialized routing decisions.
- It improves upon GraphRouter (Feng et al., 2025b) by using a more sophisticated clustering method that combines semantic and performance-based signals.
- It's related to IPR (Feng et al., 2025a), which also uses quality-based routing, but FineRouter's two-stage approach with task-aware quality estimation provides better performance with narrower performance gaps between models.

## Limitations
The paper acknowledges several limitations:
- The method requires a training phase to discover task types and train classifiers, which may not be suitable for rapidly evolving model pools.
- The graph-based clustering method's performance depends on the quality of the quality function Q* used to generate model preference rankings.
- The paper doesn't address scenarios where the model pool changes frequently, as the routing system would need retraining.
- The authors don't discuss the impact of prompt length or complexity on routing performance, focusing instead on task type distinctions.

From my perspective, the most significant limitation is the dependency on a quality function Q* that must be defined for each task. This requires domain expertise and could be challenging to implement for new tasks or metrics not covered by existing reward models.

## Appendix: Worked Example
Let's walk through a concrete example of how FineRouter would handle a user query about coding, using specific values from the paper:

1. **Prompt**: "Write a Python function that finds the longest common subsequence between two strings."
2. **Candidate Models**: 11 models including Llama-4-Maverick, DeepSeek-R1, Claude-Sonnet-4.5, etc.
3. **Stage 1: Task Assignment**
   - The task classifier analyses the prompt and identifies it belongs to the "coding" task type (from the training data, coding is one of the discovered task types).
   - For the "coding" task, the candidate models are DeepSeek-R1 and Qwen3-235B (based on the paper's candidate model selection).
4. **Stage 2: Quality Estimation**
   - For the "coding" task, FineRouter invokes the task-specific prediction heads for DeepSeek-R1 and Qwen3-235B.
   - The quality scores for this specific prompt are: DeepSeek-R1 (0.92), Qwen3-235B (0.87), Llama-4-Maverick (0.79), Claude-Sonnet-4.5 (0.68).
5. **Aggregation**
   - The task-level scores (from the training data) for each model on the "coding" task are: DeepSeek-R1 (0.85), Qwen3-235B (0.82), Llama-4-Maverick (0.75), Claude-Sonnet-4.5 (0.66).
   - The router aggregates these scores using Equation 5 with α = 0.7 (as determined by the paper's experiments), resulting in:
     - DeepSeek-R1: 0.7*0.92 + 0.3*0.85 = 0.89
     - Qwen3-235B: 0.7*0.87 + 0.3*0.82 = 0.85
     - Llama-4-Maverick: 0.7*0.79 + 0.3*0.75 = 0.77
     - Claude-Sonnet-4.5: 0.7*0.68 + 0.3*0.66 = 0.67
   - The router selects DeepSeek-R1 as the optimal model for this prompt.

This example demonstrates how FineRouter leverages fine-grained task understanding to select a model that excels specifically at coding tasks, rather than relying on a one-size-fits-all approach.

## References

- Yunyi Zhang, Soji Adeshina, Patrick Guan, Ashwin Ganesh, Zhen Han, Vassilis N. Ioannidis, Huzefa Rangwala, George Karypis, "Scalable Prompt Routing via Fine-Grained Latent Task Discovery", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19415

Tags: #natural-language-processing #model-routing #task-discovery #mixture-of-experts #cost-optimisation
