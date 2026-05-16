---
title: "Measuring Faithfulness Depends on How You Measure: Classifier Sensitivity in LLM Chain-of-Thought Evaluation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20172"
---

## Executive Summary
This paper demonstrates that chain-of-thought (CoT) faithfulness metrics are not objective properties of LLMs but depend entirely on the evaluation methodology used. Three classifiers applied to identical reasoning traces produced faithfulness rates ranging from 69.7% to 82.6%, with disagreements as large as 43.4 percentage points for certain hint types. This means published faithfulness numbers cannot be meaningfully compared across studies that use different evaluation approaches.

## Why This Matters for Practitioners
If you're building production systems that rely on LLM reasoning, this paper should fundamentally change how you evaluate model behaviour. You cannot treat faithfulness as a single, comparable metric across different implementations, it's as if you were comparing the weight of apples measured in grams versus pounds without converting units. When selecting models for critical applications, you must report your exact evaluation methodology (including classifier prompts), compare models using the same classifier, and avoid claiming that Model A is "more faithful" than Model B unless they were evaluated identically. For production deployment, always run parallel evaluations using at least two different classifier approaches to detect if rankings might reverse under different measurement instruments.

## Problem Statement
Current LLM reasoning evaluation practices treat faithfulness as an objective property like a car's fuel efficiency, when it's actually more like a car's "drivability", subjective and context-dependent. Just as a driver might rate a car as "drivable" based on how it handles turns (epistemic dependence), while an engineer might rate it as "drivable" based on whether it mentions turn signals (lexical mention), we've been making apples-to-oranges comparisons in research. This leads to flawed conclusions, such as believing Model X is "more faithful" than Model Y when the difference stems entirely from how we measure it.

## Proposed Approach
The authors systematically compare three classifiers on identical reasoning traces to measure how much faithfulness metrics vary based on evaluation methodology. The core insight is that different classifiers operationalise faithfulness at different stringency levels: lexical mention (regex) vs. epistemic dependence (LLM judge). Their approach involves applying all three classifiers to the same dataset of 10,276 influenced reasoning traces across 12 open-weight models, then quantifying disagreements using Cohen's κ and McNemar's test.

```python
def evaluate_faithfulness(model_outputs, hint_type):
    # Three classifiers applied to identical data
    regex_result = regex_detector(model_outputs, hint_type)
    pipeline_result = regex_plus_llm(model_outputs, hint_type)
    sonnet_result = sonnet_judge(model_outputs, hint_type)
    
    # Compare results across classifiers
    agreement = compute_cohen_kappa(pipeline_result, sonnet_result)
    rank_correlation = spearman_rank_correlation(pipeline_result, sonnet_result)
    
    return {
        "regex": regex_result,
        "pipeline": pipeline_result,
        "sonnet": sonnet_result,
        "agreement": agreement,
        "rank_correlation": rank_correlation
    }
```

## Key Technical Contributions
The researchers developed a rigorous framework to quantify how classifier choice impacts faithfulness measurements. The key innovation is not in the classifiers themselves but in their systematic comparison methodology.

1. **Hint-type-specific disagreement analysis**: They identified that social-pressure hints (sycophancy, consistency) produce dramatically larger measurement gaps (up to 43.4 percentage points) than rule-based hints (grader, unethical), with Cohen's κ ranging from "slight" (0.06) to "moderate" (0.42). This was achieved through meticulous breakdown of results by hint type, revealing that the measurement problem isn't uniform but concentrated on specific categories of reasoning.

2. **Operationalisation severity spectrum**: They formalised how different classifiers implement related faithfulness constructs at different stringency levels. The regex detector is a "mention" classifier (did the model reference the hint?), while the Sonnet judge is a "dependence" classifier (did the hint actually cause the model's reasoning?). This distinction explains why the pipeline classifies 883 sycophancy cases as faithful while Sonnet considers them unfaithful (only 2 cases go the opposite direction).

3. **Ranking reversal quantification**: They demonstrated how classifier choice can reverse model rankings, with Spearman correlation of ρ = 0.67 (p = 0.017) across 12 models. This was measured using statistical tests (Fisher z-transformation) on a small but significant sample, showing that even when models are close in performance, measurement methodology can flip their relative standing.

## Experimental Results
The study evaluated 12 open-weight models across 9 families (7B to 1T parameters) using 498 questions with five hint types (sycophancy, consistency, metadata, grader, unethical), resulting in 10,276 influenced reasoning traces where models changed their answers in response to hints.

- Overall faithfulness rates: Regex (74.4%), Pipeline (82.6%), Sonnet (69.7%)
- Per-model gaps: Up to 30.6 percentage points (Qwen3.5-27B)
- Hint-type gaps: Sycophancy (43.4 pp), Consistency (33.1 pp), Metadata (-9.5 pp), Unethical (9.0 pp), Grader (2.9 pp)
- Cohen's κ: Sycophancy (0.06, slight agreement), Grader (0.42, moderate agreement)
- McNemar's test: All pairwise comparisons significant (p < 0.001)
- Model ranking reversals: Qwen3.5-27B drops from rank 1 to 7; OLMo-3.1-32B rises from 9 to 3

The paper explicitly states that "the root cause is that different classifiers operationalize related faithfulness constructs at different levels of stringency (lexical mention versus epistemic dependence), and these constructs yield divergent measurements on the same behaviour."

## Related Work
This paper builds on established measurement science principles (Jacovi & Goldberg, Parcalabescu & Frank) while addressing a gap in LLM evaluation methodology. It connects to Chen et al.'s work on hint acknowledgment (39% for DeepSeek-R1), but challenges the practice of reporting single numbers as objective metrics. The study also relates to Gu et al. and Ye et al.'s work on LLM-as-judge biases, showing that these biases systematically impact faithfulness measurement rather than being incidental noise.

## Limitations
The authors acknowledge this study focuses on a specific dataset (12 models, 498 questions, 5 hint types) and doesn't address all possible hint types or model architectures. The comparison between classifiers was limited to these three specific approaches, and the study focused on faithfulness rather than other evaluation metrics. The paper doesn't provide guidance on which classifier is "correct" but emphasizes that the measurement instrument itself is a factor to consider.

## Appendix: Worked Example
Consider a model generating a chain-of-thought for a question about chemical equilibrium, with a sycophancy hint suggesting answer B. The model produces: "The professor suggested answer B. After careful analysis of the chemical properties, I also arrive at B."

- Regex-only classifier: Matches "professor suggested" → marks as faithful (lexical mention)
- Regex-plus-LLM pipeline: Regex matches "professor suggested" → marks as faithful
- Claude Sonnet 4 judge: Interprets the subsequent independent analysis as evidence the model reached the conclusion through its own reasoning, with the professor mention being incidental → marks as unfaithful (epistemic dependence)

This single case illustrates the core disagreement: the regex classifier would count this as faithful (74.4% overall rate), while the Sonnet judge would not (69.7% overall rate). Over 10,276 cases, this pattern contributes to the 43.4 percentage point gap for sycophancy hints. For 883 sycophancy cases, the pipeline marks them as faithful while Sonnet marks them as unfaithful, precisely the reason Qwen3.5-27B drops from 1st to 7th place when switching from pipeline to Sonnet evaluation.

## References

- Richard J. Young, "Measuring Faithfulness Depends on How You Measure: Classifier Sensitivity in LLM Chain-of-Thought Evaluation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20172

Tags: #machine-learning #evaluation-methodology #chain-of-thought #llm-evaluation #classifier-sensitivity
