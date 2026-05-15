---
title: "Vocabulary shapes cross-lingual variation of word-order learnability in language models"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19427"
---

## Executive Summary
This paper demonstrates that vocabulary structure, particularly coverage metrics, strongly predicts language model learnability across languages, challenging the assumption that fixed vs. free word-order typology explains cross-lingual differences. The authors create a continuous spectrum of synthetic word-order variants using word-level shuffling, showing vocabulary design has a greater impact on model performance than word-order flexibility.

## Why This Matters for Practitioners
If you're implementing multilingual language models, prioritise vocabulary design over word-order assumptions. For languages with free word order (like Czech), optimise for higher subword coverage of low-frequency words through larger vocabulary sizes (|V| > 8,000), which reduces surprisal when word order is irregular. For engineering teams, this means: 1) When tokenising morphologically rich languages, increase vocabulary size to 16,000+ to maintain robustness against word-order variations, 2) For languages with complex morphology, focus on coverage metrics rather than assuming fixed word order, and 3) Evaluate model performance using vocabulary coverage metrics during preprocessing to predict cross-lingual learnability.

## Problem Statement
Current language models struggle with languages that allow flexible word order (like Czech) because they're implicitly trained on English-like fixed word order. Imagine trying to navigate a city where street signs appear in random order, but the city's underlying layout (vocabulary structure) reveals the correct path, your navigation fails when signs are out of order, but works when you understand the city's structure. Most models treat word position as primary syntactic indicator rather than recognising vocabulary as the true structural driver.

## Proposed Approach
The authors create a continuous spectrum of synthetic word-order variants using the Mallows permutation model, which provides a single continuous parameter θ controlling preference for the original word order. By shuffling at word level (not subword level), they preserve morphology while systematically varying word-order regularity. Language models trained on these variants allow precise measurement of how vocabulary structure affects learnability.

```python
def create_word_order_variant(sentence, theta):
    """
    Generate synthetic word-order variant using Mallows model.
    
    Args:
        sentence: Input sentence as word list
        theta: Continuous preference parameter for original order (θ > 0) or reversed order (θ < 0)
        
    Returns:
        Permutation of the sentence according to Mallows model
    """
    n = len(sentence)
    # Calculate probability distribution based on Kendall distance
    if theta > 0:
        perm = sample_mallows_permutation(n, theta=theta)
    elif theta < 0:
        perm = sample_mallows_permutation(n, theta=abs(theta), reverse=True)
    else:  # theta == 0
        perm = random.shuffle(sentence)
    return [sentence[i] for i in perm]
```

## Key Technical Contributions
The authors make two key technical contributions that shift our understanding of language model learnability:

1. They introduce the Mallows model as a continuous measure of word-order regularity, providing a single parameter θ that controls the degree of shuffling while preserving morphology. Unlike prior work using discrete subword shuffling that breaks lexical units, this approach maintains global text entropy, allowing direct measurement of how vocabulary structure affects learnability.

2. They demonstrate through multivariate PLS regression that vocabulary coverage metrics explain 79% of variance in model surprisal (R² = 0.79) across languages and perturbation orders, far surpassing categorical word-order typology. The primary vocabulary component explains original/reverse-order surprisal (mean R² = 0.65), while complex morphology (unique word types, word length) explains irregular order robustness.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The authors trained language models on ten European languages (five free-word-order: Czech, Finnish, Estonian, Hungarian, Latvian; five fixed-word-order: English, French, Portuguese, Swedish, Danish) using ByteLevel-BPE tokenization (|V| = 16,000). Key results:

- Vocabulary metrics explained 79% of variance in surprisal (R² = 0.79) across all θ values, with mean explained variance per θ slice at 79% (range 66-86%)
- Coverage explained original/reverse-order surprisal with mean R² = 0.65 (range 0.26-0.76)
- Morphological complexity (unique word types, word length) was necessary for explaining irregular word-order robustness (R² = 0.79 for both components combined)
- Free-word-order languages (Czech, Finnish) used more low-frequency subwords, resulting in more slowly increasing subword coverage (60% coverage for top 100 words vs 75% for English)
- Vocabulary size |V| > 8,000 separated free/fixed word-order languages in baseline surprisal (Sorig)

The paper didn't explicitly report statistical significance measures for these findings, but used leave-one-language-out cross-validation to establish predictive performance.

## Related Work
The paper builds on synthetic language experiments (Kallini et al., 2024; Xu et al., 2025) that perturb word order while preserving vocabulary, addressing two key limitations: 1) Prior work used subword shuffling that broke morphological integrity, and 2) Disparate shuffling methods with discrete parameters limited comparison. By using word-level shuffling with continuous θ, they isolate vocabulary's role in learnability without conflating word-order and morphology effects.

## Limitations
The study focused exclusively on ten European languages from Europarl (parliamentary speeches), which may not represent global language diversity. The authors acknowledge that model surprisal may not perfectly correlate with human language processing difficulty. The small sample size (n=10 languages) limits generalisation to all languages, though the leave-one-language-out cross-validation approach strengthens their claims.

## Appendix: Worked Example
Consider the Czech sentence "Robot maluje kočku." (The robot paints the cat.) with 5 words. Using the Mallows model with θ = 0 (fully irregular word order), the sentence becomes "kočku maluje robot." The model's surprisal for this permutation is measured.

Czech has free word order due to case marking but exhibits different vocabulary structure from English. Czech has 60% coverage for top 100 words (vs English's 75%), meaning Czech's vocabulary is more fine-grained with more low-frequency words. When word order is randomized (θ = 0), the model's surprisal increases by 1.2 for Czech, while English shows a 0.8 increase at the same θ.

This difference occurs because Czech's fine-grained vocabulary (more low-frequency words) causes greater surprisal when word order is disrupted. At vocabulary size |V| = 16,000, Czech's subword coverage increases more slowly (60% for top 100 words) than English (75%), resulting in higher surprisal change (ΔS = 1.2) compared to English (ΔS = 0.8) at irregular word order.

## References

- Jonas Mayer Martins, Jaap Jumelet, Viola Priesemann, Lisa Beinborn, "Vocabulary shapes cross-lingual variation of word-order learnability in language models", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19427

Tags: #natural-language-processing #language-models #vocabulary-structure #cross-lingual #word-order
