---
title: "Span-Level Machine Translation Meta-Evaluation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19921"
---

## Executive Summary
This paper introduces a new metric called "match with partial overlap and partial credit" (MPP) for evaluating machine translation error detection systems. It reveals that commonly used metrics like exact match (EM) and match with partial overlap (MP) produce misleading results due to biases toward longer error spans or task sparsity. Practitioners should care because using flawed metrics can lead to adopting suboptimal error detection systems, ultimately reducing translation quality in production pipelines.

## Why This Matters for Practitioners
If you're building or maintaining production translation systems that use auto-evaluators for quality control, error detection, or data filtering, this paper reveals that your current evaluation metrics might be actively misleading you. For example, if your team uses macro-averaged MP (which many implementations do), you might be incentivising models that under-detect errors to maximise F1 scores on zero-error translations. This could lead to rolling out translation systems with persistent, undetected errors. The paper shows that with MPP and micro-averaging, Qwen3 235b outperforms Claude Sonnet 4.5 (22.37 vs 19.92 F1) on the MQM test set. Therefore, engineers should immediately audit their evaluation metrics for error detection systems and transition to MPP with micro-averaging to ensure they're building better translation quality pipelines.

## Problem Statement
Evaluating error detection in machine translation is like trying to judge a surgeon's work based only on whether they correctly identified the general location of a problem, not the exact surgical site. Current metrics treat longer error spans as more significant (like judging a surgeon by how much of the body they "fixed" rather than which specific issue they addressed), and they're prone to being gamed by models that deliberately under-detect errors to inflate scores on translations with no errors. The field has evolved from simple scalar scores to detailed error localization, but without a robust way to evaluate these detailed error detectors, teams are making critical decisions based on flawed metrics.

## Proposed Approach
The authors propose a new evaluation strategy for span-level machine translation error detection that treats all error spans equally regardless of length and is robust to the sparsity of errors in translation data. This involves two key innovations: first, assigning partial credit based on character overlap between detected and ground truth error spans; second, using micro-averaging across spans rather than across translations. The MPP metric calculates precision and recall for each matched span pair before averaging across all spans, ensuring each error has equal weight.

```python
def mpp_score(hypothesis_spans, ground_truth_spans):
    # Find optimal one-to-one matching maximising F1
    matching = find_optimal_matching(hypothesis_spans, ground_truth_spans)
    
    # Calculate precision and recall for each span pair
    precision = 0
    recall = 0
    for (h_span, g_span) in matching:
        overlap = len(intersect(h_span, g_span))
        precision += overlap / len(h_span)
        recall += overlap / len(g_span)
    
    # Average across all spans
    precision /= len(matching)
    recall /= len(matching)
    
    # Calculate F1 as harmonic mean
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1
```

## Key Technical Contributions
The paper's core innovations go beyond just proposing a new metric, they identify systematic flaws in existing evaluation approaches and provide a robust alternative.

1. **Character-based normalization for error span matching**: Unlike previous methods that treated longer spans as more significant, MPP normalizes partial credit by span length, ensuring a 5-character error gets the same relative weight as a 20-character error. For example, if a model detects a 5-character error with 4/5 character overlap (80% precision) and a 20-character error with 16/20 character overlap (80% precision), MPP treats both as 80% precision rather than weighting the longer error more heavily.

2. **Micro-averaging for error detection sparsity**: The paper demonstrates that macro-averaging (averaging scores across translations) is fundamentally flawed because many translations contain no errors. Micro-averaging (aggregating statistics across all error spans) ensures that models favouring under-detection of errors (to score high on zero-error translations) don't receive inflated scores. Their experiments show that under macro-averaging, models with deliberate error removal ("Remove-1" sentinels) achieved F1 scores up to 50% higher than they should.

3. **Formal identification of metric biases**: The authors provide clear formal definitions of how different metric implementations can be manipulated (e.g., MP can be gamed by returning longer error spans), and they demonstrate these biases with empirical evidence rather than speculation. This transforms evaluation from a subjective process into a verifiable one.

## Experimental Results
The paper evaluates 11 auto-evaluators on MQM test sets using MPP with micro-averaging (Table 2), revealing significant performance differences:

- Qwen3 235b achieves the highest F1 score at 22.37, while AutoLQA41.sec (the worst) scores 9.30
- GemSpanEval-pri (fine-tuned on Gemma-3 27b) ranks second at 19.84 F1
- The difference between top and bottom models is substantial (22.37 vs 9.30 F1)

The authors also demonstrate the flaws in existing metrics:
- MP with macro-averaging shows a 25-30% F1 inflation for "Remove-1" sentinels (models that deliberately remove errors to maximise zero-error translations)
- EM drops to near zero F1 with even minor span extension (e.g., adding 1 character to a detected span)

The paper establishes MPP with micro-averaging as the only metric that reliably reflects true error detection capability across all tested scenarios.

## Related Work
The paper positions itself against the WMT (Workshop on Machine Translation) shared tasks, which have used inconsistent evaluation strategies. The WMT 2019-2020 editions used variants of MPP but with macro-averaging, while WMT 2023-2025 used strategies that favour longer error spans. Previous work like Perrella et al. (2022) used Span Hit Metrics (similar to MP with τ=1), and Kocmi et al. (2024b) measured inter-annotator agreement using metrics similar to RMP. The paper shows these approaches systematically bias results, making cross-study comparisons unreliable.

## Limitations
The authors acknowledge that their evaluation is limited to MQM and ESA annotation protocols, and they don't explore how their metric would perform with other error detection frameworks. The paper also focuses on error detection rather than translation quality evaluation, so the metric doesn't directly assess translation adequacy. Additionally, the experiments are based on publicly available MQM test sets from 2022-2024, so the results might not generalise to newer translation systems or different language pairs.

## Appendix: Worked Example
Let's walk through a concrete example of MPP calculation with actual numbers from the paper. Suppose we have a translation with two ground truth error spans: "quick" (5 characters) and "fox" (3 characters). An auto-evaluator detects two spans: "quick brown" (10 characters) and "fox" (3 characters).

For the "quick" span:
- Overlap length: 5 characters (perfect match for "quick")
- Precision: 5/10 = 0.5
- Recall: 5/5 = 1.0

For the "fox" span:
- Overlap length: 3 characters (perfect match)
- Precision: 3/3 = 1.0
- Recall: 3/3 = 1.0

Now, we average across spans for micro-averaged precision and recall:
- Precision = (0.5 + 1.0) / 2 = 0.75
- Recall = (1.0 + 1.0) / 2 = 1.0
- F1 = 2 × (0.75 × 1.0) / (0.75 + 1.0) = 0.857

This example shows how MPP normalises for span length (the 10-character span has lower precision than the 3-character span, but we don't weight it more heavily) and how micro-averaging ensures each error has equal importance.

## References

- Stefano Perrella, Eric Morales Agostinho, Hugo Zaragoza, "Span-Level Machine Translation Meta-Evaluation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19921

Tags: #natural-language-processing #machine-translation #evaluation-metrics #span-level-analysis #error-detection
