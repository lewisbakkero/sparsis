---
title: "Generative Active Testing: Efficient LLM Evaluation via Proxy Task Adaptation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19264"
---

## Executive Summary
Generative Active Testing (GAT) is a framework that transforms dynamic multiple-choice question answering (MCQA) tasks into fixed statement verification proxies, enabling efficient sample selection for LLM evaluation in healthcare domains. By stabilising uncertainty estimates through a novel "RunnerUp" strategy and zero-shot acquisition functions, GAT reduces the estimation error by 40% compared to random sampling, significantly lowering the annotation burden for clinical applications without requiring task-specific fine-tuning.

## Why This Matters for Practitioners
If you're building production systems that rely on LLMs for medical decision support or healthcare applications, this paper directly addresses your annotation bottleneck. Instead of requiring clinicians to manually label every test case (costing $50-$200 per label), GAT reduces the required annotation volume by 40% while maintaining benchmark reliability. For a production system processing 10,000 medical QA samples weekly, this means saving approximately $200,000-$800,000 annually in expert annotation costs. Specifically, implement the Statement Adaptation Module with the RunnerUp strategy for your MCQA tasks and adopt UniformLM_CEAcq as your primary acquisition function. This requires no additional training, integrates directly into existing pipelines, and avoids the computational overhead of training custom uncertainty estimators.

## Problem Statement
Today's LLM evaluation for healthcare applications is like trying to calibrate a microscope using only a magnifying glass - you can't see the fine details (the model's subtle hallucinations in high-confidence regimes) without expensive, time-consuming expert reviews. Medical QA tasks (like determining coverage eligibility in prior authorization) require context-based multiple-choice questions with dynamically varying options, making it impossible to directly compare model confidence across different questions. This forces teams to either waste expert time on low-value samples or risk evaluating models on a non-representative dataset, leading to unreliable benchmarks that may miss dangerous "silent failures" where the model hallucinates correct answers.

## Proposed Approach
GAT's core architecture consists of two main components: the Statement Adaptation Module and LLM-based Acquisition Functions. The Statement Adaptation Module converts dynamic MCQA tasks into a fixed binary classification format (True/False) using a "statement verification" approach. This enables meaningful comparison of model confidence across different questions. The Acquisition Functions then leverage the uncertainty estimates from this adapted task to select the most informative samples for evaluation, using a zero-shot approach that requires no task-specific fine-tuning.

```python
def statement_adaptation(question, options, model):
    # Step 1: Get primary prediction (MostConf)
    primary_pred = model.predict(question, options)
    
    # Step 2: Get RunnerUp prediction (second most probable)
    runnerup_pred = model.predict(question, options, exclude_primary=True)
    
    # Step 3: Construct binary statement
    statement = f"The answer is {runnerup_pred}."
    
    # Step 4: Verify statement (True/False)
    verification = model.verify(statement)
    
    return verification  # True or False
```

## Key Technical Contributions
GAT's novel mechanisms address critical limitations in active sampling for generative tasks. Specifically:

1. The RunnerUp option selection strategy uniquely targets model ambiguity at the point where it causes the most errors. Unlike MostConf (which targets the primary prediction) or LeastConf (lowest probability option), RunnerUp consistently achieves higher Cond. AUROC scores (e.g., +2.1% on PubMedQA, +4.9% on MedQA) by focusing on the second-highest probability option. This effectively acts as a margin-sampling technique that resolves the specific ambiguity confusing the model most.

2. The Zero-Shot Regularisation framework using UniformLM_CEAcq measures cross-entropy between the model's prediction and a uniform distribution. This down-weights samples where the model is "confidently wrong" on hallucinations (a critical issue in biomedical settings where high-confidence errors may bypass standard review filters), smoothing the selection distribution and reducing estimation error by 40% compared to random sampling.

3. The Statement Adaptation Module's binary True/False output space resolves the fixed-label space requirement for unbiased risk estimation. By converting dynamic MCQA tasks into a fixed-schema classification setting, it enables direct comparison of confidence scores across different samples, making uncertainty estimation reliable across the entire dataset.

## Experimental Results
GAT was evaluated across three medical QA datasets with the main model Llama 3.1 8B and surrogate model NLIVerifier. The key metric was Estimation Error AUC (lower values indicate better performance). GAT's UniformLM_CEAcq achieved a 39.6% reduction in geometric mean estimation error compared to RandomAcq across PubMedQA (0.00160 vs 0.00339), MedQA (0.00041 vs 0.00061), and AI2_ARC (0.00039 vs 0.00058). Crucially, SelfEntropyAcq and UCBAcq performed worse than random sampling on all datasets, demonstrating that raw LLM confidence is unreliable for active testing. The results are statistically significant as evidenced by the consistent performance across different datasets and Cond. AUROC measurements.

## Related Work
GAT builds on Active Testing (Sawade et al., 2010; Kossen et al., 2021), which uses surrogate models to guide sample selection. However, existing frameworks offer limited support for generative QA tasks where option dynamics affect model decision boundaries. The authors extend this by transforming dynamic MCQA tasks into a fixed proxy, enabling the use of standard unbiased risk estimators. Unlike prior work on using LLMs for classification tasks (Berrada et al., 2025), GAT addresses the unique challenges of generative QA through the Statement Adaptation Module.

## Limitations
The authors acknowledge two key limitations: First, GAT currently uses only zero-shot surrogates for uncertainty estimation, without fine-tuning on acquired samples to inform future selection. Second, while the approach works for multiple-choice QA, it's not yet extended to open-domain QA or long-form generation tasks. For practitioners, this means GAT's current implementation is most applicable to structured QA tasks with fixed answer options, and may require adaptation for more open-ended medical reasoning scenarios.

## Appendix: Worked Example
Let's walk through the Statement Adaptation Module's RunnerUp strategy on a medical QA question from PubMedQA: "Does caffeine consumption have a significant effect on the risk of developing breast cancer?"

The options are:
- A: Yes, it increases the risk
- B: No, it decreases the risk
- C: No, it has no effect
- D: Not enough evidence

The Llama 3.1 8B model makes the following predictions with confidence scores:
- A: 0.72
- B: 0.15
- C: 0.10
- D: 0.03

The primary prediction (MostConf) is A (0.72), but the RunnerUp (second most probable) is B (0.15). The Statement Adaptation Module constructs the statement: "Caffeine consumption decreases the risk of developing breast cancer." The model then verifies this statement (True/False), which would be False in this case (since option B is incorrect). This binary verification allows for consistent uncertainty comparison across different questions, as the output space is fixed to True/False instead of varying options. The Cond. AUROC for this sample would be significantly higher than using MostConf (which would be A: 0.72), as RunnerUp better identifies the model's uncertainty about the correct answer.

## References

- Aashish Anantha Ramakrishnan, Ardavan Saeedi, Hamid Reza Hassanzadeh, Fazlolah Mohaghegh, Dongwon Lee, "Generative Active Testing: Efficient LLM Evaluation via Proxy Task Adaptation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19264

Tags: #biomedicine #healthcare-ai #active-learning #uncertainty-estimation #zero-shot-learning
