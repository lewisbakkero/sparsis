---
title: "TempPerturb-Eval: On the Joint Effects of Internal Temperature and External Perturbations in RAG Robustness"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2512.01183"
---

## Executive Summary
This paper investigates how temperature settings in language models interact with external perturbations (simulating noisy retrieval) in RAG systems. The authors demonstrate high-temperature settings consistently amplify vulnerability to perturbations, with different model families exhibiting distinct temperature sensitivity profiles. For practitioners, this means temperature tuning requires model-specific constraints to maintain robustness under real-world retrieval noise.

## Why This Matters for Practitioners
If you're deploying RAG systems in production where retrieval quality can fluctuate (e.g., due to indexing delays or data freshness), this paper provides concrete temperature limits to prevent performance cliffs. For GPT-based systems, cap temperature at T ≤1.4 to avoid sharp degradation when retrieval is noisy; for Llama-based systems, maintain T ≤0.6. In your deployment pipelines, add temperature validation checks that enforce these ceilings based on the LLM type. When evaluating new models, prioritize BERTScore over EM/F1 due to its sensitivity to nuanced perturbations, this metric will better reflect real-world robustness.

## Problem Statement
Today's RAG evaluations treat retrieval quality and generation parameters like temperature as isolated variables, much like checking a car's brakes while ignoring its steering wheel. This approach misses how these elements interact in practice: noisy retrieval (simulating real-world retrieval errors) combined with high temperature can cause catastrophic output degradation, while the same perturbation at low temperature might remain undetectable.

## Proposed Approach
The authors developed a comprehensive framework that systematically varies both perturbation types (simulating retrieval errors) and temperature settings across LLM runs. They tested three perturbation types (sentence replacement, sentence removal, NER replacement) on HotpotQA with multiple LLMs at different temperature settings, measuring semantic similarity through BERTScore. The framework establishes a diagnostic benchmark for RAG robustness that accounts for this critical interaction.

```python
def temperature_perturbation_analysis(model, temperature_range, perturbation_types):
    baseline_performance = {}
    for temperature in temperature_range:
        for perturbation in perturbation_types:
            performance = []
            for _ in range(3):  # Account for LLM stochasticity
                perturbed_context = apply_perturbation(original_context, perturbation)
                response = generate_response(model, perturbed_context, temperature)
                performance.append(compute_bertscore(response, ground_truth))
            baseline_performance[(temperature, perturbation)] = {
                'mean': np.mean(performance),
                'std': np.std(performance)
            }
    return baseline_performance
```

## Key Technical Contributions
The authors make three concrete contributions to RAG robustness evaluation:

1. **A diagnostic benchmark with 440 experimental conditions** - They established a systematic framework testing all combinations of 5 models (GPT-3.5, GPT-4o, Llama-3.1-8B, Llama-3.2-1B, deepseek-reasoner), 11 temperature values (0.0 to 2.0), 3 perturbation types, and 2 question types (bridge/comparison) across 600 HotpotQA samples. This provides the first comprehensive assessment of temperature-perturbation interactions.

2. **Model-specific temperature sensitivity profiles** - The authors quantified how different model architectures respond to temperature-perturbation combinations. For example, they identified that Llama-3.2-1B exhibits minimal temperature sensitivity for comparison questions, while GPT-4o shows a sharp performance cliff at T ≥1.4 for all perturbation types, with degradation accelerating beyond T = 1.8.

3. **Practical temperature tuning guidelines** - Based on their analysis, they provide model-specific temperature ceilings: deepseek-reasoner (T ≤2.0), GPT models (T ≤1.4), and Llama models (T ≤0.6). These aren't arbitrary but derived from empirical performance degradation curves.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The authors evaluated on HotpotQA's training set (600 samples: 100 per fact count category × 2 question types), using BERTScore as primary metric (with ROUGE-1/2/L as secondary). Key findings:

- At T = 2.0, GPT-4o's BERTScore dropped 32% for Sentence Replacement perturbations compared to baseline (T = 0.0), while Llama-3.1-8B dropped 18%
- Deepseek-reasoner maintained consistent performance across all temperatures (±3% BERTScore variance), outperforming GPT-4o at high temperatures
- For GPT-4o: Sentence Replacement caused 41% performance loss at T = 2.0 (vs. 21% at T = 0.6), showing non-linear temperature sensitivity
- Llama-3.1-8B exhibited 28% more robustness at T = 0.6 than at T = 1.4 under Sentence Removal (p < 0.05)
- The authors explicitly state they did not measure statistical significance for all comparisons, though they note "statistically indistinguishable" cases where p > 0.05

## Related Work
The authors build on existing RAG evaluation frameworks like RAG-Ex (Sudhi et al., 2024) but highlight their critical limitation: these only evaluate perturbations without considering temperature interactions. They position their work as the first comprehensive study of this interaction, extending information retrieval perturbation taxonomies (Raval & Verma, 2020; Wu et al., 2023) into the RAG context with temperature as a co-variant.

## Limitations
The authors acknowledge they tested only three perturbation types, leaving out other realistic retrieval errors like document shuffling. The work focuses on the HotpotQA dataset (multi-hop QA) but doesn't test open-ended generation tasks or non-English languages. The evaluation uses GPT-4o for reference answer formatting, which may introduce bias, though the authors note they manually verified a subset to confirm no hallucinations.

## Appendix: Worked Example
Consider a bridge question requiring two supporting documents: "The Eiffel Tower is in Paris" (document 1) and "Paris is the capital of France" (document 2). A Sentence Replacement perturbation would replace document 2 with "The Eiffel Tower is a famous landmark" (irrelevant to capital city question). At T = 0.6 (low temperature), the model might generate: "Paris is the capital of France" (BERTScore 0.85 to ground truth). At T = 2.0 (high temperature), the same perturbation causes: "The Eiffel Tower is a famous landmark in Paris" (BERTScore 0.62), reflecting how high temperature amplifies perturbation effects. For Llama-3.1-8B, this gap widens from 0.13 at T = 0.6 to 0.32 at T = 2.0, demonstrating why temperature cap T ≤0.6 is critical for this model.

## References

- Yongxin Zhou, Philippe Mulhem, Didier Schwab, "TempPerturb-Eval: On the Joint Effects of Internal Temperature and External Perturbations in RAG Robustness", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2512.01183

Tags: #information-retrieval #language-models #temperature-tuning #perturbation-analysis #rag-systems
