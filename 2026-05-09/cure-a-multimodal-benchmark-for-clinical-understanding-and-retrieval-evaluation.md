---
title: "CURE: A Multimodal Benchmark for Clinical Understanding and Retrieval Evaluation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19274"
---

## Executive Summary

The CURE benchmark creates a clinical evaluation framework that disentangles multimodal reasoning from evidence retrieval in medical AI systems. It benchmarks 500 physician-annotated clinical cases against four evidence paradigms, revealing that current models achieve 73.4% accuracy when given physician-cited references but plummet to 25.4% with independent retrieval, highlighting retrieval as the primary deployment bottleneck in clinical AI.

## Why This Matters for Practitioners

If you're building clinical decision support systems, CURE reveals a critical implementation insight: your model's reasoning capability is often not the limiting factor, retrieval quality is. In production systems, this means prioritising evidence retrieval quality over model architecture upgrades. For example, when using RAG pipelines, invest in curated biomedical literature rather than larger model parameters. This paper shows that improving retrieval precision (e.g., from 23.0% coverage in standard RAG to 38.2% in agent-based retrieval) directly increases clinically relevant output. Your next engineering sprint should focus on evidence grounding rather than model scaling.

## Problem Statement

Current medical AI evaluation is like testing a chef with a pre-chopped, pre-seasoned meal, measuring only the final dish, not the chef's ability to source ingredients or follow a recipe. Existing benchmarks evaluate MLLMs in end-to-end answering scenarios, conflating three critical components: (1) multimodal understanding of images and text, (2) precise evidence retrieval from clinical literature, and (3) evidence-grounded diagnosis. This means when a system fails, you don't know whether it's because the model can't interpret an MRI, can't find the right reference article, or can't apply the retrieved evidence correctly.

## Proposed Approach

CURE addresses this by constructing a benchmark that isolates evidence retrieval from clinical reasoning through four controlled evidence paradigms. Each clinical case includes medical images, clinical history, and physician-cited literature references. Models are evaluated under four settings:
- **Base**: Only clinical history and images
- **Physician Reference**: Pre-provided reference evidence
- **RAG**: Standard retrieval-augmented generation
- **Agent Retrieval**: Multi-step agentic search

The benchmark explicitly decouples these components to identify where failures occur. Figure 1 in the paper shows this pipeline: cases are paired with PubMed reference evidence, and models are evaluated under different retrieval paradigms to disentangle multimodal understanding from evidence-grounded diagnosis performance.

```python
def evaluate_cure_case(case):
    # Case includes clinical history, medical images, and physician-cited references
    base_performance = model(base_input=case.history + case.images)
    reference_performance = model(
        base_input=case.history + case.images,
        external_context=case.physician_references
    )
    rag_performance = model(
        base_input=case.history + case.images,
        external_context=retrieve_rag(case.history + case.images)
    )
    agent_performance = model(
        base_input=case.history + case.images,
        external_context=agent_retrieve(case.history + case.images)
    )
    return {
        'base': base_performance,
        'reference': reference_performance,
        'rag': rag_performance,
        'agent': agent_performance
    }
```

## Key Technical Contributions

CURE introduces a systematic way to evaluate MLLMs in clinical contexts by explicitly isolating evidence retrieval from reasoning. The key innovations are:

1. **Physician-Cited Reference as Ground Truth**: The benchmark uses PubMed IDs (PMIDs) explicitly cited in clinical reports rather than LLM-generated summaries. This ensures evaluation is grounded in actual clinical practice, avoiding the bias of generated references. For example, each case is mapped to PubMed references directly cited by physicians in the original report, creating a standardized evidence base that reflects real-world diagnostic practices.

2. **Retrieval Coverage as a Key Metric**: CURE introduces coverage as a critical success metric, measuring the percentage of cases where retrieved evidence includes the physician-cited references. The paper shows that RAG achieves only 23.0% coverage compared to 92.1% with physician references, directly linking retrieval quality to diagnostic accuracy. This metric allows engineers to quantify the "evidence gap" in their systems.

3. **Faithfulness Analysis Framework**: The paper develops a quadrants framework to analyse diagnostic behaviour: evidence-based correct (64.5% with physician references), evidence-misled (57.3% with RAG), knowledge-based correct (2.1% across all settings), and hallucination (31.6% with RAG). This analysis reveals that models are often too faithful to retrieved evidence, so when retrieval fails to include diagnostic literature, the result becomes a grounded but incorrect diagnosis.

## Experimental Results

CURE evaluated 16 models across four evidence paradigms on 500 clinical cases. The most significant finding is the stark performance gap between models with physician references versus those with independent retrieval:

- **Closed-ended diagnosis**: Models achieve 73.4% accuracy with physician references (e.g., GPT-5.2) but drop to 25.4% with independent retrieval (Base setting)
- **Open-ended diagnosis**: Hit@1 accuracy ranges from 73.4% (with physician references) to 25.4% (with Base)
- **RAG vs. Agent Retrieval**: Agent Retrieval shows improved coverage (38.2% vs. 23.0% for RAG) and better evidence-based accuracy (14.0% vs. 7.0%)
- **Textual findings**: Adding radiologist-authored findings (not diagnostic conclusions) consistently improves performance by 10-20% across all settings

The paper doesn't report statistical significance testing for these results, which is a limitation in the evaluation methodology.

## Related Work

CURE positions itself against existing medical AI benchmarks like MedXpertQA and MMMU, which primarily assess end-to-end answering without providing a systematic testbed for evidence-grounded retrieval. While RAG-based systems have proliferated in clinical contexts, CURE is the first benchmark to explicitly decouple evidence retrieval from reasoning. It builds on prior work in evidence-based medicine (e.g., guidelines for diagnostic reasoning) but extends it to machine learning evaluation.

## Limitations

The benchmark uses Eurorad cases published in 2025, which postdate current MLLM knowledge cutoffs, effectively eliminating data contamination but limiting applicability to real-time clinical scenarios. The paper doesn't test the benchmark on rare conditions or extremely high-stakes cases (e.g., stroke or sepsis). Additionally, the evaluation doesn't account for the time costs of retrieval, which is critical for real-time clinical decision support systems.

## Appendix: Worked Example

Consider a case with a 46-year-old male patient with HIV and hepatitis C, presenting with hypoplastic fingernails and CT findings showing bilateral patellar hypoplasia. The physician-cited references (PMIDs: 28612165, 20143059, 38022029, 27450397) correspond to literature describing the diagnostic connection between these findings and a specific syndrome.

When evaluating with Physician Reference evidence, the model receives the abstracts of these four references. The model correctly identifies the syndrome (e.g., "osteochondrodysplasia") with 73.4% accuracy, as the evidence directly matches the findings. 

With Base setting (no evidence), the model relies solely on the clinical history and images. It might generate "osteoporosis" (incorrect diagnosis), as the images show bone abnormalities but the model lacks the specific reference.

With RAG, the system retrieves a broader set of PubMed articles. The retrieved evidence might include general bone disorder literature (e.g., "osteoporosis treatment") but lacks the specific clinical association. The model becomes "evidence-misled," generating "osteoporosis" (57.3% of cases), as the retrieved evidence is relevant but not diagnostic.

With Agent Retrieval, the system performs multi-step search. It might query "hypoplastic fingernails and patellar hypoplasia in HIV patients" and retrieve the correct references (38.2% coverage). The model then identifies the syndrome with 14.0% evidence-based correctness, improving over RAG but still falling short of physician references.

## References

- **Code:** https://github.com/yanniangu/CURE.
- Yannian Gu, Zhongzhen Huang, Linjie Mu, Xizhuo Zhang, Shaoting Zhang, Xiaofan Zhang, "CURE: A Multimodal Benchmark for Clinical Understanding and Retrieval Evaluation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19274

Tags: #biomedicine #diagnosis-support #retrieval-augmented-generation #evidence-grounded-ai #multimodal-clinical-ai
