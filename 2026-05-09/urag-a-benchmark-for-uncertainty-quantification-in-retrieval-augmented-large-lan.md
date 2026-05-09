---
title: "URAG: A Benchmark for Uncertainty Quantification in Retrieval-Augmented Large Language Models"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19281"
---

## Executive Summary
URAG introduces the first benchmark for evaluating uncertainty in retrieval-augmented generation (RAG) systems across five domains and eight standard RAG methods. By reformulating open-ended tasks into multiple-choice question answering, it enables principled uncertainty quantification through conformal prediction. Practitioners should care because understanding RAG uncertainty can prevent overconfident hallucinations in production systems, especially in high-stakes domains like healthcare and finance.

## Why This Matters for Practitioners
If you're deploying RAG systems in healthcare applications, this paper shows that simply adding retrieval can amplify confident errors: when the LLM already has weak parametric knowledge (e.g., about pregnancy guidelines), noisy retrieval can cause overconfident yet incorrect responses (as shown in Figure 1). Engineers should implement validation against a confidence threshold (e.g., set size >2 for medical queries) and prioritise modular RAG methods like Fusion or Naive RAG over complex pipelines like REPLUG, which show 3.36 average prediction-set size versus 2.42 for simpler methods. For production systems, always evaluate both accuracy and uncertainty metrics, not just accuracy, to avoid deploying systems that are confident but wrong.

## Problem Statement
Today's RAG systems suffer from "confidence blindness", they generate answers with high certainty but may be factually incorrect due to retrieval noise or over-reliance on partial evidence. Imagine a medical chatbot confidently stating "coffee is safe during pregnancy" while ignoring contradictory evidence in its retrieved documents, as in Figure 1. This isn't a theoretical risk; it's a direct consequence of how current RAG evaluations focus solely on correctness without measuring uncertainty.

## Proposed Approach
URAG reformulates open-ended generation tasks into multiple-choice question answering (MCQA) to enable principled uncertainty quantification using conformal prediction. It generates high-quality distractors through an iterative pipeline that ensures distractors are factually incorrect yet plausible, then applies conformal prediction metrics (LAC and APS) to measure prediction-set size and coverage rate. The benchmark spans five diverse domains and evaluates eight standard RAG methods.

```python
def generate_mcqa_dataset(query: str, correct_answer: str, retrieval_corpus: List[str]):
    retrieved_docs = retriever(query, retrieval_corpus)
    wrong_answers = []
    for _ in range(3):
        wrong_answer = llm_prompt(
            f"Generate a plausible but factually incorrect answer to {query} that matches the semantic type of {correct_answer}"
        )
        if not is_valid_distractor(wrong_answer, correct_answer):
            continue
        wrong_answers.append(wrong_answer)
    return {
        "question": query,
        "correct_answer": correct_answer,
        "distractors": wrong_answers,
        "supporting_docs": [generate_supporting_doc(a) for a in wrong_answers]
    }
```

## Key Technical Contributions
URAG's novelty lies in its systematic approach to uncertainty benchmarking for RAG systems. The core innovations include:

1. **MCQA construction pipeline**: The paper develops a novel iterative method to generate high-quality, confusing distractors for multiple-choice questions. This involves prompting LLMs to create factually incorrect but plausible answers, then using natural language inference (NLI) to confirm each distractor's confusability with the correct answer. The process evolves from naive prompts to highly effective ones, with Table 1 showing progression from 0% to 96% of questions having at least one entailed distractor.

2. **Domain-agnostic evaluation pipeline**: By reformulating tasks into MCQA format, URAG enables statistically valid uncertainty quantification through conformal prediction. This overcomes the fundamental challenge of measuring uncertainty in open-ended generation, where output space is vast and confidence metrics correlate poorly with factual correctness.

3. **Dual evaluation setting**: The benchmark evaluates RAG systems in two complementary settings: one where the LLM already knows the answer (parametric knowledge sufficient), and another where it does not (parametric knowledge insufficient). This reveals how retrieval reshapes model confidence, showing that retrieval depth and confidence cues can amplify confident errors.

## Experimental Results
URAG evaluates eight standard RAG methods across five domains (healthcare, programming, science, math, general text) and eight datasets. Key findings include:

- Accuracy-uncertainty relationship: Accuracy gains often coincide with reduced uncertainty (e.g., Fusion achieves 0.52 accuracy with 2.64 prediction-set size in Healthcare), but this relationship breaks under noisy retrieval (e.g., in Healthcare domain where retrieval noise is high).

- Simple modular methods outperform complex pipelines: Fusion, Naive RAG, and HyDE achieve high average accuracy (≈0.63) with low uncertainty (SS ≈2.42), while REPLUG shows high accuracy (≈0.67) but higher uncertainty (SS ≈3.36).

- No single method is universally reliable: For example, Fusion excels in Healthcare (Acc 0.52) and Code (Acc 0.86) but performs moderately in Math (Acc 0.37), while Self-RAG shows consistency across domains.

- Self-aware prompting: When exposed to confidence scores, RAT shows significant accuracy degradation (from 0.56 to 0.40), while other methods show reduced uncertainty on most benchmarks.

Table 2 shows specific metrics: Fusion achieves highest average accuracy (0.63) with lowest uncertainty (SS 2.42), while FiD shows worst calibration (SS 3.61).

## Related Work
URAG extends prior RAG benchmarking efforts like RAGBench, CRAG, and SafeRAG, but focuses specifically on uncertainty quantification, something these benchmarks lack. Unlike prior work that measures accuracy, robustness, or safety, URAG co-reports accuracy and uncertainty, enabling principled analysis of calibration, over/under-reliance on external evidence, and reliability of RAG-enabled systems.

## Limitations
The paper acknowledges limitations: the benchmark uses a single LLM (Llama-3.1-8B-Instruct) for evaluation, which may not generalise to other model sizes. The MCQA construction pipeline is computationally expensive for large datasets, though the paper reports it's manageable for their scale (e.g., 34K documents in ODEX). The authors note that their analysis doesn't cover all RAG architectures (e.g., multimodal RAG), and they recommend future work to expand the benchmark to include these.

## Appendix: Worked Example
Let's walk through the MCQA construction process for a healthcare question about pregnancy and coffee:

1. **Query**: "Can pregnant women drink coffee?"
2. **Correct answer**: "Coffee is safe in moderation during pregnancy."
3. **Retrieved documents**: 
   - Document A: "Coffee is safe in pregnancy"
   - Document B: "Too much caffeine may be harmful during pregnancy"
4. **Generation of wrong answers**:
   - First attempt: "Coffee should be completely avoided during pregnancy" (too obviously wrong)
   - Second attempt: "Coffee is safe during pregnancy but should be limited to 1 cup daily" (plausible but incorrect)
   - Third attempt: "Coffee is safe during pregnancy as long as it's decaffeinated" (plausible and confusing)
5. **Validation**:
   - NLI model confirms Document B entails "Too much caffeine may be harmful during pregnancy" but not the wrong answer "Coffee is safe during pregnancy as long as it's decaffeinated"
   - Final distractors: "Coffee should be completely avoided" (too obvious, discarded), "Coffee is safe in moderation" (correct), "Coffee is safe during pregnancy but should be limited to 1 cup daily" (valid distractor)
6. **Final MCQA**:
   - Question: "Can pregnant women drink coffee?"
   - Options:
     - A: Coffee is safe in pregnancy
     - B: Coffee should be completely avoided during pregnancy
     - C: Coffee is safe during pregnancy but should be limited to 1 cup daily
     - D: Too much caffeine may be harmful during pregnancy

The conformal prediction algorithm would then assign confidence scores to each option, with the correct answer (A) receiving the highest probability. In this case, the LLM might assign high probability to A, leading to a small prediction set (low uncertainty) even though Document B provides contradictory evidence.

## References

- Vinh Nguyen, Cuong Dang, Jiahao Zhang, Hoa Tran, Minh Tran, Trinh Chau, Thai Le, Lu Cheng, Suhang Wang, "URAG: A Benchmark for Uncertainty Quantification in Retrieval-Augmented Large Language Models", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19281

Tags: #healthcare-computing #retrieval-augmented-generation #uncertainty-quantification #conformal-prediction #multi-domain-evaluation
