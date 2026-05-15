---
title: "FDARxBench: Benchmarking Regulatory and Clinical Reasoning on FDA Generic Drug Assessment"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19539"
---

## Executive Summary
FDARxBench is a 17K+ question-answering benchmark grounded in real FDA drug labels, designed to evaluate LLMs on regulatory-grade reasoning, particularly their ability to cite sources precisely, handle multi-section drug label reasoning, and safely refuse unanswerable questions. For engineers building clinical AI systems, this benchmark reveals critical gaps in current LLMs' grounding and refusal capabilities that could lead to dangerous hallucinations in regulated healthcare environments.

## Why This Matters for Practitioners
If you're deploying LLMs for drug information systems, this paper shows that even state-of-the-art models like GPT-5.2 achieve only 47% accuracy on multi-hop drug label questions requiring cross-section reasoning when given the full label, significantly lower than their 72% accuracy on standard biomedical QA. This isn't just a research curiosity: FDA regulators must verify every answer's citation, and models that hallucinate or fail to refuse unanswerable questions could lead to regulatory non-compliance or patient safety issues. Practitioners should immediately: 1) implement retrieval-augmented generation (RAG) with precise citation tracking, 2) add refusal validation tests to their QA pipelines, and 3) avoid using closed-book models for regulatory applications until this gap is addressed.

## Problem Statement
Current biomedical QA benchmarks are like testing a surgeon's skills with a toy scalpel, they measure basic competence but ignore the high-stakes precision required in real operating rooms. For FDA drug assessment, regulators need models that can trace answers to exact label sections (not just "section 4.2"), handle multi-section reasoning (like connecting "food effect" to "dosage adjustments"), and refuse to answer when information is missing, something current LLMs can't reliably do. The problem is like asking a barista to replicate a coffee recipe from memory without referencing the actual menu: they'll probably get the basics right but miss critical details that could change the outcome.

## Proposed Approach
FDARxBench creates a structured pipeline to generate regulatory-grade QA questions from FDA drug labels, evaluated across four evidence settings. The core innovation is a four-step process: context selection, LLM question generation with expert feedback, LLM-as-judge filtering, and final validation. The evaluation framework tests models in closed-book, full-label, oracle-passage, and retrieval-augmented settings to isolate grounding, reasoning, and citation capabilities.

```python
def generate_fdarxbench_questions(drug_labels):
    questions = []
    for label in drug_labels:
        # Step 1: Context selection
        sections = select_sections(label, k=random(1,2))
        
        # Step 2: LLM question generation with task-specific prompts
        qa_pair = llm_generate(
            template=question_types[sections],
            context=extract_section_content(sections)
        )
        
        # Step 3: Expert feedback loop (via LLM-as-judge)
        validated = llm_as_judge(
            qa_pair,
            criteria=expert_guidelines
        )
        
        # Step 4: Rule-based filtering
        if not is_valid(qa_pair) or not validated:
            continue
        questions.append(qa_pair)
    return questions
```

## Key Technical Contributions
This paper's novel contributions go beyond standard QA benchmarks by embedding regulatory requirements into the evaluation framework:

1. **Regulatory-grade question taxonomy**: The authors define three question types (factual, multi-hop, refusal) that reflect real FDA assessment needs. Factual questions require single-section evidence, multi-hop questions integrate information across two sections (with a validity check: removing either section makes the question unanswerable), and refusal questions programmatically insert out-of-scope biomedical terms to test safe abstention.

2. **Expert-guided dataset curation pipeline**: By partnering with FDA regulatory assessors, they developed a four-step process where an LLM generates initial questions, experts provide feedback on relevance (precision=0.968, F1=0.800), and the LLM-as-judge applies this feedback to filter questions. This creates a scalable pipeline that maintains regulatory relevance while reducing human curation costs.

3. **Multi-evidence evaluation protocol**: Unlike standard benchmarks, FDARxBench evaluates four distinct settings: closed-book (no label context), full-label (entire label as context), oracle passages (gold-standard citations), and retrieval (top-k chunks). This isolates grounding failures from retrieval limitations, revealing that the largest gap isn't model knowledge (as shown by the oracle-passage accuracy) but the ability to cite precisely (Cite F1 peaks at 0.53).

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The benchmark evaluated eight models across four settings, revealing critical limitations in grounding and refusal:

- **Closed-book accuracy**: Average 22% (factual) and 34% (multi-hop), showing models lack drug label knowledge
- **Oracle passages accuracy**: 78-82% (factual) and 65-82% (multi-hop), confirming models could answer if given correct context
- **Full-label accuracy**: Only 53% (factual) and 40% (multi-hop) on average, with best multi-hop at 47% (GPT-5.2)
- **Citation overlap (Cite F1)**: Peaks at 53% (factual) and 46% (multi-hop), indicating models often cite plausible but incorrect passages
- **Refusal behaviour**: F1 ranges from 71% (Qwen-14B) to 80% (Llama-8B), with high-performing models often over-refusing (refusal recall ≈0.99)

BM25 retrieval outperforms dense methods (recall@1=0.56 factual vs. 0.42 cosine similarity), suggesting drug-label evidence selection is driven by lexical overlap rather than semantic similarity.

## Related Work
FDARxBench differentiates from existing biomedical benchmarks by focusing on regulatory requirements, not just medical knowledge. Unlike BIOASQ or PUBMEDQA (which target biomedical literature questions), HEALTHSEARCHQA (which targets consumers), or clinical EHR QA tasks, FDARxBench captures three regulatory-specific aspects: (1) fine-grained provenance tracking, (2) cross-section reasoning requirements, and (3) explicit refusal behaviour. The authors show this is the first benchmark designed specifically to evaluate LLMs for FDA generic drug assessment workflows.

## Limitations
The authors acknowledge five key limitations: (1) chunking isolates sections, potentially narrowing context for clinically meaningful questions; (2) redundant sections can cause multiple valid citations, confusing retrieval evaluation; (3) multi-hop construction remains challenging (many candidate questions are filtered for invalid hop structure); (4) evaluation uses LLM-based graders rather than human adjudication for borderline cases; and (5) refusal questions rely on heuristic templates rather than natural unanswerable queries.

Practitioners should be aware that the benchmark's 17K+ questions may not capture all regulatory edge cases, particularly for complex multi-section reasoning (e.g., questions linking food effect to pharmacokinetics, efficacy, and safety).

## Appendix: Worked Example
Let's walk through how a multi-hop question is constructed for Zorvolex, a fictional drug, using the pipeline:

1. **Context selection**: The pipeline randomly selects two sections from the drug label: "Food Effect" (section 2.3) and "Dosage and Administration" (section 2.4).
2. **LLM question generation**: Given the context, the LLM generates: "If Zorvolex is taken with food, how does this affect dosage adjustment requirements?" 
3. **Expert validation**: FDA assessors confirm this question is decision-relevant (directly affects usage) and verifiable (typically stated in "food effect" sections).
4. **Filtering**: The question is validated as requiring cross-section reasoning (removing either section makes it unanswerable).

During evaluation with GPT-5.2:
- **Full-label setting**: Model answers "Dosage must be reduced by 25% when taken with food" (correct, but cites section 2.4 instead of the actual section 2.3)
- **Citation overlap**: The model's cited passage (section 2.4) doesn't match the gold standard (section 2.3), giving a Cite F1 of 0.45 (not perfect, but plausible)
- **Refusal test**: The model correctly refuses an unanswerable question about "Zorvolex causing cancer in mice" (since pre-clinical data isn't in the label), achieving a refusal F1 of 0.79 for this item

This example shows the core tension: models can answer correctly but cite wrong sections (low Cite F1), and refusal behaviour doesn't necessarily correlate with answer accuracy.

## References

- **Code:** https://github.com/xiongbetty/FDARxBench
- Betty Xiong, Jillian Fisher, Benjamin Newman, Meng Hu, Shivangi Gupta, Yejin Choi, Lanyan Fang, Russ B Altman, "FDARxBench: Benchmarking Regulatory and Clinical Reasoning on FDA Generic Drug Assessment", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19539

Tags: #biomedicine #regulatory-compliance #document-grounded-qa #multi-hop-reasoning #citation-tracking
