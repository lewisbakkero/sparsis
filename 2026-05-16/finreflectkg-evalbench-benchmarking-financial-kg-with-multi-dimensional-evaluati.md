---
title: "FinReflectKG -- EvalBench: Benchmarking Financial KG with Multi-Dimensional Evaluation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2510.05710"
---

## Executive Summary
FinReflectKG-EvalBench introduces the first standardized benchmark and bias-aware evaluation framework for financial knowledge graph (KG) extraction from SEC 10-K filings. This addresses the critical gap in reliable, reproducible evaluation protocols for financial KGs, where unvalidated extraction methods can propagate subtle errors that lead to incorrect risk signals, misleading investment insights, or regulatory exposure. Practitioners should care because existing evaluation approaches fail to capture the nuanced quality dimensions required for high-stakes financial applications.

## Why This Matters for Practitioners
If you're currently evaluating financial KG extraction pipelines with simple metrics like exact triple matching, you're missing critical dimensions that impact real-world utility. This paper demonstrates that reflection-based extraction achieves 74.27% comprehensiveness versus 58.75% for single-pass, but at the cost of slightly lower faithfulness (84.80% vs 85.11%). For your production system, you should:
1. Implement multi-dimensional evaluation (faithfulness, precision, relevance, comprehensiveness) rather than relying on single metrics
2. Choose extraction strategies based on application requirements: reflection for coverage-sensitive tasks like risk assessment (74.27% comprehensiveness), single-pass for factually critical applications like regulatory compliance (85.11% faithfulness)
3. Incorporate explicit bias controls into your LLM-as-Judge evaluation system (position independence, leniency control, world-knowledge prohibition)
4. Track inter-judge agreement (Krippendorff's α) as a quality metric - the paper shows α=0.88 for faithfulness indicates reliable judgment

## Problem Statement
Building a financial knowledge graph from unstructured SEC filings is like constructing a bridge with missing blueprint specifications. Current evaluation approaches treat the problem as simple matching, ignoring that financial documents contain dense, context-dependent information where a single misinterpreted triple can propagate through risk assessment models, leading to incorrect credit ratings or misallocated capital. Today's evaluation frameworks lack the nuanced dimensions required to capture semantic correctness, contextual relevance, and coverage completeness in high-stakes financial contexts.

## Proposed Approach
FinReflectKG-EvalBench integrates schema-aware extraction with a conservative, bias-aware evaluation protocol. The system consists of three main components: a corpus of SEC 10-K filings, extraction modes (single-pass, multi-pass, reflection), and an LLM-as-Judge evaluation ensemble. The evaluation framework assesses candidate triples across four complementary dimensions: faithfulness, precision, relevance, and comprehensiveness.

The reflection-based extraction mode uses an agentic iterative workflow where extraction and feedback loops refine triples until inconsistencies are resolved or a maximum iteration limit is reached. Evaluation employs a heterogeneous LLM-as-Judge ensemble configured for deterministic decoding, applying explicit bias controls to mitigate prompt sensitivity.

```python
def evaluate_extraction_triples(extraction_mode, source_text, triples):
    # Evaluate using four dimensions: faithfulness, precision, relevance, comprehensiveness
    results = {}
    
    if extraction_mode == "reflection":
        # Iterative refinement until inconsistency resolved or max iterations
        for iteration in range(MAX_ITERATIONS):
            refined_triples = refine_triples(triples, source_text)
            if not has_inconsistencies(refined_triples):
                break
            triples = refined_triples
            
    # Evaluate using LLM-as-Judge ensemble with bias controls
    verdicts = evaluate_with_llm_judge(
        triples, 
        source_text,
        bias_controls={
            "conservative": True,
            "locality": True,
            "position_independent": True,
            "verbosity_independent": True
        }
    )
    
    # Aggregate results across dimensions
    results["faithfulness"] = aggregate_verdicts(verdicts["faithfulness"], "micro")
    results["precision"] = aggregate_verdicts(verdicts["precision"], "micro")
    results["relevance"] = aggregate_verdicts(verdicts["relevance"], "micro")
    results["comprehensiveness"] = aggregate_verdicts(verdicts["comprehensiveness"], "macro")
    
    return results
```

## Key Technical Contributions
The authors introduce several novel mechanisms that go beyond existing evaluation frameworks:

1. **Bias-aware evaluation protocol:** Instead of relying on standard LLM-as-Judge approaches, they explicitly design controls for four key biases: position effects (judges shouldn't favour candidates based on presentation order), leniency (judges shouldn't be overly generous), verbosity (judges shouldn't favour longer outputs), and world-knowledge reliance (judges shouldn't use external knowledge beyond the text). This is implemented through strict protocol enforcement in the LLM judge prompts, with conservative defaults when evidence is ambiguous.

2. **Commit-then-justify paradigm:** The LLM judges first produce a structured verdict (0 or 1) before providing a concise justification (≤15 words), enabling both automatic evaluation and human-readable error analysis. This is critical for financial applications where auditability is required, with the paper noting that this approach "enables actionable correction paths" for iterative self-improvement.

3. **Inter-judge agreement as protocol stability metric:** Rather than treating disagreement among judges as noise, they use Krippendorff's α to measure agreement across dimensions, with α=0.88 for faithfulness indicating a stable and reliable evaluation protocol. This is particularly important in financial contexts where evaluation consistency has regulatory implications.

4. **Multi-dimensional evaluation framework:** Moving beyond single metrics like exact match or F1 scores, they evaluate along four complementary dimensions, recognising that these criteria must be interpreted jointly to capture the full spectrum of trade-offs in financial KG construction. The paper explicitly states that "these criteria must ultimately be interpreted jointly in order to capture the full spectrum of trade-offs."

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The paper evaluated three extraction modes on SEC 10-K filings from S&P 100 companies for fiscal year 2024, using a heterogeneous LLM-as-Judge ensemble (QWEN3-235B-A22B, GPT-OSS-120B, LLAMA-3.3-NEMOTRON-SUPER-49B-V1.5) with deterministic decoding (temperature=0.0).

The reflection-based extraction mode achieved:
- 74.27% comprehensiveness (vs 58.75% for single-pass)
- 64.23% precision (vs 60.61% for single-pass)
- 90.44% relevance (vs 84.35% for single-pass)

Single-pass achieved the highest faithfulness (85.11% vs 84.80% for reflection). Multi-pass showed intermediate performance across all dimensions.

Inter-judge agreement was measured using Krippendorff's α:
- Faithfulness: α=0.88 (strong agreement)
- Precision: α=0.84 (strong agreement)
- Relevance: α=0.77 (good agreement)
- Comprehensiveness: α=0.71 (moderate agreement)

The authors note that comprehensiveness and relevance require judgment about salience and coverage, making them inherently more sensitive to extraction granularity under strict locality constraints.

## Related Work
The authors position their work at the intersection of financial knowledge graph construction and LLM-based evaluation. They acknowledge that while several works have explored KG construction from financial documents (e.g., FR2KG, GoLLIE), none provide a standardized benchmark or evaluation protocol for financial KG extraction. Recent LLM-as-Judge frameworks (e.g., AlpacaEval) focus on instruction-following evaluation, not structured KG extraction. The authors explicitly address the gap in bias-aware evaluation for financial KGs, noting that existing LLM-as-Judge approaches are prone to position bias, leniency bias, and style-based biases that can lead to inaccurate evaluations in high-stakes financial contexts.

## Limitations
The authors acknowledge several limitations: the benchmark is currently scoped to U.S. SEC Form 10-K filings from S&P 100 companies for fiscal year 2024, which limits generalisation to other issuers, time periods, jurisdictions, and document types (e.g., 10-Q, 8-K, earnings call transcripts). The evaluation relies on an LLM-as-Judge ensemble, which may still exhibit residual failure modes despite explicit bias controls. The metrics capture complementary aspects of extraction quality but don't fully characterise downstream utility, as chunk-level comprehensiveness doesn't measure document-level coverage, cross-chunk consistency, or entity resolution over entire filings.

## Appendix: Worked Example
Let's walk through the evaluation of a single text span from a SEC 10-K filing using the FinReflectKG-EvalBench protocol.

**Text Span:** "The company reported revenue of $1.2 billion for the fiscal year 2024, a 15% increase from the previous year's $1.04 billion. The CEO, Jane Smith, was promoted to her current position in 2021."

**Candidate Triple 1 (Single-pass):** ("The company", "reported_revenue", "$1.2 billion")
**Candidate Triple 2 (Reflection):** ("The company", "reported_revenue", "$1.2 billion"), ("The company", "revenue_increase", "15%"), ("CEO", "named", "Jane Smith"), ("Jane Smith", "promoted_to", "CEO position"), ("Jane Smith", "promoted_in", "2021")

**Evaluation Process:**
1. **Faithfulness:** Does the triple directly reflect the text without inference?
   - Triple 1: "The company reported revenue of $1.2 billion" → matches text → faithfulness score: 1
   - Triple 2: All triples match text directly → faithfulness score: 1 for all

2. **Precision:** Are the triples specific?
   - Triple 1: "$1.2 billion" is specific → precision: 1
   - Triple 2: All triples use specific values → precision: 1 for all

3. **Relevance:** Do triples relate to the main theme of the span (financial performance)?
   - Triple 1: Financial performance (revenue) → relevance: 1
   - Triple 2: All triples relate to financial performance or corporate structure → relevance: 1 for all

4. **Comprehensiveness (at chunk level):**
   - Triple 1: Only covers revenue figure → comprehensiveness: partial (2/3)
   - Triple 2: Covers revenue, revenue increase, CEO name, promotion, and promotion year → comprehensiveness: good (3/3)

**Aggregated Results for the Span:**
- Faithfulness: 1.0 (all triples faithful)
- Precision: 1.0 (all triples specific)
- Relevance: 1.0 (all triples relevant)
- Comprehensiveness: 0.87 (4/5 triples represent core facts)

**Note:** The comprehensiveness score is calculated based on the three-level scale (good, partial, poor), with "good" meaning all core facts covered, "partial" meaning most core facts covered, and "poor" meaning few core facts covered. The paper doesn't specify the exact threshold for each level, but it's clear that Triple 2 covers more core facts than Triple 1.

## References

- Fabrizio Dimino, Abhinav Arun, Bhaskarjit Sarmah, Stefano Pasquali, "FinReflectKG -- EvalBench: Benchmarking Financial KG with Multi-Dimensional Evaluation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2510.05710

Tags: #financial-technology #knowledge-graphs #evaluation-frameworks #llm-as-judge #bias-aware-evaluation #multi-dimensional-evaluation
