---
title: "Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.12180"
---

## Executive Summary
MADQA establishes the first rigorous benchmark for evaluating multimodal agents' strategic reasoning over document collections, revealing that top agents achieve comparable accuracy to humans but rely on brute-force search patterns rather than strategic navigation, failing to close the nearly 20% performance gap to oracle-level reasoning. This benchmark exposes the limitations of current agentic systems in document-intensive workflows.

## Why This Matters for Practitioners
If you're building enterprise document search systems (e.g., legal, financial, or HR document processing), this paper shows that simply improving retrieval accuracy isn't sufficient, agents must learn strategic navigation patterns to match human efficiency. The nearly 20% accuracy gap between agents and oracle performance indicates that current systems waste computational resources through unproductive search loops rather than strategic planning. You should prioritise developing systems that measure and reward strategic navigation patterns (not just retrieval accuracy) by implementing MADQA's evaluation protocol to benchmark your agents' effort calibration and evidence grounding. For production systems, this means designing evaluation metrics that track both accuracy-effort trade-offs and evidence attribution quality to avoid costly, inefficient search patterns.

## Problem Statement
Today's document search systems resemble a librarian randomly flipping through stacks of books to find a single fact, retrieving pages through brute-force search rather than strategic navigation. Current benchmarks fail to capture how humans mentally navigate document collections by decomposing questions, planning retrieval trajectories, and synthesising evidence across multiple pages or documents. This leads to systems that can match accuracy but waste computational resources on unproductive search loops, creating unnecessary latency and cost in production environments.

## Proposed Approach
MADQA defines a formal task for Agentic Document Collection Visual Question Answering with six core properties distinguishing it from standard document QA. The system iteratively retrieves pages, reasons over visual and textual content, and aggregates evidence from multiple pages to produce a grounded answer with attribution (see Figure 1 in the paper). The evaluation protocol measures the accuracy-effort trade-off through a novel Kuiper statistic that quantifies how well agents calibrate their search effort to problem complexity.

```python
def evaluate_agent(agent, questions, corpus):
    # For each question, record agent's search trajectory
    trajectories = []
    for q in questions:
        # Agent performs iterative search
        evidence_set = []
        steps = 0
        while not answer_is_confirmed:
            page = agent.retrieve(q, evidence_set)
            evidence_set.append(page)
            steps += 1
            
        # Record accuracy and evidence
        trajectories.append({
            "question": q,
            "evidence": evidence_set,
            "accuracy": evaluate_answer(agent.answer, q, corpus),
            "steps": steps
        })
    
    # Compute Kuiper statistic for effort calibration
    kuiper = compute_kuiper(trajectories)
    return {
        "accuracy": mean_accuracy(trajectories),
        "page_f1": compute_page_f1(trajectories),
        "kuiper": kuiper
    }
```

## Key Technical Contributions
The paper introduces three critical contributions that advance our understanding of document-retrieval agents:

1. **Agentic Document Collection Visual Question Answering Formalisation**: The authors define six properties that distinguish their task from standard document QA, particularly the *agentic property* that necessitates planning (decomposing queries), navigation (iterating on findings), and aggregation (synthesising partial answers). This formalisation explicitly requires evidence sets that cannot be retrieved by a single query, forcing agents to demonstrate strategic planning rather than brute-force search.

2. **Construct Validity Framework for Benchmarking**: They operationalise Classical Test Theory to create a benchmark with strong rank correlation to the full dataset while preserving headroom (20% of test set items remain unsolvable by current models). This ensures the benchmark remains relevant as models improve and provides a principled way to select development and test sets based on difficulty and discrimination.

3. **Effort Calibration Metric via Kuiper Statistic**: The authors introduce a novel metric measuring how agents balance effort (steps) against accuracy. The Kuiper statistic quantifies the range of deviations between accuracy and the global mean accuracy across different effort levels. A low Kuiper score indicates stable, "effort-invariant" performance, while a high score reveals poor calibration where agents waste effort on difficult queries they ultimately fail to solve.

## Experimental Results
The MADQA benchmark comprises 2,250 human-authored questions over 800 heterogeneous PDFs (12.2M tokens total). The evaluation reveals significant gaps: the best agents (Gemini 3 Pro BM25 Agent) achieve 82.2% accuracy compared to human oracle performance at 99.4%, creating an 18% accuracy gap. Crucially, agents fail to close this gap despite matching human accuracy in raw scores, they succeed on different questions and rely on brute-force search to compensate for weak strategic planning.

Table 3 shows that while agents outperform static RAG systems (e.g., Gemini 3 Pro BM25 Agent: 82.2% vs. Gemini 3 Pro File Search: 78.6%), they persist in unproductive loops. For multi-hop reasoning (X-Doc), the best agent (Claude Sonnet 4.5) achieves 82.0% accuracy compared to human oracle performance of 98.0%, revealing a 16% gap. The Kuiper statistic (lower is better) shows agents have higher effort calibration than humans (Claude Sonnet: 35.1 vs. humans: 14.6), indicating agents waste more effort on complex queries.

## Related Work
MADQA distinguishes itself from prior benchmarks by addressing three limitations: (1) format, most benchmarks use HTML or plain text, ignoring visual comprehension required for real documents; (2) scope, domain-specific benchmarks like FinRAGBench-V restrict evaluation to narrow verticals; (3) data provenance, many benchmarks use MLLM-generated questions or recycled documents. Unlike ViDoRE v3 (Loison et al., 2026), which mitigates MLLM bias through human verification, MADQA uses fully human-authored questions over fresh documents, eliminating questions shaped by generating model bias.

## Limitations
The authors acknowledge the benchmark focuses on English documents from specific domains (financial, legal, government), which limits generalisability to non-English or culturally different document types. The study evaluates only the current state of multimodal agentic systems, leaving open questions about how future agents might close the accuracy gap. The paper doesn't address how to optimise for strategic navigation versus brute-force search in agent design, though the benchmark provides the tools for such research.

## Appendix: Worked Example
Let's walk through the strategic navigation process for a sample X-Doc question: "What was the total excess permit revenue in Minnesota for the 2014-2019 period?" (Figure 3 in paper).

1. **Initial question decomposition**: The agent decomposes the query into three sub-questions: (a) "What was the excess permit revenue in Minnesota in 2014-2018?", (b) "What was the excess permit revenue in Minnesota in 2019?", and (c) "Sum these values."

2. **First retrieval**: The agent performs a search for "Minnesota excess permit revenue 2014-2018" and retrieves page 42 of the "2018 Annual Report" (which covers 2014-2018), finding $2.3 million.

3. **Second retrieval**: The agent searches "Minnesota excess permit revenue 2019" and retrieves page 17 of the "2019 Annual Report" (which covers 2019), finding $0.8 million.

4. **Evidence synthesis**: The agent aggregates the findings from both documents and calculates the total as $3.1 million.

5. **Effort calibration**: This query required two search steps (effort = 2), with a high accuracy (99% on the correct answer). The Kuiper statistic would measure this as a point in the "efficient region" (upward segment in the cumulative difference curve) where the agent's low effort matched high accuracy.

In contrast, a brute-force agent might perform 50+ search iterations for the same question, wasting computational resources on irrelevant documents (e.g., searching for "Minnesota permit revenue" without decomposing the temporal scope), resulting in a high Kuiper score (poor calibration) despite eventually arriving at the correct answer.

## References

- Łukasz Borchmann, Jordy Van Landeghem, Michał Turski, Shreyansh Padarha, Ryan Othniel Kearns, Adam Mahdi, Niels Rogge, Clémentine Fourrier, Siwei Han, Huaxiu Yao, Artemis Llabrés, Yiming Xu, Dimosthenis Karatzas, Hao Zhang, Anupam Datta, "Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.12180

Tags: #information-retrieval #multi-agent #document-understanding #strategic-navigation #efficiency-calibration
