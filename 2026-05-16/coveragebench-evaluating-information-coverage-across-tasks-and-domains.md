---
title: "CoverageBench: Evaluating Information Coverage across Tasks and Domains"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20034"
---

## Executive Summary
CoverageBench provides a unified testbed for evaluating information coverage across multiple retrieval tasks and domains, addressing a critical gap in RAG system evaluation. It transforms existing TREC collections into coverage-focused benchmarks with nugget-level annotations, enabling engineers to measure whether retrieval systems surface diverse, complementary information rather than just relevant documents.

## Why This Matters for Practitioners
If you're building or evaluating RAG systems in production, this paper reveals that traditional metrics like precision and recall are insufficient, they reward retrieving more relevant documents without ensuring coverage of diverse information aspects. Many RAG systems fail to surface comprehensive information because they optimise for relevance rather than coverage, which directly impacts answer quality and bias. For example, a RAG system might retrieve 10 documents all discussing the same historical event, scoring well on traditional metrics but failing to cover diverse perspectives on that event. Engineers should incorporate coverage evaluation into their testing pipeline to ensure their systems surface comprehensive information, especially for applications requiring nuanced answers like legal research or medical decision support. You can immediately start using CoverageBench to test your retrieval models against these new metrics rather than just standard relevance-based benchmarks.

## Problem Statement
Current retrieval systems operate like a single-minded librarian who can only find books about one specific aspect of a topic, ignoring the broader context. Traditional metrics reward systems for retrieving more documents about the same aspect (e.g., "How did the 1929 stock market crash affect Wall Street?"), while failing to reward systems that surface complementary aspects (e.g., "How did the 1929 stock market crash affect Main Street, European markets, and global economic policies?"). This leads to RAG systems that produce narrow, potentially biased answers because they're optimised to find relevant documents, not diverse information units.

## Proposed Approach
CoverageBench constructs a unified suite of coverage evaluation collections by transforming existing ad hoc retrieval collections. It introduces nugget-level annotations representing discrete information units essential for comprehensive answers, enabling evaluation of whether retrieved results collectively cover all necessary aspects of an information need. For each dataset, the core components include topics (queries), nuggets (discrete information units), document collections, and relevance labels.

```python
def create_coverage_benchmark(original_collection):
    # Transform existing collection into coverage-focused benchmark
    topics = extract_topics(original_collection)
    nuggets = derive_nuggets(topics, original_collection)
    document_collection = get_public_documents(original_collection)
    relevance_labels = create_nugget_qrels(document_collection, nuggets)
    
    # Release all components on Hugging Face Datasets
    huggingface_release({
        "topics": topics,
        "nuggets": nuggets,
        "documents": document_collection,
        "qrels": relevance_labels,
        "baselines": generate_baselines(document_collection, nuggets)
    })
    return benchmark
```

## Key Technical Contributions
CoverageBench solves the challenge of creating coverage evaluation benchmarks by reusing existing collections with minimal augmentation. The specific mechanisms enabling this approach are:

1. **Nugget derivation methodology**: For collections without existing nugget annotations (Fair Ranking, CAsT), they derive nuggets from existing annotations using LLM judges. For Fair Ranking, they convert demographic attribute values (e.g., geographic regions, popularity levels) into nuggets. For CAsT, they treat each conversational turn as a subtopic. The LLM judge (Llama-3.3-70B-Instruct) assesses whether documents contain information corresponding to each nugget, achieving 69% precision and 90% recall against original relevance judgments.

2. **Baseline generation for coverage evaluation**: They provide 6 baselines per dataset (2 initial retrieval with BM25 and Qwen3-8B, plus 4 reranked runs with Rank1-7B and Qwen3-Reranker-8B), enabling direct comparison of coverage performance across different retrieval approaches. The baselines allow researchers to measure how much coverage improves with reranking.

3. **Unified testbed across diverse domains**: By incorporating seven datasets spanning different information needs (multilingual report generation, fair ranking, conversational assistance, and long-form report generation), CoverageBench offers a comprehensive evaluation framework rather than a single-task solution. The datasets vary in document size (565k to 113M passages) and nugget complexity (3-62 nuggets per query), creating a robust benchmark for coverage evaluation.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
CoverageBench comprises seven datasets with 334 total topics. The datasets vary significantly: CRUX-MultiNews contains 100 topics with 14.2 nuggets per query on average, while Fair Ranking has 50 topics with 29.7 nuggets per query. The document collections range from 565,015 passages (CRUX-MultiNews) to 113,520,750 segments (RAG). The authors evaluated six retrieval configurations (BM25, Qwen3-8B with and without reranking) across all datasets.

For RAG 2024, the LLM judge (Llama-3.3-70B-Instruct) achieved 69% precision and 90% recall against original relevance judgments when creating nugget-level qrels. For Fair Ranking, they achieved 72% recall on the inferred support set. The authors didn't report statistical significance testing for coverage improvements, though they note that "the same information" in retrieved results would be considered redundant by coverage metrics.

## Related Work
CoverageBench builds on several existing efforts but addresses their limitations. It extends the TREC-6 interactive track's concept of coverage (1996) but provides a standardized, unified testbed rather than isolated examples. It incorporates elements from NTCIR's IMINE tasks (subtopic discovery) and TREC's diversity ranking tasks (e.g., TREC web track, TREC NeuCLIR), but transforms them into a consistent coverage evaluation framework. Unlike the TREC RAG Track, which had limited query sets and narrow domains, CoverageBench provides a broader testbed with seven datasets spanning multiple genres and tasks. It also extends the CRUX framework by adding nugget-level annotations for coverage evaluation.

## Limitations
The authors acknowledge that coverage evaluation requires substantial annotation effort, which limits the number of datasets they could include. They had to rely on LLM judges for nugget derivation in some datasets, introducing potential bias. The paper doesn't explore how coverage evaluation scales to extremely large datasets beyond the current benchmark (over 113M document segments in RAG). They also note that the "sparsity of collections may in part be because algorithms need to be sufficiently advanced to distinguish aspects within documents," suggesting that coverage evaluation might be limited by current retrieval technology. The authors don't address how coverage metrics correlate with end-to-end RAG system performance beyond the baseline tests reported.

## Appendix: Worked Example
Let's walk through how CoverageBench evaluates information coverage for a specific query in the Fair Ranking dataset. The query "Architecture and architectural styles from diverse world regions across different time periods including both famous and lesser-known architects" has 62 nuggets (the highest in CoverageBench) representing geographic regions and popularity levels.

1. **Query processing**: The query is rewritten as a natural language information-seeking query targeting coverage across demographic facets (geographic regions and popularity levels).
2. **Nugget derivation**: Each unique value in the demographic annotations becomes a nugget. For example, geographic regions (South America, South-eastern Asia, Southern Africa) and popularity levels (High, Low, Medium-High) are derived from the dataset.
3. **Document relevance assessment**: For each document in the collection, the LLM judge determines whether the document contains information corresponding to each nugget. If a document contains information about "South America" and "High popularity," it covers two nuggets.
4. **Coverage calculation**: For the top-10 retrieved documents, the system calculates the fraction of nuggets covered (e.g., if 30 out of 62 nuggets are covered, coverage = 30/62 = 48.4%).

In this example, a retrieval model might achieve 48.4% coverage (30 of 62 nuggets covered), while another might achieve 27.3% (17 of 62 nuggets covered). Traditional metrics like nDCG would have scored the first model higher because it retrieved more relevant documents, but coverage evaluation reveals the second model actually provides more diverse information.

## References

- Saron Samuel, Andrew Yates, Dawn Lawrie, Ian Soboroff, Trevor Adriaanse, Benjamin Van Durme, Eugene Yang, "CoverageBench: Evaluating Information Coverage across Tasks and Domains", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20034

Tags: #information-retrieval #retrieval-augmented-generation #coverage-evaluation #nugget-based-evaluation #retrieval-metrics
