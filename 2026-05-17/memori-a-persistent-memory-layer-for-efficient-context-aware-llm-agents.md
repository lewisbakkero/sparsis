---
title: "Memori: A Persistent Memory Layer for Efficient, Context-Aware LLM Agents"
venue: "Context-Aware LLM Agents""
paper_url: "https://arxiv.org/abs/2603.19935"
---

## Executive Summary
Memori introduces a persistent memory layer for LLM agents that structures unstructured conversation into semantic triples and summaries, achieving 81.95% accuracy on the LoCoMo benchmark while using just 1,294 tokens per query (4.97% of full context). This represents a 67% reduction in tokens compared to Zep while delivering higher accuracy, directly solving the cost-performance tradeoff that plagues production LLM agents.

## Why This Matters for Practitioners
If you're running customer support chatbots or personal assistants that need to remember user preferences across sessions, Memori lets you dramatically reduce API costs without sacrificing accuracy. For instance, instead of injecting 26,031 tokens per query (full context) that costs $0.0208 per turn with GPT-4.1-mini, Memori uses only 1,294 tokens ($0.001035 per turn) while improving accuracy from 79.09% (Zep) to 81.95%. This means you can deploy longer-term agents without blowing your API budget, effectively turning memory from a cost centre into a profit centre. The key engineering decision this paper informs: stop treating memory as raw context storage, and start treating it as a data structuring problem.

## Problem Statement
Current memory systems for LLM agents operate like an unsorted garage: every conversation history gets piled on top of the last, creating a chaotic mess where the LLM must sift through mountains of irrelevant text to find what's needed. This isn't just inefficient, it's actively harmful. As context grows, models suffer from "context rot," where critical information exists but isn't effectively used (e.g., a user's coffee preference mentioned hours ago gets lost in the noise). The cost isn't just financial ($0.0208 per turn for full context), but also functional, agents become less reliable as they struggle to find the needle in the haystack.

## Proposed Approach
Memori operates as a decoupled memory layer positioned between application logic and the LLM, using an "Advanced Augmentation" pipeline to convert raw conversations into structured semantic representations. The system processes conversations offline to create two linked memory assets: semantic triples (subject-predicate-object) for precise facts, and conversation summaries for narrative context. These assets are stored in a vector index for efficient retrieval, with a hybrid search approach combining cosine similarity of embeddings with BM25 keyword matching.

```python
def retrieve_memory(query, conversation_id):
    # Hybrid search: combine semantic and keyword matching
    semantic_results = vector_search(query_embedding, index="triples")
    keyword_results = bm25_search(query, index="summaries")
    
    # Merge results with weighted scoring
    merged_results = merge_results(
        semantic_results, 
        keyword_results,
        weights=(0.7, 0.3)
    )
    
    # Return top k results with context links
    return get_contextual_snippets(merged_results, conversation_id)
```

## Key Technical Contributions
Memori's core innovation lies in its dual-layer memory representation, which fundamentally shifts how we think about LLM memory. Unlike traditional approaches that store raw text or use simple embeddings, Memori's Advanced Augmentation creates:

1. **Semantic triples with contextual anchoring**: Memori decomposes conversations into atomic knowledge units (subject-predicate-object), each linked to its exact conversational context. This eliminates noise (e.g., "I like coffee but I'm not sure about tea" becomes "user prefers coffee") and enables precise retrieval. Unlike prior approaches that store raw text, Memori's triples are explicitly tied to timestamped messages, ensuring context isn't lost during retrieval.

2. **Dynamic summary integration**: While semantic triples capture granular facts, Memori generates conversation summaries that maintain narrative flow. Each triple is linked to its proper summary, allowing the LLM to understand context (e.g., a preference for cappuccino might be due to a morning routine described in the summary). This dual-layer system bridges the gap between fine-grained facts and contextual narrative without inflating context windows.

3. **Hybrid search for optimal retrieval**: Memori uses a hybrid search combining cosine similarity of embeddings (for semantic understanding) with BM25 keyword matching (for exact term matching). This approach ensures the system retrieves both highly relevant facts and important contextual terms that might not be captured by embeddings alone, significantly improving accuracy over pure embedding-based approaches.

## Experimental Results
Memori achieved 81.95% overall accuracy on the LoCoMo benchmark, outperforming competing memory systems while using substantially fewer tokens:
- **Accuracy**: 81.95% (vs. Zep's 79.09%, LangMem's 78.05%, Mem0's 62.47%)
- **Token usage**: 1,294 tokens per query (4.97% of full context)
- **Cost efficiency**: 67% fewer tokens than Zep (3,911 tokens), over 20× savings compared to full-context (26,031 tokens)

The Full-Context ceiling achieved 87.52% accuracy, but is financially unsustainable for production at $0.0208 per turn. Memori narrows this gap by 5.57% while drastically reducing token costs. The paper does not report statistical significance testing for the accuracy improvements, though the results were computed as averages of three rounds.

## Related Work
Memori builds on and improves over existing memory systems like Zep (a persistent memory system for agents), LangMem (a memory framework for language models), and Mem0 (a scalable long-term memory system for agents). While these systems also aim to manage memory for LLM agents, they typically rely on storing raw conversation chunks or simple embeddings, leading to high token costs and contextual noise. Memori fundamentally shifts the paradigm from "store more context" to "structure context better," using semantic triples and summaries to create a high-signal knowledge base.

## Limitations
The authors acknowledge that Memori's current approach has limitations in temporal reasoning compared to some competitors (LangMem achieved 86.92% vs. Memori's 80.37% in Temporal reasoning). They suggest this is because isolated semantic triples miss temporal context, though summaries help rebuild timelines. The paper also notes that open-domain reasoning (63.54% accuracy for Memori vs. 67.71% for LangMem) remains challenging for all retrieval-based systems, as it requires broader synthesis across contexts. For production systems, this means Memori may not be ideal for tasks requiring broad synthesis of unstructured information.

## Appendix: Worked Example
Let's walk through how Memori processes a two-message conversation to create structured memory:

1. **Raw conversation**:
   - Message 1 (10:00 AM): "I love coffee, especially espresso. My favorite is the cappuccino."
   - Message 2 (10:05 AM): "I'm going to the cafe downtown for a coffee meeting at 2 PM."

2. **Advanced Augmentation processing**:
   - Semantic triples extraction:
     * ("user", "prefers", "espresso")
     * ("user", "prefers", "cappuccino")
     * ("user", "meeting location", "cafe downtown")
     * ("user", "meeting time", "2 PM")
   - Conversation summary: "User expressed preference for coffee, specifically cappuccino, and scheduled a cafe meeting for 2 PM."

3. **Storage**:
   - Triples are stored with temporal reference to their originating messages (10:00 and 10:05).
   - Summary is stored as a single entry linked to both messages.

4. **Retrieval for new query**:
   - User asks: "What time is your cafe meeting?"
   - Hybrid search retrieves the triple ("user", "meeting time", "2 PM") and its linked summary.
   - LLM prompts use only the triple and summary (1,294 tokens total), not the full raw conversation.

This process allows Memori to deliver the same contextual understanding as full conversation history while using a fraction of the context, proving that memory quality depends on representation, not context size.

## References

- **Code:** https://github.com/MemoriLabs/Memori
- Luiz C. Borro, Luiz A. B. Macarini, Gordon Tindall, Michael Montero, Adam B. Struck, "Memori: A Persistent Memory Layer for Efficient, Context-Aware LLM Agents", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19935

Tags: #information-retrieval #agent-systems #cost-optimisation #semantic-triples #hybrid-retrieval
