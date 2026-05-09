---
title: "L-PRISMA: An Extension of PRISMA in the Era of Generative Artificial Intelligence (GenAI)"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19236"
---

## Executive Summary
L-PRISMA extends the PRISMA framework for systematic literature reviews to incorporate GenAI while preserving scientific rigor. It introduces a statistical pre-screening phase that uses semantic similarity scores to determine which documents should undergo human review versus GenAI-assisted processing. This enables broader literature coverage without compromising reproducibility, transparency, or auditability.

## Why This Matters for Practitioners
If you're responsible for conducting literature reviews in your engineering team, whether for product research, technical debt assessment, or identifying emerging patterns, L-PRISMA offers a practical framework to integrate GenAI without sacrificing methodological integrity. Specifically, when you need to cover large volumes of literature (e.g., 10,000+ papers in a domain), implement a statistical pre-screening phase with defined thresholds before human review. Document these statistical boundaries thoroughly to ensure reproducibility, and maintain clear separation between human-reviewed and GenAI-assisted results in your final report. This approach prevents the common pitfall of "AI hallucination" leading to inaccurate literature coverage while avoiding the time-consuming manual screening that currently constrains systematic reviews.

## Problem Statement
Traditional systematic reviews using PRISMA are like manually sorting through a library's entire collection to find relevant books, thorough but time-consuming. When GenAI is introduced to automate this process, it's like having an AI librarian who can quickly scan all books but may misremember titles (hallucinate) or have biased recommendations. The problem is that PRISMA wasn't designed to handle this new reality: GenAI's non-deterministic outputs compromise reproducibility, and hallucinations can introduce bias into the literature review process. This creates a fundamental conflict between scalability and scientific rigor that needs resolution.

## Proposed Approach
L-PRISMA integrates GenAI into systematic review workflows through a hybrid framework that maintains human oversight as the core component while using GenAI to expand the scope of literature coverage. The framework adds a statistical pre-screening phase before the standard screening phase, using semantic similarity scores to determine which documents should undergo human review versus GenAI-assisted processing. This ensures reproducibility through deterministic statistical boundaries while preserving the transparency required by PRISMA.

```python
def statistical_pre_screening(abstracts, review_intent):
    """
    Apply statistical pre-screening to determine which documents require human review
    based on semantic similarity scores with the review intent.
    
    Args:
        abstracts: List of document abstracts
        review_intent: Statement describing the review's purpose
        
    Returns:
        human_review_list: Documents requiring human review
        genai_review_list: Documents suitable for GenAI processing
    """
    # Calculate semantic similarity score for each abstract
    similarity_scores = [semantic_similarity(abstract, review_intent) for abstract in abstracts]
    
    # Fit mixture model to similarity score distribution
    pi_H, pi_L, pH, pL = fit_mixture_model(similarity_scores)
    
    # Determine decision boundary using quartile points
    threshold = calculate_threshold(similarity_scores, pi_H, pi_L)
    
    human_review_list = [abstract for score, abstract in zip(similarity_scores, abstracts) if score >= threshold]
    genai_review_list = [abstract for score, abstract in zip(similarity_scores, abstracts) if score < threshold]
    
    return human_review_list, genai_review_list
```

## Key Technical Contributions
L-PRISMA introduces specific technical mechanisms that address the tension between GenAI scalability and scientific rigor:

1. The statistical pre-screening approach based on mixture models (Equation 1) that determines document relevance through a deterministic process. Unlike other proposed approaches that rely on fine-tuned LLMs, this method requires no deep statistical knowledge from researchers and uses standard statistical boundary rules (e.g., quartiles) to separate highly relevant from weakly relevant documents.

2. A transparent reporting framework that documents the statistical boundaries used for filtering, including the mixture model parameters (πH, πL) and the threshold calculation method. This ensures reproducibility by allowing other researchers to replicate the exact filtering process.

3. The explicit separation of human-reviewed and GenAI-assisted results in the final review, with clear reporting of both in the systematic review framework. This prevents the common issue where AI-generated summaries are treated as equivalent to human-reviewed evidence.

## Experimental Results
The paper does not provide detailed experimental results or quantitative comparisons with baselines in the provided text. It describes a framework rather than presenting empirical validation of the statistical pre-screening approach. The authors conducted a use case example (Section IV) but did not report specific numbers on performance improvements, accuracy metrics, or statistical significance measures.

## Related Work
L-PRISMA builds upon and improves existing approaches to AI-assisted systematic reviews. Teo [40] proposed a domain-specific fine-tuned LLM-based extension for PRISMA, but this requires resource-intensive fine-tuning of models, making it inaccessible to non-ML specialists. Tools like Rayyan, Covidence, and ASReview use AI for refinement but lack the systematic approach provided by PRISMA and do not address reproducibility concerns. L-PRISMA specifically addresses the gaps identified in Gundersen et al. [11] (reproducibility) and Haibe-Kains et al. [12] (non-deterministic AI methods) by introducing a statistical layer that ensures reproducibility while maintaining human oversight.

## Limitations
The paper acknowledges that LLMs have a tendency to hallucinate and generate false results confidently, requiring human moderation of all GenAI outputs. The framework needs validation across multiple domains beyond the initial Education-focused use case. The paper doesn't address how to handle cases where the statistical pre-screening might miss highly relevant documents that would have been captured by more nuanced human judgment.

## Appendix: Worked Example
Let's walk through how L-PRISMA's statistical pre-screening would work in practice for an engineering team seeking literature on "GenAI for text similarity in education":

1. **Define review intent**: "Find papers discussing GenAI applications for text similarity metrics in educational technology contexts."

2. **Collect abstracts**: 1,000 abstracts from IEEE and ACM databases (see Table II for initial query results).

3. **Calculate semantic similarity scores** using a transformer-based model:
   - The review intent statement is embedded.
   - Each abstract is embedded.
   - Cosine similarity scores are calculated (range 0-1).

4. **Fit mixture model** to the similarity scores:
   - The distribution of scores follows a mixture of two normal distributions (pH and pL).
   - πH (weight for highly relevant articles) = 0.55
   - πL (weight for weakly relevant articles) = 0.45
   - Model fit: p(s) = 0.55 * N(0.85, 0.05) + 0.45 * N(0.35, 0.1)

5. **Determine threshold**: Using quartiles, the threshold is set at 0.62 (80% of highly relevant documents fall above this threshold).

6. **Filter documents**:
   - Documents with similarity score ≥ 0.62: 247 abstracts (for human review)
   - Documents with similarity score < 0.62: 753 abstracts (for GenAI-assisted processing)

7. **GenAI processing**: The 753 documents are summarised by GenAI with a prompt: "Summarise this paper's approach to text similarity metrics in educational technology, highlighting methodology and key findings."

8. **Human moderation**: All GenAI summaries are reviewed by human experts for consistency and accuracy before inclusion in the final systematic review.

See Appendix for a step-by-step worked example showing how data flows through the statistical pre-screening process with specific values at each stage.

## References

- Samar Shailendra, Rajan Kadel, Aakanksha Sharma, Islam Mohammad Tahidul, Urvashi Rahul Saxena, "L-PRISMA: An Extension of PRISMA in the Era of Generative Artificial Intelligence (GenAI)", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19236

Tags: #information-retrieval #systematic-reviews #reproducible-ai #statistical-pre-screening #hybrid-framework
