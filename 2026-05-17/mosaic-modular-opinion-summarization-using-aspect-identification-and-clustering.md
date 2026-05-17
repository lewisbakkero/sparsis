---
title: "MOSAIC: Modular Opinion Summarization using Aspect Identification and Clustering"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19277"
---

## Executive Summary
MOSAIC is a modular opinion summarisation framework that decomposes review analysis into interpretable components, theme discovery, structured opinion extraction, and grounded summary generation, rather than producing monolithic summaries. It improves customer experience by surfacing intermediate outputs like review themes directly on product pages, validated through A/B tests showing measurable revenue impacts. Engineers building review-driven product features should consider this approach for more transparent, measurable, and scalable user feedback processing.

## Why This Matters for Practitioners
If you're building or maintaining product pages with user reviews, MOSAIC's modular approach means you can deploy granular insights incrementally rather than waiting for full summary systems. For example, surfacing "Review Themes" (like "Couple-friendly" or "Amazing Sights") as interactive filters increased revenue per visitor by 1.5% in live tests, with no additional infrastructure cost beyond standard review processing pipelines. This is especially valuable for engineering teams that need to demonstrate quick wins while building out more complex features, start with theme discovery (20% of the work), measure engagement impact, then incrementally add opinion extraction and clustering. Crucially, this approach avoids the "black box" problem of end-to-end summarisation systems, making it easier to debug when insights don't align with user expectations.

## Problem Statement
Current review summarisation systems are like a single-sentence weather forecast for a 5-day trip, they tell you "sunny" but don't reveal the temperature swings, rain showers, or wind speed that actually affect your plans. Similarly, traditional approaches produce monolithic summaries that obscure the underlying themes, making it hard to understand why a particular hotel review says "excellent service" without knowing whether that refers to the front desk, housekeeping, or restaurant staff.

## Proposed Approach
MOSAIC decomposes review summarisation into three interconnected modules: Theme Discovery & Standardization (identifying themes from reviews), Theme-Constrained Opinion Extraction (extracting structured opinions related to those themes), and Opinion-Aware Review Summarisation (generating summaries using only theme-relevant opinions). This sequential processing ensures transparency and allows for incremental deployment.

Here's a simplified pseudocode for the core opinion clustering mechanism:

```python
def opinion_clustering(themes, opinions, sentiment):
    # Cluster opinions at product-theme-sentiment level
    clusters = HDBSCAN(
        data=opinions,
        min_samples=5,
        cluster_selection_epsilon=0.05
    )
    
    # For each cluster, retain 3 diverse, representative opinions
    representative_opinions = {}
    for cluster_id, cluster in clusters.items():
        # Choose 3 diverse opinions (e.g., by semantic distance)
        representatives = choose_diverse_opinions(cluster, k=3)
        representative_opinions[cluster_id] = representatives
    
    return representative_opinions
```

## Key Technical Contributions
MOSAIC's modular design fundamentally shifts how review summarisation is implemented in production systems. Each component addresses a specific pain point in the traditional approach:

1. **Theme Refinement Pipeline with Human-in-the-Loop Validation**  
   The theme standardization process uses frequency-based filtering, semantic deduplication using BERT embeddings, and optional human validation, only flagging new themes that exceed a frequency threshold and are semantically distinct from existing ones. This ensures the system maintains scalability while allowing for domain-specific adjustments when needed (e.g., replacing "Logistics" with more specific themes like "Tour Pacing" or "Tour Itinerary" if the theme is dominated by a single aspect cluster). The human validation step isn't required for all themes, only those that exceed a frequency threshold and aren't semantically similar to existing themes, making it practical for large-scale deployment.

2. **Opinion Clustering for Robustness**  
   MOSAIC introduces clustering of opinions at the product-theme-sentiment level using HDBSCAN with fixed parameters (cluster_selection_epsilon=0.05, min_samples=5). For each cluster, it retains only three diverse, representative opinions, significantly reducing redundancy and improving summary faithfulness. This is crucial because the authors demonstrate that LLMs are sensitive to opinion ordering in prompts, and redundancy can lead to hallucinations or omission of minority viewpoints. The clustering process directly addresses this issue by neutralizing the volatility caused by opinion ordering.

3. **Intermediate Outputs as Production Features**  
   Unlike previous work that focuses solely on final summaries, MOSAIC surfaces intermediate outputs (like themes and sentiment-aware review sorting) directly to users. This approach enables incremental deployment, teams can start with just theme discovery (using a few-shot LLM approach) and measure engagement impact before investing in full summarisation. The A/B tests showed a 1.5% revenue per visitor increase from interactive review themes, demonstrating that granular insights provide immediate value even before the final summary is ready.

## Experimental Results
The paper evaluates MOSAIC on two public datasets (SPACE for hotel reviews, PeerSum for scientific reviews) and their new TRECS dataset (344 tour products, 140K reviews).

On PeerSum with GPT-4o, MOSAIC achieved:
- Coverage: 0.99 (vs. 0.95 for the strongest baseline)
- G-Eval: 0.84 (vs. 0.76)
- AlignScore-R: 0.81 (vs. 0.68)
- AlignScore-M: 0.16 (vs. 0.06)

On SPACE with GPT-4o, MOSAIC matched or outperformed baselines in coverage (1.00 vs. 1.00 for strongest baseline) and improved AlignScore-M by 7.9% with opinion clustering.

The most significant finding was from their synthetic redundancy tests (Figure 4), which showed that removing redundant opinions through clustering substantially improves both coverage and faithfulness across all models (GPT-4o-mini, GPT-4.1-mini, Llama-3.1-70B).

The online A/B tests showed statistically significant results (p < 0.1):
- Review sorting by theme and sentiment: 1% uplift in conversion rate
- Interactive review themes: 1.5% increase in revenue per visitor

## Related Work
MOSAIC builds on recent work decomposing summarisation into intermediate steps (Li et al., 2025; Zhou et al., 2025), but addresses key limitations of those approaches. Unlike Li et al. (2025), MOSAIC handles large-scale theme extraction and addresses opinion redundancy. Unlike Zhou et al. (2025), MOSAIC introduces opinion clustering as a system-level component and demonstrates its impact on improving summary faithfulness. The paper also acknowledges and improves upon the limitations of the SPACE dataset by releasing TRECS, an open-source dataset with 36 unique themes compared to SPACE's 6.

## Limitations
The paper acknowledges that the TRECS dataset doesn't fully capture extreme opinion redundancy found in high-volume product pages (e.g., thousands of reviews repeating the same point), which is why they created a synthetic benchmark for stress testing. The authors note that the full impact of opinion clustering may be under-stated due to limited redundancy in TRECS. Additionally, the human-in-the-loop validation step is optional and only applied to new themes exceeding a frequency threshold and not semantically similar to existing themes, which might miss subtle domain-specific nuances in less frequent themes.

## Appendix: Worked Example
Let's walk through how MOSAIC processes a set of reviews for a tour product called "Paris City Tour" using the theme "Guide" as an example.

1. **Initial Theme Discovery**:  
   The system processes 400 reviews (average per product in TRECS) using few-shot GPT-4o-mini prompting. It identifies 12 themes (including "Guide", "Pickup", "Sights", etc.) with "Guide" appearing most frequently (220 reviews).

2. **Theme Refinement**:  
   The "Guide" theme undergoes frequency-based filtering (220 occurrences is above threshold), semantic deduplication (no other themes with similarity >0.88), and human validation (no need for human review as it's a common theme).

3. **Opinion Extraction**:  
   For the "Guide" theme, the system extracts 220 structured ABSA tuples (theme, aspect, opinion, sentiment) from the reviews. For example:
   - Theme: "Guide", Aspect: "Friendliness", Opinion: "Very friendly and helpful", Sentiment: positive
   - Theme: "Guide", Aspect: "Knowledge", Opinion: "Knows a lot about history", Sentiment: positive
   - Theme: "Guide", Aspect: "Pacing", Opinion: "Too fast for some", Sentiment: negative

4. **Opinion Clustering**:  
   Using HDBSCAN with parameters cluster_selection_epsilon=0.05 and min_samples=5, the system clusters the 220 opinions into 5 clusters. For each cluster, it selects 3 diverse representative opinions:
   - Cluster 1 (positive about friendliness): ["Very friendly and helpful", "Always smiled", "Made us feel welcome"]
   - Cluster 2 (positive about knowledge): ["Knows a lot about history", "Shared interesting stories", "Good at explaining"]
   - Cluster 3 (negative about pacing): ["Too fast for some", "Would like more time", "Rushed through sites"]

5. **Summary Generation**:  
   The system generates a theme-level summary using only the 9 representative opinions (3 per cluster):
   "Travelers frequently praised the guide's friendliness and helpfulness, with many mentioning they felt welcome. Guides also demonstrated strong historical knowledge, sharing interesting stories to enhance the experience. Some travelers felt the pacing was too fast, suggesting more time at key sites would be beneficial."

This process ensures the summary captures nuanced opinions without being skewed by repetitive reviews, with the system maintaining high coverage (all three aspects were addressed) and faithfulness (no opinions were added that weren't in the reviews).

## References

- Piyush Kumar Singh, Jayesh Choudhari, "MOSAIC: Modular Opinion Summarization using Aspect Identification and Clustering", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19277

Tags: #information-retrieval #aspect-based-sentiment #opinion-clustering #modular-systems
