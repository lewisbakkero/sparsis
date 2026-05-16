---
title: "Semantic-Driven Topic Modeling for Analyzing Creativity in Virtual Brainstorming"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2509.16835"
---

## Executive Summary
This paper presents a semantic-driven topic modelling framework that automatically identifies coherent themes from virtual brainstorming transcripts, outperforming traditional models like LDA and BERTopic with an average topic coherence of 0.687 (CV). For engineers building collaborative tools, this offers a production-ready pipeline to extract interpretable insights from unstructured meeting data without manual coding.

## Why This Matters for Practitioners
If you're developing meeting analytics tools for distributed teams (e.g., Zoom transcript processors or idea management platforms), this framework directly solves the problem of noisy topic extraction in creative discussions. Traditional LDA-based approaches produce semantically incoherent topics (e.g., "parking" and "dining" appearing together), while this method’s semantic embeddings reveal meaningful clusters like "academic support" (topic 1) and "campus facilities" (topic 4), with 0.687 coherence vs. LDA’s 0.528 in their Table 2. Implement this four-step pipeline: Sentence-BERT embeddings → UMAP reduction → HDBSCAN clustering → cosine-similarity-based topic extraction. Skip manual coding of ideas, your system can now automatically rank topic depth (e.g., "financial aid" cluster 3: 46 ideas, coherence 0.733) and diversity (e.g., cluster 6: "social inclusion" with 56 ideas), enabling data-driven decisions on where to focus team efforts.

## Problem Statement
Traditional topic models treat brainstorming transcripts like a chaotic library where books (ideas) are shelved by physical proximity (word co-occurrence), not meaning. Like sorting a pile of scattered LEGO bricks by colour instead of building the intended structure, LDA groups "parking tickets" with "dining menus" (both appear in the same university discussion) but fails to reveal that parking (cluster 4) and dining (cluster 5) represent separate operational themes. This leads to useless summaries like "the group talked about campus food and parking," missing the nuanced exploration of distinct problem domains.

## Proposed Approach
The framework processes ideas through four sequential modules: (1) Sentence-BERT generates semantic embeddings; (2) UMAP reduces dimensionality to 2D; (3) HDBSCAN clusters semantically similar ideas while filtering outliers; (4) Topic extraction selects top words per cluster using cosine similarity. This pipeline transforms raw transcripts into interpretable topic maps where clusters reveal both convergent (deep dives) and divergent (broad exploration) thinking patterns.

```python
def extract_topics(transcript):
    # Step 1: Embed sentences (Sentence-BERT)
    embeddings = sentence_bert.embed(transcript)
    
    # Step 2: Dimensionality reduction (UMAP)
    reduced = umap.reduce(embeddings, n_components=2)
    
    # Step 3: Cluster with HDBSCAN (filters noise)
    clusters = hdbscan.cluster(reduced)
    
    # Step 4: Extract top words per cluster (see Eq. 1)
    topics = []
    for cluster in clusters:
        top_words = calculate_top_words(cluster, k=10)
        topics.append({cluster_id: top_words})
    
    return refine_topics(topics)  # Merges similar topics
```

## Key Technical Contributions
The framework’s novelty lies in how it leverages semantic embeddings for both clustering and topic extraction, unlike prior work that uses embeddings only for feature representation. 

1. **Contextual topic word selection**: Instead of using word frequency (like BERTopic), it calculates average cosine similarity between candidate words and sentences within each cluster (Equation 1). For cluster 1 ("education"), "academic" scored 0.736 (highest) because it co-occurred semantically with "degree" and "student" across 134 ideas, unlike LDA’s "student" (0.312) in cluster 1, which lacked context.

2. **Dynamic topic refinement**: After initial extraction, it merges clusters with low cosine similarity to other topics (e.g., cluster 7 "class/lab" (0.488) merged with cluster 9 "schedule" (0.464) because their topic words shared semantic overlap). This reduced topics from 10 to a user-specified count without losing coherence, unlike static k-means.

3. **Outlier filtering via HDBSCAN**: The algorithm labels weakly related ideas (e.g., "sfdeg" or "21hy%") as noise (label -1), excluding them from topic extraction. In their data, 14.8% of ideas were filtered as outliers (Figure 4), directly improving coherence by removing irrelevant noise.

## Experimental Results
Evaluated on 13 student groups (3, 5 participants each) generating 762 ideas during 60-minute Zoom sessions. The framework achieved **0.687 average topic coherence (CV)**, significantly outperforming baselines (LDA: 0.528, ETM: 0.581, BERTopic: 0.607) across 2, 10 topics (Table 2). For example, topic 3 ("financial aid") reached coherence 0.733, 30% higher than LDA’s 0.564 for the equivalent topic. Coherence was measured using Gensim’s CV metric (score range -1 to 1), where higher values indicate better word co-occurrence within a topic. The paper does not report statistical significance tests, but the consistent margin (0.08, 0.16) across topic counts suggests robustness.

## Related Work
The work extends Mersha et al.’s earlier semantic framework (2022) by applying it to structured synchronous meetings (not Slack conversations), addressing the gap in real-time collaboration analysis. It positions against probabilistic models (LDA) by showing semantic embeddings capture deeper relationships (e.g., "scholarship" and "tuition" co-occurred in 92% of cluster 3 ideas vs. LDA’s 68%), and advances BERTopic by adding iterative topic refinement for better interpretability.

## Limitations
The evaluation used university improvement brainstorming (a single domain), so generalizability to other contexts (e.g., product design sprints) is untested. The framework requires pre-processed transcripts (not raw audio), and its 2D UMAP projection limits scalability for real-time analysis (though UMAP is efficient for typical meeting lengths). The paper doesn’t address bias in topic extraction, e.g., whether "student" dominates clusters due to demographic skew.

## Appendix: Worked Example
Consider cluster 1 (134 ideas) from the university brainstorming dataset. After Sentence-BERT embedding and UMAP reduction, 134 sentence vectors cluster together. The framework calculates average cosine similarity for each candidate word (e.g., "education", "academic") against all sentences in the cluster using Equation 1:

- "academic" similarity: 0.736 (co-occurred with "degree" in 42 ideas, "study" in 38)
- "parking" similarity: 0.021 (only appeared once with "ticket")

The top 10 words are sorted by similarity: `["academic", "degree", "student", "curriculum", "benefit", ...]`. This yields topic "academic support" (coherence 0.736), revealing deep exploration of educational infrastructure, unlike LDA’s "student" topic (coherence 0.312), which bundled unrelated ideas like "student" (in "student housing") and "student" (in "student job").

## References

- Melkamu Abay Mersha, Jugal Kalita, "Semantic-Driven Topic Modeling for Analyzing Creativity in Virtual Brainstorming", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2509.16835

Tags: #information-retrieval #collaborative-systems #sentence-bert #umap #hdbscan
