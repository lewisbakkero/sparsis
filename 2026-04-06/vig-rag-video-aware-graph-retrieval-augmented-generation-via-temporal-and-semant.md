---
title: "ViG-RAG: Video-aware Graph Retrieval-Augmented Generation via Temporal and Semantic Hybrid Reasoning"
category: "AI Applications"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36963"
---

## Executive Summary

**ViG-RAG** introduces a probabilistic temporal knowledge graph framework that structures video transcripts into entities with timestamps and confidence scores, enabling more accurate retrieval and generation for long-form video understanding. It addresses two key limitations in existing video RAG systems: fragmented knowledge representations and static query matching. For engineers, this isn't just a marginal gain—it’s a paradigm shift that yields up to **11.2% accuracy gains** on major video QA benchmarks by treating video as a dynamic, interconnected timeline rather than a pile of static clips.


## Why This Matters for Practitioners

If you're currently building video search or video Q&A systems using simple text embeddings or segment-based approaches, you've likely hit a wall. Traditional RAG treats video clips as isolated "documents," losing the narrative thread that connects them.

ViG-RAG demonstrates that structuring content with **temporal and confidence information** can substantially improve accuracy. Instead of indexing each 30-second video clip as a single text chunk, consider the **Probabilistic Temporal Knowledge Graph (PTKG)**. This approach captures how entities relate across time and assigns confidence to facts. It allows your system to handle complex queries like: 
> *"Show me the moment where the scientist explained the chemical reaction in the lecture video that came after the introduction."* Current systems struggle here because they lack temporal continuity. ViG-RAG can be implemented as a plugin to existing video models with minimal retraining, making it a highly practical upgrade for production-grade AI.


## Problem Statement

Current video understanding systems treat each video segment like a single, isolated puzzle piece. When you ask a question like *"How did the protagonist's relationship with their mentor develop throughout the film?"*, these systems fail because they don't model how scenes relate to each other over time. It's like trying to understand a story from disconnected movie stills without knowing the sequence—each fact is presented in a vacuum, missing the narrative flow that makes the answer meaningful.


## Proposed Approach

ViG-RAG structures video content into a **Probabilistic Temporal Knowledge Graph (PTKG)**. The mechanism works by:

1.  **Segmentation**: Dividing videos into 30-second clips.
2.  **Multimodal Extraction**: Using ASR for transcripts and sampling frames (5 for initial analysis, 15 for caption extraction).
3.  **Graph Construction**: Feeding transcripts and frames into a Vision-Language Model (VLM) to create quintuples:
    $$\mathcal{K} = (e_h, r, e_t, \tau, c)$$
    Where $e_h$ and $e_t$ are entities, $r$ is the relation, $\tau$ is the timestamp, and $c$ is the confidence score.
4.  **Dual-Level Retrieval**: Combining semantic matching (textual) and temporal coherence (sequence).
5.  **Adaptive Selection**: Applying a **Gaussian Mixture Model (GMM)** to adaptively select top-K candidates.

```python
def gmm_top_k_selection(similarity_scores, max_components=5):
    """Determines optimal top-K candidates using GMM without handcrafted thresholds."""
    # Fit GMM to similarity scores
    gmm = GaussianMixture(n_components=max_components, covariance_type='spherical')
    gmm.fit(np.array(similarity_scores).reshape(-1, 1))
    
    # Determine optimal components using BIC
    k_opt = find_optimal_components(similarity_scores, gmm)
    
    # Find highest-mean component (high-confidence region)
    k_star = np.argmax(gmm.means_.flatten())
    
    # Calculate posterior probability for each score
    posteriors = gmm.predict_proba(np.array(similarity_scores).reshape(-1, 1))[:, k_star]
    
    # Return indices sorted by posterior probability
    return np.argsort(posteriors)[::-1][:k_opt]
```

## Key Technical Contributions

ViG-RAG makes three key technical contributions that address the limitations of existing video RAG systems:

1.  **Probabilistic Temporal Knowledge Graph (PTKG)**: The mechanism constructs a graph where each fact includes a timestamp and confidence score, rather than treating video segments as isolated facts. It segments transcripts into structured units, extracts entities and relations using an LLM, then builds a graph with quintuples ($e_h, r, e_t, \tau, c$). This enables more nuanced reasoning across time by preserving temporal context and uncertainty about factual evidence, unlike previous methods that used static entity-relation pairs.

2.  **Temporal-Semantic Dual-Level Retrieval**: This mechanism combines two filtering functions: `Text-F` evaluates semantic alignment between query and clip content using structured prompting in LLMs, while `Temp-F` assesses temporal coherence by checking if retrieved clips maintain meaningful long-range dependencies. These are combined as $\text{topK}(\alpha \cdot \text{Text-F} + (1-\alpha) \cdot \text{Temp-F})$ to select video segments, enabling the system to recognize narrative flow alongside keyword matching.

3.  **Adaptive GMM-Based Top-K Selection**: The mechanism uses a **Gaussian Mixture Model** to automatically identify high-confidence segments without handcrafted thresholds. It fits a K-component univariate GMM to similarity scores, determines optimal components using the **Bayesian Information Criterion (BIC)**, identifies the component with the highest mean (representing the high-confidence region), and selects candidates based on posterior probability.



---

## Experimental Results

ViG-RAG was evaluated on three major benchmarks: **LongerVideos**, **Video-MME**, and **LongVideoBench**.

* **Accuracy Gains**: On Video-MME, ViG-RAG improved accuracy by **11.2%** for LLava-NeXT-Video (from 43.0% to 54.2%) and by **7.5%** for Qwen2-VL (from 64.9% to 72.4%).
* **Win Rates**: On the LongerVideos dataset (20+ sets across educational and entertainment genres), ViG-RAG achieved a **43.1% overall win rate** across all metrics (comprehensiveness, clarity, depth, relevance, and practical value), outperforming GraphRAG-g (21.9%) and VideoRAG (35.1%).
* **Robustness**: The results indicate that the hybrid reasoning approach is particularly effective for videos of arbitrary length, where traditional chunk-based retrieval often loses the "story thread."

---

## Related Work

ViG-RAG positions itself as an evolution of Retrieval-Augmented Generation for multimodal content:

* **GraphRAG**: While GraphRAG introduced knowledge graphs to RAG, it primarily represents knowledge as static entity-relation pairs. ViG-RAG improves this by adding **temporal $(\tau)$** and **confidence $(c)$** dimensions.
* **VideoRAG**: Previous VideoRAG iterations utilized knowledge graphs for retrieval but struggled with complex component associations and relied on fixed thresholds for top-K selection. ViG-RAG replaces these with adaptive GMM-based filtering.

---

## Limitations

Despite its performance, the paper suggests (or leaves open) several limitations:

1.  **Computational Cost**: The paper does not explicitly provide latency or memory usage metrics. Processing 15 frames per segment and fitting GMMs in real-time may be resource-intensive for production environments.
2.  **Edge Cases**: The evaluation does not deeply explore performance on videos with low-quality audio (affecting ASR)

## Appendix: Worked Example
Let's walk through a concrete example of how ViG-RAG processes a 3-minute educational video. The video is divided into six 30-second segments as specified in implementation details.

**Segment 3 (60-90 seconds):**
- Audio transcript (from ASR): "The catalyst speeds up the chemical reaction while reducing energy requirements, making industrial applications more efficient."
- Visual frames (sampled 15 frames): Shows a scientist demonstrating a chemical reaction in a lab setting.
- VLM-generated structured caption: "Catalyst (entity) speeds up (relation) chemical reaction (entity), timestamp: 60-90s, confidence: 0.85."

**PTKG addition:**
- This becomes a quintuple in the PTKG: (Catalyst, speeds up, chemical reaction, 60-90s, 0.85).

**Query: "How does the catalyst affect the reaction?"**

**Retrieval process:**
1. Textual retrieval identifies segments containing "catalyst" and "reaction" (segments 2, 3, and 5).
2. Text-F scores (semantic alignment):
   - Segment 3: 0.92
   - Segment 2: 0.78
   - Segment 5: 0.85
3. Temp-F scores (temporal coherence):
   - Segment 3: 0.8
   - Segment 2: 0.7 (earlier context)
   - Segment 5: 0.9 (later context)
4. Combined scores (α = 0.7):
   - Segment 3: 0.92*0.7 + 0.8*0.3 = 0.878
   - Segment 2: 0.78*0.7 + 0.7*0.3 = 0.746
   - Segment 5: 0.85*0.7 + 0.9*0.3 = 0.835
5. GMM-based filtering:
   - Similarity scores: [0.878, 0.835, 0.746]
   - GMM determines optimal components and identifies high-confidence region
   - Posterior probabilities: [0.92, 0.87, 0.65]
   - Top 3 segments selected: [Segment 3, Segment 5, Segment 2]

**Response generation:**
- Semantic anchors (Kq): "catalyst", "reaction"
- Expanded contextual field (Cp): "chemical reaction, industrial efficiency"
- Retrieved segments (Ŝ): Segment 3 (60-90s), Segment 5 (120-150s), Segment 2 (30-60s)
- VLM generates: "The catalyst speeds up the chemical reaction while reducing energy requirements, making industrial applications more efficient (as shown in the demonstration at 60-90s), and this efficiency gain was demonstrated in the lab setting where the reaction was first introduced (30-60s) and later applied in practical contexts (120-150s)."


## References

- Zongsheng Cao, Anran Liu, Yangfan He, Jing Li, Bo Zhang, Zigan Wang, "ViG-RAG: Video-aware Graph Retrieval-Augmented Generation via Temporal and Semantic Hybrid Reasoning", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36963
