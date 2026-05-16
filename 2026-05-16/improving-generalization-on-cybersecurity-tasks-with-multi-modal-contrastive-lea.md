---
title: "Improving Generalization on Cybersecurity Tasks with Multi-Modal Contrastive Learning"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20181"
---

## Executive Summary
SALM (Semantically Aligned Language Models) tackles the critical problem of poor generalisation in cybersecurity ML systems by using multi-modal contrastive learning to ground payload classification in textual vulnerability descriptions. It achieves 68.1% accuracy on challenging temporal splits (a 5.8% improvement over best baseline), demonstrating that aligning payloads with semantic text representations reduces shortcut learning.

## Why This Matters for Practitioners
If you're building or maintaining cybersecurity systems that rely on ML for threat classification, this paper reveals why your models likely fail in production: they learn superficial patterns rather than semantic concepts. The 5.8% accuracy improvement on temporal splits (from 62.3% to 68.1%) translates directly to fewer false negatives in real-world deployments. Specifically, you should consider implementing a two-stage contrastive learning pipeline for your classification tasks, especially when dealing with temporal shifts in attack patterns. Start by collecting vulnerability descriptions as a secondary data source, then implement a text-payload alignment process using frozen LLM embeddings. Avoid end-to-end fine-tuning on imbalanced datasets, this paper shows TF-IDF with Random Forest outperforms fine-tuned approaches for many classes.

## Problem Statement
Current cybersecurity ML systems suffer from a "credibility crisis" where models excel on controlled benchmarks but collapse on temporal shifts, much like a weather forecast model that predicts perfect sunshine for today but fails completely when tomorrow's weather patterns change. The root cause is that models learn superficial correlations (shortcuts) instead of underlying cybersecurity semantics, making them brittle when facing novel attack patterns.

## Proposed Approach
SALM uses two stages to transfer semantic knowledge from text to payloads:
1. **Stage 1:** Construct a semantic embedding space using contrastive learning on vulnerability descriptions
2. **Stage 2:** Align payloads to this space through cross-modal alignment with a frozen text encoder

This creates a unified embedding space organized by vulnerability type, enabling semantic retrieval at inference. The approach leverages abundant textual vulnerability reports to guide classification of data-scarce payloads.

```python
# Pseudocode for SALM's two-stage training
def train_salm(text_descriptions, payloads):
    # Stage 1: Build vulnerability-aware text space
    text_encoder = initialize_text_encoder()
    for epoch in range(stage1_epochs):
        triplets = generate_triplets(text_descriptions)
        loss = compute_triplet_loss(triplets, text_encoder)
        update(text_encoder, loss)
    
    # Stage 2: Align payloads to text space
    payload_encoder = initialize_payload_encoder()
    frozen_text_encoder = freeze(text_encoder)
    for epoch in range(stage2_epochs):
        text_payload_pairs = create_pairs(text_descriptions, payloads)
        loss = compute_alignment_loss(text_payload_pairs, frozen_text_encoder, payload_encoder)
        update(payload_encoder, loss)
    
    return text_encoder, payload_encoder
```

## Key Technical Contributions
SALM's innovations go beyond simple text-payload alignment:

1. **Structured semantic space via triplet loss on descriptions**: Unlike pre-trained LLMs that encode general knowledge, SALM explicitly structures the embedding space around vulnerability-type boundaries using contrastive learning on descriptions. This reorganizes the space so that descriptions of the same vulnerability type cluster together, while different types are clearly separated, evident from Figure 3 in the paper where clusters emerge around generic textual anchors.

2. **Frozen text encoder for stable semantic transfer**: By freezing the text encoder after Stage 1, SALM prevents catastrophic forgetting while providing stable target embeddings for payload alignment. This is crucial because payloads are computationally expensive to process (up to 16KB each), forcing smaller batch sizes in Stage 2 versus the larger batch sizes used in Stage 1.

3. **Semantic retrieval with generic labels**: At inference, SALM uses generic textual labels ("SQL injection attack") rather than training set labels to avoid shortcut learning. This forces the model to generalise to abstract class descriptions not seen during training, enabling better zero-shot capability for new vulnerability types.

## Experimental Results
SALM was evaluated on two datasets: a large-scale private corpus (29,675 descriptions, 601,518 payloads across 15 classes) and a synthetic benchmark (11,000 balanced samples across 11 classes).

On the challenging temporal split (simulating zero-day conditions):
- SALM achieved 68.1% accuracy (vs. 65.7% for TF-IDF+RF, 62.3% for FT CodeBERT+MLP)
- The improvement translates to a 5.8% absolute gain over the best baseline
- On the synthetic benchmark, SALM reached 24.4% accuracy (vs. 20.6% for best baseline)

The authors note that accuracy and macro F1 diverge significantly, SALM's 68.1% accuracy is driven by high-frequency classes (Code-execution, Injection), while its macro F1 of 30.1% reflects persistent challenges with rare classes. This divergence highlights that the core problem isn't just temporal shift but class imbalance.

## Related Work
The paper positions itself against the growing literature on "credibility crisis" in ML, citing the seminal work of Arp et al. on pitfalls in cybersecurity ML and Pendlebury et al. on temporal bias in malware classification. SALM builds on cross-modal alignment techniques from vision-language models (e.g., [11], [14]), adapting them to cybersecurity's unique data asymmetry between abundant textual descriptions and scarce payload data. It differs from prior work by focusing on transfer from text to payloads rather than other modalities.

## Limitations
The authors explicitly acknowledge that 68.1% accuracy remains far from production-grade reliability, attributing this to three key factors:
1. The vendor's class definitions mix abstraction levels (exploitation techniques, attack outcomes, and vectors)
2. Textual descriptions lack detail for some classes
3. The dataset exhibits strong class imbalance (e.g., Code-execution: 167k payloads; Trojan: 72)

The temporal split tests generalisation within seen types, not zero-shot transfer to entirely new categories. While SALM naturally supports zero-shot transfer, validating this capability requires dedicated experiments.

## Appendix: Worked Example
Let's walk through SALM's core mechanism using specific values from the paper:

1. **Stage 1: Building the semantic space**
   - For "SQL injection" vulnerability, the training split provides 5,823 descriptions (average 127 tokens)
   - Triplets are sampled with anchor (da) and positive (dp) sharing "SQL injection" label, negative (dn) from other classes
   - The text encoder (initialized from instructor-base) learns to cluster "SQL injection" descriptions together
   - After training, the t-SNE projection in Figure 3 shows clear clusters around the generic anchor "SQL injection attack" (star in Figure 3b)

2. **Stage 2: Aligning payloads to text space**
   - 517,692 text-payload pairs are created from the training split
   - A payload encoder (initialized from same model) is trained to minimise ||fp(p) - ft(d)||²
   - For a specific HTTP request-response payload: 
     - `GET /api/search?q=' OR 1=1-- HTTP/1.1`
     - The payload encoder produces a 768-dimensional embedding
     - The frozen text encoder maps "SQL injection attack" to a 768-dimensional vector
     - Cosine distance computes similarity between these vectors
   - If the payload's embedding is closest to "SQL injection attack"'s vector, it's classified as SQL injection

3. **Inference: Semantic retrieval**
   - Given a new payload, the model computes distances to 15 generic class anchors
   - For "SQL injection" anchor: distance = 0.21
   - For "XSS" anchor: distance = 0.43
   - For "Directory Traversal" anchor: distance = 0.56
   - The model predicts "SQL injection" (smallest distance)

This mechanism explains why SALM outperforms baselines on classes with clear structural patterns (e.g., Dir-traversal F1: 78.6 vs 63.0 for best baseline) while struggling with rare classes (Webshell F1: 0.3).

## References

- **Code:** https://github.com/SmartData-Polito/LLM
- Jianan Huang, Rodolfo V. Valentim, Luca Vassio, Matteo Boffa, Marco Mellia, Idilio Drago, Dario Rossi, "Improving Generalization on Cybersecurity Tasks with Multi-Modal Contrastive Learning", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20181

Tags: #cybersecurity #threat-classification #contrastive-learning #cross-modal-alignment #semantic-retrieval
