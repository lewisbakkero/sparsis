---
title: "Grounded Multimodal Retrieval-Augmented Drafting of Radiology Impressions Using Case-Based Similarity Search"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.17765"
---

## Executive Summary
This paper introduces a multimodal retrieval-augmented generation (RAG) system for radiology report drafting that reduces hallucinations by grounding outputs in verified historical cases. The approach fuses image and text embeddings through late integration (α=0.5), achieving 95.6% Recall@5 compared to 63.3% for image-only retrieval on a curated MIMIC-CXR dataset. It enforces citation traceability and confidence-based refusal, making it suitable for clinical deployment.

## Why This Matters for Practitioners
If you're building clinical AI systems that generate text (like radiology reports), this paper provides actionable engineering guidance to prevent hallucinations without compromising performance. Specifically: (1) Use multimodal fusion (image + text embeddings) instead of image-only retrieval to boost Recall@5 by 52% (0.633 → 0.956), but keep deployment simple by querying with images against a multimodally-indexed database; (2) Implement citation verification (with ≥86.7% coverage) as a mandatory output constraint to enable auditability; (3) Deploy a confidence-gating mechanism (refuse if top-1 similarity < 0.95) that achieves 0% refusal rate on in-distribution data while automatically rejecting out-of-domain inputs like non-chest X-rays. These steps transform academic experiments into production-ready systems without requiring radiology-specialised LLMs.

## Problem Statement
Today's generative radiology systems resemble a chef who invents dishes from memory without checking the recipe book, they might sound convincing but hallucinate details (e.g., describing a "pulmonary nodule" on a normal chest X-ray). The problem isn't just inaccuracy; it's trust. Clinicians can't verify unsupported claims, making such systems unusable in real workflows, unlike case-based reasoning where prior examples directly inform interpretation.

## Proposed Approach
The system uses contrastive image-text embeddings (CLIP) for multimodal alignment, fuses these embeddings for case retrieval, and generates drafts constrained by retrieved evidence with explicit citations. Core components: (1) CLIP encoders for image (ViT-B/32) and text; (2) FAISS index for efficient similarity search; (3) Grounded drafting with citation verification; (4) Confidence-based refusal. The architecture ensures every generated statement references historical cases, creating a closed loop between evidence and output.

```python
def draft_radiology_impression(image_path, k=5):
    # Generate image embedding (CLIP ViT-B/32)
    image_emb = clip_image_encoder(load_image(image_path))
    
    # Retrieve top-K cases from FAISS index (built with fused embeddings)
    top_k_cases = faiss_index.search(image_emb, k)
    
    # Format evidence with case citations for grounding
    evidence = "\n".join([f"[Case {i+1}] {case.impression}" for i, case in enumerate(top_k_cases)])
    
    # Generate draft (fallback to deterministic summariser if needed)
    draft = llm.generate(f"Draft radiology impression based on:\n{evidence}", 
                         fallback=det_summariser(evidence))
    
    # Verify citations and confidence
    if not draft.contains_citations() or top_k_cases[0].similarity < 0.95:
        return "Refusal: Low confidence (similarity < 0.95)"
    return draft
```

## Key Technical Contributions
The paper's technical innovations enable reliable clinical deployment through specific implementation choices:

1. **Late fusion without additional training**: Instead of training a fusion network, they leverage CLIP's pre-aligned embedding space to combine image and text via simple weighted averaging (α=0.5). This avoids costly training while achieving higher recall (95.6% vs 63.3% Recall@5) by exploiting CLIP's inherent multimodal alignment, unlike prior medical RAG systems requiring domain-specific fusion.

2. **Deterministic citation fallback**: When LLM output fails to include citations, a rule-based summariser synthesises evidence without hallucination. This ensures all drafts reference historical cases (achieving 86.7% average citation coverage) while avoiding complex LLM fine-tuning.

3. **Confidence gating as default safety**: Top-1 retrieval similarity directly drives refusal decisions (threshold = 0.95). This eliminates the need for separate safety models, with 0% refusal rate on in-distribution data (Table 3) but automatic rejection of non-chest images (validated in Section 7).

4. **Deployment-focused containerisation**: Packaging as a FastAPI service with Docker (including dependency files) bridges academic research and production. This ensures reproducibility, unlike many academic systems that stop at offline metrics.

## Experimental Results
On the curated MIMIC-CXR subset (2,696 image-impression pairs), multimodal fusion (α=0.5) achieved Recall@5 = 0.956 versus image-only retrieval (Recall@5 = 0.633), a 52.3% relative improvement. The system demonstrated strong safety metrics: 0% refusal rate (all in-distribution cases retrieved confidently), average top-1 similarity = 0.980, and average citation coverage = 0.867 (Table 3). The paper does not report statistical significance testing for the Recall@5 improvement, though the magnitude (0.956 vs 0.633) is substantial.

## Related Work
This work bridges two research strands: (1) multimodal pretraining (CLIP) for medical imaging [11,12,13], and (2) RAG for clinical text generation [7,8,14,15]. Unlike prior medical RAG systems [14,15] that used text-only retrieval, it uniquely fuses image and report text. It also improves on [14] by adding citation verification and confidence gating, addressing critical gaps in clinical safety that prior work overlooked.

## Limitations
The authors acknowledge four limitations: (1) dataset subset (2,696 pairs from 2,000 sampled MIMIC-CXR studies, not full dataset); (2) drafting uses a lightweight language model (not radiology-specialised); (3) no radiologist validation of generated outputs; (4) citation coverage is a proxy for factual correctness. From an engineering perspective, the lack of radiologist review means clinical safety hasn't been validated in real-world settings, a critical gap before production deployment.

## Appendix: Worked Example
Consider a chest X-ray with finding "mild bibasilar atelectasis" (query). The system retrieves three historical cases:
- Case 1: "Bibasilar atelectasis. Otherwise, no acute cardiopulmonary abnormality." (similarity: 0.98)
- Case 2: "Mild bibasilar atelectasis. No other acute findings." (similarity: 0.97)
- Case 3: "Mild bibasilar atelectasis. Otherwise, no acute cardiopulmonary process." (similarity: 0.96)

Evidence snippets are formatted with case citations:
```
[Case 1] Bibasilar atelectasis. Otherwise, no acute cardiopulmonary abnormality.
[Case 2] Mild bibasilar atelectasis. No other acute findings.
[Case 3] Mild bibasilar atelectasis. Otherwise, no acute cardiopulmonary process.
```

The deterministic fallback summariser synthesises a grounded draft:
"Mild bibasilar atelectasis. [Case 1][Case 2] No acute cardiopulmonary abnormality. [Case 1][Case 3]"

Citation coverage is 100% (all three cases included), and top-1 similarity (0.98) exceeds the 0.95 threshold, so the draft is generated without refusal. This matches the qualitative example in Table 4.

## References

- Himadri Samanta, "Grounded Multimodal Retrieval-Augmented Drafting of Radiology Impressions Using Case-Based Similarity Search", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.17765

Tags: #biomedicine #medical-imaging #information-retrieval #retrieval-augmented-generation #case-based-reasoning
