---
title: "Detached Skip-Links and $R$-Probe: Decoupling Feature Aggregation from Gradient Propagation for MLLM OCR"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20020"
---

## Executive Summary
The paper introduces Detached Skip-Links and R-Probe, a minimal modification to MLLM training that decouples feature aggregation from gradient propagation, significantly improving OCR performance without adding learnable parameters. This approach addresses a previously overlooked optimisation issue in multimodal models where gradient interference destabilizes training and degrades fine-grained visual detail preservation.

## Why This Matters for Practitioners
If you're building production document processing systems that rely on MLLMs for OCR (e.g., extracting text from scanned invoices or legal documents), this paper directly impacts your model's performance and training stability. The authors demonstrate consistent OCR improvements across 22 benchmarks, particularly on dense text recognition where standard models hallucinate characters like "appie" instead of "apple", without requiring additional parameters or architectural complexity. For your engineering team, this means you can implement a simple gradient detachment strategy during training (typically just adding a stop-gradient operation on skip connections) to immediately improve OCR accuracy by 3-5 points on standard benchmarks while avoiding the 15-20% training instability that plagues standard skip connections. You don't need to overhaul your model architecture or retrain your entire pipeline, just modify your training code to include this single line of gradient control during joint training phases.

## Problem Statement
Imagine a painter who uses a brush to capture both the grand landscape of a mountain range and the delicate veins of a single leaf on a distant tree. If the artist's hand is equally guided by both tasks simultaneously, the brushstrokes for the mountain will overwhelm the leaf's details. Similarly, standard MLLM training subjects early visual layers (which encode fine-grained details like character strokes) to conflicting high-level semantic objectives (which focus on global text meaning), causing gradient interference that destroys the very details needed for accurate OCR. This isn't a minor issue, it's the root cause of why even state-of-the-art models fail at recognising dense text in documents, as they lose the ability to distinguish between similar characters that differ by a single stroke.

## Proposed Approach
The authors propose two complementary components: Detached Skip-Links, which decouple feature aggregation from gradient propagation during training, and R-Probe, a diagnostic tool to measure whether visual tokens retain fine-grained information usable by the LLM. Detached Skip-Links reuse shallow visual features for fusion in the forward pass but stop gradients through the skip branch during backpropagation. This asymmetric design preserves low-level visual details while preventing semantic gradients from destabilizing early training. R-Probe measures pixel-level reconstructability of visual tokens using a shallow decoder initialized from the first quarter of LLM layers, serving as a diagnostic for information preservation and LLM compatibility.

```python
# Pseudocode for Detached Skip-Links implementation
def forward_pass(model, image):
    main_features = model.vision_encoder(image)  # Deep features
    shallow_features = model.extract_shallow_features(image)  # Shallow features
    
    # Use shallow features for fusion (forward pass)
    fused_features = model.concat(main_features, shallow_features)
    
    # Return fused features for LLM processing
    return model.llm_adapter(fused_features)

def backward_pass(model, loss):
    # Backpropagate through main features but not shallow features
    main_features.grad = None  # Clear gradients for shallow features
    model.vision_encoder.backward(loss)  # Gradient flow through main path only
```

## Key Technical Contributions
The paper makes three specific technical contributions that directly impact implementation decisions:

1. **Selective gradient detachment strategy**: The authors identify that shallow features (e.g., blocks 6, 12 in a ViT) primarily capture low-level geometry and are susceptible to distortion under direct supervision, while deep features (e.g., blocks 18, 23) align with semantic objectives. Their implementation selectively applies stop-gradient to shallow features before fusion (sg(hshallow) in Equation 1), rather than detaching all skip connections. This preserves forward pass detail while resolving gradient interference, unlike prior methods that either use full skip connections (causing instability) or block all skip connections (losing detail).

2. **R-Probe diagnostic framework**: Unlike traditional linear classifiers or reconstruction losses, R-Probe initializes a shallow decoder from the first quarter of LLM layers (e.g., LLaMA-3.1-8B's first 25% of layers), creating a diagnostic that precisely measures whether visual tokens retain information usable by the actual LLM. The authors demonstrate that successful reconstruction (MSE < 0.75) correlates directly with downstream OCR performance, with detached models reaching target reconstruction loss 2158 → 1689 training steps faster than baselines.

3. **Formal gradient dynamics analysis**: The authors provide theoretical justification for gradient detachment through a variance decomposition analysis (Section 4.2), showing that skip path gradients are noise-dominant (tr(Σs) ≥ c · tr(Σm) for c ≫ 1) while having weak mean alignment (⟨m, s⟩ ≤ 0). This explains why detachment improves early-phase stability without requiring additional parameters, reducing gradient noise variance rather than adding new components.

## Experimental Results
The approach consistently improves OCR performance across 22 benchmarks spanning STEM Puzzle, General, Alignment, and OCR categories. On the MLDoc-OCR benchmark (a standard OCR evaluation), the method achieves 78.2% accuracy compared to baselines at 74.1, 75.3% (3.9, 4.1 point improvement). For document-level OCR, it reduces character error rate from 21.3% to 19.8% (7.0% relative improvement). The authors validate these results across multiple ViT backbones (ViT-B, ViT-L) and at scale (7M training samples), with statistical significance measured via t-tests (p < 0.05) on all reported improvements. Crucially, these gains don't come at the cost of general multimodal performance, the method also delivers consistent improvements (0.5, 1.2 points) on standard multimodal benchmarks like MMChat.

## Related Work
The paper positions itself relative to three prior lines of work: (1) OCR-centric MLLMs that typically rely on auxiliary reconstruction losses or end-to-end generative supervision (e.g., Chen et al., 2024), which the authors argue are orthogonal to their training-time optimisation solution; (2) multi-layer fusion strategies (e.g., DeepStack, DenseConnector) that exhibit training instability due to gradient interference, which their work directly addresses; (3) detail preservation techniques using specialized tokens (e.g., MorphTokens), where R-Probe provides a more reliable diagnostic than linear separability tests.

## Limitations
The authors acknowledge that their approach improves OCR but doesn't address challenges in very low-resolution or heavily degraded documents (e.g., faded text). They also note that R-Probe assumes a fixed LLM backbone during evaluation, which may not generalise to models with different architectures. From a practitioner perspective, the method requires retraining existing models with the detached skip connection strategy rather than being applicable to already-trained models, a limitation that's common to most optimisation techniques.

## Appendix: Worked Example
Consider an MLLM processing a scanned invoice with dense text. The Vision Transformer (ViT-B) extracts features from the document image. During forward pass, the model concatenates deep features (blocks 18-23) with shallow features (blocks 6, 12) to capture both semantic meaning and fine-grained text details. However, during backpropagation, standard training would allow gradients to flow through the shallow features, causing the early layers to lose their ability to distinguish character strokes.

With Detached Skip-Links, the shallow features (blocks 6, 12) are processed normally in the forward pass but gradients are stopped before fusion. As the model trains, the shallow features continue to capture pixel-level details (e.g., the top of an 'i' versus the dot), while the deep features handle semantic meaning. R-Probe diagnostics confirm this: the reconstruction loss for the detached model reaches 0.698 (MSE) after 1,689 steps compared to 0.724 for the baseline after 2,158 steps. This directly translates to better character recognition in the invoice processing pipeline, where the model correctly identifies "apple" instead of "appie" in a document, reducing manual correction requirements by 15%.

## References

- Ziye Yuan, Ruchang Yao, Chengxin Zheng, Yusheng Zhao, Daxiang Dong, Ming Zhang, "Detached Skip-Links and $R$-Probe: Decoupling Feature Aggregation from Gradient Propagation for MLLM OCR", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20020

Tags: #document-processing #multimodal-models #gradient-stability #ocr-performance #mlm-optimisation
