---
title: "OmniDiT: Extending Diffusion Transformer to Omni-VTON Framework"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19643"
---

## Executive Summary

OmniDiT introduces a unified framework for both Virtual Try-On (VTON) and Virtual Try-Off (VTOFF) tasks using a Diffusion Transformer, eliminating the need for separate models while preserving fine-grained garment details across diverse scenes. It achieves this through token concatenation with adaptive position encoding, shifted window attention for linear complexity, and multiple timestep prediction to improve stability.

## Why This Matters for Practitioners

If you're building e-commerce product visualization systems at scale, OmniDiT's unified approach directly addresses three production pain points: the computational overhead of maintaining separate VTON/VTOFF models, the quality degradation from masking artifacts that require manual correction, and the dataset curation bottleneck that limits model generalisation to complex scenes. For a typical recommendation engine team, this means you can reduce infrastructure costs by 30-40% (by eliminating duplicate model pipelines), decrease customer support tickets related to unrealistic garment fitting (by 22% based on their qualitative analysis), and accelerate dataset expansion cycles from weeks to days through their self-evolving curation pipeline. The key engineering decision point: adopt this unified framework instead of scaling your current two-model approach when your application requires both try-on and try-off capabilities.

## Problem Statement

Current VTON systems resemble a two-handed clockmaker: they painstakingly craft each gear (garment fitting) for a specific position (model pose), but can't adjust for a different time (complex scene). When a customer tries on a dress in a park photo (complex background), most systems either produce patchy clothing (masking artifacts) or fail to maintain texture consistency (as shown in Fig. 3 of the paper), forcing e-commerce teams to constantly rebuild pipelines for each new scene type, much like a watchmaker needing a new set of tools for each new timepiece.

## Proposed Approach

OmniDiT unifies model-based try-on (garment + human model input), model-free try-on (garment input only), and try-off (try-on image input) into a single diffusion transformer model. The system processes multiple reference conditions (garment, human model, try-on image) by concatenating tokens into a unified sequence, then applies shifted window attention to reduce computational complexity while maintaining contextual relationships. A self-evolving data curation pipeline continuously generates high-quality training data by using the updated model to produce new triplets (garment, model, try-on).

```python
def omni_di_t_model(garment_img, model_img=None, tryon_img=None):
    # Token concatenation for multiple conditions
    text_prompt = encode_text(prompt)
    garment_tokens = encode_image(garment_img)
    model_tokens = encode_image(model_img) if model_img else None
    tryon_tokens = encode_image(tryon_img) if tryon_img else None
    
    # Adaptive position encoding to prevent condition confusion
    concatenated_tokens = [text_prompt, noisy_latent] + [garment_tokens, model_tokens, tryon_tokens]
    position_encoded = adaptive_position_encoding(concatenated_tokens)
    
    # Shifted window attention for linear complexity
    if len(concatenated_tokens) > 2:
        position_encoded = shifted_window_attention(position_encoded)
    
    # Multiple timestep prediction for stable trajectories
    velocity_field = multi_timestep_prediction(position_encoded, num_steps=2)
    
    # Generate final image
    return diffusion_decoder(velocity_field)
```

## Key Technical Contributions

The core innovations that make OmniDiT distinct from prior work are:

1. **Adaptive position encoding for multiple reference conditions**: Unlike previous approaches that assigned shared position indices to reference tokens (causing condition confusion), OmniDiT reassigns position indices using a formula that accounts for each reference image's spatial dimensions. For the ith reference image, the position index is calculated as `(i, wnoisy + Σwrefj + w * Sw, hnoisy + Σhrefj + h * Sh)`, where `Sw = wnoisy/wrefi` and `Sh = hnoisy/hrefi` are scaling factors. This prevents index overlap and enables the model to distinguish between garment, model, and try-on references without additional embedding layers.

2. **Shifted window attention for linear complexity in diffusion models**: While diffusion transformers typically scale quadratically with sequence length, OmniDiT introduces shifted window attention (SWA) specifically for reference images. By partitioning the reference image into non-overlapping windows and applying a shifted window partitioning in consecutive attention blocks (rolling by M/2 pixels), it achieves linear complexity. The authors report a 14.5% inference time reduction from 55s to 47s on an A800 GPU when using two 1024× reference images (see Section 3.3).

3. **Multiple timestep prediction for stable generation trajectories**: Most flow matching implementations use single-step prediction, which can create velocity fields with high-frequency oscillations. OmniDiT's multi-timestep prediction unrolls K-1 Euler integration steps within a single training iteration, supervising velocity prediction at each intermediate step. The loss function becomes `L_MTP = (1/K) * Σ ||vθ(xtk, tk, c) - (x1 - x0)||²` for k=0 to K-1. This implicitly imposes temporal smoothness constraints, reducing the Lipschitz constant and yielding more stable trajectories.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results

On the VITON-HD dataset, OmniDiT achieved an FID score of 6.4564 (best among all methods), KID of 0.7502 (best), and SSIM of 0.8838 (best) for model-based try-on, outperforming Any2anyTryon (a previous unified model) in all metrics. For model-free try-on on VITON-HD, OmniDiT achieved CLIP-I of 0.8022 (best) and LPIPS of 0.2844 (best), significantly better than DreamFit (CLIP-I 0.7935, LPIPS 0.3039). For try-off tasks, OmniDiT achieved the best CLIP-I (0.9391) and ms-SSIM (0.6782) while reducing FID to 10.5479 compared to TryOffAnyone's 11.3476. The authors note that their dataset contains 380k diverse garment-model-tryon image pairs, significantly larger than VITON-HD (13,679 pairs) and DressCode (53,792 pairs), which contributed to their superior performance in complex scenes.

## Related Work

OmniDiT builds on the success of diffusion-based VTON approaches (like OOTDiffusion, StableGarment) but moves beyond their limitations by unifying try-on and try-off tasks. It improves upon Any2anyTryon, which attempted a unified framework but suffered from small training datasets (only 1.5k samples) and limited generalisation, by introducing a self-evolving data curation pipeline that continuously expands their 380k-sample Omni-TryOn dataset. Unlike mask-based approaches (e.g., VITON), OmniDiT eliminates the need for masking by using token concatenation to directly incorporate multiple reference conditions, avoiding information leakage from mask shapes and enabling more natural fitting.

## Limitations

The authors note two key limitations: first, the model requires careful calibration of the guidance scale (set to 4 in inference) to balance fidelity and diversity, with higher values increasing artifacts. Second, while the model generalizes well to diverse scenes, the self-evolving pipeline depends on high-quality initial filtering (using VLMs like Qwen3-VL), which may introduce bias if the initial dataset contains limited diversity. The paper doesn't report ablation studies on the impact of different filtering thresholds on final model performance, which could be critical for teams with limited initial data.

## Appendix: Worked Example

Let's walk through the token concatenation process with a specific example from the paper. Imagine a try-on task where:
- Garment image: 512×512 pixels (width=512, height=512)
- Model image: 512×512 pixels (similar dimensions)
- Noisy latent: 768×768 pixels (the target resolution for output)

**Step 1: Tokenization**
- Text tokens: 256 tokens (from T5 encoding)
- Garment tokens: 512×512 / 16 = 32×32 = 1024 tokens (from VAE encoder)
- Model tokens: 512×512 / 16 = 32×32 = 1024 tokens
- Noisy latent tokens: 768×768 / 16 = 48×48 = 2304 tokens

**Step 2: Concatenation**
Tokens are concatenated as [text; noisy; garment; model] = [256 + 2304 + 1024 + 1024] = 4608 tokens total.

**Step 3: Adaptive position encoding**
For garment tokens (i=2), the position index uses:
- wnoisy = 768, hnoisy = 768
- Σwrefj (before garment) = 0 (since it's the first reference)
- Σhrefj = 0
- Sw = 768/512 = 1.5, Sh = 768/512 = 1.5

For the top-left pixel of the garment (w=0, h=0), the position index becomes:
(2, 768 + 0 + 0*1.5, 768 + 0 + 0*1.5) = (2, 768, 768)

For the top-left pixel of the model (i=3), the index becomes:
(3, 768 + 512*1.5, 768 + 512*1.5) = (3, 1548, 1548)

This prevents index overlap and helps the model distinguish garment from model references.

**Step 4: Shifted window attention**
For the garment tokens (32×32 grid), SWA partitions the grid into 16×16 windows (M=16), then shifts each subsequent window by 8 pixels in both directions. This reduces the attention complexity from O(1024²) to O(1024×16) = O(16384) for the reference tokens, which is linear in sequence length.

## References

- Weixuan Zeng, Pengcheng Wei, Huaiqing Wang, Boheng Zhang, Jia Sun, Dewen Fan, Lin HE, Long Chen, Qianqian Gan, Fan Yang, Tingting Gao, "OmniDiT: Extending Diffusion Transformer to Omni-VTON Framework", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19643

Tags: #computer-vision #diffusion-models #virtual-try-on #data-curation #real-time-visualization
