---
title: "Dual Prompt-Driven Feature Encoding for Nighttime UAV Tracking"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19628"
---

## Executive Summary
This paper introduces DPTracker, a dual prompt-driven feature encoding method that enhances nighttime UAV tracking by explicitly integrating illumination and viewpoint cues into the feature encoding process. It outperforms state-of-the-art methods by up to +2.7% in precision and +2.3% in normalized precision on key nighttime UAV tracking benchmarks without requiring extensive retraining.

## Why This Matters for Practitioners
If you're building or maintaining UAV tracking systems for applications like night-time surveillance, search and rescue, or autonomous drone navigation, this paper directly addresses a critical reliability gap. Existing solutions like low-light enhancement or domain adaptation often fail to provide robust feature representations for nighttime tracking, leading to unreliable performance. DPTracker offers a practical path to improve tracking robustness by adding just a few lightweight modules to existing ViT-based trackers, no need for complete retraining on scarce nighttime data. You can integrate this approach into your existing tracking pipeline to significantly reduce tracking failures during nighttime operations, with minimal computational overhead (only adding ~2.5% more parameters compared to SOTA methods).

## Problem Statement
Existing feature encoding methods for UAV tracking treat nighttime conditions as a simple degradation problem, like trying to read a smudged book by candlelight while constantly changing the angle you hold it at. The core issue isn't just poor image quality, it's that the feature encoding itself ignores the critical illumination and viewpoint cues that would help the system understand what it's seeing. Current approaches either try to fix the image quality (before feature extraction) or apply domain adaptation without addressing the specific challenges of illumination and viewpoint variations in aerial tracking.

## Proposed Approach
DPTracker introduces a dual prompt-driven feature encoding block (DPBlock) that integrates two prompters directly into the ViT backbone: a pyramid illumination prompter (PIP) to handle illumination variations and a dynamic viewpoint prompter (DVP) to address viewpoint changes. The system establishes prompt-conditioned feature adaptation and context-aware prompt evolution, where the generated prompt tokens modulate features, enabling the model to attend to critical visual cues under degraded conditions.

The overall architecture incorporates these components within a standard ViT-based tracking pipeline: input image → patch embedding → DPBlock (for prompt-feature interaction) → tracker head. The key innovation is the bidirectional interaction between prompt tokens and features, where features guide prompt refinement and prompts guide feature adaptation.

```python
def prompt_conditioned_feature_adaptation(F_prev, P_prev, alpha_f=0.7, beta_f=0.3, alpha_p=0.6, beta_p=0.4):
    E_sim = layer_norm(mlp(F_prev + P_prev))
    E_dif = layer_norm(mlp(F_prev - P_prev))
    
    F_current = F_prev + alpha_f * E_sim - beta_f * E_dif
    P_current = P_prev - alpha_p * E_sim + beta_p * E_dif
    
    return F_current, P_current
```

## Key Technical Contributions
The paper's core innovations provide specific technical mechanisms for improving nighttime UAV tracking:

1. **Prompt-conditioned feature adaptation** uses element-wise summation and subtraction between prompt tokens and visual features to model both similarity (shared features) and difference (distinctive features), with learnable coefficients controlling the contribution of each component. This allows the system to both reinforce relevant features and suppress irrelevant noise in low-light conditions.

2. **Pyramid illumination prompter (PIP)** approximates the Laplacian pyramid structure using learnable convolutional components rather than hand-crafted operators. It decomposes images into multi-scale frequency bands (low, mid, high frequencies), where low frequencies capture illumination information while mid/high frequencies capture structural details. This hierarchical approach effectively extracts frequency-aware illumination features.

3. **Dynamic viewpoint prompter (DVP)** uses deformable convolution to learn adaptive sampling offsets for feature extraction. Unlike standard convolutions with fixed grid-based sampling, DVP introduces learnable spatial offsets that enable each kernel position to sample from geometrically relevant regions. This allows the system to capture viewpoint changes without requiring explicit viewpoint annotations.

## Experimental Results
The authors evaluated DPTracker on three key benchmarks:

- **UAVDark135**: DPTracker-B achieved 72.3% precision (vs. DCPT's 70.3%), 65.2% normalized precision (vs. 63.9%), and 58.1% success rate (vs. 57.1%), outperforming SOTA by +2.0%, +1.3%, and +1.0% respectively.
- **DarkTrack2021**: DPTracker-B achieved 69.8% precision (vs. DCPT's 67.1%), 62.2% normalized precision (vs. 59.9%), and 55.8% success rate (vs. 53.7%).
- **NAT2021-test**: DPTracker-B achieved 73.9% precision (vs. DCPT's 69.4%), 62.5% normalized precision (vs. 58.4%), and 55.9% success rate (vs. 52.0%).

The attribute-based evaluation (Fig. 5) showed DPTracker-B consistently outperformed other methods in low ambient intensity (51.3%) and illumination variation (52.9%) scenarios.

## Related Work
The paper positions itself between two main categories: low-light image enhancement (which treats the problem as an image quality issue before tracking) and domain adaptation (which tries to bridge the daytime-nighttime distribution gap). It also distinguishes itself from prior prompt tuning methods (like DCPT) by explicitly incorporating both illumination and viewpoint information into the feature encoding stage, rather than just adding a single "darkness cue" prompt.

## Limitations
The paper doesn't explicitly state limitations, but the evaluation was conducted on three specific benchmarks (UAVDark135, DarkTrack2021, NAT2021-test), so generalisation to other nighttime UAV tracking scenarios isn't fully verified. The method requires training on a mix of daytime and nighttime data, though less than training-from-scratch approaches. The real-world testing was limited to a Parrot UAV with a fixed camera orientation, so it doesn't capture the full range of dynamic UAV movements.

## Appendix: Worked Example
Let's walk through how DPTracker processes a single frame from a nighttime UAV sequence:

We start with a low-light image from a UAV tracking sequence. The image dimensions are 256×256 (standard for the benchmarks).

1. **Patch embedding**: The image is divided into 16×16 non-overlapping patches (256/16=16 patches per dimension, totaling 256 patches), each projected into 768-dimensional embeddings.

2. **Illumination prompt extraction**: The pyramid illumination prompter (PIP) processes the image:
   - Level 0 (coarsest): Input image (256×256) → Degradation-aware blur kernel (Gaussian approximation) → Blurred image (256×256)
   - Level 1: Input (256×256) → Blurred (128×128) → Laplacian component (128×128)
   - Level 2: Input (128×128) → Blurred (64×64) → Laplacian component (64×64)
   - Level 3: Input (64×64) → Blurred (32×32) → Laplacian component (32×32)
   - The Laplacian components are concatenated channel-wise to form the illumination prompt tokens (512×768 in dimension).

3. **Viewpoint prompt extraction**: The dynamic viewpoint prompter (DVP) processes the image:
   - Standard convolution layer produces coarse viewpoint prompt tokens (16×16×768) from small image patches.
   - Deformable convolution layer predicts offsets (2×K×H×W, K=5 sampling points) and samples features adaptively.
   - Output: Fine-grained viewpoint prompt tokens (16×16×768) capturing geometric variations from dynamic UAV viewpoints.

4. **Prompt-feature interaction**: The DPBlock applies the prompt-conditioned feature adaptation:
   - For each visual feature vector (768-dimensional), the system calculates similarity (E_sim) and difference (E_dif) with the corresponding prompt tokens.
   - Features are updated using learned coefficients: F_current = F_prev + 0.7×E_sim - 0.3×E_dif.
   - Prompt tokens are refined based on the new feature context: P_current = P_prev - 0.6×E_sim + 0.4×E_dif.

The updated features (with illumination and viewpoint awareness) then proceed to the tracker head for bounding box prediction.

See Key Technical Contributions for the implementation details of the prompt-conditioned feature adaptation.

## References

- Yiheng Wang, Changhong Fu, Liangliang Yao, Haobo Zuo, Zijie Zhang, "Dual Prompt-Driven Feature Encoding for Nighttime UAV Tracking", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19628

Tags: #computer-vision #nighttime-tracking #uav-systems #prompt-tuning #feature-encoding #deformable-convolution
