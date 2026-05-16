---
title: "Enhancing Alignment for Unified Multimodal Models via Semantically-Grounded Supervision"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19807"
---

## Executive Summary
Semantically-Grounded Supervision (SeGroS) resolves the granularity mismatch between sparse text prompts and dense visual tokens in Unified Multimodal Models (UMMs) by constructing two complementary supervision signals: semantic visual hints and semantically-grounded corrupted input. This fine-tuning framework significantly improves generation fidelity and cross-modal alignment across various UMM architectures without requiring architectural changes to the base models.

## Why This Matters for Practitioners
If you're implementing text-to-image generation pipelines in production systems, you're likely encountering subtle alignment issues where models produce images that match the general concept but miss critical visual details. For example, a prompt like "a young child sitting near pita bread next to a tree" might generate a scene with an accurate child but incorrect pita bread texture or tree position. SeGroS addresses this by ensuring reconstruction loss focuses on semantically relevant regions rather than background details, which means you can achieve higher quality outputs with fewer training iterations. Crucially, SeGroS can be applied as a fine-tuning step to existing UMMs like Show-o, Harmon, or OpenUni without architectural modifications, making it immediately applicable to production systems. For instance, on Harmon-1.5B, SeGroS improved CompBench scores by 1.46 points (from 87.2 to 88.66) without increasing inference latency.

## Problem Statement
Imagine directing a chef to prepare a specific dish using only vague instructions like "something with chicken and vegetables" instead of precise details like "chicken stir-fry with broccoli and carrots." The chef would have to make countless assumptions about ingredients, cooking style, and presentation, leading to inconsistent results. Similarly, current UMMs suffer from a granularity mismatch where text prompts provide sparse semantic constraints while the model must reconstruct dense visual tokens. This forces models to fit incidental visual details rather than learning robust semantic alignment, resulting in generations that match the general concept but miss specific visual details.

## Proposed Approach
SeGroS constructs a visual grounding map that quantifies the alignment between text tokens and image patches to create two complementary supervision signals. The framework consists of three main steps: (1) discriminative text token filtering to identify linguistically salient text tokens with strong visual correspondences, (2) constructing a visual grounding map that measures similarity between filtered text tokens and image patches, and (3) using the map to generate semantic visual hints for conditioning and semantically-grounded corrupted input for reconstruction targets.

```python
def semantically_ground_supervision(text_prompt, image):
    # Step 1: Discriminative text token filtering
    text_embeddings = tokenize(text_prompt)
    intra_modal_affinity = compute_intra_modal_affinity(text_embeddings)
    inter_modal_affinity = compute_inter_modal_affinity(text_embeddings, image_embeddings)
    discriminative_tokens = select_top_k_tokens(intra_modal_affinity, inter_modal_affinity, ratio=0.4)
    
    # Step 2: Visual grounding map construction
    grounding_map = compute_grounding_map(discriminative_tokens, image_embeddings)
    
    # Step 3: Construct complementary signals
    visual_hints = get_top_k_regions(grounding_map, ratio=0.3)
    corrupted_input = get_bottom_k_regions(grounding_map, mask_ratio=0.7)
    
    return visual_hints, corrupted_input
```

## Key Technical Contributions
SeGroS introduces several novel mechanisms to address the granularity mismatch in UMM training. Here's how these mechanisms work at the implementation level:

1. **Discriminative Text Token Filtering**: SeGroS filters for tokens that are both linguistically salient within the text prompt (measured by intra-modal affinity) and visually grounded in the image (measured by inter-modal affinity). For each text token, the intra-modal affinity score is calculated as the sum of attention weights it receives from all other tokens in the sequence, while the inter-modal affinity score is the sum of attention weights the token receives from visual tokens. These scores are normalized and summed to determine each token's discriminative importance score, with the top 40% of tokens (determined by the text preservation ratio ρ=0.4) selected for further processing.

2. **Visual Grounding Map Construction**: The grounding map measures patch-level similarity between filtered text tokens and visual features. For each image patch i, the grounding score mi is computed as the weighted sum of text-to-image attention probabilities, normalized to [0,1]. To prevent overfitting to the same regions across epochs, a small uniform noise ξ ~ U([0,0.5]) is added to the normalized scores before selecting regions, ensuring different regions are masked in different training iterations.

3. **Semantically-Selective Construction of Training Signals**: Unlike previous approaches that use all visual tokens as hints or apply random masking, SeGroS selects high-grounding regions for visual hints (top 30-40% of regions) and low-grounding regions for unmasked context (bottom 50-70% of regions), with the remaining regions (core semantic areas) masked for reconstruction. This ensures reconstruction loss concentrates on semantically relevant areas while reducing the impact of background noise.

4. **Joint Objective for Unified Training**: The final objective combines a reconstruction loss on the semantically-grounded corrupted input (with visual hints providing additional conditioning) and a standard I2T loss to preserve multimodal understanding capabilities. The reconstruction loss is evaluated only on masked regions, while the I2T loss ensures the model maintains the ability to generate text descriptions from images.

## Experimental Results
SeGroS was evaluated across three UMM families (Show-o, Harmon, OpenUni) on GenEval, DPGBench, and CompBench. On Show-o-256, SeGroS improved GenEval by 0.7 points (98.1 vs 97.4), DPGBench by 4.3 points (72.5 vs 68.2), and CompBench by 4.6 points (62.22 vs 57.6) over the base model. The improvements were consistent across different model sizes and resolutions: Harmon-1.5B achieved 88.66 on CompBench compared to Reca's 87.2 (88.66 vs 87.2), representing a 1.46-point improvement. The authors conducted ablation studies showing that selectively using only the top 30% of visual hints (compared to all visual tokens) improved GenEval from 78.7→79.2 on Harmon 0.5B, validating their approach to reduce supervisory redundancy.

## Related Work
SeGroS builds on recent work like Reca [43], which introduced image-conditioned training for UMMs, but addresses its key limitation of supervisory redundancy by providing semantically selective conditioning. Unlike Show-o [44] and Harmon [39], which rely on image reconstruction from random masking, SeGroS explicitly constructs training signals that align with the text prompt's semantic content. The paper positions itself as addressing the fundamental granularity mismatch that other approaches have overlooked, demonstrating that even small improvements in supervision quality can lead to significant gains in generation fidelity.

## Limitations
The authors do not test SeGroS on larger models beyond 3.6B parameters or on tasks requiring complex compositionality beyond the benchmarks used. They also don't specify how the framework scales to extremely long text prompts or images with very high resolution. While the method is architecture-agnostic, it might require adjustment for models with very different tokenization schemes. The paper doesn't address potential issues with the noise injection in grounding map construction, though the authors state that uniform noise of ξ ~ U([0,0.5]) was sufficient to prevent overfitting to fixed regions.

## Appendix: Worked Example
Let's walk through an example with a text prompt "a young child is sitting near some pita bread next to a tree" and an image of this scene. First, the discriminative text token filtering identifies "child," "pita bread," and "tree" as having high intra-modal affinity (they're central to the scene) and high inter-modal affinity (they're visually distinct in the image). For the visual grounding map, each image patch is assigned a grounding score based on its similarity to these filtered text tokens. The top 30% of patches (highest scores) are selected as visual hints, preserving details of the child, pita bread, and tree. The bottom 50% of patches (lower scores) are retained as unmasked context (background elements like grass and sky). The remaining 20% of patches (core semantic areas) are masked for reconstruction. During training, the model focuses on reconstructing the child, pita bread, and tree while using the visual hints to guide the generation. This ensures reconstruction loss is concentrated on semantically relevant regions rather than background details, as demonstrated in Table 1 where SeGroS improved CompBench scores by 4.6 points over the base model.

## References

- Jiyeong Kim, Yerim So, Hyesong Choi, Uiwon Hwang, Dongbo Min, "Enhancing Alignment for Unified Multimodal Models via Semantically-Grounded Supervision", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19807

Tags: #multimodal-models #text-to-image-generation #cross-modal-alignment #semantically-grounded-supervision #fine-tuning
