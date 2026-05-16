---
title: "Warm-Start Flow Matching for Guaranteed Fast Text/Image Generation"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19360"
---

## Executive Summary
Warm-Start Flow Matching (WS-FM) accelerates flow matching model inference by 2-5x without quality degradation, using a lightweight draft model to produce initial samples that require fewer refinement steps. This approach guarantees speed-ups by starting the flow matching process from an advanced time step (t0 > 0) rather than pure noise, significantly reducing computation in production generative systems.

## Why This Matters for Practitioners
If you're operating text or image generation services using flow matching or diffusion models, WS-FM offers immediate inference speed improvements with minimal implementation effort. For instance, on the Text-8 dataset, your existing DFM model could reduce average sentence generation time from 6.56 seconds to just 1.36 seconds (a 5x speed-up) by adding a small LSTM draft model (4.2M parameters) and fine-tuning your existing flow matching model for a starting time t0=0.8. This requires only modifying your data pipeline to collect draft-refined pairs (using nearest neighbours or a lightweight LLM like Gemma3 for refinement) and slightly adjusting your training protocol - no need to rebuild your entire generative model. The approach delivers measurable latency reduction while maintaining or even improving sample quality, directly translating to lower operational costs and higher throughput for your production services.

## Problem Statement
Imagine trying to assemble a complex puzzle where you start with all pieces scattered randomly across the table (pure noise), then painstakingly move each piece into position (many diffusion steps). WS-FM is like having a partially assembled version of the puzzle already in front of you (from a fast draft model), allowing you to focus only on the final, critical pieces rather than rearranging everything from scratch. Current flow matching models waste computation on transforming random noise into meaningful samples, requiring many steps that become expensive at scale.

## Proposed Approach
WS-FM replaces the traditional pure noise initial distribution with draft samples from a lightweight model, allowing flow matching to start closer to the target distribution (t0 > 0 rather than t0 = 0). The system uses a two-stage pipeline: 
1. Train a fast draft model (e.g., small LSTM for text, DC-GAN for images) on target data
2. Generate draft samples, refine them to match target distribution, and use these as starting points for flow matching

The key insight is that sample generation time from the lightweight model is negligible compared to the flow matching model, so the initial distribution already contains meaningful information about the target distribution, eliminating the need for many initial diffusion steps.

```python
def warm_start_flow_matching(draft_model, fm_model, t0=0.8, num_steps=20):
    # Generate draft samples from lightweight model
    draft_samples = draft_model.generate(num_samples=100)
    
    # Refine draft samples to match target distribution
    refined_samples = refine(draft_samples, target_distribution)
    
    # Collect paired data (draft_sample, refined_sample)
    paired_data = [(draft, refined) for draft, refined in zip(draft_samples, refined_samples)]
    
    # Train or fine-tune flow matching model using paired data
    fm_model.train(paired_data, start_time=t0)
    
    # Generate samples starting from draft distribution at time t0
    samples = fm_model.generate(num_steps=int(num_steps * (1 - t0)))
    return samples
```

## Key Technical Contributions
The paper introduces a novel approach to flow matching inference that fundamentally changes how we think about the initial distribution. Here's how it specifically works:

1. **Draft-to-Refinement Pipeline**: The draft model (e.g., LSTM for text, DC-GAN for images) is trained to produce samples close to the target distribution, but with negligible computational cost compared to the flow matching model. For text, a 4.2M-parameter LSTM generates draft samples in nearly negligible time (0.002s per sentence on their hardware), allowing the system to bypass the first 80% of diffusion steps.

2. **Time-Step Adjustment Mechanism**: Unlike conventional flow matching that starts at t=0, WS-FM starts at a user-defined time t0 (e.g., 0.8) where the draft samples already represent a high-quality distribution. The number of required flow matching steps is reduced from N to N*(1-t0), guaranteeing speed-ups of 1/(1-t0) without quality degradation, as validated in their experiments.

3. **Refinement Strategy Design**: The paper demonstrates two refinement approaches, nearest neighbour search in target data (for images) and LLM-based refinement (for text), that collect paired data (draft, refined) to train the flow matching model on the modified time interval [t0, 1]. This pairing strategy ensures the flow matching model can effectively learn to refine low-quality draft samples to high-quality outputs.

4. **Compatibility with Existing Frameworks**: WS-FM seamlessly integrates with existing discrete flow matching (DFM) frameworks like those described by Gat et al. (2024), requiring only minimal modifications to the training pipeline (see Figure 2) and inference process (Figure 3), making it accessible for immediate adoption.

## Experimental Results
On Text-8 text generation, WS-DFM with t0=0.8 achieved a 5x speed-up (1.36s vs 6.56s) while maintaining comparable next-token NLL scores (6.54 vs 6.58) to the baseline DFM model (see Table 2). For Wikitext-103, WS-DFM with t0=0.8 achieved a 5x speed-up (1.70s vs 8.33s) with improved perplexity (67.86 vs 69.06). The paper reports these improvements as statistically significant, with the authors using GPT-J-6B to measure NLL and perplexity as metrics. Notably, WS-DFM outperformed the Gemma3 27B LLM used for refinement on Text-8 (NLL 6.54 vs 6.54), suggesting the refinement process effectively learns from the draft samples.

## Related Work
The paper positions WS-FM as an improvement over conventional flow matching algorithms that rely on pure noise initial distributions. It acknowledges the work of Lipman et al. (2023a;b), Kim (2025), and Gat et al. (2024) as foundational for flow matching, but identifies their inefficiency in requiring many steps to transform noise into meaningful samples as the key limitation. WS-FM builds on the discrete flow matching framework (DFM) described by Gat et al. (2024), extending it to address the inference efficiency problem without requiring a new theoretical framework.

## Limitations
The paper only tested WS-FM on text and image generation tasks using discrete flow matching, leaving open questions about applicability to continuous-state flow matching or other modalities like audio. The authors acknowledge that the quality of the draft model heavily influences the achievable speed-up, with better draft models requiring fewer refinement steps (e.g., the "pretty good" draft model in the two-moons experiment achieved a 10x speed-up while the "poor" model only achieved 1.5x). The experiments used relatively small datasets (Text-8, Wikitext-103) and limited training data for the refinement process (256K pairs), suggesting potential quality ceilings if scaled to larger datasets.

## Appendix: Worked Example
Let's walk through the text generation process for a single English sentence using WS-DFM on Text-8, with t0=0.8 (5x speed-up):

1. **Draft Model Generation**: A lightweight 4.2M-parameter LSTM generates a draft sentence of 256 characters in 0.002 seconds. The draft might be: "the quick brown fox jumps over the lazy dog" (but with grammatical errors or odd phrasing).

2. **Refinement Strategy**: The draft is refined using a Gemma3 27B LLM with the prompt: "Refine the following text to look more natural and grammatically correct in English. The output needs to contain lowercase letters or spaces only. Please just output the answer: 'the quick brown fox jumps over the lazy dog'."

3. **Refined Output**: The LLM outputs: "the quick brown fox jumps over the lazy dog" (a natural English sentence).

4. **Flow Matching Process**: Instead of starting at t=0 (20 steps for the original DFM), WS-DFM starts at t0=0.8, requiring only 4 steps (20 × 0.2 = 4) to transform the draft into the refined output. Each step takes approximately 0.34 seconds (1.36s total), compared to 6.56 seconds for the original DFM's 20 steps.

5. **Quality Assessment**: The refined output is evaluated by GPT-J-6B, measuring an NLL of 6.54 (compared to the baseline DFM's 6.58), showing no quality degradation despite the 5x speed-up.

## References

- **Code:** https://github.com/ollama/ollama
- Minyoung Kim, "Warm-Start Flow Matching for Guaranteed Fast Text/Image Generation", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19360

Tags: #generative-models #flow-matching #text-generation #image-generation #inference-optimisation
