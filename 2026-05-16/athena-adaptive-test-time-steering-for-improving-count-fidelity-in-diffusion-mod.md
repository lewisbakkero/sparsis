---
title: "ATHENA: Adaptive Test-Time Steering for Improving Count Fidelity in Diffusion Models"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19676"
---

## Executive Summary
ATHENA is a model-agnostic framework that improves object count fidelity in text-to-image diffusion models during test time without requiring architectural changes or retraining. It addresses a persistent issue where diffusion models fail to generate images with specified object counts (e.g., "eight cows"), which undermines applications like synthetic data generation for object detection. For engineers building production systems using diffusion models, this means achieving more precise control over image generation with minimal overhead.

## Why This Matters for Practitioners
If you're building production systems that utilise text-to-image diffusion models for generating synthetic data in object detection pipelines, this paper suggests you can significantly improve count accuracy without additional training overhead. For example, if your current system generates images with 7 cows when asked for 8, you can implement ATHENA to achieve the correct count by adding a single additional forward pass for count estimation, avoiding costly retraining of your diffusion models. The framework requires only minor pipeline modifications, with minimal impact on runtime (up to 2.5× faster than competing baselines), making it practical for integration into existing production systems.

## Problem Statement
Imagine trying to assemble a Lego tower with specific numbers of each colour block, but the instructions only say "make a tower" without specifying quantities. Early decisions about which blocks to place affect the final composition, making it impossible to fix later without dismantling the whole structure. Similarly, diffusion models make early stochastic decisions about object count that become difficult to correct later, causing systematic errors in count fidelity.

## Proposed Approach
ATHENA intervenes early in the diffusion sampling process by estimating object counts from intermediate representations and applying count-aware noise corrections before structural errors become difficult to revise. It works by evaluating the denoiser twice during sampling - once with the original prompt and once with a control prompt - then combining these evaluations to steer the generation trajectory. The framework offers three progressively more advanced variants:

- ATHENA-Static: Applies fixed steering using a count-agnostic control prompt
- ATHENA-Feedback: Uses intermediate count estimates to inform the control prompt
- ATHENA-Adaptive: Adaptively adjusts steering strength based on error direction

```python
def athena_adaptive(diffusion_model, prompt, target_count):
    # Estimate count at intermediate diffusion step
    count_estimate = estimate_count(diffusion_model, prompt, step=50)
    
    # If estimate doesn't match target, apply initial steering
    if count_estimate != target_count:
        # Apply initial steering
        steered_image = apply_steering(diffusion_model, prompt, target_count, gamma=0.5)
        
        # Get new count estimate
        new_count_estimate = estimate_count(diffusion_model, prompt, image=steered_image, step=50)
        
        # Adjust steering strength based on error direction
        if new_count_estimate == target_count:
            return steered_image
        elif (new_count_estimate > target_count) == (count_estimate > target_count):
            # Error direction remains unchanged, increase steering strength
            gamma = 1.0
        else:
            # Error direction flipped, decrease steering strength
            gamma = 0.25
            
        # Apply final steering with adjusted strength
        return apply_steering(diffusion_model, prompt, target_count, gamma=gamma)
    else:
        # Already correct, no need for steering
        return generate_image(diffusion_model, prompt)
```

## Key Technical Contributions
ATHENA's core innovation lies in its ability to improve count fidelity through adaptive test-time steering without model modifications. The key technical contributions include:

1. **Early intervention with count-aware noise correction**: Unlike previous methods that apply corrections late in the sampling process, ATHENA estimates object counts from intermediate representations at a fixed step (as shown in Figure 2) and applies count-aware noise corrections early in the denoising trajectory. This prevents structural errors from becoming irreversible, leveraging the observation that early stochastic decisions largely determine object multiplicity and spatial layout.

2. **Prompt-based steering mechanism**: ATHENA's steering mechanism avoids gradient-based optimisation or model-specific internals by modifying the prompt condition during sampling. It evaluates the denoiser twice - once with the original prompt and once with a control prompt - combining these evaluations to form a steered noise estimate. This approach is model-agnostic, requires no additional training, and preserves visual quality while enabling corrective control.

3. **Adaptive steering strength adjustment**: ATHENA-Adaptive, the most effective variant, uses a single feedback step to adjust steering strength based on error direction. After initial steering, if the count still doesn't match the target, the method analyses whether the error direction has changed to determine whether to increase or decrease steering strength. This approach requires at most two steered trajectories and avoids the computational burden of iterative optimisation, while maintaining strong count accuracy.

## Experimental Results
ATHENA was evaluated across three diffusion backbones (SDXL, SD 3.5 Large, FLUX.1-dev) using three datasets (CoCoCount, CoCoCount-E, and the new ATHENA dataset). The key results include:

- On FLUX.1-dev, ATHENA-Adaptive improved count accuracy by 15.2 percentage points (from 39.4% to 54.0%) compared to unsteered sampling on the CoCoCount dataset.
- Across all models and datasets, the method achieved up to a 22% improvement in count fidelity over the unsteered baseline.
- ATHENA-Adaptive reduced memory usage by approximately 4× and achieved up to 2.5× faster image generation relative to Counting Guidance (a prominent baseline).
- On the ATHENA dataset (with four levels of increasing complexity), ATHENA-Adaptive achieved 56.1% accuracy on SD 3.5 Large compared to 38.9% for unsteered sampling.

## Related Work
ATHENA builds upon and improves over previous methods for count control in diffusion models. Unlike CountGen, which requires training an auxiliary layout-modification model specific to SDXL, ATHENA is model-agnostic and requires no additional training. Compared to Counting Guidance, which applies classifier guidance at each denoising step, ATHENA avoids gradient-based optimisation and preserves visual quality. While CountCluster enforces count control through early cross-attention, it assumes uncluttered scenes and fails with overlapping objects. ATHENA's key advancement is its ability to provide adaptive, test-time steering across diverse diffusion backbones with minimal computational overhead.

## Limitations
The authors acknowledge that ATHENA currently focuses on object count fidelity for specific categories specified in the prompt, and doesn't address more complex relational constraints (though the ATHENA dataset includes some relational prompts). The method also relies on external count estimation (using GroundingDINO), which may introduce additional latency in production systems. The paper doesn't explore how ATHENA performs with extremely high target counts (beyond 10), though the authors note that prior work reports degradation beyond this range. Additionally, the framework has not been evaluated in production environments with varying hardware configurations, which could affect the precise runtime characteristics.

## Appendix: Worked Example
Let's walk through ATHENA-Adaptive for a prompt "five apples on a table" where the target count is 5:

1. The diffusion process runs for 50 steps (estimation_step) without steering.
2. At step 50, the latent representation is decoded, and GroundingDINO estimates the object count as 3 (the model generated too few apples).
3. The framework compares the estimated count (3) to the target (5) and applies initial steering with γ = 0.5.
4. After steering, the count estimate becomes 4 (still too low, but closer to target).
5. The error direction has not changed (both estimates were below target), so the steering strength is doubled to γ = 1.0.
6. The framework then applies the final steering with γ = 1.0 for the remaining steps (steering_horizon = 20).
7. The final image shows exactly 5 apples on the table, as intended.

This process requires only two additional forward passes for count estimation and steering (compared to one for the baseline), with minimal impact on the overall generation time (as shown in Table 2, the runtimes remain within acceptable ranges for production use).

## References

- **Code:** https://github.com/MShahabSepehri/ATHENA.
- Mohammad Shahab Sepehri, Asal Mehradfar, Berk Tinaz, Salman Avestimehr, Mahdi Soltanolkotabi, "ATHENA: Adaptive Test-Time Steering for Improving Count Fidelity in Diffusion Models", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19676

Tags: #computer-vision #generative-models #diffusion-models #test-time-steering #count-fidelity
