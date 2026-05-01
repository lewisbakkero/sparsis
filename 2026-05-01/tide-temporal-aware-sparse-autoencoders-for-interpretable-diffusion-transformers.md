---
title: "TIDE: Temporal-Aware Sparse Autoencoders for Interpretable Diffusion Transformers in Image Generation"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/37006"
---

## Executive Summary
TIDE introduces a framework for extracting sparse, interpretable activation features across timesteps in Diffusion Transformers (DiTs), revealing how these models inherently learn hierarchical semantics (3D structure, object class, and fine-grained concepts) during pretraining. This enables practical applications like safe image editing and style transfer while maintaining reasonable generation quality.

## Why This Matters for Practitioners
If you're maintaining a production image generation system based on DiTs, understanding internal features is crucial for implementing safe, controllable editing features. TIDE allows you to identify and manipulate specific visual elements (like "shadow" for depth estimation or "beak" for eagle classification) without compromising generation quality. For instance, you could implement a "style eraser" that removes Rococo architectural elements while preserving the underlying image content, a capability previously impossible with black-box diffusion models. The minimal quality degradation (FID increase of only +0.15 with default settings) means you can integrate this without significantly impacting your user experience.

## Problem Statement
Imagine trying to fix a car engine's timing system by only observing the dashboard lights, your engine might run, but you'd never know which components were malfunctioning. Similarly, Diffusion Transformers (DiTs) produce high-quality images but remain fundamentally opaque, making it impossible to understand why certain visual elements appear or to reliably modify them. Existing interpretation tools built for U-Net models don't transfer to DiTs because they depend on spatially structured feature maps, which DiTs don't use. This creates a critical blind spot for engineers who need to build trustworthy, controllable generative systems.

## Proposed Approach
TIDE adapts sparse autoencoders (SAEs) to DiTs by incorporating temporal awareness into the architecture. It extracts activations from specific transformer layers during both forward diffusion and reverse denoising processes, then uses these to train specialized SAEs. The framework applies timestep-dependent modulation to handle time-varying activation patterns across diffusion steps, enabling more accurate reconstruction while maintaining sparsity.

```python
def tide_sa_autoencoder(activation, timestep):
    # Apply temporal modulation using lightweight MLPs initialized from DiT's adaptive LayerNorm
    scale = mlpscale(timestep)
    shift = mlpshift(timestep)
    modulated_activation = activation * (1 + scale) + shift
    
    # Apply TopK sparsity constraint
    sparse_activation = topk(modulated_activation, k=sparsity_ratio)
    
    # Reconstruct activation
    reconstructed = decoder(sparse_activation)
    
    return reconstructed
```

## Key Technical Contributions
TIDE's core innovations specifically address DiT's unique challenges:

1. **Temporal-Aware Architecture**: Unlike standard SAEs, TIDE incorporates timestep-dependent modulation (xmod = x · (1 + scale(t)) + shift(t)) directly into the SAE architecture. The scale and shift functions use lightweight MLPs initialized from DiT's pre-trained adaptive LayerNorm, allowing the model to adaptively align activations across different diffusion timesteps without requiring substantial parameter modifications.

2. **Token-Level Sampling Strategy**: TIDE uses random sampling of 1/16 of tokens at the token level during training, which significantly improves training efficiency and generalisation. As Table 1 shows, this sampling reduces MSE from 3.3e-3 to 2.5e-3 while boosting cosine similarity on OOD validation sets from 0.935 to 0.962. This focuses learning on more informative local features while mitigating global noise.

3. **Dead Latent Prevention**: TIDE prevents dead latents (non-activating latent variables) through careful initialization (encoder initialized as transpose of decoder) and periodic revival of inactive latents. This ensures the latent space remains expressive and maintains the sparsity-reconstruction trade-off even with high-dimensional activations.

## Experimental Results
The authors evaluated TIDE using MSE, cosine similarity, and diffusion loss. TIDE achieved an MSE of 3.1e-3 and cosine similarity of 0.970 (Table 1), outperforming standard SAEs (MSE 3.3e-3, cosine similarity 0.968). With token sampling, TIDE's MSE dropped to 2.5e-3 with cosine similarity 0.972. The diffusion loss comparison (Figure 4c-d) shows TIDE's reconstructed features achieve diffusion loss comparable to the original model, surpassing other SAE-based methods. For practical applications, the FID increase was minimal (+0.15 with default configuration), while maintaining strong semantic alignment (AlignScore 0.647 vs baseline 0.688).

## Related Work
TIDE builds on sparse autoencoder (SAE) work for interpretability in language models (Ng et al. 2011; Bricken et al. 2023) and vision-language models (Daujotas 2024a), but specifically addresses DiT's unique challenges: time-varying activation patterns, high token dimensionality, and multimodal input alignment. Unlike previous SAE applications to diffusion models (Ijishakin et al. 2024; Surkov et al. 2024), TIDE incorporates temporal dynamics and token-level sampling to handle DiT's iterative denoising process.

## Limitations
The paper doesn't test TIDE on extremely high-resolution images beyond 1024x1024 or on models with significantly different architectures. The authors acknowledge that the current implementation is limited to single-image generation rather than video or sequential content. The minimal quality degradation (+0.15 FID increase) is acceptable for most applications, but engineering teams should verify this trade-off in their specific use cases. The evaluation focuses on standard benchmarks, so real-world performance on specialized tasks may vary.

## Appendix: Worked Example
Let's walk through a concrete example from the paper's methodology using the default configuration (5% sparsity, token sampling, 16d latent dimension). Consider an image of an eagle with the text prompt "a majestic eagle flying over mountains" at timestep t=100 (out of 1000 total timesteps):

1. **Activation Extraction**: The DiT model extracts activations from the penultimate layer, resulting in a 73728-dimensional vector (16d = 73728) for the eagle image.
   
2. **Token-Level Sampling**: TIDE randomly samples 1/16 of the tokens (4608 tokens out of 73728), focusing on the most informative local features related to the eagle's "beak" and "wing" regions.

3. **Temporal Modulation**: The lightweight MLPs compute scale(t=100) = 0.25 and shift(t=100) = 0.1 based on the timestep. The modulated activation becomes: activation * (1 + 0.25) + 0.1 = activation * 1.25 + 0.1.

4. **Top-K Sparsity**: The top 5% of activations (3686 tokens) are selected based on magnitude, effectively isolating features like "beak," "wing," and "feathers."

5. **Reconstruction**: The decoder reconstructs the activation from these 3686 sparse features, achieving an MSE of 3.1e-3 (Table 1) and cosine similarity of 0.970.

6. **Editing Application**: To remove the eagle's "beak" feature, we zero out the relevant top-K tokens. This modifies the latent space without creating out-of-distribution artifacts, as shown in Figure 5, allowing precise feature manipulation for applications like safe image editing.

See Appendix for a step-by-step worked example with concrete numbers.

## References

- Victor Shea-Jay Huang, Le Zhuo, Yi Xin, Zhaokai Wang, Fu-Yun Wang, Yuchi Wang, "TIDE: Temporal-Aware Sparse Autoencoders for Interpretable Diffusion Transformers in Image Generation", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/37006

Tags: #computer-vision #generative-models #model-interpretability #sparse-autoencoders #temporal-aware-learning
