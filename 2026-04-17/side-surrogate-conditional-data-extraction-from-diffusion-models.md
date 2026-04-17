---
title: "SIDE: Surrogate Conditional Data Extraction from Diffusion Models"
category: "AI Applications"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36972"
---

## Executive Summary

SIDE introduces a method to extract training data from both conditional and unconditional diffusion models, challenging the assumption that unconditional models are safe from data extraction attacks. This is critical for practitioners because it reveals a fundamental vulnerability in diffusion models used across production systems, requiring stronger privacy measures regardless of model conditioning.

## Why This Matters for Practitioners

If you're deploying a diffusion model in production without explicit conditioning (e.g., for image generation tasks), this paper demonstrates that your model's training data is vulnerable to extraction attacks, just like conditional models. This means you can no longer rely on "unconditional" models as a privacy safeguard. For engineers: immediately review your model's privacy profile using the authors' memorisation divergence metrics, and implement anti-memorisation guidance during sampling (as described in Chen et al. 2024) rather than relying on conditioning alone. If you're using Stable Diffusion 1.5 or similar models, prioritise fine-tuning with LoRA adapters as a privacy countermeasure, as the paper shows this can reduce vulnerability to extraction attacks.

## Problem Statement

Imagine building a house with a door that appears locked from the outside, but the architects never considered that someone could simply climb through the window. That's the current state of diffusion model privacy: conditional models were thought to be "locked" (with explicit prompts as a security measure), while unconditional models were thought to be "windowless" (safe by default). SIDE reveals that the window wasn't closed, it was just hidden.

## Proposed Approach

SIDE constructs surrogate conditions from the model's own internal structure to guide extraction. The process involves generating synthetic images from the target model, clustering these images using a pre-trained feature extractor, and using the cluster centroids as surrogate conditions to steer the diffusion process toward memorized samples. For small models, it trains a time-dependent classifier; for large models like Stable Diffusion, it uses LoRA fine-tuning to adapt the model itself.

```python
def SIDE(dpm, feature_extractor, clusters, guidance_scale=1.0):
    # Generate synthetic dataset
    synthetic_images = generate_images(dpm, num_samples=50000)
    features = feature_extractor(synthetic_images)
    
    # Cluster and filter clusters
    clusters = kmeans(features, n_clusters=100)
    valid_clusters = filter_clusters(clusters, cohesion_threshold=0.5)
    
    # Create surrogate conditions
    surrogate_conditions = [cluster.centroid for cluster in valid_clusters]
    
    # Train conditional guidance
    if dpm is small:
        classifier = train_classifier(synthetic_images, surrogate_conditions)
    else:
        dpm = lora_finetune(dpm, synthetic_images, surrogate_conditions)
    
    # Extract with surrogate conditions
    extracted_images = []
    for _ in range(51200):
        target_cluster = random.choice(surrogate_conditions)
        image = denoise_with_guidance(dpm, target_cluster, guidance_scale)
        extracted_images.append(image)
    
    return extracted_images
```

## Key Technical Contributions

SIDE's core innovation isn't just creating surrogate conditions, it's how these conditions align with the model's internal representation of data. The authors make three key technical contributions that directly challenge previous assumptions.

1. **Implicit Label Construction via Clustering**: The approach doesn't require access to the original training data. By clustering synthetic images generated from the model using a pre-trained feature extractor (ResNet34 in the paper), SIDE identifies high-cohesion clusters (measured by cosine similarity) that represent memorised data regions. The paper specifies a cohesion threshold of 0.5 to filter clusters, ensuring only high-quality clusters serve as surrogate conditions. This is fundamentally different from previous work that required explicit prompts or class labels to guide extraction.

2. **Dual Guidance Mechanisms for Model Scale**: For small-scale diffusion models (like those trained on CIFAR-10), SIDE trains a time-dependent classifier to provide gradient guidance. For large models like Stable Diffusion, it uses LoRA fine-tuning instead of training a separate classifier, which is computationally prohibitive. The paper shows LoRA fine-tuning with rank 512 achieves superior results, demonstrating that direct model adaptation is more effective than external guidance for large-scale systems.

3. **Theoretical Framework for Conditioning and Memorisation**: The authors derive a KL divergence-based memorisation measure (Definition 1) that quantifies how closely a model's distribution aligns with the training data distribution. Their proof (Theorem 1) demonstrates that conditioning on informative labels, whether explicit (like text prompts) or implicit (like cluster information), amplifies memorisation. This explains why SIDE works: it creates "informative" labels through clustering, which the model has internally encoded for memorisation.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results

SIDE outperforms both unconditional (Carlini UnCond) and conditional (Carlini Cond) baselines across six datasets. On CelebA-25000, SIDE achieves a low-similarity AMS of 20.527% compared to Carlini Cond's 8.712% (Table 1). For high-similarity extraction (SSCD > 0.6), SIDE achieves 0.030% UMS on CelebA-25000 versus 0.010% for Carlini Cond. The paper reports these results for 51,200 generated images on CelebA-HQ-FI and 512,000 on LAION-5B. Statistical significance isn't explicitly stated, but the consistent outperformance across all datasets and similarity levels suggests strong evidence for SIDE's effectiveness.

## Related Work

SIDE builds on recent work showing conditional DPMs are highly susceptible to data extraction attacks (Carlini et al. 2023), but challenges the assumption that unconditional models are safe. It improves upon Carlini's baselines by demonstrating that surrogate conditions can outperform explicit prompts for conditional models and enable extraction from unconditional models. The authors also extend prior memorisation analysis (Somepalli et al. 2023) with a theoretical framework explaining why conditioning amplifies memorisation, regardless of whether the conditioning is explicit or surrogate.

## Limitations

The authors acknowledge SIDE doesn't address privacy in black-box settings (though they note a black-box extension in the appendix). The paper focuses on image extraction with no evaluation on text or multimodal models. I'm particularly concerned that the authors don't address whether their extraction method can be detected or mitigated, it's purely a vulnerability demonstration. For production systems, this means SIDE reveals a threat but doesn't provide a complete solution for preventing extraction.

## Appendix: Worked Example

Let's walk through SIDE's process using the CelebA-25000 dataset from the paper. We'll focus on extracting a single image from the "high-similarity" category (SSCD > 0.6).

1. **Synthetic dataset generation**: The paper generates 50,000 synthetic images from the target diffusion model (DDIM scheduler with batch size 64). For CelebA-25000, this takes approximately 1,000 epochs.

2. **Feature extraction and clustering**: Using a ResNet34 feature extractor, these images are converted to 512-dimensional vectors. The paper uses KMeans with 100 clusters (SSCD feature extractor with 100 clusters). After clustering, they remove low-cohesion clusters using a cosine similarity threshold of 0.5.

3. **Cluster centroid processing**: Suppose cluster 47 has 421 images with high cohesion (cosine similarity of 0.55 to centroid). This cluster represents a group of similar faces (e.g., "blonde women with glasses"). The centroid vector for this cluster becomes our surrogate condition.

4. **Guidance application**: During extraction, SIDE selects target cluster 47 randomly. For each denoising step (t from T down to 1), it applies guidance:
   - For small models: `sguided = sθ(xt, t) + λ · ∇xt log pϕ(47 | xt)`
   - For Stable Diffusion (large model): It uses LoRA fine-tuning with rank 512, so `sguided = sθ+Δθ(xt, t, 47)`

5. **Similarity measurement**: The extracted image is compared to the training set using SSDC. When the similarity score exceeds 0.6, it's classified as high-similarity extraction. On CelebA-25000, SIDE achieves 0.030% UMS for high-similarity extraction (Table 1), meaning 0.030% of the generated images closely match training samples at the highest similarity level.


## References
Tags: #diffusion-models #data-extraction #privacy-security #memorisation
- Yunhao Chen, Shuejie Wang, Difan Zou, "SIDE: Surrogate Conditional Data Extraction from Diffusion Models", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36972
