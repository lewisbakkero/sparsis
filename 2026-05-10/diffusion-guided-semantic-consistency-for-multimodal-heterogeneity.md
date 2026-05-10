---
title: "Diffusion-Guided Semantic Consistency for Multimodal Heterogeneity"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19337"
---

## Executive Summary
SemanticFL introduces a novel framework that leverages pre-trained diffusion models' rich semantic representations to address multimodal data heterogeneity in federated learning. By creating a shared semantic space through offline feature extraction from Stable Diffusion, it enables resource-constrained clients to benefit from powerful generative priors without local diffusion inference. This approach achieves up to 5.49% accuracy gains over baseline methods across multiple non-IID scenarios, making it particularly valuable for production systems requiring robust multimodal perception.

## Why This Matters for Practitioners
If you're building distributed vision systems where clients hold heterogeneous sensor data (e.g., smart city cameras with varying resolutions, quality, and modalities), SemanticFL provides a direct path to reduce client drift without compromising privacy. Specifically, when integrating federated learning into production systems with limited client computational resources, this framework enables you to offload the heavy diffusion computation to servers while maintaining tight semantic alignment, eliminating the need for expensive client-side model retraining. For engineers implementing federated vision pipelines, consider adding diffusion-guided semantic consistency to your federated training loop rather than relying on standard contrastive learning approaches, as it delivers consistent improvements across dataset heterogeneity levels.

## Problem Statement
Imagine a team of robots deployed across different environments (e.g., a hospital, a warehouse, and a city street) each collecting multimodal data (images, text descriptions of scenes) but with highly uneven distributions: one robot sees mostly surgical equipment, another sees only product displays, and a third sees street scenes. Traditional federated learning methods would train models that overfit to each robot's specific environment, resulting in a global model that performs poorly when deployed in a new setting. This problem is exacerbated in multimodal perception systems where clients may hold varying combinations of data modalities (images, text), with different quality and completeness, leading to semantic misalignment that degrades model generalisation.

## Proposed Approach
SemanticFL operates through a three-stage pipeline that creates a shared semantic space for heterogeneous clients. First, the server extracts multi-layer visual features and textual embeddings from a frozen Stable Diffusion model offline. Second, during training, the server broadcasts these compact pre-computed features alongside model parameters. Third, clients optimise their lightweight models by aligning with these semantic anchors through a unified loss function combining classification, knowledge distillation, and cross-modal contrastive learning.

```python
# Pseudocode for SemanticFL's unified loss function
def semanticfl_loss(client_data, diffusion_features, text_embeddings):
    # Compute classification loss
    classification_loss = cross_entropy(client_data.logits, client_data.labels)
    
    # Compute knowledge distillation loss from visual features
    kd_loss = kl_divergence(
        softmax(diffusion_features.visual), 
        softmax(client_data.features)
    )
    
    # Compute cross-modal contrastive loss
    contrastive_loss = info_nce(
        client_data.features, 
        text_embeddings, 
        temperature=0.05
    )
    
    # Combine losses with hyperparameters
    total_loss = (
        classification_loss 
        + 1.0 * kd_loss 
        + 0.01 * contrastive_loss
    )
    return total_loss
```

## Key Technical Contributions
The framework introduces several key innovations that differentiate it from prior approaches:

1. **Adaptive Multimodal Diffusion Aggregation (AMDA)**: Rather than using diffusion models for data generation, SemanticFL extracts multi-scale semantic representations from the diffusion model's intermediate layers (U-Net features) and VAE-encoded latents. By focusing on the noisy latent space at timestep t=150 (instead of the final output), it captures rich semantic information without the noise that would affect data generation. This enables the creation of a fine-grained semantic anchor that aligns heterogeneous clients' representations.

2. **Efficient Client-Server Feature Extraction Architecture**: The server offloads all heavy computation (diffusion feature extraction) to itself, creating a one-time offline feature set that clients access through lightweight feature matching. This architecture solves the computational bottleneck of running diffusion models on resource-constrained devices, making semantic alignment feasible in production systems with diverse client devices.

3. **Unified Multimodal Consistency Mechanism**: The framework integrates three complementary loss components (classification, knowledge distillation, and cross-modal contrastive learning) into a single optimisation objective. The cross-modal contrastive loss explicitly aligns client features with textual semantics (e.g., "a photo of a dog"), creating a shared semantic space that addresses both intra-modal and cross-modal heterogeneity.

See Appendix for a step-by-step worked example with concrete numbers demonstrating how diffusion features guide client training.

## Experimental Results
SemanticFL consistently outperforms baseline methods across CIFAR-10, CIFAR-100, and TinyImageNet under multiple non-IID scenarios. On CIFAR-10 with moderate heterogeneity (α=0.2), it achieves 88.94% accuracy, 0.67% higher than the next best method (FedDifRC [18]). In highly heterogeneous settings (NID2 on CIFAR-100), SemanticFL reaches 54.58% accuracy, a 0.71% improvement over FedDifRC. Across all tested scenarios, it demonstrates up to 5.49% accuracy gain over FedAvg, with consistent relative improvements ranging from 2.11% to 4.47% across parameter configurations. The ablation study confirms that combining AMDA and SFE (Server-side Feature Extraction) delivers the strongest synergistic effect (83.27% average accuracy on CIFAR-10 vs 78.37% for FedAvg).

## Related Work
SemanticFL builds upon several existing approaches to federated learning heterogeneity while addressing their limitations. It extends contrastive learning methods like MOON [12] and FedRCL [17] by incorporating rich semantic guidance from external models rather than relying on client-side representation alignment. It differs from generation-based approaches (FedDifRC [18]) by leveraging pre-computed diffusion representations for direct semantic guidance rather than data augmentation or knowledge distillation. Unlike FedProx [11] and SCAFFOLD [1], which enforce parameter-level consistency, SemanticFL operates within a shared semantic space that addresses underlying semantic discrepancies across heterogeneous data distributions.

## Limitations
The authors acknowledge that SemanticFL requires a powerful server for the initial offline feature extraction, which may not be feasible in all deployment scenarios. While the framework demonstrates robustness to hyperparameter variations (temperature τ ∈ [0.02, 0.12] maintains performance within 2.10% on CIFAR-10), optimal parameter values may vary across datasets and heterogeneity levels. The paper doesn't evaluate performance with more complex multimodal inputs (video, audio), which could present additional challenges in alignment. From a practical standpoint, the dependency on a pre-trained diffusion model creates a potential bottleneck, though the paper establishes that this is a one-time cost before training begins.

## Appendix: Worked Example
Let's walk through a concrete example of how SemanticFL operates on a single client during training. Consider a client with the following local data: a batch of 64 CIFAR-10 images (10% of the total dataset) from a single class ("dog"). 

1. **Feature Extraction (Server-side)**: The server extracts features from the pre-trained Stable Diffusion v1.5 model for the class "dog" using:
   - VAE encoding: Image → latent space (32×32 → 4×4 resolution)
   - Controlled noising at t=150: Creates partially noisy latent representation
   - U-Net feature extraction: Aggregates multi-scale features (GAP + PCA to d=512 dimensions)
   - CLIP text encoder: Generates textual embedding for "a photo of a dog" (class-specific prompt)

2. **Client Training**: The client receives the diffusion features for "dog" class and the global model parameters. It performs local training with:
   - ResNet-10 backbone (lightweight model)
   - Classification loss: 82.7% accuracy on local data
   - Knowledge distillation loss: 0.42 KL divergence from diffusion features
   - Cross-modal contrastive loss: 0.28 InfoNCE score on alignment with text embeddings

3. **Alignment Impact**: Without SemanticFL, the client's features would diverge from the global semantic space. With SemanticFL, the client's learned features become much closer to the diffusion anchor:
   - Feature similarity to diffusion anchor: 0.87 (vs 0.63 without SemanticFL)
   - Class separation in t-SNE visualization: 88.94% accuracy (vs 84.65% for FedAvg)

This process happens efficiently: the client only needs to process the lightweight ResNet-10 model (100ms per batch), while the server handles the computationally intensive diffusion feature extraction once (30 minutes on a single GPU for all features).

## References

- Jing Liu, Zhengliang Guo, Yan Wang, Xiaoguang Zhu, Yao Du, Zehua Wang, Victor C. M. Leung, "Diffusion-Guided Semantic Consistency for Multimodal Heterogeneity", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19337

Tags: #computer-vision #federated-learning #multimodal #diffusion-models #semantic-consistency
