---
title: "Guiding Diffusion-based Reconstruction with Contrastive Signals for Balanced Visual Representation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.04803"
---

## Executive Summary
DCR (Diffusion Contrastive Reconstruction) enhances CLIP's visual representations by simultaneously improving discriminative ability (D-Ability) and detail perceptual ability (P-Ability). This addresses a critical bottleneck where diffusion-based reconstruction methods improve fine-grained detail understanding but degrade class separability. For production engineers, this means achieving better performance on both recognition and detailed visual reasoning tasks without needing separate models.

## Why This Matters for Practitioners
For engineers deploying CLIP-based vision systems in production, this paper reveals a fundamental trade-off: diffusion-based reconstruction methods (which improve P-Ability) often degrade D-Ability (class separability), leading to suboptimal performance on recognition tasks. If you're using CLIP for multi-modal LLMs, object recognition, or detailed visual question answering, you should consider fine-tuning your CLIP backbone with DCR rather than relying on existing diffusion-based reconstruction methods. The implementation requires only two training stages using existing diffusion models as the foundation, with minimal additional computational overhead. Specifically, replace your current diffusion-based reconstruction pipeline with DCR's two-stage protocol: first align the projector (Stage 1), then enhance the visual encoder (Stage 2). This approach avoids the gradient conflicts that plague naive combinations of contrastive learning and reconstruction.

## Problem Statement
Current CLIP-based vision systems are like a chef who can perfectly replicate a dish (P-Ability) but can't distinguish between similar dishes (D-Ability). When an image's subtle details change (e.g., a snowman with a black hat versus a silver hat), the system can reproduce the fine details flawlessly but fails to recognise the same object as a distinct category. This leads to systems that excel at answering detailed questions about image content but struggle with basic object recognition when visual variations occur. The CLIP model's limited understanding capacity creates a bottleneck where visual representations that excel at one task (e.g., detailed visual reasoning) underperform at another (e.g., category recognition), forcing engineers to choose between specialized models rather than having a balanced solution.

## Proposed Approach
DCR integrates contrastive learning directly into the diffusion reconstruction process by injecting contrastive signals derived from reconstructed images rather than original inputs. This creates a single unified optimisation objective that naturally balances D-Ability and P-Ability without gradient conflicts. The approach involves two training stages: first aligning the projector (Stage 1), then enhancing the visual encoder (Stage 2). Instead of adding contrastive learning on top of reconstruction (which causes gradient conflicts), DCR performs contrastive learning on the predicted noise from the diffusion process.

```python
def dcr_loss(anchor_noise, positive_noise, negative_noises):
    """DCR loss implementation as described in the paper."""
    # Compute cosine similarity between anchor and other noises
    def cosine_sim(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    
    # Compute contrastive loss using predicted noise
    loss = 0
    for p in [positive_noise, gt_noise]:  # gt_noise is ground-truth diffusion noise
        numerator = np.exp(cosine_sim(anchor_noise, p) / temperature)
        denominator = 0
        for n in negative_noises + [p]:
            denominator += np.exp(cosine_sim(anchor_noise, n) / temperature)
        loss -= np.log(numerator / denominator)
    return -loss / (len([positive_noise, gt_noise]))
```

## Key Technical Contributions
The authors make three key technical contributions beyond the high-level approach:

1. **Gradient conflict analysis**: The authors demonstrate that 86.3% of training steps exhibit negative cosine similarity between gradients of contrastive and reconstruction objectives, indicating pervasive gradient conflict. This analysis explains why naive combinations of contrastive learning and reconstruction fail to balance D-Ability and P-Ability.

2. **Contrastive signals on reconstructed images**: DCR injects contrastive signals derived from reconstructed images rather than original inputs. For each image, the anchor is the reconstruction from its own features, the positive is the reconstruction from an augmented view, and negatives are reconstructions from other images in the batch. This design creates a single optimisation objective that naturally balances both capabilities.

3. **Two-stage training protocol**: The method uses a two-stage schedule to transfer diffusion knowledge into the vision encoder gradually. Stage 1 aligns the projector (freezing the visual encoder), while Stage 2 enhances the visual encoder (freezing the projector). This prevents gradient interference during training and ensures the vision encoder learns from the diffusion model's optimisation.

## Experimental Results
DCR was evaluated across six CLIP backbones and multiple vision benchmarks. On the MMVP-VLM benchmark for P-Ability (Table 1), DCR outperformed all baselines: on OpenAI ViT-L-14@224, DCR achieved 33.3% average performance compared to 32.6% for un2CLIP (best baseline). The improvement was consistent across multiple visual patterns like orientation (☼), presence of specific features (Û), and quantity ().

For D-Ability, DCR achieved higher scores on six standard zero-shot clustering benchmarks (Table 2). On SigLIP ViT-SO-14, DCR achieved 0.85 NMI, 0.89 ACC, and 0.78 ARI compared to GenHancer's 0.80 NMI, 0.78 ACC, and 0.70 ARI. The authors also provided qualitative evidence in Figure 4 showing DCR's improved ability to distinguish snowmen with different hat colours.

## Related Work
The paper positions DCR as addressing a gap in existing representation learning methods. Prior work focused on either improving D-Ability (e.g., using contrastive learning) or P-Ability (e.g., using diffusion-based reconstruction), but failed to balance both. The authors compared with DIVA, GenHancer, and un2CLIP, which all focus on improving P-Ability through diffusion-based reconstruction but neglect D-Ability, leading to suboptimal representations. GenHancer and un2CLIP also require retraining specialized diffusion models from scratch, incurring substantial computational costs. DCR, in contrast, leverages existing pretrained diffusion models as the foundation, dramatically reducing training overhead while achieving a better balance between capabilities.

## Limitations
The paper acknowledges that DCR requires access to a pre-trained diffusion model (like Stable Diffusion), though it can leverage existing ones without retraining. The method also requires two training stages, though the authors note this is minimal compared to the cost of retraining diffusion models. The paper doesn't explicitly discuss the impact on inference latency, though the authors note that the diffusion model remains frozen during DCR training, which should minimise inference overhead. The authors also don't test DCR on extremely large-scale benchmarks with millions of images, though their experiments on standard benchmarks provide strong evidence of effectiveness.

## Appendix: Worked Example
Let's walk through the DCR process for an image of a snowman with a black hat. The CLIP visual encoder processes this image into a feature vector z. In Stage 1 (projector alignment), we condition the diffusion model on z to reconstruct the image, aligning the visual guidance with the diffusion model's text guidance. In Stage 2 (encoder enhancement), we augment the image with random crop (showing more of the hat's black colour) to create x+, and condition the diffusion model on fϕ(x+) (the CLIP feature of the augmented image) to predict noise ˆϵ+. The original image becomes the anchor (ˆϵ), the augmented image becomes the positive (ˆϵ+), and other images in the batch (e.g., a bird, a cat) become negatives (ˆϵj−). The contrastive loss then encourages features from similar images (snowmen) to produce similar predicted noises, while features from different images (snowman vs. bird) produce dissimilar noises. For the snowman with black hat vs. silver hat, the anchor noise (black hat) and positive noise (augmented black hat) will be closer than both to the negative noise (bird). This process simultaneously improves both D-Ability (class separability) and P-Ability (detailed hat colour understanding).

## References

- **Code:** https://github.com/boyuh/DCR.
- Boyu Han, Qianqian Xu, Shilong Bao, Zhiyong Yang, Ruochen Cui, Xilin Zhao, Qingming Huang, "Guiding Diffusion-based Reconstruction with Contrastive Signals for Balanced Visual Representation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.04803

Tags: #computer-vision #visual-representation #multimodal-ai #diffusion-models #contrastive-learning #representation-enhancement
