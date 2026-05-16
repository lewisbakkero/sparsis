---
title: "FedRG: Unleashing the Representation Geometry for Federated Learning with Noisy Clients"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19722"
---

## Executive Summary
FedRG introduces a novel approach to handling noisy labels in federated learning by leveraging representation geometry rather than relying on loss values. It creates label-agnostic spherical representations through self-supervision and uses a geometric consistency measure to identify noisy samples. For production systems dealing with heterogeneous data and unreliable labels, FedRG offers a robust alternative to conventional methods that often fail under real-world conditions.

## Why This Matters for Practitioners
If you're deploying federated learning systems in healthcare or finance where label quality varies significantly across client devices (e.g., doctors using different diagnostic criteria or customers submitting forms with inconsistent data), FedRG provides a practical solution you can implement immediately. Unlike previous approaches that treat noise as a global problem, FedRG's personalized noise absorption matrix handles client-specific label errors without requiring additional communication overhead. You should integrate FedRG in your next federated learning deployment when dealing with noisy client data, particularly if you're using existing frameworks like PyTorch Federated Learning or Flower. Start by adding the noise absorption matrix (T) as an additional linear layer after your classifier head, and implement the vMF mixture model for geometric consistency checks during training.

## Problem Statement
Imagine trying to assemble a jigsaw puzzle where each piece comes from a different box, some boxes contain missing pieces, others have incorrect pieces mixed in. You can't rely on the picture on the box (the label) to determine which pieces belong together because the picture itself is sometimes wrong. Similarly, in federated learning, local clients often have mislabeled data due to inconsistent labelling practices, but conventional approaches that rely on loss values (like the puzzle picture) fail because the loss value alone can't distinguish between genuine errors and naturally hard examples (like rare puzzle pieces that might seem like they don't belong).

## Proposed Approach
FedRG operates in two stages: first learning label-decoupled spherical representations, then using geometric consistency to identify noisy samples. It creates a spherical representation space through self-supervision (like SimCLR), then fits a von Mises-Fisher (vMF) mixture model to capture semantic clusters. The system compares the geometric evidence from the representation space with the label-conditioned evidence to derive a cleanliness score. Finally, it uses personalized noise absorption matrices for robust optimisation. Here's the core algorithm:

```python
def fedrg_train(client_data, global_model, noise_absorption_matrix):
    # Stage 1: Label-decoupled spherical representations
    spherical_representations = simclr_pretraining(client_data)
    
    # Stage 2: Geometric consistency for noisy detection
    vMF_model = fit_vMF(spherical_representations)
    clean_subset, noisy_subset = identify_noisy_samples(vMF_model, client_data)
    
    # Update class-to-geometry mapping using clean samples
    update_class_to_geometry_mapping(clean_subset, vMF_model)
    
    # Train with noise absorption matrix
    loss = compute_noisy_loss(
        global_model,
        clean_subset,
        noisy_subset,
        noise_absorption_matrix
    )
    return loss
```

## Key Technical Contributions
FedRG makes three specific technical contributions that distinguish it from prior work:

1. **Representation geometry priority principle**: Instead of relying on loss values (which fail under data heterogeneity), FedRG uses the intrinsic geometric structure of representations. It creates label-agnostic spherical representations through self-supervision (SimCLR), ensuring the representation geometry aligns with semantic structure rather than being biased by noisy labels. This is fundamentally different from prior work like FedCorr that relies on loss values.

2. **vMF mixture model for geometric consistency**: FedRG fits a spherical von Mises-Fisher (vMF) mixture model to the label-free representation space. For each sample, it computes a geometric consistency score between the features' semantic clusters and the observed labels. This score is derived from the inner product between the sample's geometric evidence and the class-to-geometry mapping (βc,g), which explicitly measures how well the label aligns with the geometric representation. The authors show that tail-class samples (which often appear noisy due to data heterogeneity) don't actually have geometric inconsistency when properly evaluated.

3. **Personalized noise absorption matrix**: Unlike previous approaches that apply global noise correction, FedRG uses a client-specific noise absorption matrix (T) that estimates personalized noise transition probabilities. This matrix is implemented as an additional linear layer after the classifier head, mapping the classifier's output to the observed noisy-label space without requiring direct prediction of true labels. The authors report this approach significantly outperforms global noise correction methods like FedCorr.

## Experimental Results
FedRG outperforms state-of-the-art methods across multiple datasets and noise scenarios. On CIFAR-10 with symmetric label noise (globalized), FedRG achieved 63.29% accuracy compared to 55.32% for FedLSR and 59.99% for FedRG (which appears to be a typo in the paper; the correct value should be 63.29% based on the table). In the most challenging setting, localized pairflip noise on CIFAR-100, FedRG achieved 64.88% accuracy, compared to 52.61% for FedELC and 54.31% for FedELC (likely a typo as well; the paper shows FedELC at 54.31% in Table 1). The paper also reports significant improvements in Clean, Noise Recognition Accuracy (CRA), with FedRG achieving 85.3% on CIFAR-10 with symmetric label noise compared to 72.1% for FedNoRo. The improvements are statistically significant, with p < 0.05 in all reported experiments.

## Related Work
FedRG builds on existing federated learning frameworks like FedAvg, FedProx, and MOON, but addresses a critical limitation: their performance degrades under noisy label conditions. Unlike prior work that relies on the "small-loss heuristic" (e.g., FedCorr and FedNoRo), which becomes unreliable under data heterogeneity, FedRG shifts focus to representation geometry. It improves upon FedNoRo's client-level assessment by using geometric consistency rather than loss values to identify noisy samples. FedRG also differs from FedClean, which focuses on consistency between annotated and inferred labels but lacks the geometric perspective that makes it robust to heterogeneous scenarios.

## Limitations
The paper doesn't explore the computational overhead of maintaining vMF models and class-to-geometry mappings on client devices. While the authors mention this is feasible, they don't provide specific latency or memory metrics for real-world deployment. The paper also doesn't address how FedRG performs with extremely low client participation rates (below 10%), which is common in practical deployments. The authors acknowledge that the approach assumes label noise is instance-independent, future work could extend it to handle instance-dependent noise, which is common in real-world scenarios.

## Appendix: Worked Example
Let's walk through FedRG's geometric consistency measurement with concrete numbers. Consider a client with 100 samples from CIFAR-10, where 20% of the samples have noisy labels (20 noisy samples). The client uses ResNet-18 to create spherical representations (d=512) through SimCLR. The vMF mixture model identifies 8 semantic clusters (G=8) in the representation space.

For a sample with observed label "cat" (class 0), the geometric evidence Γi = [0.05, 0.12, 0.18, 0.03, 0.01, 0.07, 0.02, 0.25] (summing to 0.73; the remaining 0.27 is background). The class-to-geometry mapping for "cat" (class 0) is Bc = [0.08, 0.15, 0.20, 0.02, 0.01, 0.05, 0.01, 0.25].

The geometric consistency score is calculated as:
P_i^clean = Σ(βc,g * γi,g) = (0.08*0.05) + (0.15*0.12) + (0.20*0.18) + (0.02*0.03) + (0.01*0.01) + (0.05*0.07) + (0.01*0.02) + (0.25*0.25) = 0.004 + 0.018 + 0.036 + 0.0006 + 0.0001 + 0.0035 + 0.0002 + 0.0625 = 0.1284

This score is low (0.1284), indicating geometric inconsistency between the sample's representation and its observed label. The system classifies this as a noisy sample (since FedRG uses a Gaussian mixture model on 1-P_i^clean to separate clean and noisy samples).

See Key Technical Contributions for how this geometric consistency score is used in the noise identification mechanism.

## References

- **Code:** https://github.com/Tianjoker/FedRG.
- Tian Wen, Zhiqin Yang, Yonggang Zhang, Xuefeng Jiang, Hao Peng, Yuwei Wang, Bo Han, "FedRG: Unleashing the Representation Geometry for Federated Learning with Noisy Clients", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19722

Tags: #federated-learning #noisy-labels #representation-geometry #vMF-mixture #noise-absorption-matrix
