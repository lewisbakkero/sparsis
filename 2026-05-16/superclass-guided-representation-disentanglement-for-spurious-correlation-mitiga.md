---
title: "Superclass-Guided Representation Disentanglement for Spurious Correlation Mitigation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2508.08570"
---

## Executive Summary
SupER introduces a novel approach to mitigating spurious correlations in machine learning models without requiring group annotations during training. It leverages superclass labels (e.g., "bird" vs "background") from pretrained vision-language models to guide feature disentanglement, enabling robustness to unseen groups at test time. For practitioners, this resolves a critical pain point in domain generalisation where traditional approaches fail when test groups differ from training groups.

## Why This Matters for Practitioners
If you're building a production image classification system that must generalise across different environments (e.g., medical imaging across hospitals or product recognition across lighting conditions), you'll often face spurious correlations that degrade performance when deployed in new settings. Current solutions either require expensive group annotations during training or fail when test-time groups differ from training groups. SupER solves this by using superclass semantic information from pretrained models to guide feature disentanglement, allowing your model to focus on core features that generalise better. This means you can deploy models with significantly better worst-case accuracy across diverse environments without the overhead of annotating group information, making your system more robust in real-world deployments.

## Problem Statement
Imagine training a bird classifier using images where waterbirds consistently appear on watery backgrounds and landbirds on land backgrounds. The model learns to rely on background features as a shortcut, predicting "waterbird" based on watery backgrounds instead of the bird features themselves. This works fine until you deploy it on a test set with landbirds on watery backgrounds (a new group not seen during training), where the model fails catastrophically. This spurious correlation problem is exacerbated in real-world systems where background features vary unpredictably across environments, yet most solutions require explicit group annotations or assume identical group distributions between training and test.

## Proposed Approach
SupER combines feature disentanglement with superclass guidance from a pretrained vision-language model using gradient-based attention alignment. The architecture consists of four key components:
1. A β-VAE that disentangles input features into superclass-relevant (zrel) and superclass-irrelevant (zirr) components
2. Two classifiers (ωrel and ωirr) that predict labels from zrel and zirr respectively
3. A CLIP-guided mechanism that generates attribution maps to guide disentanglement
4. An L2 regularisation term that encourages using diverse superclass-relevant features

Here's the training algorithm for SupER:

```python
def train_supER(Ds, model, learning_rate, epochs):
    for epoch in range(epochs):
        shuffle(Ds)
        for batch in Ds:
            for (x, y) in batch:
                beta_vae_loss = compute_beta_vae_loss(x)
                attention_loss = compute_attention_alignment_loss(x, y)
                rel_classifier_loss = compute_classifier_loss(x, y, ωrel)
                irr_classifier_loss = compute_classifier_loss(x, y, ωirr)
                total_loss = rel_classifier_loss + irr_classifier_loss - λ1 * beta_vae_loss + λ2 * attention_loss + λ3 * ||ωrel||^2
                update_model_parameters(total_loss, learning_rate)
```

## Key Technical Contributions
SupER introduces several novel mechanisms that enable robustness to spurious correlations without group annotations:

1. Superclass-guided feature disentanglement: The β-VAE disentangles features into zrel (superclass-relevant) and zirr (superclass-irrelevant) components, guided by CLIP's attention mechanism. For each image, CLIP generates attribution maps highlighting regions corresponding to superclass labels (e.g., "bird" for the superclass), which are then aligned with the gradient-based attention maps from ωrel and ωirr. This ensures zrel captures semantic features relevant to the superclass while zirr captures irrelevant features.

2. Minimax-optimal feature usage strategy: The theoretical analysis proves that SupER should discard superclass-irrelevant features (zirr) and use a diverse set of superclass-relevant features (zrel). This is implemented by excluding ωirr during inference and adding an L2 penalty on ωrel during training, which encourages smoother and more evenly distributed weights across informative features (see Table 30 for empirical validation).

3. Robustness to biases in the guiding model: Unlike prior work that directly uses pretrained models like CLIP, SupER mitigates CLIP's own biases by using the superclass label as a semantic prior. The superclass-level guidance avoids CLIP's spurious correlations that arise when conditioning on fine-grained labels, and the β-VAE's encouragement of independent latent factors overrides occasional attribution errors from CLIP (see Figure 3 for visual validation).

4. Handling unseen groups at test time: SupER's architecture allows it to handle test-time groups that were absent during training. For example, on Waterbirds-100% where two groups are entirely absent during training, SupER achieves 79.7% worst-group accuracy compared to the best baseline (JTT) at 61.3% (a 17.6% improvement), demonstrating robustness to the most challenging scenarios.

## Experimental Results
SupER significantly outperforms baselines across multiple domain generalisation benchmarks. On Waterbirds-100%, SupER achieves 79.7% worst-group accuracy, surpassing the best baseline (JTT) by 17.6% (61.3%). On Spawrious M2M-hard, SupER achieves 79.9% worst-group accuracy compared to the best baseline (GroupDRO) at 54.1%, a 25.8% improvement. The standard deviation of worst-group accuracy across Spawrious subsets is 2.7%, significantly lower than other baselines (e.g., 14.1% for UW). SupER also shows consistent improvements over the CLIP teacher, achieving 82.9% mean worst-group accuracy across all Spawrious subsets compared to CLIP's 72.9% (a 10% absolute improvement).

## Related Work
SupER positions itself as the first group-label-free framework for mitigating spurious correlations in settings where test-time groups may differ from training groups. It improves upon prior work in two key areas: (1) it doesn't require group annotations during training (unlike GroupDRO, UW, and DFR), and (2) it handles unseen groups at test time (unlike most baselines that assume identical group distributions).

SupER builds on the idea of feature disentanglement (e.g., from [17, 46, 50]) but goes beyond by using superclass guidance instead of group annotations. It also extends theoretical analysis of spurious correlations [2, 44, 45, 53] with a finer partition using superclass information, providing the first minimax-optimal feature-usage strategy for this scenario.

## Limitations
The paper doesn't explicitly state limitations beyond the scope of the experimental validation. The method requires a pretrained vision-language model (CLIP) for guidance, which might not be available for all tasks. The evaluation was limited to image classification tasks; it's unclear how well it would generalise to other modalities like text or audio. The paper doesn't test SupER on extremely large-scale datasets or with extreme computational constraints.

## Appendix: Worked Example
Let's walk through how SupER processes a waterbird image on watery background:

1. **Input**: An image of a waterbird on watery background (label: "waterbird", background: "water")
2. **Superclass guidance**: CLIP generates attribution maps highlighting "bird" features (superclass-relevant) and "background" features (superclass-irrelevant), using prompts like "a waterbird" and "a watery background"
3. **Feature disentanglement**: The β-VAE disentangles the image features into:
   - zrel: Captures "bird" semantic features (e.g., beak, wings, feather patterns)
   - zirr: Captures "background" features (e.g., water texture, reflection patterns)
4. **Classification**: 
   - ωrel (superclass-relevant classifier) processes zrel to predict "waterbird"
   - ωirr (superclass-irrelevant classifier) processes zirr but is excluded during inference
5. **Result**: The model correctly identifies the bird type (focusing on bird features, not background), achieving 84.4% worst-group accuracy on Waterbirds-95% compared to ERM's 64.9% (a 19.5% improvement)

The L2 penalty on ωrel during training ensures the model uses a diverse set of superclass-relevant features (e.g., not just the beak but also wing patterns and feather textures), leading to more robust representations that generalise better to new backgrounds.

## References

- **Code:** https://github.com/crliuuuuu/SupER.
- Chenruo Liu, Hongjun Liu, Zeyu Lai, Yiqiu Shen, Chen Zhao, Qi Lei, "Superclass-Guided Representation Disentanglement for Spurious Correlation Mitigation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2508.08570

Tags: #computer-vision #domain-generalisation #spurious-correlations #superclass-guided-disentanglement #minimax-optimality
