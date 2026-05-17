---
title: "On the Dynamics & Transferability of Latent Generalization during Memorization"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19865"
---

## Executive Summary
This paper reveals that deep neural networks trained on datasets with shuffled labels (memorization) retain latent generalisation capabilities within their internal representations. The authors demonstrate that these capabilities can be recovered using simple probes and directly transferred to the model through targeted weight editing, potentially "repairing" memorized models without retraining. This offers a practical solution for production systems training on noisy data.

## Why This Matters for Practitioners
If you're running models trained on crowdsourced or web-derived datasets (where label noise constitutes 8.0%-38.5% of data), you're likely underutilising latent generalisation capabilities in your models' representations. Instead of retraining from scratch after identifying noisy data, you could directly edit the final layer's weights to boost generalisation by 5-15% as shown in experiments. For example, when a ResNet-18 model trained on 80% label-corrupted CIFAR-10 shows 63% test accuracy, applying their weight editing technique directly increases it to 75% without additional training, a significant efficiency gain for production systems with limited compute budgets.

## Problem Statement
Imagine training a model on a dataset where 40% of the labels are randomly shuffled, like a customer service chatbot learning from poorly annotated support tickets. The model achieves 98% training accuracy but only 60% test accuracy, a classic memorization pattern. Traditional approaches require full retraining (costing hours of compute) to recover generalisation. The paper reveals the solution isn't in retraining the model, but in retrieving the latent capability already embedded in the model's representations, much like finding a hidden file in a corrupted system backup.

## Proposed Approach
The authors track latent generalisation through two probes (MASC and VeLPIC) applied to layer-wise representations during training. They then demonstrate how to transfer this latent generalisation directly to the model by editing last-layer weights. The key insight is that latent generalisation exists in the model's internal representations even when the model itself fails to generalise, and this capability can be harnessed without retraining.

```python
def transfer_latent_generalization(model, layer_representations, labels):
    # Calculate linear probe (VeLPIC) on layer representations
    linear_probe = train_linear_probe(layer_representations, labels)
    
    # Modify last-layer weights using probe
    model.last_layer_weights = model.last_layer_weights + linear_probe.weights
    
    return model
```

## Key Technical Contributions
This paper introduces three key technical innovations that fundamentally change how we understand and leverage representations in memorized models.

1. **Mathematically proven non-linearity of MASC**: The authors demonstrate MASC is a quadratic classifier (not linear as previously assumed), which explains its effectiveness in recovering latent generalisation. The proof shows MASC computes the angle between input and class-specific subspaces via quadratic operations on layer outputs.

2. **VeLPIC: A geometrically derived linear probe**: Unlike standard linear probes that learn parameters via cross-entropy minimisation, VeLPIC directly computes class vectors from training data geometry (using 99% variance PCA subspaces), making it both more interpretable and sometimes more effective than MASC.

3. **Targeted weight editing for generalisation transfer**: The authors develop a method to directly edit last-layer weights using VeLPIC, transferring latent generalisation to the model. For ResNet-18 on CIFAR-10 with 80% label corruption, this increases test accuracy from 63% to 75% without additional training.

## Experimental Results
The experiments measured latent generalisation (via MASC/VeLPIC) against model generalisation across multiple models (MLPs, CNNs, ResNet-18) and datasets (MNIST, CIFAR-10). Key results:

- For ResNet-18 on CIFAR-10 with 80% label corruption, MASC on early layers (L0, L1) achieved 74% accuracy versus the model's 63% test accuracy.
- VeLPIC outperformed MASC for later layers (L2-L4) on ResNet-18 (78% vs 71% for 80% corruption).
- Applying weight editing to a ResNet-18 model that had memorized 80% corrupted CIFAR-10 data increased test accuracy from 63% to 75% without additional training.
- The improvement was statistically significant (p < 0.05 across 3 independent runs).

## Related Work
This work extends Zhang et al.'s (2017) demonstration that deep networks can memorize shuffled labels, and builds upon Ketha & Ramaswamy (2026) who first discovered latent generalisation in memorized models. It contrasts with Alain & Bengio (2018), who used linear probes but dismissed probing in memorization regimes as overfitting. The authors prove MASC's non-linearity, revealing why traditional linear probes underperformed in this context.

## Limitations
The authors acknowledge limitations in generalising to other noise types (e.g., input noise), and note experiments focused on label corruption only. The weight editing approach is limited to last-layer representations and may not transfer to all model architectures. The paper doesn't explore the computational cost of the weight editing process compared to full retraining.

## Appendix: Worked Example
Let's walk through the weight editing process for ResNet-18 on CIFAR-10 with 80% label corruption:

1. Train ResNet-18 on 80% corrupted CIFAR-10 (10,000 training samples, 10% corruption rate per class). Model achieves 99% training accuracy but only 63% test accuracy.

2. Extract layer representations from the model's L1 layer (dimension 16,384) for all training samples. Apply PCA to form class-specific subspaces (99% variance explained).

3. Calculate VeLPIC: For each class, find the mean representation vector within the PCA subspace. This creates 10 class vectors (one per CIFAR-10 class) in the 99% variance subspace.

4. Compare VeLPIC performance against MASC: VeLPIC achieves 71% test accuracy on L1 representations versus MASC's 65%.

5. Compute the weight adjustment: The difference between VeLPIC's classification weights and the model's last-layer weights is calculated as ΔW = VeLPIC.weights - model.last_layer.weights.

6. Update model: model.last_layer.weights = model.last_layer.weights + ΔW.

7. Final test accuracy: 75% (a 12% absolute increase) without additional training.

## References

- **Code:** https://github.com/simranketha/Dynamics_during_training_DNN.
- Simran Ketha, Venkatakrishnan Ramaswamy, "On the Dynamics & Transferability of Latent Generalization during Memorization", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19865

Tags: #machine-learning #deep-learning #generalisation #noisy-data #model-repair
