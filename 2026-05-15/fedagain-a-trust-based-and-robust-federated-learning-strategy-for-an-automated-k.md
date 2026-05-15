---
title: "FedAgain: A Trust-Based and Robust Federated Learning Strategy for an Automated Kidney Stone Identification in Ureteroscopy"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19512"
---

## Executive Summary
FedAgain is a trust-based federated learning strategy designed to enhance robustness and generalisation for automated kidney stone identification from endoscopic images. It integrates a dual trust mechanism that combines benchmark reliability and model divergence to dynamically weight client contributions, mitigating the impact of noisy or adversarial updates during aggregation. This approach enables collaborative model training across multiple institutions while preserving data privacy and promoting stable convergence under real-world medical imaging conditions.

## Why This Matters for Practitioners
If you're building or maintaining medical AI systems that require cross-institutional collaboration without sharing patient data, FedAgain provides a practical solution to improve model robustness against image corruption and client heterogeneity. The paper demonstrates that FedAgain outperforms standard federated learning baselines under non-IID data and corrupted-client scenarios, which are common in real-world medical imaging deployments. Engineers should consider integrating trust-weighting mechanisms like FedAgain's dual-signal approach into their federated learning pipelines, especially when dealing with endoscopic or other image acquisition systems prone to motion blur, specular reflections, and variable illumination conditions. This could reduce the need for extensive data cleaning pipelines and improve model reliability across diverse hospital settings without compromising patient privacy.

## Problem Statement
Today's federated medical image analysis systems often assume clean, homogeneous data across institutions, but in reality, endoscopic images suffer from consistent artifacts like motion blur, specular reflections, and variable illumination, similar to trying to build a reliable navigation system for drivers who frequently encounter different weather conditions, road surfaces, and vehicle types without accounting for these variations. Current federated learning approaches treat these image corruptions as noise to be filtered out, rather than integrating robustness into the training loop itself, leading to models that fail when deployed in real clinical settings despite good performance on clean benchmarks.

## Proposed Approach
FedAgain enhances federated learning by incorporating a dual trust mechanism that combines benchmark reliability and model divergence to dynamically weight client contributions during aggregation. The system works as follows: each client computes a reconstruction error (anomaly score) using a convolutional autoencoder on its local data, and also calculates the divergence of its local model weights from the global model. The server uses these two signals to compute a trust weight for each client, which is then used to weight the client's contribution during aggregation. This transforms anomaly detection from a passive filter into an active component of the federated learning loop.

```python
def compute_trust_weight(client_data, global_model, local_model):
    # Compute reconstruction error (anomaly score) using autoencoder
    reconstruction_error = autoencoder_reconstruction_error(client_data, global_model)
    
    # Compute weight divergence from global model
    weight_divergence = np.linalg.norm(local_model - global_model, ord=2)
    
    # Calculate trust weight with ε to avoid division by zero
    trust_weight = 1 / (reconstruction_error * weight_divergence + EPSILON)
    
    return trust_weight
```

## Key Technical Contributions
FedAgain introduces a novel approach to robust federated learning in medical imaging by embedding trust mechanisms directly into the aggregation process. The key technical contributions are:

1. **Dual Trust Mechanism**: FedAgain combines two signals for trust assessment, benchmark reliability (reconstruction error from a client's local data) and model divergence (difference between client and global model weights). This dual approach allows the system to dynamically weight client contributions without needing explicit outlier detection or attack-specific tuning, making it adaptable to real-world medical image corruptions that aren't adversarial but still degrade image quality.

2. **Integrated Anomaly Detection**: Unlike previous approaches that use autoencoders only as a passive filter for data cleaning, FedAgain leverages the autoencoder's reconstruction error directly within the federated learning loop to inform the aggregation strategy. This makes robustness a core part of the training process rather than a post-hoc adjustment, which is particularly valuable in endoscopic imaging where artifacts like motion blur and specular reflections are persistent and domain-specific.

3. **Robustness to Image Corruption**: FedAgain specifically addresses the unique challenges of endoscopic image corruption (motion blur, specular reflections, occlusions by instruments, and variable illumination), which are common but often overlooked in medical federated learning. The paper demonstrates that this targeted approach leads to significant improvements over standard federated learning methods in real-world conditions.

4. **Practical Implementation**: The authors provide a reproducible PyTorch implementation with support for non-IID Dirichlet splits, which makes it easier for practitioners to integrate FedAgain into existing federated learning pipelines without significant re-engineering.

## Experimental Results
FedAgain was evaluated across five datasets, including two canonical benchmarks (MNIST and CIFAR-10), two private multi-institutional kidney stone datasets, and one public dataset (MyStone). The experiments tested robustness under non-IID data distributions and corrupted-client scenarios with corruption rates varying from 10% to 50%.

The paper reports that FedAgain consistently outperformed standard federated learning baselines (FedAvg and FedProx) under non-IID data and corrupted-client settings. Specifically, it achieved higher accuracy on kidney stone classification tasks compared to FedAvg and FedProx, particularly under higher corruption rates. For example, with 50% corrupted clients, FedAgain maintained an accuracy of approximately 85% on the kidney stone datasets, while FedAvg dropped to around 70%.

The paper also conducted ablation studies that isolated the contributions of the trust-weighting scheme and federated training, demonstrating that both are essential for improved reliability in heterogeneous and corrupted-client federated scenarios.

## Related Work
FedAgain positions itself as a bridge between robust learning and federated learning, addressing gaps highlighted in recent medical federated learning surveys. Previous works like (Reyes-Amezcua et al. (2024)) had begun to tackle robust federated learning for medical imaging but mainly focused on training with corrupted data rather than integrating robustness into the aggregation process. FedAgain builds on the FedAgain algorithm (Reyes-Amezcua et al. (2025)), which introduced a dual-signal autoencoder mechanism into the federated loop, but extends it specifically for kidney stone identification and medical imaging in ureteroscopy.

The paper compares against standard federated learning baselines (FedAvg, FedProx), robust aggregation baselines (coordinate-wise median, Bulyan), and demonstrates that FedAgain provides better performance under realistic medical imaging conditions that include persistent, domain-specific image corruptions.

## Limitations
The paper acknowledges that its evaluation primarily focused on image-level corruptions rather than more complex clinical scenarios. The authors note that they did not test FedAgain in a fully deployed medical setting, though the experiments were designed to reflect real-world conditions. Additionally, the paper doesn't specifically address how FedAgain would perform with extremely limited client participation (e.g., only 2-3 clients), which is common in medical settings where institutions may be hesitant to participate in federated learning.

## Appendix: Worked Example
Let's walk through a concrete example of FedAgain's dual trust mechanism with realistic numbers. Suppose we have a federated learning system with 3 hospitals (clients) collaborating on kidney stone identification:

1. **Client 1** (Hospital A): Has good, relatively clean endoscopic images with a reconstruction error of 0.15 (low error = reliable data) and weight divergence of 0.2 (small difference from global model).
2. **Client 2** (Hospital B): Has images with noticeable motion blur, reconstruction error of 0.4 (higher error = less reliable data) and weight divergence of 0.6 (larger difference from global model).
3. **Client 3** (Hospital C): Has images with severe specular reflections, reconstruction error of 0.7 (high error = very unreliable data) and weight divergence of 0.3 (moderate difference from global model).

Using FedAgain's trust-weighting mechanism:

- Trust weight for Client 1: 1 / (0.15 * 0.2 + 0.0001) = 1 / 0.0301 ≈ 33.22
- Trust weight for Client 2: 1 / (0.4 * 0.6 + 0.0001) = 1 / 0.2401 ≈ 4.16
- Trust weight for Client 3: 1 / (0.7 * 0.3 + 0.0001) = 1 / 0.2101 ≈ 4.76

The server normalizes these weights so they sum to 1, giving Client 1 significantly more influence in the aggregation process than the other clients. This means the global model will be more heavily influenced by the reliable data from Hospital A, while the less reliable data from Hospitals B and C will have less impact.

This example demonstrates how FedAgain dynamically weights client contributions based on data quality and model consistency, preventing unreliable data sources from dragging down overall model performance.

## References

- Ivan Reyes-Amezcua, Francisco Lopez-Tiro, Clément Larose, Christian Daul, Andres Mendez-Vazquez, Gilberto Ochoa-Ruiz, "FedAgain: A Trust-Based and Robust Federated Learning Strategy for an Automated Kidney Stone Identification in Ureteroscopy", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19512

Tags: #biomedicine #kidney-stone-identification #federated-learning #medical-imaging #trust-based-algorithms
