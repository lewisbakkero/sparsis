---
title: "Towards Distance-Invariant Radio Frequency Fingerprinting via Augmented Unsupervised Learning"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/37005"
---

## Executive Summary
This paper introduces the first unsupervised framework for distance-invariant radio frequency fingerprinting (RFF) that eliminates dependence on labelled target-domain data. It achieves 40% higher identification accuracy than state-of-the-art methods while maintaining computational efficiency suitable for edge deployment. Practitioners building secure IoT systems should care because this approach provides robust device identification without the costly overhead of collecting and annotating distance-specific data.

## Why This Matters for Practitioners
If you're deploying IoT security systems in real-world environments with varying device-receiver distances (like smart city sensor networks or industrial monitoring), this paper directly addresses a critical pain point: current RFF methods fail when devices move farther apart or change positions. Instead of requiring expensive data collection campaigns for every potential distance (which could take months of field testing for large installations), your system can now automatically adapt to new distance configurations using only unlabeled data from the target environment. This means you can deploy secure identification systems in dynamic settings without needing to retrain with new labelled data for each new deployment scenario. For example, a smart agriculture system with sensors moving between fields can maintain secure identification without requiring manual reconfiguration or data collection at each new location.

## Problem Statement
Imagine trying to recognise a person's face in a photo taken from different distances, when they're 1 meter away, you see fine details, but at 10 meters, the image becomes blurry, motion-distorted, and noise dominates, making facial recognition impossible. Current RFF systems face the same problem with wireless signals: as distance increases, the signal-to-noise ratio degrades, and complex multipath effects obscure the physical signatures that should identify devices. Traditional approaches require collecting and labelling data at every possible distance, which is impractical for large-scale IoT deployments.

## Proposed Approach
The framework consists of three main stages: preprocessing raw RF samples, applying dual alignment contrastive learning on source domain data, and performing unsupervised domain adaptation on unlabeled target domain data. Preprocessing normalizes and scales RF signals to reduce distance-dependent variations. The source model uses physics-inspired data augmentation to simulate wireless channel effects, followed by dual alignment contrastive learning to explicitly decouple device-specific traits from distance-related distortions. Finally, a pseudo-labelling-based domain adaptation module refines the model using high-confidence predictions on unlabeled target samples.

```python
def distance_invariant_rf_fingerprinting(source_samples, target_samples):
    # Preprocessing
    source_preprocessed = preprocess(source_samples)
    target_preprocessed = preprocess(target_samples)
    
    # Source model training
    source_model = train_source_model(
        source_preprocessed,
        augmentation_strategies=physics_inspired_augmentations
    )
    
    # Target adaptation
    target_model = adapt_to_target(
        source_model,
        target_preprocessed,
        pseudo_labeling_strategy=dynamic_cluster_based,
        info_maximization=True
    )
    return target_model
```

## Key Technical Contributions
The paper's technical innovations address the core challenge of distance-invariant RF fingerprinting through specific mechanisms:

1. **Physics-inspired data augmentation**: The authors developed six augmentation techniques that explicitly model real wireless propagation effects: signal scaling (to mimic path loss), permutation (to emulate multipath), window slicing (for channel fading), time warping (for time-varying delays), magnitude warping (for amplitude variations), and window warping (for local signal distortions). Unlike generic image augmentations, these are grounded in radio wave physics, ensuring the augmentations produce realistic channel variations that the model must learn to ignore.

2. **Dual alignment contrastive learning**: The framework introduces both class-level and prototype-level alignment losses to explicitly decouple device-specific features from distance-dependent distortions. The class-level alignment loss ensures features from the same device cluster together despite distance variations, while the prototype-level alignment loss anchors features to class-specific references, tightening the clustering of features for each device around its prototype. This dual approach creates a more robust feature space where device identification is less sensitive to distance.

3. **Progressive pseudo-labelling adaptation**: The paper proposes a dynamic cluster-based pseudo-labelling strategy that generates reliable supervision signals from unlabeled target data. Unlike methods that directly use source-domain classifiers for label assignment, this approach leverages target data's intrinsic structure by initializing prototypes from source classifier confidence scores, then refining them through iterative reassignment. The information maximization loss further prevents error propagation by maximising per-sample confidence while maintaining class diversity.

## Experimental Results
The framework was evaluated on two public datasets: ORACLE (16 USRP devices across 9 distances from 2 to 50 feet) and LoRa (5 identical IoT devices at 4 distances: 5m, 10m, 15m, 20m). Testing compared against four baselines: OpenRFI, VC-SEI, MPTN (modified prototypical network), and ConvTran.

Key results:
- On ORACLE.1 with 1 source domain trained and 8 target domains tested, the framework achieved 95.2% accuracy in the source domain and 65.0% in the target domain (vs. OpenRFI's 50.9% and 31.3%).
- The average cross-domain accuracy improved from 50.9% (source-only) to 65.0% (with adaptation), representing a 14.1% absolute improvement.
- The framework outperformed all baselines by 40% in identification accuracy across all distance configurations (as stated in the abstract).
- On the LoRa dataset, it achieved 99.3% source accuracy and 93.4% target accuracy, compared to VC-SEI's 92.0% and 49.0%.

The paper does not specify statistical significance tests for the 40% improvement claim, though Table 1 shows consistent improvement across multiple configurations.

## Related Work
The paper positions itself as the first to address distance-invariant RFF without labelled target-domain data, building on three prior research threads:
- Deep learning-based RFF: The authors acknowledge prior work using CNNs, LSTMs, and Transformers for RF fingerprinting but note these fail under distance variations (Wu et al. 2021; Yao et al. 2025).
- Contrastive learning for RFF: They cite recent work using contrastive learning (Zha et al. 2023; Wang et al. 2024) but note these assume same-location deployment and fail with spatial variations.
- Domain adaptation for RFF: They build on domain adaptation work (Zhang et al. 2022; Yin et al. 2023) for temporal variations but note these require labelled target data, while their method works fully unsupervised.

## Limitations
The authors acknowledge several limitations:
- The experiments focused on spatial variations (distance) but did not test other environmental factors like different obstacles or weather conditions.
- The method was evaluated on public datasets with limited diversity; real-world IoT deployments might encounter more complex channel conditions.
- The paper doesn't specify how the system would handle simultaneous distance changes across multiple devices in a dense environment.

My assessment: The lack of real-world field testing in diverse environments is a significant gap. While the results on standard datasets are promising, actual deployment would require testing in environments with dynamic obstacles (e.g., moving vehicles in smart city scenarios) that could affect signal propagation in ways the paper doesn't address.

## Appendix: Worked Example
Let's walk through a single RF sample's journey through the system with concrete values. The sample is an IQ signal from a USRP X310 device at 10 feet, captured with a USRP B210 receiver.

1. **Preprocessing**: The raw complex-valued signal has a dynamic range spanning 0.01 to 1.2 (unnormalized). First, RMS normalization scales the signal: `√(1/T ∑|x[t]|²) = 0.35`, so each sample element is divided by 0.35. Then min-max scaling normalizes the range to [0, 1], resulting in values spanning 0.02 to 1.0.

2. **Data Augmentation**: The system applies four random augmentations from the six defined:
   - Signal scaling: Multiplies by N(1, 0.12) = 0.93
   - Permutation: Splits signal into 3 segments and randomly reorders them
   - Window slicing: Takes 70% of the signal and expands with linear interpolation
   - Time warping: Applies random warping factor = 0.85

3. **Dual Alignment**: The backbone (MSAN) produces feature embeddings. For the class-level alignment, the system identifies positive pairs (augmented views of the same device) and computes cosine similarity between embeddings. If two augmented views of the same device have a cosine similarity of 0.82 (vs. 0.31 for different devices), the class-level loss is reduced. For prototype-level alignment, the system anchors features to class-specific prototypes (e.g., device 5's prototype is [0.43, -0.11, 0.72, ...]) and computes prototype assignment distributions.

4. **Pseudo-Labelling**: For a target sample at 30 feet (unlabeled), the source model's classifier produces initial predictions with confidence scores. The system computes initial prototypes using confidence-weighted averaging: for class 5, `μ⁽⁰⁾₅ = (0.82·[0.51, -0.09, 0.68] + 0.76·[0.49, -0.12, 0.70]) / (0.82 + 0.76) = [0.50, -0.10, 0.69]`. Samples are then assigned to the closest prototype, generating pseudo-labels for training the target classifier.

See Section 4.2 for the detailed mechanism of dual alignment contrastive learning.

## References

- Shiyue Huang, Yuchen Su, Hongbo Liu, Zikang Ding, Xuewan He, Yanzhi Ren, Haitao Jia, "Towards Distance-Invariant Radio Frequency Fingerprinting via Augmented Unsupervised Learning", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/37005

Tags: #iot-security #radio-frequency-fingerprinting #unsupervised-learning #domain-adaptation #edge-ai
