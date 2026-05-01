---
title: "DeepSenseMoE: Harnessing Power of Time Series Foundation Models for Few-Shot Human Activity Recognition"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36990"
---

## Executive Summary
DeepSenseMoE introduces a novel parameter-efficient fine-tuning module that unlocks the potential of general-purpose Time Series Foundation Models (TSFMs) for few-shot human activity recognition using wearable sensors. It solves the dual challenges of sensor annotation scarcity and heterogeneous sensor data distribution through multi-scale convolutional experts with shared expert isolation and hierarchical supervised contrastive alignment. Engineers working with resource-constrained wearable systems should care because it achieves up to 9.5% accuracy gains with less than 1% additional parameters compared to state-of-the-art methods.

## Why This Matters for Practitioners
If you're building a wearable health monitoring system with limited labelled sensor data (common in healthcare applications where manual annotation is expensive and time-consuming), this paper suggests you should consider fine-tuning pre-trained TSFMs instead of training from scratch. For instance, when working with the MotionSense dataset (which contains 11,843 samples across 12 activities from 13 subjects with varying sensor placements), DeepSenseMoE achieves 91.7% accuracy in 5-shot learning, outperforming the best baseline (TS2ACT) by 9.5%, while using only 0.043 million additional parameters compared to the standard Adapter approach (0.238 million). This means you can deploy more accurate activity recognition models in resource-constrained environments without needing massive labelled datasets or significantly increasing model size.

## Problem Statement
Imagine trying to teach a new language to someone who only speaks one dialect, using a dictionary written entirely in a different dialect. That's the challenge of applying general time series models to wearable activity recognition today. The "dialect" problem arises because sensor data from wearable devices has two unique characteristics: first, labelling sensor signals is like trying to distinguish subtle differences in a language you've never learned, just as it's easy to tell a dog from a cat in an image, differentiating between walking upstairs versus downstairs in IMU data requires painstaking annotation; second, the "dialect" of sensor data varies wildly across devices, body locations, and users, like regional accents changing pronunciation, identical activities recorded with the same sensor on different people produce completely different signal patterns. These characteristics mean traditional fine-tuning approaches fail because they can't handle this distributional shift.

## Proposed Approach
DeepSenseMoE integrates with pre-trained Time Series Foundation Models (specifically MOMENT) as an adapter module that dynamically routes sensor inputs through multiple specialized convolutional experts while maintaining a shared expert for common activity knowledge. The architecture consists of three core components: multi-scale convolutional experts that capture varying sensor contexts, a shared-expert isolation mechanism that compresses common knowledge into a single expert, and hierarchical supervised contrastive alignment that ensures all experts learn discriminative features.

```python
def deepsense_moe(input_features, top_m=2):
    # Multi-scale convolutional experts with different filter sizes
    expert_outputs = []
    for kernel_size in [3, 5, 7, 9, 11]:
        expert_output = depthwise_conv(input_features, kernel_size)
        expert_outputs.append(expert_output)
    
    # Router selects top-m experts per sample
    router_scores = router(input_features)
    top_indices = top_m_indices(router_scores)
    
    # Shared expert isolation mechanism
    shared_expert = depthwise_conv(input_features, fixed_kernel_size)
    
    # Dynamic output combining shared and top experts
    final_output = (1/(top_m+1)) * shared_expert
    for i in top_indices:
        final_output += (1/(top_m+1)) * router_scores[i] * expert_outputs[i]
    
    return final_output
```

## Key Technical Contributions
The core innovation lies in how DeepSenseMoE handles the heterogeneity of wearable sensor data through three specific mechanisms:

1. **Multi-scale convolutional experts**: Instead of using linear projections like standard adapters, DeepSenseMoE employs parallel depth-wise convolutional branches with filter sizes ranging from 3 to 11 (odd numbers), each responsible for capturing different time-scale patterns in sensor data. The authors explain that "CNNs inherently possess different cognition of sensor features at different filter scales" (Chen et al. 2021), so by using multi-scale convolutional experts, they can better model the varying temporal patterns across different activities.

2. **Shared-expert isolation mechanism**: Unlike traditional MoE where all experts compete for activation, DeepSenseMoE isolates one expert (Esh) that remains always active, compressing common activity knowledge into this single shared expert. This "reduces redundancy among routed experts" and "effectively mitigates parameter redundancy" while maintaining specialized knowledge in the routed experts. The authors note that "given high variability in sensor input space, conventional routing strategy may make multiple experts converge in acquiring too similar knowledge."

3. **Hierarchical supervised contrastive alignment**: For each layer, they introduce layer-wise contrastive learning between shared and routed experts, treating outputs from the same activity label as positive pairs and different labels as negative pairs. This "forces all experts to learn increasingly discriminative features" and prevents "expert collapse" where the router consistently selects the same expert. The contrastive loss function is defined as:
   Lsca = −∑(z(i)_p · z(i)_p / τ) / ∑(z(i)_p · Ei(xr)/τ)
   where z(i)_p represents the embedding from expert i for activity p.

## Experimental Results
DeepSenseMoE was evaluated on three HAR benchmarks: HHAR (1,319 samples across 12 activities), MotionSense (11,843 samples across 12 activities), and PAMAP2 (41,924 samples across 18 activities). In 5-shot learning, DeepSenseMoE achieved 91.7% accuracy on MotionSense (outperforming TS2ACT by 9.5%), 82.1% on HHAR (outperforming TS2ACT by 7.0%), and 90.5% on PAMAP2 (outperforming TS2ACT by 4.5%). In full-supervised settings, it achieved 98.9% on MotionSense, 96.6% on HHAR, and 98.6% on PAMAP2. The authors report "up to 9.5% accuracy gains over state-of-the-art" with "only <1% additional trainable parameters" (0.043 million vs. Adapter's 0.238 million).

The paper doesn't specify whether accuracy gains are statistically significant (e.g., p-values), though they mention "all results are averaged over three independent runs" to ensure statistical reliability. The comparison includes eight baselines spanning from fully supervised (DCNN, TCN, ConformerHAR) to semi-/unsupervised learning (FixMatch, SimCLR, MDC, TS2ACT, Vi2ACT), with Vi2ACT being the previous state-of-the-art.

## Related Work
DeepSenseMoE positions itself at the intersection of Time Series Foundation Models and Human Activity Recognition. It explicitly builds on MOMENT (Goswami et al. 2024), the first family of open-source TSFM, but notes "there is no existing literature on fine-tuning time series-based foundation models for wearable activity recognition." Unlike previous adapter methods (Hu et al. 2022; Houlsby et al. 2019) designed for text and image with "fixed layer parameters," DeepSenseMoE specifically addresses the "heterogeneous sensor signals" problem in wearable activity recognition. The authors note that "existing adapters... simply compress upstream features with linear projection, where the fixed layer parameters cannot be well fine-tuned to fully match the varying distribution of diverse activity recognition tasks."

## Limitations
The paper focuses exclusively on the MOMENT backbone, with no evaluation of other TSFM architectures. The authors acknowledge that "the majority of previous TSFMs are closed-source, resulting in limited access to the model itself," which means their work applies only to open-source TSFMs. The evaluation was limited to three HAR benchmarks, and the authors note that "this work lays a solid foundation to accelerate development and deployment of TSFMs in activity recognition," suggesting future work should extend to more diverse datasets and applications. The paper doesn't test the approach on real-time wearable systems with latency constraints, though the parameter efficiency suggests it could be suitable for edge deployment.

## Appendix: Worked Example
Let's walk through a single sensor input through DeepSenseMoE with concrete values. Consider a sensor input sequence of 500 time steps (standard processing window size) with 9 sensor channels (a common configuration in wearable devices), represented as x ∈ R^(500×9).

1. **Multi-scale convolutional experts**: The input passes through five parallel depth-wise convolutional branches with kernel sizes 3, 5, 7, 9, and 11. Each branch applies a depth-wise convolution with 128 filters:
   - 3×1 DW: Output size (500×128)
   - 5×1 DW: Output size (500×128)
   - 7×1 DW: Output size (500×128)
   - 9×1 DW: Output size (500×128)
   - 11×1 DW: Output size (500×128)

2. **Router selection**: The router computes scores for each expert using a 5-channel convolution followed by global average pooling and softmax. For this sample, it selects the top-2 experts: the 5×1 DW and 7×1 DW branches, with scores of 0.6 and 0.4 respectively.

3. **Shared expert isolation**: A fixed kernel size (e.g., 7×1) convolution branch is always active, producing a shared expert output with the same dimensions (500×128).

4. **Dynamic output combination**: The final output is calculated as:
   - Shared expert contribution: (1/3) × [shared_expert_output]
   - 5×1 DW expert: (0.6/3) × [5×1 DW output]
   - 7×1 DW expert: (0.4/3) × [7×1 DW output]

5. **Hierarchical contrastive alignment**: For each layer in the MOMENT backbone, this output is aligned with other experts using the contrastive loss. If the activity label is "walking," all outputs from experts processing walking samples are treated as positive pairs, while outputs from other activities (e.g., "running") are negative pairs. The contrastive loss function ensures these feature representations remain distinct across different activities.

This process happens at every encoder layer in the MOMENT backbone, with the final output being the classification prediction. The parameter efficiency comes from using only 0.043 million additional parameters compared to the standard Adapter (0.238 million), primarily through the shared expert isolation mechanism that reduces redundancy.

## References

- **Code:** https://github.com/FuZenan/DeepSenseMoE
- Zenan Fu, Dongzhou Cheng, Lei Zhang, Wenbo Huang3, Zhenghao Chen, Hao Wu, "DeepSenseMoE: Harnessing Power of Time Series Foundation Models for Few-Shot Human Activity Recognition", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36990

Tags: #biomedicine #activity-recognition #time-series-forecasting #parameter-efficient-fine-tuning #mixture-of-experts
