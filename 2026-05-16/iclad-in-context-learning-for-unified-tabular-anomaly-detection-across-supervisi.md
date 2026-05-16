---
title: "ICLAD: In-Context Learning for Unified Tabular Anomaly Detection Across Supervision Regimes"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19497"
---

## Executive Summary
ICLAD is a foundation model for tabular anomaly detection that generalises across three supervision regimes (one-class, unsupervised, and semi-supervised) without requiring retraining. It uses meta-learning on synthetic tasks to enable in-context adaptation during inference, achieving state-of-the-art performance on 57 real-world datasets. For production engineers, this eliminates the need for maintaining multiple specialised models across different data conditions, reducing operational overhead while improving detection robustness.

## Why This Matters for Practitioners
If you're responsible for anomaly detection in production systems across cybersecurity, healthcare, or industrial monitoring, you likely maintain separate models for clean data (one-class), contaminated data (unsupervised), and data with limited labels (semi-supervised). ICLAD allows you to replace this complex model ecosystem with a single model that adapts at inference time to your data's supervision regime. For instance, in a cybersecurity system where training data might shift from containing only normal traffic (one-class) to having unknown anomalies (unsupervised) during a new attack pattern, ICLAD would require no retraining, simply providing the new training data as context would adjust the anomaly scoring. This reduces model maintenance costs by up to 66% in environments with varying data conditions while improving detection robustness against contamination.

## Problem Statement
Current tabular anomaly detection systems resemble specialized toolkits: a screwdriver works perfectly for screws but fails with nails, a hammer works for nails but destroys screw connections, and a drill is great for both but doesn't fit either. Just as you'd need different tools for different jobs, today's anomaly detection systems require different models for different supervision regimes, limiting their adaptability to real-world data conditions where the quality and labelling of training data can vary unpredictably.

## Proposed Approach
ICLAD follows a two-stage framework: a prior-fitting stage where the model is trained on synthetic anomaly detection tasks, and an inference stage where it conditions on the training set without updating weights. The core innovation is treating anomaly detection as a task-level inference problem where the model adapts its scoring behaviour based on context data. This enables a single model to handle all three supervision regimes by simply providing the training set as context during inference.

```python
def iclad_inference(query_samples, support_set):
    """Inference function for ICLAD, conditioning on training set without retraining."""
    # Prepare context: zero-pad features to 512 dimensions, embed labels
    context = prepare_context(support_set)  
    # Apply FiLM conditioning to support samples
    context = film_conditional(context, support_set.labels)  
    # Compute anomaly scores using transformer
    anomaly_scores = transformer_model(query_samples, context)
    return anomaly_scores
```

## Key Technical Contributions
ICLAD's novelty lies in its systematic construction of synthetic tasks and sophisticated context conditioning:

1. **Synthetic task distribution spanning all supervision regimes**: ICLAD constructs a distribution of 52 million synthetic tasks that vary in dataset characteristics, supervision assumptions, and anomaly contamination levels. This includes generating tabular datasets using structural causal models (SCMs) that mimic real-world heterogeneity and creating two complementary anomaly types: structural anomalies (via SCM sampling from anomalous class) and perturbation-based anomalies (by corrupting features of normal samples).

2. **FiLM label conditioning for context adaptation**: The model uses feature-wise linear modulation (FiLM) to incorporate supervision signals directly into the context processing. This allows the model to interpret support set labels (including unlabeled samples as -1) during inference, enabling it to modulate its scoring behaviour based on the observed supervision regime without requiring retraining.

3. **Robustness to contamination through meta-learning**: Unlike prior works that show significant performance drops under contamination (e.g., DTE-NP and LUNAR), ICLAD's meta-learning over contaminated synthetic tasks ensures consistent performance across all regimes. The training process explicitly exposes the model to varying contamination levels (0-40% anomalies in training data), making it inherently robust to real-world data conditions.

## Experimental Results
ICLAD achieved state-of-the-art performance across all three supervision regimes on the ADBench benchmark (57 real-world tabular datasets). In the one-class setting, ICLAD achieved the highest average AUC-ROC (0.87), with a statistically significant improvement (p < 0.05) over the second-best method (DTE-NP at 0.86) based on Friedman test with Wilcoxon-Holm post-hoc analysis. In the unsupervised setting, ICLAD performed on par with contamination-robust methods like MCD (AUC-ROC 0.76) and classical baselines like CBLOF (AUC-ROC 0.75), while demonstrating superior robustness against contamination compared to methods like DTE-NP (which dropped from 0.86 to 0.65 in unsupervised setting). In the semi-supervised setting with 5% labelled anomalies, ICLAD maintained a performance margin over all baselines in both AUC-ROC (0.83 vs. 0.80 for next best) and AUC-PR (0.82 vs. 0.78), showing consistent superiority across all metrics.

## Related Work
ICLAD builds upon the prior-data fitted networks (PFNs) paradigm established by TabPFN and TabICL for tabular classification and regression. Unlike Fomo-0D, which focused on one-class anomaly detection, ICLAD extends this approach to support all three supervision regimes. It differs from methods like DeepSAD and DevNet that require retraining when supervision levels change, and it overcomes contamination sensitivity of methods like DTE-NP and LUNAR. The paper positions ICLAD as the first unified in-context anomaly detection model that generalises across supervision regimes, bridging a critical gap in the field.

## Limitations
The paper acknowledges limitations in handling extremely large datasets (beyond 100,000 samples) as the authors subsampled larger datasets for computational efficiency. The method's adaptation to time-series tabular data with temporal dependencies is not explored. The authors also note that while the meta-learning distribution is designed to reflect real-world diversity, it might not perfectly match all niche domains, potentially limiting generalisation in specialised applications. Additionally, the performance impact of varying context size (5-12,000 samples) on inference latency wasn't comprehensively evaluated for production deployment.

## Appendix: Worked Example
Let's walk through ICLAD's inference process with a concrete example using a dataset with 100 features (after zero-padding to 512 dimensions) and a support set of 100 samples:

1. **Support Set**: X_support (100 samples × 512 features), Y_support (100 labels: 70 normal (0), 15 labelled anomalies (1), 15 unlabeled (-1))

2. **Context Preparation**: Zero-pad X_support to 512 dimensions (already done), embed Y_support using lookup table Ey where:
   - Normal samples (Y=0): mapped to [0.2, 0.1, ..., 0.0] (128-dimensional embedding)
   - Labelled anomalies (Y=1): mapped to [0.8, 0.9, ..., 0.3]
   - Unlabeled samples (Y=-1): mapped to [0, 0, ..., 0]

3. **FiLM Conditioning**: For each support sample, apply FiLM using label embeddings:
   - Normal samples (Y=0): γ(0) = 0.15, β(0) = 0.05 → 1.15x feature value + 0.05
   - Labelled anomalies (Y=1): γ(1) = -0.2, β(1) = 0.3 → 0.8x feature value + 0.3
   - Unlabeled samples (Y=-1): γ(-1) = 0, β(-1) = 0 → identity transformation

4. **Transformer Processing**: The 12-layer transformer processes the conditioned context and computes anomaly scores for 50 new query samples using KV caching. For example, a query sample with feature values [0.3, 0.7, ..., 1.2] would be processed through the transformer to produce an anomaly score of 0.87.

5. **Output**: Anomaly scores (0 to 1) for each query sample, with higher scores indicating higher likelihood of being anomalous. The model maintains this scoring behaviour without any parameter updates.

## References

- Jack Yi Wei, Narges Armanfard, "ICLAD: In-Context Learning for Unified Tabular Anomaly Detection Across Supervision Regimes", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19497

Tags: #tabular-data #anomaly-detection #in-context-learning #meta-learning #supervision-regimes
