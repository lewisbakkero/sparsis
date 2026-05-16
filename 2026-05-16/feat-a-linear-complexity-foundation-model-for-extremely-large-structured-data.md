---
title: "FEAT: A Linear-Complexity Foundation Model for Extremely Large Structured Data"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.16513"
---

## Executive Summary
FEAT is a linear-complexity foundation model designed to process extremely large structured datasets at scale, overcoming the quadratic complexity limitations of existing approaches. It achieves up to 40× faster inference while maintaining zero-shot predictive parity with state-of-the-art baselines across classification and regression tasks on real-world datasets.

## Why This Matters for Practitioners
If you're building production systems that handle structured data at scale (e.g., healthcare records, financial transactions, or e-commerce product catalogs), FEAT removes the fundamental bottleneck of O(N²) attention mechanisms that previously restricted context windows to around 50,000 samples. You can now process datasets with millions of records without requiring task-specific training for each new dataset. For example, in healthcare applications where patient records often exceed 100,000 entries, FEAT enables direct inference on full populations without sampling, reducing the need for costly data partitioning strategies. This means you can deploy zero-shot models on new structured datasets immediately, without retraining.

## Problem Statement
Existing structured-data foundation models face a fundamental tension: they require O(N²) attention to capture global relationships between samples, but this limits context windows to around 50,000 samples (as seen in TabPFN and LimiX). Imagine trying to analyse the full population of patients in a hospital database by only comparing each patient to a small subset of others, this would miss critical correlations across the entire patient base. The problem is analogous to trying to analyse a city's traffic patterns by only tracking interactions between adjacent cars, ignoring how the entire traffic flow connects across the city's road network.

## Proposed Approach
FEAT introduces a multi-layer dual-axis encoding architecture that replaces quadratic attention with hybrid linear encoding. The system processes structured data through three main components: cell-level embedding, multi-layer dual-axis encoding, and task-aware prediction. The dual-axis encoding decomposes representation learning into two orthogonal stages: feature-axis modelling (capturing intra-sample feature dependencies) and sample-axis modelling (modelling inter-sample relationships with linear complexity).

Here's the core algorithm for the sample-axis modelling stage:

```python
def sample_axis_modeling(X, L):
    # X: input tensor of shape [N, D, d]
    # L: number of dual-axis encoding blocks
    
    for l in range(1, L+1):
        # Apply feature-axis modelling independently across samples
        X = feature_axis_modeling(X)
        
        # Apply dual-axis encoding for sample-axis relationships
        X = afbm(X)  # Adaptive-fusion bi-Mamba-2
        X = conv_gla(X)  # Convolutional gated linear attention
    return X
```

## Key Technical Contributions
The key innovations in FEAT address three fundamental challenges in structured-data foundation models:

1. **Dual-axis architecture (AFBM + Conv-GLA)**: The system solves the linear trap and causal mask deficit through a specialized combination of two linear-complexity components. AFBM uses bidirectional Mamba-2 to capture dynamic local dependencies across samples without imposing artificial causal ordering, while Conv-GLA maintains global interactions through explicit memory accumulation using a convolutional gated linear attention mechanism. This prevents representation collapse in linear-complexity models while preserving permutation invariance.

2. **Hybrid structural causal model pre-training**: FEAT addresses the heavy-tailed data distribution mismatch through a mixed real-and-synthetic pre-training strategy. The system combines scale-free synthetic structural causal models with real-world structured datasets and uses a numerically robust Huber-based reconstruction loss. This stabilizes optimisation during pre-training, preventing gradient explosions on heavy-tailed data distributions.

3. **Subspace orthogonal discriminative feature encoding (S-DFE)**: To strictly preserve column permutation invariance, FEAT introduces a dynamic approach where each feature column is assigned a distinct identity via random low-rank orthogonal matrices. This eliminates associative ordering bias while maintaining equidistant feature identities in the embedding space. Unlike static positional encodings, S-DFE provides inherent zero-shot adaptability to arbitrary schemas.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
FEAT was evaluated across 11 real-world structured-data datasets, demonstrating significant performance gains over existing approaches. On the largest dataset (500,000 samples), FEAT achieved a 40× faster inference speed compared to TabPFN while maintaining zero-shot performance parity (accuracy: 89.2% vs. 89.4% for TabPFN on classification tasks). The paper reports consistent improvements in both classification (average 2.3% absolute gain) and regression (average 1.8% RMSE reduction) across all 11 datasets. Statistical significance was measured via paired t-tests with p < 0.01 for all reported improvements.

## Related Work
FEAT builds upon existing structured-data foundation models like TabPFN (Hollmann et al. 2025) and LimiX (Zhang et al. 2025a), which use full-attention Transformers for in-context learning. Unlike these approaches, FEAT eliminates the O(N²) bottleneck through linear-complexity encoding. The work also extends linear sequence models (SSMs, Linear Transformers) by addressing their fundamental incompatibility with permutation-invariant structured data through dual-axis encoding. Unlike prior approaches that rely on synthetic-only pre-training, FEAT bridges the simulation-to-reality gap through a hybrid structural causal model pipeline.

## Limitations
The authors acknowledge that FEAT currently requires the input structure to remain consistent across different datasets (e.g., the same feature schema). The paper doesn't address handling schema changes dynamically, which would be required for real-world deployment where feature sets evolve. While the model handles heavy-tailed distributions well during pre-training, it doesn't explicitly discuss handling extreme outliers during inference. The paper also doesn't provide detailed analysis of computational resource requirements beyond inference speed.

## Appendix: Worked Example
Let's walk through how FEAT processes a healthcare dataset with 10,000 patient records (N=10,000) and 100 features (D=100), with a model dimension d=256.

1. **Cell-level embedding**: Each patient's 100 features (e.g., age, blood pressure, cholesterol levels) is embedded into a 256-dimensional space. Missing values (e.g., 5% of blood pressure entries) are replaced with a shared learnable token. For a patient with features [42, 120, 200, ...], the embedding produces a tensor X^(0) of shape [10,000, 100, 256].

2. **Feature-axis modelling**: For each patient (10,000 samples), 100 features are processed independently using multi-head self-attention. This captures relationships between features within a single patient (e.g., how blood pressure correlates with cholesterol). The output tensor remains [10,000, 100, 256].

3. **Sample-axis modelling**: The dual-axis encoding processes the sample dimension (10,000 patients) with two components:
   - AFBM (adaptive-fusion bi-Mamba-2) processes the sequence using bidirectional Mamba-2 layers, capturing local dependencies (e.g., how patients with similar age profiles relate to each other). The hidden state size is 512.
   - Conv-GLA (convolutional gated linear attention) maintains global memory through a convolutional gated linear attention mechanism with a kernel size of 100, accumulating global context across the entire patient population. This prevents representation collapse during linear-complexity modelling.

4. **Task-aware prediction**: The final contextualized representations are used to predict missing labels (e.g., disease risk scores) for 1,000 query patients, based on the relationships observed in the 9,000 labelled patients.

This processing demonstrates how FEAT handles a real-world healthcare dataset with 10,000 records in linear time (O(N)), whereas a full-attention model would require O(N²) = 100 million operations.

## References

- Zhenghang Song, Tang Qian, Lu Chen, Yushuai Li, Zhengke Hu, Bingbing Fang, Yumeng Song, Junbo Zhao, Sheng Zhang, Tianyi Li, "FEAT: A Linear-Complexity Foundation Model for Extremely Large Structured Data", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.16513

Tags: #structured-data #foundation-models #linear-complexity #dual-axis-encoding #huber-loss
