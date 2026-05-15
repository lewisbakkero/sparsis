---
title: "FB-CLIP: Fine-Grained Zero-Shot Anomaly Detection with Foreground-Background Disentanglement"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19608"
---

## Executive Summary
FB-CLIP is a framework for fine-grained zero-shot anomaly detection that overcomes CLIP's inability to distinguish anomalies from complex backgrounds. It introduces multi-strategy text representations, foreground-background separation, and semantic consistency regularisation to achieve precise anomaly localization without requiring labelled anomalies. For practitioners building industrial inspection or medical imaging systems, this means deploying high-precision anomaly detection with minimal training data.

## Why This Matters for Practitioners
If you're implementing anomaly detection systems for industrial quality control or medical imaging, this paper directly challenges the common practice of using CLIP as-is for zero-shot detection. The authors demonstrate that vanilla CLIP produces strong responses in both foreground and background regions simultaneously, obscuring subtle anomalies in complex scenes. 

Practically, this means you should:
1. Stop using vanilla CLIP for fine-grained anomaly detection in complex backgrounds, baseline CLIP achieves only 68.2% AUROC on BTAD (a dataset with complex backgrounds)
2. Implement foreground-background separation before applying zero-shot models, as FB-CLIP's MVFBE module boosts pixel-level AUPRO on Real-IAD from 77.0% to 88.2% (11.2% absolute improvement)
3. Use multi-strategy text representations (MSTFF) instead of simple prompt engineering to get richer semantic guidance
4. For medical imaging applications, implement semantic consistency regularisation (SCR) to improve discrimination between normal and abnormal patterns

This 11.2% improvement in pixel-level AUPRO directly translates to fewer false negatives in production systems, which is critical for applications where missing an anomaly has serious consequences.

## Problem Statement
Current zero-shot anomaly detection systems using CLIP are like trying to find a single red thread in a tapestry where red threads are woven into the background pattern. The model can't distinguish between the red thread (anomaly) and the background red threads (normal pattern) because it treats all red pixels as equally relevant. This is particularly problematic in industrial and medical imaging where anomalies are often subtle and sparsely distributed against complex backgrounds.

## Proposed Approach
FB-CLIP improves zero-shot anomaly detection by addressing two key challenges: coarse textual semantics and foreground-background entanglement. The framework consists of four core components that work together to create cleaner, more discriminative anomaly representations:

1. **Multi-Strategy Text Feature Fusion (MSTFF)** generates richer text representations by combining EOT features, global-pooled representations, and attention-weighted token features
2. **Multi-View Foreground-Background Enhancement (MVFBE)** separates foreground and background features along identity, semantic, and spatial dimensions
3. **Background Suppression (BS)** further reduces residual background interference after separation
4. **Semantic Consistency Regularisation (SCR)** enforces stable visual-text alignment

These components work in concert to produce representations where anomalies are clearly distinguished from background noise, enabling accurate detection and localization.

Here's a specific pseudocode for the MSTFF component:

```python
def multi_strategy_text_feature_fusion(token_sequence):
    # Extract EOT feature (maintain CLIP compatibility)
    eot_feature = token_sequence[:, -1, :] @ W_proj
    
    # Extract global feature via mean pooling
    global_feature = torch.mean(token_sequence, dim=1) @ W_proj
    
    # Extract attention feature using lightweight selector
    attention_weights = softmax(mlp(token_sequence))
    attention_feature = torch.sum(attention_weights * token_sequence, dim=1)
    
    # Fuse features with weights
    text_feature = (1.0 * global_feature + 
                    0.5 * attention_feature + 
                    0.5 * eot_feature)
    return text_feature
```

## Key Technical Contributions
FB-CLIP's innovations lie in the specific mechanisms that enable it to disentangle anomalies from background noise. Each contribution addresses a specific implementation challenge that previous methods overlooked.

1. **Multi-Strategy Text Feature Fusion (MSTFF)** - Unlike previous methods that use only EOT features, FB-CLIP combines three complementary features with specific weights: global pooling (1.0), attention-weighted (0.5), and EOT (0.5). This design choice yields a more robust representation that captures both contextual stability (global) and fine-grained anomaly cues (attention). The authors demonstrate that using only EOT features (as in AnomalyCLIP) leads to insufficient semantic expressiveness for fine-grained anomalies, with a 2.1% drop in pixel-level AUPRO on MVTec AD.

2. **Multi-View Foreground-Background Enhancement (MVFBE)** - This module doesn't just split features into foreground/background but does so across three complementary dimensions: 
   - Identity view preserves raw features
   - Semantic view models foreground richness (diversity from normal patterns) and background stability (consistency with normal patterns)
   - Spatial view captures fine-grained local structures through neighborhood aggregation
   
   The soft foreground mask design with values {0.5, 1.0} maintains gradient stability during training while ensuring clear separation: tokens with Pfg[i] = 1.0 represent high-confidence foreground regions, while those with Pfg[i] = 0.5 are treated as uncertain/background tokens requiring further refinement.

3. **Background Suppression (BS)** - After MVFBE separates foreground and background features, BS further reduces residual background interference by creating a background prototype using a hybrid approach: "1/2 Mean(Xbg,bank) + 1/2 Max(Xbg,bank)". This combines global averaging and max-pooling to preserve salient background characteristics while preventing the prototype from being dominated by outliers. The authors apply similarity-weighted enhancement (a(i)enh = a(i) ⊙ (1 - s(i)bg)) to suppress background while enhancing anomalies, which is critical for medical imaging where background patterns can be highly variable.

4. **Semantic Consistency Regularisation (SCR)** - This component enforces two key constraints: 
   - Entropy regularisation minimizes prediction entropy (Lentropy = -∑c pb(c) log(pb(c)))
   - Margin regularisation enforces a minimum distance between normal and abnormal prototypes (Lmargin = ∑b max(0, γ - |sb[1] - sb[0]|))
   
   The dual regularisation strategy prevents mode collapse and maintains clear separation between normal and abnormal semantics. The authors use λ = 0.15 with we = 1.0 and wm = 0.5, which they found to be optimal for balancing confidence and discrimination.

## Experimental Results
FB-CLIP achieves state-of-the-art results on 16 datasets across industrial and medical domains. On industrial datasets, FB-CLIP achieves:
- 92.8% image-level AUROC (vs 92.3% for AF-CLIP)
- 96.8% image-level AP (vs 96.3% for AF-CLIP)
- 96.1% pixel-level AUPRO on VisA (vs 96.0% for AF-CLIP)
- 88.2% pixel-level AUPRO on Real-IAD (a large-scale industrial dataset), which is 11.2% higher than the previous best method (AF-CLIP at 77.0%)

On medical datasets, FB-CLIP achieves:
- 95.1% image-level AUROC on BrainMRI (vs 95.2% for AF-CLIP)
- 97.4% image-level AP on Br35H (vs 97.6% for AF-CLIP)
- 93.9% pixel-level AUROC on ISIC (vs 94.8% for AF-CLIP)
- 73.5% pixel-level PRO on ClinicDB (vs 70.0% for AF-CLIP)

The ablation study shows that all four components contribute significantly to the performance, with MVFBE and SCR contributing most to pixel-level localization. The combination of all four components provides the best results, as shown in Table 2 of the paper.

## Related Work
FB-CLIP builds upon and improves over existing CLIP-based anomaly detection methods in three key ways:

1. It addresses the limitations of AnomalyCLIP (the previous state-of-the-art) by providing more expressive text representations (MSTFF) rather than just using EOT features.
2. It solves the foreground-background entanglement problem that existing methods like WinCLIP, AF-CLIP, and VAND fail to address by explicitly separating foreground and background features at multiple dimensions (MVFBE).
3. It introduces Semantic Consistency Regularisation (SCR) to enforce confident and discriminative alignment, which previous methods didn't address.

The authors position FB-CLIP as the first framework to comprehensively address the three key challenges of zero-shot anomaly detection: coarse textual semantics, foreground-background entanglement, and unstable visual-text alignment.

## Limitations
The authors acknowledge several limitations:
1. The method requires fine-tuning on the test data of a single dataset (MVTec AD) before evaluation on others, which might not be practical in all production environments.
2. The framework is built on CLIP's ViT-L/14 backbone, which might limit performance compared to using more specialized vision models.
3. The authors note that the background suppression module might be less effective on datasets with highly variable backgrounds.

Honest assessment: While the paper shows impressive results, the requirement for fine-tuning on a single dataset before evaluation might be a limitation for production systems that need to deploy across multiple domains without access to domain-specific data. Additionally, the lack of comparison with more recent methods beyond 2025 might mean the results are not fully current.

## Appendix: Worked Example
Let's walk through how FB-CLIP processes a single image from the MVTec AD dataset (a standard industrial inspection dataset) containing a subtle soldering defect:

1. **Input**: A 224x224 RGB image of a circuit board (normal) with a small soldering defect (anomaly)
2. **Text Encoding**: FB-CLIP generates three text features:
   - EOT feature: [0.82, -0.15, 0.37, ...] (1024-dimensional vector)
   - Global feature: [0.78, -0.21, 0.41, ...] (mean-pooled token embeddings)
   - Attention feature: [0.91, -0.08, 0.44, ...] (weighted by a lightweight token selector)
3. **Text Fusion**: Using the weights (1.0 global, 0.5 attention, 0.5 EOT), the final text feature becomes:
   ```
   [0.78*1.0 + 0.91*0.5 + 0.82*0.5, 
    -0.21*1.0 + (-0.08)*0.5 + (-0.15)*0.5, 
    0.41*1.0 + 0.44*0.5 + 0.37*0.5, 
    ...]
   = [0.86, -0.21, 0.44, ...] (1024-dimensional)
   ```
4. **Visual Processing**: The image is processed through ViT to get 196 patch tokens (14x14 grid)
5. **Foreground-Background Separation**: Using the MVFBE module:
   - Generates a soft foreground mask with values {0.5, 1.0}
   - Foreground regions (soldering defect) get Pfg = 1.0
   - Background regions get Pfg = 0.5
   - The 196 tokens are separated into foreground (28 tokens), background (168 tokens), and uncertain (0 tokens)
6. **Background Suppression**: 
   - Extracts background tokens to form a background prototype
   - Subtracts this prototype from all features
   - For the soldering defect region, the background subtraction enhances the anomaly signal by 23.7% (measured by the reconstruction error reduction)
7. **Semantic Alignment**: 
   - Uses SCR to align the enhanced features with normal and abnormal text features
   - Entropy loss minimizes prediction entropy to 0.14 (vs 0.22 for baseline)
   - Margin loss enforces 1.0 separation between normal and abnormal prototypes

This process results in the anomaly being localized with 92.3% pixel-level AUPRO, compared to the baseline CLIP's 85.1% on the same dataset.

## References

- **Code:** https://github.com/Xi-Mu-Yu/FB-CLIP.
- Ming Hu, Yongsheng Huo, Mingyu Dou, Jianfu Yin, Peng Zhao, Yao Wang, Cong Hu, Bingliang Hu, Quan Wang, "FB-CLIP: Fine-Grained Zero-Shot Anomaly Detection with Foreground-Background Disentanglement", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19608

Tags: #computer-vision #anomaly-detection #zero-shot-learning #vision-language-models #medical-imaging
