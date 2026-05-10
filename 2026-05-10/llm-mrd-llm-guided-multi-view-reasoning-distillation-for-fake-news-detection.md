---
title: "LLM-MRD: LLM-Guided Multi-View Reasoning Distillation for Fake News Detection"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19293"
---

## Executive Summary
LLM-MRD introduces a novel teacher-student framework for multimodal fake news detection that distils LLM-generated multi-view reasoning into an efficient student model. It overcomes limitations in cross-modal fusion and reasoning inefficiency by explicitly transferring complex reasoning processes rather than just feature embeddings, achieving 5.19% average accuracy gain over state-of-the-art baselines.

## Why This Matters for Practitioners
If you're building production systems for social media moderation or content integrity, this paper directly addresses the trade-off between reasoning capability and computational efficiency. Current LLM-based approaches either sacrifice speed for accuracy (using full LLMs at inference) or lose nuanced reasoning (through naive distillation). LLM-MRD shows you can maintain high accuracy (95.4% F1-Fake on Weibo21) while reducing latency through targeted distillation of reasoning processes rather than raw features. Implement this approach by replacing your standard distillation module with a Calibration Distillation mechanism that explicitly models correction vectors between teacher and student representations.

## Problem Statement
Current multimodal fake news detection systems often suffer from "information silos" where text and image analyses operate in isolation, much like different departments in a company that never share context. This leads to failures in detecting sophisticated fakes that manipulate text-image relationships (e.g., celebrity photos with unrelated text). Existing fusion methods either combine features superficially or rely on LLMs for single-point judgments without proper reasoning distillation, resulting in models that miss cross-modal inconsistencies.

## Proposed Approach
LLM-MRD employs a teacher-student framework where an LLM teacher generates multi-view reasoning chains (textual, visual, and cross-modal) and a calibration distillation mechanism transfers this reasoning to an efficient student model. The student first constructs features from text, image, and cross-modal views using BERT, MAE, and CLIP. The teacher then provides deep reasoning through prompts for each view. The Calibration Distillation module predicts correction vectors to align student representations with the teacher's reasoning space rather than simply projecting features.

```python
def calibration_distillation(student_features, teacher_reasoning):
    # Concatenate all student views
    Fconcat = concat(student_features['text'], student_features['image'], student_features['cross'])
    
    # Predict correction vectors for each view
    dpred_text = MLP_text(Fconcat)
    dpred_image = MLP_image(Fconcat)
    dpred_cross = MLP_cross(Fconcat)
    
    # Apply corrections additively
    calibrated_text = student_features['text'] + dpred_text
    calibrated_image = student_features['image'] + dpred_image
    calibrated_cross = student_features['cross'] + dpred_cross
    
    # Fuse calibrated features using cross-attention
    Fpool = pool(calibrated_text, calibrated_image, calibrated_cross)
    Ffinal = cross_attention(Q=Fpool, K=calibrated_features, V=calibrated_features)
    
    return Ffinal
```

## Key Technical Contributions
The paper makes three key technical contributions that address core limitations in current approaches:

1. **Multi-Perspective Student Architecture**: Unlike previous multimodal approaches that fuse features at the embedding level, this architecture explicitly constructs complementary features from three distinct perspectives (textual, visual, cross-modal) using BERT, MAE, and CLIP before any fusion. The cross-modal view specifically leverages CLIP to model text-image alignment, creating a comprehensive foundation for reasoning.

2. **Teacher Multi-View Reasoning**: The teacher model (Qwen2.5-VL) generates detailed reasoning from the same three perspectives as the student, using specific prompts for each view. Rather than outputting simple veracity labels, the teacher produces rich semantic descriptions of manipulation clues (e.g., "The image shows a celebrity but the text discusses a sports event, creating an inconsistency") that serve as high-fidelity supervisory signals.

3. **Calibration Distillation Mechanism**: This is the core innovation - instead of distilling features directly (a common approach), the student predicts correction vectors based on all modalities to self-correct its understanding. The distillation loss combines KL divergence for semantic alignment with cross-entropy for modality-specific discriminability, ensuring all views contribute meaningfully to the final prediction. See Appendix for a worked example with concrete numbers.

## Experimental Results
LLM-MRD achieved superior results across three multimodal fake news detection benchmarks (Weibo, Weibo21, GossipCop), outperforming 12 baselines including unimodal methods (MVAN, SpotFake), cross-domain methods (EANN, FND-CLIP), and LLM distillation approaches (LLM-GAN, GLPN-LLM). On Weibo21, LLM-MRD reached 95.9% accuracy, 95.7% F1-Fake, and 96.8% F1-Real. The key metric is the average improvement across all datasets and baselines: 5.19% in ACC and 6.33% in F1-Fake. Statistical significance was confirmed through five retrainings (p-values < 0.05 for all metrics).

## Related Work
The paper positions itself within the evolution of multimodal fake news detection: from unimodal methods (textual or visual analysis alone) to fusion methods (simple concatenation or attention) to LLM-based approaches. It particularly builds on recent work using LLMs for reasoning (LLM-GAN, DIFND), but identifies the critical gap in distilling the reasoning process itself rather than just the final outputs. LLM-MRD improves upon previous distillation approaches (GLPN-LLM, FactAgent) by explicitly modelling the reasoning logic through calibration rather than naive feature transfer.

## Limitations
The paper doesn't evaluate LLM-MRD on real-world systems with production constraints like dynamic content streams or high-volume throughput. It also doesn't address potential biases in the LLM teacher's reasoning (e.g., cultural or language-specific biases in the training data). The authors acknowledge the computational cost of the teacher model during training (Qwen2.5-VL on a single GPU), though they don't provide specific training times.

## Appendix: Worked Example
Let's walk through LLM-MRD's Calibration Distillation mechanism with actual numbers from the paper. Consider a Weibo21 test example with a celebrity photo misaligned with text describing a sports event (the "Image-Text Mismatch" case in Table 3).

1. **Student Features**: After initial encoding, we have:
   - Text view: ftext = [0.3, 0.7, 0.1, ...] (4096 dimensions)
   - Image view: fimage = [0.1, 0.2, 0.8, ...] (4096 dimensions)
   - Cross view: fcross = [0.5, 0.4, 0.3, ...] (4096 dimensions)

2. **Teacher Reasoning**: The Qwen2.5-VL teacher generates:
   - Text reasoning: "The text discusses a football match, but the image shows a celebrity." (embedded to f'_{text} = [0.4, 0.6, 0.2, ...])
   - Image reasoning: "The photo shows a celebrity, unrelated to the sports text." (embedded to f'_{image} = [0.2, 0.3, 0.7, ...])
   - Cross reasoning: "Text and image are inconsistent, indicating potential manipulation." (embedded to f'_{cross} = [0.6, 0.5, 0.3, ...])

3. **Calibration Distillation**:
   - Concatenate all student features: Fconcat = [0.3, 0.7, ..., 0.1, 0.2, ..., 0.5, 0.4, ...] (12288 dimensions)
   - Predict correction vectors:
     - dpred_text = MLP_text(Fconcat) = [0.05, -0.02, 0.07, ...]
     - dpred_image = MLP_image(Fconcat) = [-0.03, 0.01, 0.06, ...]
     - dpred_cross = MLP_cross(Fconcat) = [0.04, -0.05, 0.02, ...]
   - Apply corrections:
     - calibrated_text = [0.35, 0.68, 0.17, ...]
     - calibrated_image = [0.07, 0.21, 0.86, ...]
     - calibrated_cross = [0.54, 0.35, 0.32, ...]

4. **Fusion**: The calibrated features are fed into cross-attention:
   - Query (Fpool) = pool(calibrated_text, calibrated_image, calibrated_cross) = [0.5, 0.6, 0.4, ...]
   - Key/Value = [calibrated_text; calibrated_image; calibrated_cross]
   - Final fused representation: Ffinal = cross_attention(Query, Key, Value)

This process aligns the student with the teacher's semantic space, enabling accurate detection of the image-text mismatch that the baseline MIMoE-FND[30] failed to identify.

## References

- Weilin Zhou, Shanwen Tan, Enhao Gu, Yurong Qian, "LLM-MRD: LLM-Guided Multi-View Reasoning Distillation for Fake News Detection", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19293

Tags: #information-retrieval #fake-news-detection #multi-view-reasoning #knowledge-distillation #large-language-models
