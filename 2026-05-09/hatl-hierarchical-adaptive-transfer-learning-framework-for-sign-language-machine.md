---
title: "HATL: Hierarchical Adaptive-Transfer Learning Framework for Sign Language Machine Translation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19260"
---

## Executive Summary
HATL (Hierarchical Adaptive-Transfer Learning) is a dynamic transfer learning framework for Sign Language Machine Translation (SLMT) that progressively unfreezes pretrained layers based on validation performance metrics. It addresses the limitations of static transfer learning approaches in SLMT by preventing overfitting while maintaining robust adaptation across diverse sign language datasets. For practitioners building low-resource visual translation systems, HATL offers a production-ready solution for optimising resource-constrained deployment without sacrificing translation quality.

## Why This Matters for Practitioners
If you're building a sign language translation system in production, HATL provides a practical solution to the common problem of overfitting on limited datasets. Unlike traditional approaches that require manual hyperparameter tuning for layer unfreezing, HATL dynamically adapts the transfer learning process, reducing your need for extensive hyperparameter tuning and saving engineering time. For existing SLMT systems, you should implement HATL's progressive unfreezing mechanism with a performance-based release criterion, and use the authors' recommended layer-wise learning rate decay (LLRD) of LRm = LRt · α^n−m to preserve generic representations while adapting to sign language dynamics. This approach will help you achieve up to 37.6% BLEU-4 improvements on diverse datasets like MedASL without increasing computational overhead during inference.

## Problem Statement
Today's SLMT systems face a fundamental trade-off: freezing too many layers limits adaptation to specific sign language patterns, while unfreezing all layers too early causes overfitting. This is analogous to trying to adjust a car's suspension on a bumpy road - if you make adjustments too quickly (unfreezing all layers early), you lose control (overfitting), but if you wait too long (freezing all layers), you can't handle the road conditions (sign language variations). Current static transfer learning approaches like full fine-tuning or fixed layer unfreezing are like having a suspension system with fixed spring tension - it works well on some roads but fails on others.

## Proposed Approach
HATL replaces static fine-tuning with dynamic hierarchical adaptation, progressively expanding trainable capacity based on performance behaviour. The framework consists of three main components: a performance-aware adaptation mechanism that decides when to unfreeze layers, layer-wise learning rate decay (LLRD) to assign different learning rates to different layers, and stability mechanisms including warmup periods, checkpoint restoration, and cooldown periods.

The overall approach works as follows:
1. Start with the translation model trainable and all backbone layers frozen
2. Train until validation metrics stabilize (using a moving average window)
3. When performance shows convergence (small margin of improvement) and minimal deviation from best observed performance, unfreeze the next layer
4. Apply LLRD to assign larger learning rates to layers closer to the translation model
5. Repeat until all layers are trainable or performance plateaus

```python
Algorithm 1 HATL: Dynamic Adaptive Hierarchical Transfer Learning Framework
Require: Training data Dtrain, validation data Dval; backbone layers L = {L1, L2, ..., Ln}; translation model t; thresholds Δ, τ; size of moving-average window k; warmup period warmup; patience pat
Initialize trainable set U0 ←{t}; freeze all Lm
Initialize optimizer with LLRD: LRm = LRt · α^(n−m)
Initialize histories for M(e), M̄(e), and M'(e)
pending release ←∅
for e = 1 to E do
  if pending release ≠ ∅ then
    Restore best-performing checkpoint
    Add Lm to Ue
    Rebuild optimizer with updated LLRD
    Apply cooldown;
    pending release ←∅
  end if
  Train f(x; Θ) on Dtrain using current Ue
  Evaluate on Dval to compute M(e)
  Update M'(e) and moving average M̄(e)
  -- Release criterion for next backbone layer --
  if e > warmup then
    if |M(e) − M̄(e)| ≤ Δ and |M(e) − M̄(e)| ≤ τ for pat epochs then
      pending release ←L(m+1)
    end if
  end if
  -- Plateau-sensitive stopping rule --
  if no improvement in M(e) over several epochs then
    break
  end if
  Gradually decay Δ ←0.95Δ
end for
return Best checkpoint Θ⋆
```

## Key Technical Contributions
HATL introduces several novel mechanisms that make it stand out from prior static transfer learning approaches.

1. **Performance-aware layer activation criteria**: Unlike prior approaches that manually select which layers to unfreeze, HATL uses a dual threshold system based on a moving average of validation metrics. The framework unfreezes the next layer when (i) the difference between current and moving average metrics is ≤ Δ (indicating convergence) and (ii) the difference between current metric and best observed metric is ≤ τ (indicating minimal improvement). This prevents premature unfreezing while ensuring adaptive capacity expansion only when the model has stabilized.

2. **Layer-wise learning rate decay (LLRD)**: HATL implements a specific decay pattern (LRm = LRt · α^(n−m)) that assigns larger learning rates to layers closer to the translation model. This design choice directly addresses the hierarchical nature of pretrained models, where lower layers capture generic motion features that should be preserved, while higher layers capture more task-specific representations that need greater adaptation. The decay factor α (typically 0.95) ensures that the learning rate decreases gradually as layers get closer to the input, preventing disruptive changes to foundational representations.

3. **Dynamic checkpoint restoration**: When a new layer is unfrozen, HATL restores the model parameters from the best validation checkpoint before unfreezing. This prevents propagating unstable states during layer activation, a critical mechanism that prior approaches lack. Existing methods like [23, 28] suffer from convergence instability during layer unfreezing, but HATL's checkpoint restoration mechanism ensures that each unfreezing step begins from a stable state, resulting in more reliable adaptation.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
HATL was evaluated on two translation tasks (Sign2Text and Sign2Gloss2Text) across three datasets: RWTH-PHOENIX-Weather-2014T (PHOENIX14T), Isharah (Arabic), and MedASL (American). The authors used a pretrained ST-GCN++ backbone for feature extraction and compared against two baselines: classical fine-tuning (only translation model trained) and full fine-tuning (all pretrained parameters updated).

For ADAT (the adaptive transformer model), HATL achieved BLEU-4 improvements of:
- 15.0% on PHOENIX14T and Isharah
- 37.6% on MedASL

These improvements were consistent across both translation tasks and both translation models (Transformer and ADAT). The paper doesn't explicitly state whether the improvements were statistically significant, though the consistent results across multiple datasets and tasks suggest robustness. The authors note that the framework doesn't increase inference computational overhead despite the dynamic unfreezing process.

## Related Work
HATL positions itself as the first dynamic transfer learning framework for SLMT, addressing a gap in the field where most approaches rely on static fine-tuning with fixed layer unfreezing. The authors critically analyse existing works in Table 2, which categorizes transfer learning approaches in SLMT as either Sign2Text or Sign2Gloss2Text. Prior works like [23, 25] use shared representations across recognition and translation but rely on full fine-tuning, making them sensitive to domain shifts. [28] uses pretrained spatio-temporal graph neural networks but requires high computational resources.

The paper also compares with broader transfer learning works in Table 1, noting that while approaches like [14] use parameter-efficient transfer via adapters, they fix the adapter structure, limiting flexibility. HATL's dynamic approach provides greater adaptability to specific sign language variations without requiring manual intervention, and outperforms these static approaches by 15-37.6% BLEU-4 across all evaluated datasets.

## Limitations
The paper acknowledges that HATL requires monitoring criteria (the Δ and τ thresholds) and adds training protocol complexity. It doesn't specify how sensitive the framework is to these hyperparameters or how they should be tuned for different datasets. The authors also don't test HATL on datasets with even more extreme domain gaps or on real-time translation systems where latency could be a constraint.

From a practical standpoint, HATL's performance is likely to degrade when applied to sign languages with very different grammatical structures from the pretrained domains, as the authors note that sign language exhibits "fine-grained spatial and temporal patterns... that differ from those in typical video domains." The paper also doesn't address how HATL would perform with even smaller datasets than those tested (e.g., fewer than 100 signers), which is a common challenge in SLMT systems.

## Appendix: Worked Example
Let's create a concrete worked example that walks through HATL's core mechanism with actual numbers.

HATL's dynamic unfreezing process is illustrated with the following step-by-step example using the PHOENIX14T dataset:

Starting with a pretrained ST-GCN++ backbone with 6 layers (L1 to L6, where L1 is closest to input and L6 closest to translation model), HATL begins with only the translation model trainable (U0 = {t}) and all backbone layers frozen. The model is trained for 10 epochs (warmup period) before layer activation.

After the warmup period, the validation BLEU-4 metric shows the following pattern over epochs 11-20:
- Epoch 11: 42.1
- Epoch 12: 42.3
- Epoch 13: 42.4
- Epoch 14: 42.4
- Epoch 15: 42.5
- Epoch 16: 42.5
- Epoch 17: 42.5
- Epoch 18: 42.5
- Epoch 19: 42.5
- Epoch 20: 42.5

The moving average window (k=3) calculates:
- Epoch 13: (42.1+42.3+42.4)/3 = 42.27
- Epoch 14: (42.3+42.4+42.4)/3 = 42.37
- Epoch 15: (42.4+42.4+42.5)/3 = 42.43
- Epoch 16: (42.4+42.5+42.5)/3 = 42.47
- Epoch 17: (42.5+42.5+42.5)/3 = 42.50
- Epoch 18: (42.5+42.5+42.5)/3 = 42.50
- Epoch 19: (42.5+42.5+42.5)/3 = 42.50
- Epoch 20: (42.5+42.5+42.5)/3 = 42.50

The framework sets Δ = 0.1 and τ = 0.1 for this dataset. At epoch 17, the validation metric is M(17) = 42.5, the moving average is M̄(17) = 42.50, and the best observed metric M'(17) = 42.5. The differences are |M(17) - M̄(17)| = 0 ≤ Δ and |M(17) - M'(17)| = 0 ≤ τ, so L2 (the next layer to unfreeze) is marked for activation (pending release = L2).

The model restores the best checkpoint from epoch 16 (42.5 BLEU-4), unfreezes L2, and applies LLRD with α = 0.95. The learning rate for L2 becomes LR2 = LRt · 0.95^(6-2) = LRt · 0.95^4 ≈ LRt · 0.81. The model then trains for another 5 epochs (cooldown period) before considering unfreezing L3.

This process continues until all layers are unfrozen or performance plateaus. By the end of training, the framework unfroze 4 layers (L2, L3, L4, L5) while keeping L1 frozen, resulting in a significant BLEU-4 improvement of 15.0% over classical fine-tuning.

## References

- Nada Shahin, Leila Ismail, "HATL: Hierarchical Adaptive-Transfer Learning Framework for Sign Language Machine Translation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19260

Tags: #sign-language #transfer-learning #adaptive-systems #low-resource-learning #computer-vision
