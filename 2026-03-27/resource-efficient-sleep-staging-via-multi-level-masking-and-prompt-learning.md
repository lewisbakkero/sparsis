---
title: "Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning"
category: "AI Applications"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36958"
---

## Executive Summary

The authors propose MASS (Mask-Aware Sleep Staging), a neural network framework that enables accurate sleep staging using only 10% of the EEG signal typically required, addressing the critical resource constraints of wearable sleep monitoring systems. By implementing multi-level masking and hierarchical prompt learning, MASS maintains high accuracy even when significant portions of the signal are masked, making it ideal for battery-constrained devices where continuous full-signal acquisition is impractical.

## Why This Matters for Practitioners

If you're building wearable sleep monitoring systems with tight battery constraints, this paper directly addresses how to optimize data acquisition to reduce power consumption without sacrificing accuracy. The MASS framework demonstrates you can safely reduce signal acquisition to 10% of the original data (from 30-second epochs) while maintaining near-full accuracy—meaning your device can pause sampling during masked segments, significantly extending battery life. For deployment on edge devices with limited resources, implement a sampling strategy that intentionally skips data points during acquisition (not just during processing), using the masking approach described rather than trying to capture full signals. This isn't just about theoretical efficiency—it's about enabling 24/7 sleep monitoring for weeks on a single battery charge, which is essential for home health monitoring systems.

## Problem Statement

Traditional sleep staging requires full 30-second EEG recordings for every epoch—like demanding a car to run its engine at full throttle for every mile during a road trip, when what you really need is to capture just the critical moments: the turns, the traffic lights, and the stops. For wearables, this means collecting continuous full-bandwidth data at 200-250 Hz, which drains batteries in hours instead of days. It's like making your smartphone record 4K video all night long—technically possible, but completely impractical for real-world use.

## Proposed Approach

MASS solves this by training a model to work with partial data through two key mechanisms: multi-level masking and hierarchical prompt learning.

The system first divides each 30-second EEG epoch into 30 one-second patches (each patch is 200-250 data points). During training, MASS applies two types of random masking:
- Epoch-level masking: randomly masks entire sleep epochs (e.g., 80% of epochs masked, 20% visible)
- Patch-level masking: within visible epochs, randomly masks patches (e.g., 50% of patches masked per epoch)

This forces the model to learn features from incomplete data rather than relying on full sequences. To compensate for lost context, MASS uses a global prompt learning mechanism: the visible EEG patches across all visible epochs are aggregated into a single semantic anchor (a global prompt token) using a shallow Transformer with positional encoding. This global prompt is then injected into both patch-level and epoch-level modeling processes to guide feature extraction.

Here's the core algorithm for generating the global prompt:

```python
def generate_global_prompt(visible_patches):
    # visible_patches: a list of spectral-domain patch features [Nvis, da]
    Nvis = len(visible_patches)
    
    # Add CLS token at position 0
    cls_token = learnable_token  # [1, da]
    prompt_sequence = [cls_token] + visible_patches
    
    # Apply fixed positional encoding based on original positions
    positions = get_original_positions(visible_patches)  # [Nvis+1] indices
    pos_encoding = sinusoidal_encoding(positions, da)
    
    # Add positional encoding to the sequence
    sequence_with_pos = prompt_sequence + pos_encoding
    
    # Process through shallow Transformer
    prompt_output = transformer_encoder(sequence_with_pos, num_layers=4)
    
    # Extract global prompt token from CLS position
    global_prompt = prompt_output[0]  # [1, da]
    
    return global_prompt
```

## Key Technical Contributions

MASS introduces three key innovations that fundamentally change how sleep staging models can operate under resource constraints:

1. **Multi-level masking strategy**: Instead of requiring full signals, MASS intentionally masks both entire sleep epochs and patches within unmasked epochs during training. This forces the model to learn to recognize sleep stages from partial observations rather than relying on complete inputs. The approach differs from prior work because it incorporates masking as a core part of the training process (not just a test-time scenario) and enables the model to generalize across varying signal completeness.

2. **Hierarchical prompt learning**: Rather than ignoring context lost from masking, MASS aggregates visible patches across all epochs into a single global prompt using a shallow Transformer with positional encoding. This prompt serves as a semantic anchor that guides both patch-level feature extraction (within each epoch) and epoch-level modeling (across the sequence), preserving global context that would otherwise be lost.

3. **Resource-efficient deployment without performance tradeoffs**: The framework achieves state-of-the-art results at 10% signal integrity while maintaining computational complexity similar to baseline models. Unlike traditional approaches that reduce model size to save resources, MASS focuses on data efficiency—reducing the amount of data needed for acquisition rather than changing the model architecture.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results

MASS was evaluated on four public sleep staging datasets with different numbers of subjects:
- DREAMS-SUB: 20 subjects
- Sleep-EDF-20: 20 subjects
- Sleep-EDF-78: 78 subjects
- SHHS: 329 subjects

At 10% signal integrity (only 10% of the original EEG signal used for inference), MASS achieved:
- Average macro-F1: 75.58% (DREAMS-SUB), 76.62% (Sleep-EDF-20), 71.61% (Sleep-EDF-78), 70.25% (SHHS)
- Accuracy drop compared to full signal: 3.45% (DREAMS-SUB), 2.12% (Sleep-EDF-20), 2.44% (Sleep-EDF-78), 3.67% (SHHS)

MASS outperformed all baselines at 10% signal integrity:
- By up to +64.5% macro-F1 over the strongest baseline on DREAMS-SUB
- By up to +52.8% macro-F1 on Sleep-EDF-20
- By up to +41.2% macro-F1 on Sleep-EDF-78
- By up to +43.8% macro-F1 on SHHS

The paper does not explicitly state statistical significance for these results, but it does note that MASS consistently outperformed baselines across all signal integrity levels.

## Related Work

The paper positions itself against two main categories of prior work: 
- Sleep staging models that rely on complete EEG signals (DeepSleepNet, AttnSleep, TinySleepNet, CNN-Transformer-LSTM, LGSleepNet, NeuroNet) that cannot handle partial observations
- Resource-efficient approaches that focus solely on model parameters (TinySleepNet) or hardware-level power management (ADS1299, ADS1294), but not on signal acquisition efficiency

MASS is positioned as the first approach that explicitly addresses resource constraints in wearable sleep monitoring by optimizing for data efficiency through a novel masking and prompt learning strategy, rather than focusing on model size reduction or hardware design.

## Limitations

The paper doesn't explicitly state limitations, but based on the description, the following are notable gaps:
- The method requires a minimum amount of signal (10% in the paper) to maintain accuracy—that means it wouldn't work for extremely sparse signals beyond this threshold
- The approach was tested on standard EEG channels (Cz-A1, Fpz-Cz, C4-A1) but not on other sensor configurations
- The paper doesn't discuss how well the model transfers between different subject populations, which is important for real-world deployment
- The authors don't test the latency impact of the masking and prompt learning process in real-time systems

My assessment: This method is promising but likely requires adaptation for extreme resource-constrained environments (like ultra-low-power sensors with less than 10% signal integrity) and may need additional calibration for diverse populations.

## Appendix: Worked Example

Let's walk through a concrete example of how MASS processes a single 30-second sleep epoch with 10% signal integrity using the masking ratios from the paper's most extreme setting (ra = 0.8, re = 0.5).

Starting with 30-second sleep epoch (30 one-second patches):
- Full signal: 30 patches (1-30)

Apply epoch-level masking (re = 0.5): randomly mask 50% of epochs. For simplicity, assume we're processing a single epoch (the masking is applied across multiple epochs in the full system), so all 30 patches are visible in this epoch.

Apply patch-level masking (ra = 0.8): randomly mask 80% of patches, leaving 20% visible. This means 6 patches are visible (30 × 0.2 = 6).

Visible patches in order (based on the paper's masking pattern):
- Patches 3, 5, 7, 12, 20, 25 (6 patches total)

Each visible patch is transformed using power spectral density (PSD) computation to get spectral features, which are then linearly projected to 128 dimensions (da = 128 as stated in the paper). MASS operates on frequency-domain features of the patches, which is part of why it's so robust to noise—it's looking at "brain wave signatures" rather than raw, jittery voltage lines.

These 6 visible patches become:
- 6 × 128-dimensional vectors: [e1, e2, e3, e4, e5, e6]

The global prompt learning process:
1. Flatten all visible patches across the sequence (in this example, just these 6 patches)
2. Add a learnable CLS token (z0) at position 0
3. Apply fixed sinusoidal positional encoding based on original patch positions (3,5,7,12,20,25)
4. Process through a 4-layer Transformer encoder (Lp = 4 as specified)
5. Extract the output corresponding to the CLS token position as the global prompt (zprompt)

The resulting global prompt is a 128-dimensional vector that captures the semantic context of the visible patches with their original temporal positions preserved.

This global prompt is then injected into:
- The patch-level modeling: For each visible patch in the epoch, the prompt is combined with the patch features
- The epoch-level modeling: The global prompt guides the modeling of the temporal sequence across multiple epochs

This process allows the model to reconstruct the full sleep staging information from just 6 of the original 30 patches (20% of the signal), which matches the 10% signal integrity level described in the paper (since the epoch-level masking would further reduce the number of visible epochs).


## References

- **Code:** https://github.com/AnsonAiTRAY/MASS
- Lejun Ai, Yulong Li, Haodong Yi, Jixuan Xie, Yue Wang, Jia Liu, "Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36958
