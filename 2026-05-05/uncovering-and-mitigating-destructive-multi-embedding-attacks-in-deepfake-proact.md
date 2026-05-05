---
title: "Uncovering and Mitigating Destructive Multi-Embedding Attacks in Deepfake Proactive Forensics"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/37010"
---

## Executive Summary
This paper identifies a critical vulnerability in current deepfake proactive forensics systems where additional watermark embeddings can overwrite original forensic watermarks, rendering them useless. The authors propose Adversarial Interference Simulation (AIS), a plug-and-play training paradigm that explicitly simulates these multi-embedding attacks during fine-tuning, enabling systems to maintain forensic integrity even after multiple watermark applications.

## Why This Matters for Practitioners
If you're building content verification systems that rely on digital watermarks for provenance tracking (e.g., social media platforms or content distribution services), this paper reveals that your current solution may be completely broken by seemingly innocent operations like social media auto-optimisation or third-party watermarking services. Without AIS-like protection, your watermark could be overwritten during normal platform processing, making forensic tracking impossible. Practitioners should immediately evaluate whether their watermarking systems are vulnerable to MEA and consider integrating AIS as a lightweight, architecture-agnostic enhancement to their existing workflows. This isn't a theoretical concern, empirical results show that without protection, watermarks fail at random-guessing levels (BER ~50%) after a single secondary embedding.

## Problem Statement
Current deepfake proactive forensics systems operate under an idealised assumption: watermarks are embedded once and remain unchanged. This is like placing a security tag on a product at the factory, then expecting it to survive multiple retail store label changes. In reality, watermarks face repeated embedding operations, social media platforms automatically apply their own watermarks during compression, and malicious actors may deliberately insert new watermarks to obscure original forensic evidence. This vulnerability, which the authors formally define as Multi-Embedding Attacks (MEA), renders current systems fundamentally insecure in real-world deployments.

## Proposed Approach
The authors propose Adversarial Interference Simulation (AIS), a training paradigm that simulates MEA scenarios during fine-tuning rather than modifying network architecture. AIS explicitly applies simulated MEA to watermarked images and introduces a resilience-driven loss function to enforce learning of sparse and stable watermark representations. This enables models to correctly extract the original watermark even after a second embedding operation, without requiring architectural changes to existing systems.

```python
def AIS_training(En, De, X, w1, w2, L):
    # Simulate MEA: embed w2 into already watermarked image Xw1
    Xw1 = En(X, w1)
    Xw1_2 = En(Xw1, w2)
    
    # Compute resilience loss for original watermark recovery
    recovered_w1 = De(Xw1_2)
    resilience_loss = torch.mean((recovered_w1 - w1) ** 2) / L
    
    # Total loss combines standard task loss with resilience loss
    total_loss = standard_task_loss(En, De, X, w1) + 0.5 * resilience_loss
    
    # Train model with total loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

## Key Technical Contributions
The authors' key innovations address fundamental vulnerabilities in current proactive forensics systems:

1. **Formalising MEA as a systemic vulnerability**: They demonstrate that existing methods' optimisation objective (Eq. 1) creates inherent fragility to MEA because the imperceptibility term forces the encoder to minimise changes to the input image. During MEA, the second embedding treats original watermark perturbations as redundant noise, causing overwriting. This theoretical analysis reveals MEA as a general vulnerability, not an implementation flaw.

2. **Resilience-driven loss design**: AIS introduces a loss function (Eq. 7) that explicitly measures recovery error of the original watermark from doubly watermarked images. The loss weights each decoder head's contribution to recovery, effectively creating a sparsity-inducing regulariser. This differs from prior approaches that only optimise for single-embedding recovery.

3. **Sparse watermark representation learning**: The authors leverage the established benefit of sparse watermarking (Bose and Maity 2022; Deeba et al. 2020) but apply it specifically to resist MEA. The resilience loss guides the model to concentrate forensic information in stable, less-interfered feature dimensions, preventing overwriting during subsequent embeddings.

4. **Model-agnostic integration**: Unlike architectural changes, AIS can be added as a fine-tuning step to any existing encoder-decoder watermarking system. The authors validate this with six established baselines (SepMark, LampMark, WaveGuard, EditGuard, MBRS, HiDDeN), demonstrating consistent improvements without modifying original architectures.

## Experimental Results
The authors conducted extensive experiments on CelebA-HQ (256×256) and LFW datasets. Key findings:

- Before MEA, average BER across all methods was 0.50% (indicating nearly perfect recovery).
- After MEA, average BER jumped to 38.17%, with SepMark, MBRS, and HiDDeN reaching ~50% BER (statistically equivalent to random guessing).
- LampMark (with landmark-based embedding) showed the best resilience (6.42% BER after MEA) without AIS, but still suffered significant degradation.
- With AIS, BER remained low (0.02-0.11% for most methods) even after five embedding operations.
- PSNR and SSIM metrics showed minimal degradation (PSNR: 35.28 vs 36.29 for SepMark without/with AIS), confirming that AIS maintains visual quality while improving forensic robustness.

## Related Work
Existing proactive forensics methods (LampMark, SepMark, EditGuard) focus on resisting deepfake manipulations and post-processing distortions (e.g., compression, noise), but ignore the vulnerability of multiple watermark embeddings. Prior work on watermarking (e.g., HiDDeN, MBRS) optimises for robustness against content-level manipulations but doesn't consider the interference from subsequent embedding operations. The authors position AIS as a necessary complement to these methods, addressing a previously unexplored threat that undermines their fundamental premise.

## Limitations
The experiments were conducted on face images from CelebA-HQ and LFW datasets, not real-world social media platforms. The authors acknowledge that MEA might interact differently with non-face images or content types with varying visual complexity. They don't test AIS against adversarial MEA where the second embedding is specifically designed to target the original watermark. Additionally, the paper doesn't explore how AIS performs against MEA combined with other attacks like deepfake generation.

## Appendix: Worked Example
Consider an image X (256×256 RGB) with watermark w1 (128-bit message). After first embedding, the watermarked image Xw1 is produced. A MEA then occurs where a random watermark w2 is embedded, creating Xw1,2.

Without AIS:
- LampMark recovers w1 with 6.42% BER (recovery error: 6.42% of bits incorrect)
- SepMark recovers w1 with 48.62% BER (nearly random: 50% would be random guessing)

With AIS:
- LampMark achieves 0.11% BER (only 0.11% of bits incorrect)
- SepMark achieves 0.0465% BER (only 0.0465% of bits incorrect)

The resilience loss (Eq. 7) explicitly measures this recovery error during training. For SepMark with AIS, the decoder's output for the original watermark w1 from Xw1,2 (after MEA) has a mean squared error of approximately 0.0000465 (normalized by message length L=128), which corresponds to the 0.0465% BER. This demonstrates how AIS actively teaches the model to embed w1 in feature dimensions that remain stable even when w2 is later embedded.

## References

- **Code:** https://github.com/vpsg-research/MEA
- Lixin Jia, Haiyang Sun, Zhiqing Guo, Yunfeng Diao, Dan Ma, Gaobo Yang, "Uncovering and Mitigating Destructive Multi-Embedding Attacks in Deepfake Proactive Forensics", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/37010

Tags: #security-and-privacy #deepfake-detection #digital-forensics #adversarial-training #watermarking #resilient-forensics
