---
title: "TuLaBM: Tumor-Biased Latent Bridge Matching for Contrast-Enhanced MRI Synthesis"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19386"
---

## Executive Summary
TuLaBM solves the problem of synthesising contrast-enhanced MRI from non-contrast MRI without requiring gadolinium-based contrast agents (GBCAs), which reduces cost, safety concerns, and environmental impact. The method achieves clinically meaningful tumor region fidelity while maintaining inference times under 0.097 seconds per image, making it suitable for clinical deployment where diffusion models typically require minutes.

## Why This Matters for Practitioners
If you're building medical imaging pipelines that currently depend on GBCAs for tumor assessment, TuLaBM offers a practical alternative that maintains diagnostic quality while eliminating contrast agent costs and safety workflows. For production systems, this means: (1) reducing infrastructure costs by removing contrast agent procurement, (2) improving patient throughput with faster imaging workflows, and (3) mitigating liability from contrast agent side effects. Engineers can immediately integrate the latent bridge matching approach into existing MRI synthesis pipelines with minimal architectural changes, while focusing validation efforts on tumor region metrics rather than whole-image fidelity.

## Problem Statement
Current CE-MRI synthesis methods are like trying to reconstruct a complex stained-glass window from a blurry black-and-white photo: they either fail to capture the critical details (tumor boundaries) or take so long to process that they're impractical for clinical use. Existing diffusion models produce high-quality images but require multi-step sampling (11.8 seconds for I2SB), while GAN-based methods produce false positives in tumor regions (e.g., 65.37% tumor-region SSIM for Pix2Pix).

## Proposed Approach
TuLaBM transforms NC-to-CE MRI translation into a latent bridge matching problem, where the model learns to transport samples between NC-MRI and CE-MRI distributions in a lower-dimensional latent space. This eliminates the need for slow diffusion sampling and enables efficient inference. The system consists of a pre-trained VAE encoder/decoder, a latent denoiser, and the Tumor-Biased Attention Mechanism (TuBAM) that selectively amplifies tumor-relevant features during training.

```python
def tu_lbm_inference(nc_mri):
    z0 = vae_encoder(nc_mri)  # Encode NC-MRI to latent space
    z1 = latent_bridge(z0, timesteps=4)  # Bridge transport in latent space
    ce_mri = vae_decoder(z1)  # Decode to CE-MRI
    return ce_mri
```

## Key Technical Contributions
The core innovation lies in how TuLaBM handles tumor region fidelity while maintaining computational efficiency. The authors specifically address two critical gaps in medical image synthesis:

1. **Latent space bridge matching** replaces pixel-space diffusion with a single latent transport step, reducing inference time from minutes to under 0.1 seconds. Unlike I2SB (11.8s) or D3M (4.58s), TuLaBM uses only four sampling steps (vs. hundreds in diffusion models) by operating in the latent space where anatomical structures are better preserved.

2. **Tumor-Biased Attention Mechanism (TuBAM)** injects a structured bias into attention logits during training, reinforced by tumor mask projections. Specifically, it adds an additive term αtumor * ˜m * ˜m⊤ to attention logits, where ˜m is the downsampled tumor mask. This selectively strengthens interactions between tumor-associated tokens without altering the attention operator's normalization, encouraging stronger intra-tumor feature aggregation during latent evolution.

3. **Boundary-aware loss** explicitly supervises tumor interfaces using distance-to-boundary maps. The loss assigns exponentially higher weights to pixels near tumor boundaries (w(p) = exp(-Dboundary(p)/τ) * mtumor(p)), creating a gradient that sharpens tumor margins. This directly addresses the "false positive and false negative enhancement" problem mentioned in the introduction.

## Experimental Results
On BraTS2023-GLI (BraSyn), TuLaBM achieved:
- Whole-image SSIM: 92.40% (vs. 90.95% for D3M, the next best)
- Tumor-region SSIM: 88.66% (vs. 73.21% for D3M, representing a 20.5% relative improvement)
- Inference time: 0.097 seconds per image (vs. 11.8 seconds for I2SB)

On Cleveland Clinic liver MRI (zero-shot):
- Tumor-region SSIM: 58.30% (vs. 50.14% for I2SB)
- After fine-tuning on 5 liver volumes: 63.60% SSIM

The ablation study (Table 2) confirms that both TuBAM and boundary-aware loss are necessary, with removing both components causing a 4.49% drop in tumor-region SSIM.

## Related Work
The authors position TuLaBM as a natural extension of bridge matching frameworks (Shi et al., 2023), but address their key limitation: "bridge matching has been explored for medical image translation in pixel space, but our latent formulation substantially reduces computational cost." They specifically improve upon I2SB (Liu et al., 2023) and D3M (Pang et al., 2025) by moving the bridge matching to latent space and adding tumor-biased mechanisms.

## Limitations
The authors acknowledge limitations include: (1) reliance on paired NC-MRI and CE-MRI data during training, (2) evaluation limited to brain and liver MRI with no validation on other modalities, and (3) no comparison against methods that incorporate explicit anatomical constraints. The paper also doesn't report statistical significance of performance improvements, though the consistent outperformance across metrics suggests meaningful gains.

## Appendix: Worked Example
Consider an NC-MRI slice from a brain tumor scan (256×256 resolution). The VAE encoder maps this to a latent space (64×64×32). At inference time:
1. The latent vector z₀ = E(x₀) is generated from the input NC-MRI.
2. The latent bridge computes zₜ = (1-t)z₀ + t z₁ + σ√[t(1-t)]ϵ for four timesteps (t=0.0, 0.33, 0.67, 1.0).
3. During denoising, TuBAM processes the latent feature map (Hₗ=64, Wₗ=64) by flattening to a sequence (N=4096 tokens).
4. The tumor mask (binary, 256×256) is downsampled to 64×64 (mₗ), flattened to 4096 tokens (m̃), and used to compute the bias term m̃m̃ᵀ.
5. For a token corresponding to a tumor location, attention logits receive +αtumor * 1 (αtumor=0.5), while non-tumor tokens receive +αtumor * 0.
6. The boundary-aware loss computes distance-to-boundary maps (D_boundary), with pixels within 10 pixels of tumor boundaries receiving exponentially higher weights (τ=0.5).
7. The final CE-MRI is decoded from z₁ (with 0.097s inference time) and achieves 88.66% tumor-region SSIM.

## References

- Atharva Rege, Adinath Madhavrao Dukre, Numan Balci, Dwarikanath Mahapatra, Imran Razzak, "TuLaBM: Tumor-Biased Latent Bridge Matching for Contrast-Enhanced MRI Synthesis", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19386

Tags: #biomedicine #medical-imaging #diffusion-models #tumor-detection #latent-space-translation
