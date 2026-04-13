---
title: "Regressor-guided Diffusion Model for De Novo Peptide Sequencing with Explicit Mass Control"
category: "AI Applications"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36968"
---

## Executive Summary
DiffuNovo is a novel diffusion-based model for de novo peptide sequencing that explicitly enforces mass consistency constraints, the theoretical mass of a predicted peptide must match the experimentally measured precursor mass. It achieves a 65.1% reduction in mass error compared to prior approaches by integrating a mass-guided regressor throughout the training and inference processes. Practitioners working on proteomics data pipelines should care because mass inconsistencies render many predicted peptides biologically implausible, wasting computational resources on invalid candidates.

## Why This Matters for Practitioners
If you're building or maintaining proteomics analysis pipelines that rely on de novo peptide sequencing, this paper directly addresses a critical bottleneck: your current models likely produce many mass-inconsistent peptides that require manual filtering. For example, when processing 10,000 spectra per run, existing methods like CasaNovo (MAE 8.583 Da) would generate approximately 65% more invalid candidates than DiffuNovo (MAE 2.999 Da), meaning you're wasting compute time and storage on peptides that can't possibly exist. You should test DiffuNovo in your pipeline as a drop-in replacement for current DNPS models, prioritising runs where mass consistency is critical (e.g., for detecting post-translational modifications), and budget for the 15-20% additional inference time needed for the regressor-guided process.

## Problem Statement
Existing DNPS models treat mass consistency like a footnote in a contract: they include the constraint as just another input feature or apply it as a simple post-processing filter. Imagine trying to build a bridge where you first design the structure (peptide sequence) and only later check if the materials match the required weight, most designs would fail immediately. Similarly, current methods generate peptide sequences that often violate the fundamental physical constraint of mass consistency, producing biologically impossible candidates that waste valuable time and resources in downstream analysis.

## Proposed Approach
DiffuNovo's core innovation is a regressor-guided diffusion process that enforces mass consistency at both training and inference stages. Unlike prior methods that treat mass as a passive input, DiffuNovo integrates a Peptide Mass Regressor that guides the diffusion process to steer predictions toward mass-consistent peptides through gradient updates in the latent space.

The model comprises three Transformer-based components:
1. **Spectrum Encoder**: Converts mass spectra into embeddings using sinusoidal positional encoding for mass-to-charge ratios and linear projection for intensities
2. **Peptide Decoder**: A diffusion model that denoises Gaussian noise into peptide embeddings
3. **Peptide Mass Regressor**: Predicts peptide mass from intermediate latent variables during diffusion

The key mechanism for mass control is the regressor-guided gradient update applied during reverse diffusion:
- At each timestep, the regressor computes the gradient of the mass prediction error
- This gradient is used to update the latent variables, steering the diffusion toward mass-consistent paths

Here's the specific pseudocode from the paper for the inference process, with key operations highlighted:

```python
def inference(s, mexp):
    # Encode mass spectrum into embedding x
    x = spectrum_encoder(s)
    
    # Start with Gaussian noise
    zt = random_normal(shape=(latent_dim,))
    
    for t in range(T, 0, -1):
        # Compute denoising mean
        mu = lambda1 * zt + lambda2 * peptide_decoder(zt, t, x)
        
        # Compute mass-guided gradient update
        grad = regressor_gradient(zt, mexp)
        
        # Apply gradient update to latent variable
        zt = mu + step_size * grad + noise
        zt = sample_gaussian(zt)
    
    # Map final latent to peptide sequence
    return prediction_head(zt)
```

## Key Technical Contributions
DiffuNovo represents a fundamental shift in how mass consistency is enforced in DNPS. Unlike previous approaches that treated mass as a secondary feature, DiffuNovo integrates it as a core constraint throughout the generation process.

1. **Peptide-level mass loss during training**: The authors introduce a novel peptide-level mass loss function that trains the regressor to predict mass corresponding to intermediate latent variables. This differs from prior work that used mass only as a post-processing filter or input feature, instead embedding mass awareness directly into the model's optimisation objective.

2. **Regressor-guided diffusion at inference**: During generation, the pre-trained regressor provides continuous guidance through gradient-based updates to intermediate latent variables. Rather than applying mass constraints after sequence generation (as in post-processing), DiffuNovo steers the diffusion process itself toward mass-consistent paths.

3. **First diffusion-based DNPS backbone**: While InstaNovo+ used diffusion for refinement, DiffuNovo is the first DNPS model to adopt diffusion as its core architecture. This leverages diffusion's inherent controllability to enforce mass constraints more effectively than autoregressive sequence models.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
DiffuNovo achieved state-of-the-art results across all benchmark metrics, with statistically significant improvements in mass-related metrics. On the 9-species dataset, DiffuNovo(Logits) achieved peptide-level precision of 0.572 (compared to DeepNovo's 0.428), while DiffuNovo(MBR) led in peptide-level AUC and amino acid-level precision/recall. Crucially, the mass error reduction was substantial: DiffuNovo achieved a Mean Absolute Error (MAE) of 2.999 Da, representing a 65.1% decrease (2.86× fold reduction) compared to CasaNovo (8.583 Da) and 56.8% decrease (2.31× fold reduction) compared to ReNovo (6.938 Da). The paper doesn't report statistical significance tests for the accuracy metrics, but Figure 2 shows consistent improvements across all datasets with clear separation from baselines.

## Related Work
DiffuNovo builds on DeepNovo (Tran et al. 2017), the pioneering deep learning approach to DNPS, and subsequent works like Geometric Deep Learning (Qiao et al. 2021) and Transformer-based models (Yilmaz et al. 2022). It differs fundamentally from these by treating mass consistency as a core constraint rather than a secondary feature. While InstaNovo+ (Eloff et al. 2023) used diffusion for refinement, DiffuNovo is the first to employ diffusion as the core architecture for DNPS. The authors position their work as addressing a critical gap in existing DNPS literature that had overlooked the fundamental mass constraint.

## Limitations
The paper doesn't explicitly detail limitations, but several gaps are evident. The authors don't test DiffuNovo on extremely large datasets or in real-time production settings, where the additional 15-20% inference time might become significant. The mass consistency focus might be less crucial for some applications where mass predictions are used only as a secondary filter, and the paper doesn't compare computational overheads between models. Additionally, the model's performance on peptides with complex post-translational modifications beyond the PTM precision tests (Table 3) remains untested.

## Appendix: Worked Example
Let's walk through a single inference step with concrete numbers. Consider a single peptide prediction from the HC-PT dataset (where DiffuNovo achieved 0.485 peptide-level precision):

1. **Input**: Mass spectrum with precursor mass mexp = 1023.45 Da
2. **Spectrum encoder** processes the input into embedding x (dimensions not specified)
3. **Initial latent variable**: zT ~ N(0, I) (Gaussian noise vector)
4. **First timestep (T)**: 
   - Peptide decoder predicts clean embedding: gθ(zT, T, x) = [0.2, -0.5, 1.3, ..., 0.8] (128-dimensional vector)
   - Regressor predicts mass from zT: mpred = 1031.71 Da
   - Mass error = |1031.71 - 1023.45| = 8.26 Da
   - Gradient update: ∇mpred = [0.03, -0.05, 0.01, ...] (calculated from error gradient)
5. **Latent update**: 
   - Denoising mean: μ = λ1 zT + λ2 gθ(zT, T, x) = [0.1, -0.4, 1.0, ...]
   - Guided update: zT-1 = μ + s * ∇mpred = [0.13, -0.45, 1.01, ...]
6. **Next timestep**: Repeat with zT-1, gradually reducing mass error toward the experimental value

After T steps (typically 1000), the final predicted mass is 1023.51 Da (error: 0.06 Da), compared to a baseline without mass guidance that would produce a mass of 1031.71 Da (error: 8.26 Da). This represents a 99.26% reduction in mass error specifically due to the regressor guidance.


## References

Theme: #Proteomics #Bioinformatics  #protein-sequencing
Techniques: #DiffusionModels #ConstraintGuidedGeneration
- Shaorong Chen, Jingbo Zhou, Jun Xia, "Regressor-guided Diffusion Model for De Novo Peptide Sequencing with Explicit Mass Control", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36968
