---
title: "Learning Neural Operators from Partial Observations via Latent Autoregressive Modeling"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/37001"
---

## Executive Summary
This paper introduces LANO (Latent Autoregressive Neural Operator), a framework that enables neural operators to work effectively with partially observed scientific data - a common challenge in real-world applications where sensors miss large geographic areas or measurements are incomplete. LANO achieves 18-69% relative L2 error reduction compared to state-of-the-art methods on partial observation benchmarks with less than 50% missing rate, making it applicable to real-world scenarios like climate prediction.

## Why This Matters for Practitioners
If you're building production systems for scientific computing that depend on sensor data (weather forecasting, seismic analysis, or medical imaging), this paper directly addresses your core challenge: most neural operator models fail when inputs are partially observed. The key takeaway is that you must incorporate a mask-to-predict training strategy (MPT) during model development, not just during inference. For instance, when building a weather prediction system using sparse sensor networks, you should implement artificial masking during training to simulate missing regions - this approach works 26.4-68.7% better than conventional methods. The paper also demonstrates that your model should avoid decoder-side attention recalculation (which causes instability) and instead use encoder-generated attention for stable propagation from observed to unobserved regions.

## Problem Statement
Imagine trying to reconstruct a complete painting when you only have access to small, disconnected fragments of it - and each fragment is from a different part of the painting. Traditional neural operators for scientific computing work like a painter who can only see the complete painting and must reproduce it from memory, but in real-world scenarios, we often have only fragments (like weather stations missing large geographic areas). The problem isn't just the missing data - it's that the model can't learn the physical relationships between observed and unobserved regions because it has no supervision in the missing areas, and the input and output domains don't align spatially.

## Proposed Approach
LANO addresses the two core challenges through a hybrid framework combining a novel masking strategy with a physics-aware autoregressive latent space framework. It introduces two key components: a mask-to-predict training strategy (MPT) that creates artificial supervision by strategically masking observed regions during training, and a Physics-Aware Latent Propagator (PhLP) that reconstructs solutions through boundary-first autoregressive generation in latent space. Figure 2 illustrates the overall architecture, where the model processes partially observed sequences through temporal aggregation, latent operator layers with PhLP, and output projection to reconstruct the full solution field.

```python
def train_lano(model, inputs, masks):
    """Implements the mask-to-predict training strategy (MPT) described in the paper."""
    # Apply artificial masking to observed input regions
    masked_inputs = apply_mask(inputs, mask_type="random")
    
    # Predict missing regions using the model
    predictions = model(masked_inputs, masks)
    
    # Compute loss on observed regions (not missing regions)
    loss = mse_loss(predictions, inputs, masks)
    
    # Add consistency regularisation
    consistency_loss = compute_consistency(model, inputs, masks)
    
    total_loss = loss + 0.1 * consistency_loss
    return total_loss
```

## Key Technical Contributions
The innovations in LANO go beyond simply training on masked data - they specifically address the two fundamental challenges of partial observation.

1. **Mask-to-predict training strategy (MPT)**: Unlike previous work that requires fully observed training data, MPT integrates artificial masking directly into the training process. It randomly masks portions of already observed regions during training, creating pseudo-missing regions with available supervision. The model learns to predict these masked regions while maintaining consistency between original and masked inputs through explicit regularisation. This is different from previous approaches that either pre-trained on unlabeled data or required fully observed training data.

2. **Physics-Aware Latent Propagator (PhLP)**: Rather than attempting to predict all unobserved regions simultaneously (which leads to blurry outputs), PhLP uses a boundary-first autoregressive framework. It starts from observed boundary conditions and progressively propagates information outward through the latent space. The PhLP architecture uses Physics-Cross-Attention (PhCA) that directly utilizes deep features as keys (eliminating the need for explicit positional encodings), creating attention maps that aggregate spatial information into L latent tokens. This design reflects the physical structure of PDEs, enforcing physical consistency in predictions.

3. **Boundary-first reconstruction**: The PhLP propagates information from observed to unobserved regions in a structured manner, avoiding the global modelling difficulties of direct prediction. This is implemented through partial convolution (PConv) that propagates information from observed to unobserved regions, followed by self-attention before decoding. The theoretical foundation shows that PhLP maintains equivalence to learnable integral operators defined on the spatial domain, which is crucial for physical consistency.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
LANO achieves significant improvements across all benchmarks under patch-wise missingness with less than 50% missing rate, with relative L2 error reductions ranging from 17.8% to 68.7% compared to the second-best baseline. The results are summarised in Table 1 of the paper, showing consistent outperformance across three PDE-governed tasks: Navier-Stokes (fluid dynamics), Diffusion-Reaction (biological pattern formation), and ERA5 (real-world climate data). 

For example, on the Diffusion-Reaction task with patch-wise missingness at 5% missing rate, LANO achieved 0.0080 relative L2 error (vs. 0.0334 for the second-best baseline, OFormer), representing a 76.1% relative error reduction. On the same task with 50% missing rate, LANO achieved 0.0128 relative L2 error (vs. 0.0498 for the second-best baseline), showing a 74.3% relative error reduction.

The paper doesn't explicitly report statistical significance testing, though it states these are "relative error reductions" from the second-best model. The paper doesn't compare against all possible baselines in every experiment, but the benchmark includes six comprehensive experiments across three PDE-governed tasks.

## Related Work
LANO builds upon the foundation of neural operators for PDE solving but addresses a fundamental limitation: existing methods assume fully observed training data. The paper positions itself against recent neural operator work like LNO, IPOT, and GNOT, which handle partial inputs during inference but require fully observed training. Unlike approaches like DINo or CORAL that focus on sparse point-wise missingness rather than spatially correlated unobserved regions, LANO specifically addresses the challenge of spatially correlated missing regions through its novel training strategy and modelling approach.

The paper also acknowledges that previous methods like TranSolver and LSM explore latent space modelling but don't specifically address the supervision gap in unobserved regions or the dynamic spatial mismatch between inputs and outputs.

## Limitations
The paper acknowledges that their approach works effectively for up to 50% missing rate in their controlled experiments, but the authors note that the model's performance degrades with higher missing rates (over 75% missing rate in some experiments). The benchmark POBench-PDE focuses on three PDE-governed tasks, so the approach might not generalise to all scientific domains that don't follow PDE constraints.

The paper doesn't test the approach on larger, more complex PDE systems (like 3D fluid dynamics), and the real-world climate prediction results (ERA5) are limited to the specific dataset used in their benchmark. The authors also note that their model requires more memory (95.40 MB) compared to some baselines (e.g., 42.35 MB for LNO), though this is a minor trade-off for the improved accuracy.

## Appendix: Worked Example
Let's walk through how a single input flows through LANO with concrete numbers, based on the paper's description of their Diffusion-Reaction dataset with 25% patch-wise missingness.

Start with a 2D input grid measuring 64×64 spatial points (4096 total points), where 25% of the grid is missing in a structured patch pattern (16×16 blocks missing at 25% rate). The input is a sequence of T=8 historical frames, each containing 25% missing data.

**Step 1: Input Embedding (Temporal Aggregation Layer)**
- Each frame has 4096 spatial points, with 1024 points missing (25%)
- The temporal aggregation layer embeds each frame's positions and measurements into deep features
- For a single frame, the embedding process takes the 64×64 grid (with 25% missing) and generates 4096 embedding vectors of dimension C=128 (from the paper's implementation: "all models employ 8 layers unless otherwise stated")
- The input positions and measurements (25% observed) are concatenated, resulting in an input tensor of shape [1, 4096, 128] before embedding

**Step 2: Mask-to-Predict Training (MPT)**
- During training, the authors apply an additional random mask to the already partially observed input
- For this example, they apply a 50% artificial mask to the observed regions (leaving 12.5% of the total grid observable during training)
- The model is trained to predict the masked sections (now 37.5% total missing) using only the remaining 12.5% observed data
- The consistency regularisation enforces that predictions from the original input (12.5% observed) match predictions from the masked input (12.5% observed)

**Step 3: Feature Propagation (Latent Operator Layers)**
- The model processes the input through 8 latent operator layers (as described in the paper)
- Each layer uses the Physics-Aware Latent Propagator with PhCA
- At layer 1, the PhCA encoder generates attention maps S ∈ RH×No×L (H=8 heads, No=1024 observed points, L=32 latent tokens)
- The PhCA decoder processes these tokens and passes them through the self-attention layer (PConv) to propagate information from observed to unobserved regions
- After 8 layers, the feature map has progressed from observing only 12.5% of the grid to confidently predicting the entire grid

**Step 4: Output Projection**
- The final latent features (shape [1, 4096, 128]) are passed through a linear projection layer
- This outputs the full 64×64 solution field (4096 points), including the previously missing regions
- The model's prediction error is computed over the entire grid (not just the observed regions), resulting in the reported relative L2 error of 0.0080 for the Diffusion-Reaction task with 25% missing rate

This step-by-step example demonstrates how LANO progressively reconstructs the missing information by leveraging the observed regions and enforcing physical consistency through the boundary-first autoregressive framework.

## References

- **Code:** https://github.com/Kingyum-Hou/LANO
- Jingren Hou*1, Hong Wang, Pengyu Xu†1, Chang Gao1, Huafeng Liu1, Liping Jing†1, "Learning Neural Operators from Partial Observations via Latent Autoregressive Modeling", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/37001

Tags: #scientific-computing #partial-observation #neural-operators #physics-informed-ml #autoregressive-modelling
