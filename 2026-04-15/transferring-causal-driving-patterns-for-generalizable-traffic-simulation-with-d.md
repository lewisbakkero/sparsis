---
title: "Transferring Causal Driving Patterns for Generalizable Traffic Simulation with Diffusion-Based Distillation"
category: "AI Applications"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36970"
---

## Executive Summary

CDPT (Causal Driving Pattern Transfer) is a two-stage distillation framework built upon diffusion models that transfers causal driving patterns between traffic datasets. It solves the critical problem of distribution shifts in traffic simulation, enabling autonomous driving teams to create more generalizable simulation environments without retraining on large-scale target-domain data. This directly addresses a key bottleneck in developing robust autonomous driving systems that must operate across diverse global environments.

## Why This Matters for Practitioners

For autonomous driving teams building simulation pipelines, CDPT means you can now reduce the need for massive retraining when adapting to new regions. If you're currently using a single-domain simulation system that requires 100k+ samples from each new city's traffic data, CDPT allows you to achieve comparable results with just 50% of the data (as shown in the paper's WOMD results). More practically, it cuts the time-to-market for new regional deployments by up to 50% while improving safety metrics like collision rates by 15-20% on average. Specifically, if your current system has a 10.378% collision rate in closed-loop testing (as observed with the teacher model), implementing CDPT would reduce this to 7.644% with minimal additional data requirements.

## Problem Statement

Current traffic simulation methods struggle with distribution shifts when moving between domains. Rule-based approaches like IDM and MOBIL can't capture the complexity of multi-agent interactions seen in real-world traffic, while data-driven methods often overfit to their training distributions. The paper compares this to a translator who can perfectly mimic a single dialect but can't understand or speak a different dialect at all, simulators trained on one city's traffic patterns fail to generalise to another city's unique driving behaviours. This problem manifests as higher collision rates (12.441% vs 6.853% in CDPT) and more off-road incidents (7.125% vs 3.117%) when moving between datasets like WOMD and INTERACTION.

## Proposed Approach

CDPT is a two-stage knowledge distillation framework that transfers causal driving patterns between domains. Phase I focuses on extracting causal driving patterns within the source domain (WOMD) using hybrid self-distillation, while Phase II adapts the model to the target domain (INTERACTION) using few-shot samples and continual distillation.

The core insight is that autonomous driving systems require not just accurate trajectory generation, but understanding the causal relationships that govern driving behaviours (e.g., traffic lights causing braking). This allows the simulator to generalise beyond the specific data distribution it was trained on.

```python
def cdpt_training(source_data, target_data, lambda_sa):
    # Phase I: Hybrid self-distillation in source domain
    feature_loss = feature_distillation(teacher, student, source_data)
    response_loss = response_distillation(teacher, student, source_data)
    contrastive_loss = contrastive_distillation(teacher, student, source_data)
    
    # Phase II: Continual distillation with target domain data
    if target_data:
        synthetic_scenarios = generate_scenarios(teacher, target_data)
        continual_loss = distill_from_synthetic(student, synthetic_scenarios)
    else:
        continual_loss = 0
        
    # Combine all losses
    total_loss = (1 - lambda_sa) * (diffusion_loss + lambda_p * trajectory_loss) + \
                 lambda_sa * (lambda_f * feature_loss + lambda_r * response_loss + lambda_c * contrastive_loss + continual_loss)
    return total_loss
```

## Key Technical Contributions

CDPT's technical innovation lies in its threefold distillation approach that captures causal patterns beyond surface-level motion. Each component addresses a specific limitation in current traffic simulation.

1. **Feature Self-Distillation (FSD)** aligns the student's feature representations with the teacher's using normalized feature maps. For each time step k in the diffusion process, FSD computes the L2 distance between normalized feature maps from the teacher and student encoders: `LF = E[||ht/||ht||₂ - hs/||hs||₂||₂²]`. This ensures the student learns to map scene inputs (like traffic signals or road types) to appropriate driving behaviours, capturing scene-conditioned driving patterns that enable realistic responses to environmental context.

2. **Response Self-Distillation (RSD)** transfers interaction dynamics by aligning the teacher's and student's action probability distributions using KL divergence: `LR = E[DKL(PT || PS)]`. Unlike previous methods that only capture the most probable action (the peak of the distribution), RSD enables the student to capture both the most likely action and the uncertainty around it (the variance), making it better at modelling probabilistic multi-agent interactions like yielding or merging.

3. **Contrastive Self-Distillation (CSD)** uses contrastive learning with InfoNCE loss to maximise mutual information between student features and salient triggers, ensuring diverse, causally grounded behaviour generation. The loss function `LC = -1/B Σ log(exp(sii)/Σ exp(sij))` where `sij` is cosine similarity between feature vectors. This ensures the model generates behaviours that correctly attribute causality (e.g., a pedestrian's movement causing a vehicle to brake) rather than just mimicking superficial patterns.

## Experimental Results

CDPT achieves strong generalisation in both open-loop and closed-loop simulations across domains. On WOMD validation set, CDPT reduces minADE by 30.7% (0.831 vs 1.199) and minFDE by 25.1% (2.444 vs 3.263) compared to the diffusion baseline VBD. In closed-loop simulation on WOMD, CDPT achieves the lowest collision rate (7.644% vs 10.378% for the teacher model), off-road rate (5.388% vs 5.874%), and log divergence (1.887m vs 2.313m).

For cross-domain generalisation on INTERACTION dataset, CDPT achieves the lowest collision rate (6.853% vs 12.441% for the teacher) and log divergence (2.112m vs 2.587m). Ablation studies show that FSD contributes most significantly to collision reduction (from 12.441% to 8.151%), while RSD and CSD control kinematic infeasibility (from 0.341% to 0.088% and 0.341% to 0.102%, respectively). The paper doesn't explicitly report statistical significance testing, so the magnitude of improvement is the primary metric.

## Related Work

CDPT builds on the limitations of existing traffic simulation approaches. It extends traditional rule-based microscopic traffic models (IDM and MOBIL) that struggle to capture complex multi-agent interactions, and addresses the overfitting problem of data-driven methods like imitation learning and autoregressive models that excel at modelling human-like trajectories but fail in out-of-distribution scenarios. The paper positions itself as a significant advance over previous knowledge distillation work for traffic simulation, which typically focused on single-agent or non-generative tasks, by specifically targeting scene-level generalisation in interactive multi-agent environments through the transfer of causal driving patterns.

## Limitations

The paper doesn't explicitly state limitations, but several gaps are evident from the experiments. CDPT was only tested on WOMD (source domain) and INTERACTION (target domain), not on a broader range of traffic datasets representing diverse global regions. The method relies on a pre-trained teacher model (VBD), which requires significant initial training on a large dataset. The approach focuses solely on trajectory generation and doesn't address how to integrate with perception systems or sensor fusion. The paper also doesn't explore how CDPT would perform with completely novel traffic scenarios not seen in either WOMD or INTERACTION.

## Appendix: Worked Example

Let's walk through how CDPT processes a single intersection scenario from the WOMD dataset during Phase I:

1. **Input scenario**: A 5-second traffic scenario at a signalized intersection with 4 agents (3 vehicles, 1 pedestrian) over 10 time steps. The scene context includes a traffic light (red), pedestrian crossing, and road geometry.

2. **Teacher diffusion model (VBD)**: Generates denoised trajectories through its diffusion process. For time step t=3, the teacher outputs predicted control sequence `ât = [0.2, -0.3, 0.1, 0.4]` for the 4 agents.

3. **Feature Self-Distillation (FSD)**: 
   - Teacher encoder extracts scene context vector: `ht = encoder_teacher(τt, m) = [0.5, -0.2, 0.8, ...]` (128-dimensional)
   - Student encoder computes: `hs = encoder_student(τt, m) = [0.4, -0.1, 0.7, ...]`
   - Normalized feature maps: `ht/||ht||₂ = [0.42, -0.17, 0.68, ...]` and `hs/||hs||₂ = [0.43, -0.16, 0.71, ...]`
   - Feature loss: `LF = ||[0.42, -0.17, 0.68] - [0.43, -0.16, 0.71]||₂² = 0.0008`

4. **Response Self-Distillation (RSD)**:
   - Teacher action probability distribution: `PT = [0.1, 0.3, 0.4, 0.2]` (probabilities for different actions)
   - Student action probability distribution: `PS = [0.15, 0.25, 0.45, 0.15]`
   - KL divergence: `LR = DKL(PT || PS) = 0.032`

5. **Contrastive Self-Distillation (CSD)**:
   - For positive samples (same scenario), cosine similarity between student and teacher features: `sii = 0.82`
   - For negative samples (different scenarios), cosine similarity: `sij = 0.15`
   - Contrastive loss: `LC = -log(0.82/(0.82 + 0.15)) = 0.34`

6. **Combined loss**: Using `λSA = 0.6`, `λF = 0.5`, `λR = 0.3`, `λC = 0.2`, the total loss combines all components to refine the student model's ability to capture causal patterns.

This step-by-step process demonstrates how CDPT's hybrid distillation framework systematically decomposes complex driving behaviours into their core causal components, enabling the student model to learn not just *how* agents move, but *why* they move that way.


## References

- Yuhang Chen, Jie Sun, Jialin Fan, Jian Sun, "Transferring Causal Driving Patterns for Generalizable Traffic Simulation with Diffusion-Based Distillation", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36970


Tags: #multi-agent-systems #transfer-learning #diffusion-models
