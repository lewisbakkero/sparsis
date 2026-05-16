---
title: "Dementia-R1: Reinforced Pretraining and Reasoning from Unstructured Clinical Notes for Real-World Dementia Prognosis"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2601.03018"
---

## Executive Summary

Dementia-R1 is a reinforcement learning framework that enables longitudinal reasoning from unstructured clinical notes for dementia prognosis. It pre-trains models to predict verifiable clinical indices (MMSE, GDS, CDR) before determining final dementia status, achieving 84.02% AUROC on real-world unstructured data. Practitioners should adopt this approach for any longitudinal health prediction task where unstructured narrative data dominates.

## Why This Matters for Practitioners

If you're building clinical prediction systems using EHRs, consider replacing standard supervised approaches with Dementia-R1's two-stage RL framework. For production systems, this means:

1. Implementing a pre-training stage that predicts intermediate clinical indices (MMSE, GDS, CDR) from narrative notes before final prognosis
2. Using the GRPO algorithm for cold-start reinforcement learning (as detailed in Section 3.2)
3. Structuring your training data with verifiable clinical rewards instead of relying on binary outcomes
4. Monitoring performance at longer time horizons (18-24 months) where Dementia-R1 shows strongest relative gains

This approach is particularly valuable when working with unstructured clinical notes (which constitute ~80% of EHR data), as it provides a structured way to incorporate temporal reasoning into models without explicit annotation of symptom trajectories.

## Problem Statement

Current clinical prediction systems struggle with dementia prognosis because they treat it as a static snapshot rather than a longitudinal journey. Imagine trying to predict whether someone will develop dementia by only reading a single entry in their medical record - it's like trying to forecast a person's retirement by only seeing their current salary, ignoring decades of career progression. The real challenge is tracking non-monotonic cognitive decline across multiple visits from unstructured clinical notes, which is nearly impossible with standard supervised learning.

## Proposed Approach

Dementia-R1 uses a two-stage RL framework:
1. Stage 1: Cold-Start Pre-training - Train the model to predict intermediate clinical scores (MMSE, GDS, CDR) from longitudinal notes
2. Stage 2: Task Fine-tuning - Fine-tune for the final dementia prognosis using the reasoning primitives learned in Stage 1

The framework uses Group Relative Policy Optimisation (GRPO) with verifiable clinical rewards for both stages. The core innovation is using clinically established indices as intermediate rewards to guide the model's temporal reasoning before reaching the final diagnosis.

```python
def dementia_r1_framework(clinical_notes_history):
    # Stage 1: Cold-Start Pre-training
    clinical_indices = extract_indices(clinical_notes_history)  # MMSE, GDS, CDR
    
    # Predict clinical indices using GRPO with tolerance-aware rewards
    model = grpo_train(
        history=clinical_notes_history,
        targets=clinical_indices,
        reward_function=tolerance_aware_reward
    )
    
    # Stage 2: Task Fine-tuning
    dementia_label = model.predict(
        history=clinical_notes_history,
        target="dementia"
    )
    
    return dementia_label
```

## Key Technical Contributions

Dementia-R1 introduces three key technical innovations that make it effective for longitudinal clinical prediction:

1. **Tolerance-aware Reward Function**: Instead of using exact matches for clinical indices, Dementia-R1 uses a tolerance margin of δ=2 for MMSE (0-30 scale), aligning with the Minimal Detectable Change of 2-3 points for MMSE (Stein et al., 2012; Hensel et al., 2007). For coarser scales like GDS (1-7) and CDR (0-3), it enforces exact matching (δ=0). This allows the model to learn meaningful trajectories without being overly sensitive to minor fluctuations.

2. **Cold-Start RL Strategy**: The framework pre-trains the model to predict verifiable clinical indices before determining the final diagnosis. This avoids the sparse reward problem inherent in direct RL for binary prognosis tasks. As Section 3.1 states, they extract clinical indices via an auxiliary LLM, achieving 98.5% extraction accuracy.

3. **Two-Stage GRPO Optimisation**: They use GRPO for both training stages without separate value functions, which reduces optimisation complexity. The group-normalized advantage calculation (Eq. 3) encourages the model to generate reasoning paths that outperform its own group average, as validated by the consistent performance gains across all prediction horizons.

## Experimental Results

On the real-world unstructured clinical notes from Asan Medical Centre (AMC), Dementia-R1 achieved:
- 84.02% AUROC (best among all methods)
- 77.31% F1 score
- Outperformed all 7B baselines and surpassed general-purpose LLMs up to 10× larger (p < 0.01)

On Parkinson's disease dementia (PDD) prediction using the Haeundae Paik Hospital cohort:
- 78.37% AUROC
- Outperformed Qwen2.5-72B-Instruct (72.37% AUROC)

On the ADNI benchmark:
- 83.17% AUROC (highest among all LLM baselines)
- Maintained an F1 score (74.87%) comparable to substantially larger models

The paper reports statistical significance using DeLong's test (p < 0.05), McNemar's test, and bootstrap permutation (B=10,000).

## Related Work

Dementia-R1 builds on recent advances in RL-based medical LLMs but addresses a critical gap: most existing approaches target final outcomes rather than intermediate disease trajectories. While C-Reason (Kim et al., 2025) uses GRPO for sepsis management through masked value prediction, it doesn't address long-term disease progression. Dementia-R1 extends this to dementia by training the model to estimate clinical scores before determining final prognosis.

## Limitations

The paper acknowledges that the framework performs less well in cohorts with sparser clinical index annotations (like Haeundae Paik Hospital compared to AMC). The authors also note they didn't test the approach on patients with very short clinical histories (less than 6 months), as their data processing excludes such cases.

## Appendix: Worked Example

Let's walk through a concrete example of how Dementia-R1 processes a patient's clinical history:

A patient's clinical notes show:
- [2022-02] MMSE 24
- [2022-04] GDS 3
- [2022-05] MMSE 21, ApoE4(+)
- [2022-11] MMSE 19

Stage 1 (Pre-training):
1. The model processes the history up to [2022-05] (H<5) and outputs a predicted MMSE score of 20 (within tolerance δ=2 of the actual 21)
2. It also predicts GDS=3 (exact match)
3. The tolerance-aware reward function gives a reward of 1 for MMSE (|20-21| ≤ 2) and 1 for GDS (exact match)

Stage 2 (Fine-tuning):
1. The model processes the full history up to [2022-11] (H<11)
2. Using the reasoning primitives from Stage 1, it predicts dementia (1) based on the trajectory: "MMSE: 24 → 21 (Dropped), Risk: ApoE4+, GDS 4, Age: 65"
3. The model's output is validated against the ground truth diagnosis

This example demonstrates how Dementia-R1 leverages intermediate clinical indices to build a robust longitudinal understanding before making the final prognosis.

## References

- Choonghan Kim, Hyunmin Hwang, Hangeol Chang, Jaemin Kim, Jinse Park, Jae-Sung Lim, Jong Chul Ye, "Dementia-R1: Reinforced Pretraining and Reasoning from Unstructured Clinical Notes for Real-World Dementia Prognosis", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2601.03018

Tags: #biomedicine #dementia-prognosis #reinforcement-learning #clinical-nlp #longitudinal-data
