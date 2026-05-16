---
title: "Understanding Task Aggregation for Generalizable Ultrasound Foundation Models"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.18123"
---

## Executive Summary
The authors introduce M2DINO, a multi-organ, multi-task ultrasound foundation model built on DINOv3 with task-conditioned Mixture-of-Experts blocks. Their research reveals that task aggregation strategies in foundation models must account for both data scale and task characteristics, not just clinical taxonomy. This matters for engineers building medical AI systems as it provides concrete criteria to avoid negative transfer when creating unified models.

## Why This Matters for Practitioners
If you're building a medical imaging foundation model that needs to handle multiple clinical tasks (like segmentation, classification, and detection), this paper directly informs your architecture choices. For data-scarce domains like breast ultrasound (with only 2,920 training samples), avoid clinically-grouped training as it can cause negative transfer (breast lesion segmentation DSC dropped 79.7% in their experiments). Instead, adopt all-task unified training for more stable performance across small datasets. For data-rich domains like obstetrics (11,910 training samples), clinically-grouped training can offer modest gains (fetal abdomen segmentation DSC increased from 0.217 to 0.481), but still prefer all-task unified training as a more reliable fallback. Always measure task-type sensitivity first, segmentation tasks are particularly vulnerable to aggregation design flaws.

## Problem Statement
Imagine trying to train a single model that can both translate between languages and write poetry. You'd quickly find that translation requires grammatical precision while poetry demands creative expression, and forcing them together without accounting for these differences would cause the model to fail at both. Similarly, in ultrasound imaging, trying to unify heterogeneous tasks like fetal organ segmentation (dense pixel-level prediction) with lung disease classification (categorical prediction) without considering how data scale influences transfer can cause the model to underperform across the board.

## Proposed Approach
M2DINO builds on DINOv3 with task-conditioned Mixture-of-Experts blocks to create a unified framework for multi-organ, multi-task ultrasound analysis. The system uses spatial feature maps as a unified interface across all task types, avoiding the bias of global token pooling that might favour classification over segmentation. It evaluates three training paradigms: task-specific (TS), clinically-grouped (CG), and all-task unified (AU). For the MoE implementation, they condition expert selection on both token embeddings and task identifiers, routing different tasks to specialized experts within the same backbone.

```python
def task_conditioned_moe(token_embeddings, task_id):
    task_embedding = task_embedding_layer(task_id)
    gate = softmax(weighted_concat(token_embeddings, task_embedding))
    expert_outputs = [expert(token_embeddings) for expert in experts]
    return sum(gate * expert_outputs)
```

## Key Technical Contributions
M2DINO's framework introduces several key innovations:

1. Task-conditioned Mixture-of-Experts blocks that dynamically allocate capacity to different tasks based on their specific requirements. The authors condition expert selection on both token embeddings and task identifiers (using a learnable embedding for each task), allowing the model to route different tasks to specialized experts while maintaining a shared backbone. This differs from prior MoE approaches that used static routing.

2. A systematic evaluation framework that isolates the effect of task aggregation strategies by controlling for all other variables (backbone, input resolution, data processing, optimisation). This controlled comparison across 27 heterogeneous ultrasound tasks spanning segmentation, classification, detection, and regression provides empirical evidence that data scale and task characteristics govern aggregation effectiveness, not clinical taxonomy alone.

3. Practical design guidelines revealing that segmentation tasks are particularly sensitive to aggregation strategies, showing the largest performance drops (e.g., -79.7% DSC in breast lesion segmentation under CG training). This insight means engineers must evaluate segmentation tasks separately from classification and regression when designing unified models.

## Experimental Results
The paper evaluates 27 ultrasound tasks across four categories with the following dataset sizes:
- Segmentation: 12 tasks (16,615 training samples)
- Classification: 9 tasks (16,361 training samples)
- Detection: 3 tasks (4,333 training samples)
- Regression: 3 tasks (3,078 training samples)

Key findings:
- In data-rich obstetrics (OB) group (11,910 training samples), AU reduced cervical regression error (MRE: 30.4 → 15.6) and increased fetal abdomen segmentation overlap (DSC: 0.217 → 0.481).
- In data-scarce breast (2,920 samples) group, CG induced negative transfer: breast lesion segmentation DSC dropped from 0.713 to 0.145 (79.7% decrease), while AU showed comparatively stable performance (DSC: 0.713 → 0.692).
- AU exhibited more consistent performance across clinical groups: +3.76 in OB, -0.02 in Breast, and +0.07 in Lung (compared to TS), while CG showed positive gains only in OB (+2.93), but negatives in Breast (-0.29) and Lung (-0.07).

## Related Work
M2DINO builds on prior work in foundation models for medical imaging, particularly USFM and TinyUSFM, but focuses specifically on the critical question of task aggregation strategies. Unlike prior approaches that primarily reported positive results on selected task combinations, this paper systematically evaluates how aggregation strategies interact with data scale and task characteristics. The authors position their work as a necessary complement to existing foundation model research, providing concrete criteria for when unified models will actually outperform task-specific ones.

## Limitations
The authors acknowledge several limitations:
1. They focus on a single backbone (DINOv3) and predefined clinical grouping strategies, so alternative architectures or data-driven grouping might yield different results.
2. The analysis is limited to ultrasound imaging, so transfer to other modalities like radiography or pathology remains untested.
3. The experiments don't explore how the number of experts in the MoE architecture affects performance (they use a partial-MoE design with experts in later layers).
4. The paper doesn't investigate how to automatically determine the optimal grouping strategy for a given dataset.

## Appendix: Worked Example
Let's walk through the breast lesion segmentation task from the paper's experiments:

1. **Task**: Breast lesion segmentation (a segmentation task requiring dense pixel-level predictions)
2. **Training data**: 2,920 samples across all breast-related tasks (including breast lesion segmentation)
3. **Performance metrics**:
   - Task-specific (TS): DSC = 0.713
   - Clinically-grouped (CG): DSC = 0.145 (a 79.7% decrease)
   - All-task unified (AU): DSC = 0.692 (a 3.0% decrease)

The system processes a single breast ultrasound image as follows:
- The image is converted to RGB format (3 channels) for input
- DINOv3 backbone produces token embeddings Z and spatial feature maps F
- For breast lesion segmentation, the model uses F as input to a DPT-style decoder (a lightweight segmentation head)
- The MoE block routes this task to specialized experts within layers 7-12 (later transformer layers that handle task-specific representations)
- The loss function uses Dice loss (appropriate for segmentation)
- During training, the task identifier (breast lesion segmentation) conditions the MoE routing, directing the appropriate expert pathways

The significant performance drop under CG training (79.7% decrease in DSC) occurs because the breast lesion segmentation task shares capacity with other breast tasks (like lung disease classification) that have different data distributions and requirements. The all-task unified approach avoids this by using task-conditioned routing to allocate capacity more effectively across all tasks.

## References

- Fangyijie Wang, Tanya Akumu, Vien Ngoc Dang, Amelia Jiménez-Sánchez, Jieyun Bai, Guénolé Silvestre, Karim Lekadir, Kathleen M. Curran, "Understanding Task Aggregation for Generalizable Ultrasound Foundation Models", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.18123

Tags: #biomedicine #medical-imaging #multi-task-learning #mixture-of-experts
