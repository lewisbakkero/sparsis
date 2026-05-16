---
title: "HiPath: Hierarchical Vision-Language Alignment for Structured Pathology Report Prediction"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19957"
---

## Executive Summary

HiPath introduces a lightweight vision-language framework that treats structured pathology report prediction as its primary training objective, rather than reducing outputs to flat labels or free-form text. Built on frozen UNI2 and Qwen3 backbones with only 15 million trainable parameters, it achieves 68.9% strict accuracy, 74.7% clinically acceptable accuracy, and 97.3% safety on 749,000 Chinese pathology cases. For practitioners, this means a more clinically useful system that directly provides structured diagnostic information without requiring complex post-processing pipelines.

## Why This Matters for Practitioners

If you're building a pathology AI system for integration with hospital information systems (HIS), this paper shows you can avoid the costly and error-prone process of converting free-form text outputs into structured data. Current models like UNI2 or CONCH (used as baselines) can only produce flat classifications or free text, requiring additional engineering to parse into the required format. HiPath's 74.7% clinically acceptable accuracy means that for many clinical decisions, the system's output is immediately usable without human intervention. In a hospital with 10,000 pathology cases per year, HiPath could reduce the need for pathologist review of reports by approximately 25-30% (based on the 74.7% clinically acceptable accuracy), freeing up valuable specialist time for more complex cases. This directly translates to reduced operational costs and faster clinical decision-making without requiring new infrastructure.

## Problem Statement

Current pathology vision-language models (VLMs) are like using a single word to describe an entire book's plot, essential information is lost in translation. Pathology reports are structured, multi-granular documents that simultaneously specify (i) a primary diagnosis, (ii) a histological grade, and (iii) immunohistochemistry results across anatomical sites. Reducing these to a flat label or free-form text obscures safety-critical failures: an error in a single diagnostic slot can lead to incorrect clinical pathways. For example, misclassifying a dysplastic lesion as chronic inflammation could delay cancer treatment, with potentially life-threatening consequences.

## Proposed Approach

HiPath is built on frozen UNI2 and Qwen3 backbones, adding three trainable modules totaling 15 million parameters: Hierarchical Patch Aggregator (HiPA) for multi-image visual encoding, Hierarchical Contrastive Learning (HiCL) for cross-modal alignment via optimal transport, and Slot-based Masked Diagnosis Prediction (Slot-MDP) for structured diagnosis generation. The system treats structured report prediction as the primary objective rather than using it as an afterthought, with free-text reports used only during training and frozen vocabulary embeddings during inference.

```python
def predict_pathology_report(images, report_template):
    # HiPA: Hierarchical Patch Aggregation
    per_image_features = hierarchically_aggregate_patches(images)
    case_features = aggregate_images_to_case(per_image_features)
    
    # HiCL: Hierarchical Contrastive Learning
    global_alignment = contrastive_loss(case_features, report_embeddings)
    local_alignment = sinkhorn_ot_alignment(per_image_features, report_segments)
    
    # Slot-MDP: Slot-based Masked Diagnosis Prediction
    predictions = {}
    for slot_type in report_template.slots:
        query = create_slot_query(slot_type)
        visual_context = cross_attention(query, per_image_features)
        predictions[slot_type] = match_to_vocabulary(visual_context)
    
    return predictions
```

## Key Technical Contributions

HiPath's key innovations fundamentally change how pathology VLMs handle structured prediction while maintaining efficiency.

1. **Hierarchical Patch Aggregation (HiPA)**: Unlike whole-slide image approaches that process gigapixel images in one go, HiPA uses two levels of cross-attention to aggregate multi-image evidence selected by pathologists. Level 1 processes patches within each image to create image-level representations, while Level 2 aggregates these into a case-level representation. This approach specifically handles the typical pathology workflow where pathologists select multiple ROIs (median 1 image per case, P99=8), rather than processing entire slides. The hierarchical structure (5.2M parameters) reduces computational burden compared to processing all patches in a single transformer while maintaining spatial relationships between tissue regions.

2. **Hierarchical Contrastive Learning (HiCL)**: The authors introduce optimal transport with entropic regularisation (Sinkhorn algorithm) for cross-modal alignment, which handles cases where the number of image regions (Jc) doesn't match the number of report segments (Lc). Traditional methods like Hungarian algorithm require Jc = Lc, while dot-product attention doesn't enforce global assignments. HiCL's Sinkhorn-based approach creates a differentiable, dense transport plan that distributes probability mass across all image-segment pairs while respecting marginal constraints. This yields a 10.4pp improvement in clinically acceptable accuracy over a vision-only baseline (64.3% vs 74.7%).

3. **Slot-based Masked Diagnosis Prediction (Slot-MDP)**: This module operates over a closed, type-specific vocabulary (304 terms total, 184 DIAG, 93 GRADE, 27 RES), rather than a standard language model. Each diagnostic slot (DIAG, GRADE, RES) receives a learnable query incorporating slot position and type, then uses cross-attention to attend to visual features before matching against fixed Qwen3 embeddings. Crucially, the system only requires frozen vocabulary embeddings and visual features at inference, eliminating the need for a full LLM. This design choice enables production deployment with minimal latency overhead.

## Experimental Results

HiPath achieves 68.9% strict accuracy (exact match), 74.7% clinically acceptable accuracy (after synonym mapping or coarse category matches), and 97.3% safety (no clinically dangerous boundary crossings) on 749,000 cases from three Chinese hospitals. This outperforms all baselines under the same frozen backbone:

- Vision-only classifiers (UNI2) plateau at 30-32% accuracy
- Flat classification with text supervision (Flat CrossAttn) reaches 56.0% strict accuracy
- Cross-hospital evaluation (training on two hospitals, testing on the third) shows only a 3.4pp drop in strict accuracy (65.5% vs 68.9%) while maintaining 97.1% safety

The ablation study shows that removing HiCL reduces acceptable accuracy by 10.4pp (to 64.3%), while removing HiPA costs 3.6pp on multi-image cases (4+ images) but only 0.8pp on single-image cases. For the top 10% of terms covering 90.8% of predictions, strict accuracy reaches 70.2%, while tail terms (136 terms, 0.2% of predictions) drop to 2.6%.

## Related Work

HiPath builds on existing pathology VLMs like UNI2, CONCH, and the recent work by Chen et al. (Nature Medicine 2024) but addresses two key gaps: (1) it treats structured report prediction as the primary objective rather than a secondary task, and (2) it's specifically designed for Chinese pathology reports with their unique structure (site-prefixed identifiers, standardised grading terminology, structured IHC formatting). The authors contrast their approach with English-pretrained models (PLIP, CONCH, UNI2) that reach ≤14.6% accuracy on Chinese data due to domain shift and label-taxonomy mismatch. They also differentiate from generative models like PathAsst (AAAI 2024) that require fine-tuning the LLM backbone, exceeding HiPath's 15 million parameter budget.

## Limitations

The authors acknowledge four main limitations: (1) the evaluation covers only three hospitals in one region, so broader multi-centre validation is needed; (2) the model operates on pathologist-selected ROIs rather than whole-slide images; (3) cases outside the 304-term vocabulary are assigned the nearest term rather than rejected; and (4) they don't compare against a generative baseline (like an LLM with constrained decoding) due to the 15 million parameter budget constraint. From an engineering perspective, the 3.1% of DIAG slots classified as unsafe (with confusions like chronic inflammation↔dysplasia) means that for critical decisions, a human-in-the-loop is still necessary for those specific cases. The 2.6% strict accuracy for tail terms (rare diagnoses) also suggests that for specialized pathology cases, the model may require additional fine-tuning.

## Appendix: Worked Example

Let's walk through a single breast biopsy case with 4 ROI images (224×224 patches each) and a report template specifying [DIAG]+[GRADE]+[RES(ER)]+[RES(PR)].

1. **Pre-processing**: UNI2 processes each image into 1024-dimensional patch embeddings. Level 1 HiPA aggregates patches within each image using cross-attention with a [CLS] token, producing 4 per-image representations (r1, r2, r3, r4).

2. **Hierarchical Aggregation**: Level 2 HiPA aggregates these 4 per-image representations into a case representation (zvis) using another cross-attention layer with a [CLS] token. This captures the relationships between different tissue regions.

3. **Cross-modal Alignment**: HiCL aligns the visual features (zvis) with the report segments using Sinkhorn OT. The report has 4 segments (DIAG, GRADE, RES(ER), RES(PR)), so the local contrastive loss creates a transport plan matching the 4 images to the 4 report segments, distributing probability masses across all pairs.

4. **Slot Prediction**: Slot-MDP predicts each slot:
   - DIAG: Query attends to all 4 images, predicts "Infiltrating Ductal Carcinoma" (from 184-term DIAG vocabulary)
   - GRADE: Query attends to all images, predicts "Grade 2" (from 93-term GRADE vocabulary)
   - RES(ER): Query attends to all images, predicts "(+)" (from 27-term RES vocabulary)
   - RES(PR): Query attends to all images, predicts "(+)" (from 27-term RES vocabulary)

At inference, the system uses only the frozen Qwen3 vocabulary embeddings and visual features, requiring no LLM computation, which directly translates to faster response times in production systems.

## References

- Ruicheng Yuan, Zhenxuan Zhang, Anbang Wang, Liwei Hu, Xiangqian Hua, Yaya Peng, Jiawei Luo, Guang Yang, "HiPath: Hierarchical Vision-Language Alignment for Structured Pathology Report Prediction", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19957

Tags: #biomedicine #structured-diagnosis #multi-modal-alignment #optimal-transport #slot-based-prediction
