---
title: "ODySSeI: An Open-Source End-to-End Framework for Automated Detection, Segmentation, and Severity Estimation of Lesions in Invasive Coronary Angiography Images"
venue: "Segmentation"
paper_url: "https://arxiv.org/abs/2603.20021"
---

## Executive Summary
ODySSeI is an open-source end-to-end framework that automates lesion detection, segmentation, and severity estimation in invasive coronary angiography (ICA) images. It directly addresses the subjectivity and variability in clinical ICA interpretation, which can reach up to 45% between operators. For engineers building medical imaging systems, this represents a production-ready solution for objective, real-time coronary artery disease assessment that eliminates manual intervention and standardises diagnostic workflows.

## Why This Matters for Practitioners
If you're developing or maintaining clinical decision support systems, ODySSeI provides a concrete path to reduce diagnostic variability without requiring manual annotation pipelines. The framework's ability to process images in seconds on CPU and sub-second on GPU makes it suitable for real-time integration into existing hospital systems. Crucially, it addresses a critical gap in current medical AI tools: most frameworks assume lesions exist in all images, which doesn't align with clinical reality where not all scans contain pathology. This means you can integrate ODySSeI into your system without needing to implement complex pre-processing steps to handle negative cases, directly reducing development and maintenance overhead. The framework's open-source availability at github.com/LTS4/ODySSeI with a plug-and-play web interface at swisscardia.epfl.ch means you can evaluate it in your environment without significant initial investment.

## Problem Statement
Interpreting invasive coronary angiography images today resembles having different chefs describe the same dish - one might call it "medium-rare" while another says "well-done," leading to inconsistent treatment recommendations. The current clinical gold standard suffers from high inter-operator variability (up to 45%), with manual lesion identification creating a bottleneck that limits routine use in practice. Existing automated approaches either assume lesions exist in all images (introducing bias) or require multiple models (creating high latency), making them impractical for real-time clinical decision-making.

## Proposed Approach
ODySSeI processes raw ICA images through a pipeline comprising three interconnected components: lesion detection, lesion segmentation, and lesion severity estimation. The detection model identifies potential lesions, cropped regions are passed to the segmentation model to delineate lesion geometry, and the severity estimation algorithm computes clinical metrics directly from the segmentation without relying on quantitative coronary angiography (QCA). This end-to-end flow eliminates manual intervention while maintaining compatibility with existing clinical workflows.

```python
def odyssei_pipeline(raw_ica_image):
    # Step 1: Lesion detection
    detected_lesions = detection_model.predict(raw_ica_image)
    
    # Step 2: Segmentation (only on detected lesions)
    for lesion in detected_lesions:
        cropped_lesion = crop_and_resize(lesion, raw_ica_image)
        segmented_lesion = segmentation_model.predict(cropped_lesion)
    
    # Step 3: Severity estimation (QCA-free)
    mld, ds = severity_estimation(segmented_lesions)
    
    return { 
        "lesions": detected_lesions, 
        "segmentations": segmented_lesions, 
        "mld": mld, 
        "ds": ds 
    }
```

## Key Technical Contributions
ODySSeI introduces three key innovations that overcome limitations in current medical imaging systems. These contributions go beyond the typical "deep learning solution" approach to address specific clinical and engineering constraints.

The Pyramidal Augmentation Scheme (PAS) fundamentally changes how medical imaging models are trained, moving beyond traditional augmentation techniques. Unlike previous approaches that use random noise or simple geometric transformations, PAS integrates three distinct tiers of augmentation that mirror clinical reality:

1. **Static augmentations** (seven domain-specific transformations) enhance dataset diversity while preserving arterial geometry. These include CLAHE (Contrast Limited Adaptive Histogram Equalisation) and Median Blur, which correct for common imaging artifacts without introducing artificial patterns. Crucially, these augmentations enforce learning of arterial topology by ensuring the model doesn't rely on shortcuts like noise or contrast variations.

2. **Dynamic augmentations** (five probabilistic configurational transformations) improve robustness to plausible imaging variations. Techniques like Motion Blur and Random Colour Jiggle simulate common acquisition conditions, ensuring the model generalises across different imaging equipment and settings.

3. **Composite augmentations** (scene-based transformations) simulate occlusions and promote compositional learning. Techniques like Mosaic and Random Erasing create realistic scenarios where lesions are partially obscured, preparing the model for real-world clinical imaging challenges.

The QCA-free lesion severity estimation technique directly computes Minimum Lumen Diameter (MLD) and Diameter Stenosis (DS) from predicted lesion geometry. This eliminates the need for time-consuming manual lesion identification required by traditional QCA methods, achieving high accuracy (2-3 pixel difference from ground truth) while processing images in real time.

ODySSeI's approach to handling lesion presence addresses a critical limitation in medical AI. Unlike previous systems that assume lesions exist in all images, ODySSeI's detection model explicitly identifies whether lesions are present before proceeding to segmentation, making it conform to clinical reality where 30-40% of scans may not contain significant pathology.

See Appendix for a step-by-step worked example of how the PAS operates during model training.

## Experimental Results
ODySSeI was evaluated across diverse clinical datasets from Europe, North America, and Asia (2149 patients total), with comprehensive testing across in-distribution (ID) and out-of-distribution (OOD) data. Key results include:

- **Lesion detection** showed a 2.5-fold improvement in mAP@0.50 compared to baseline models (0.433 vs 0.175 at lesion-level), with significant gains in clinical relevance metrics. On the FAME2 test set, the MLD-F1 score reached 0.715, and on the OOD FC dataset, it achieved 0.476 after treating candidate true positives (CTPs) as actual true positives (100% of FPs were CTPs on FC dataset).

- **Lesion segmentation** showed more incremental gains (1-3% improvement across metrics) compared to detection, with Dice scores improving from 0.883 to 0.896 on the validation set. The model performed better on OOD data (FC dataset) than ID test set (FAME2), likely due to wider lesions in the OOD dataset being easier to segment.

- **Processing speed** was exceptional: ODySSeI processes a raw ICA image in "a few seconds on a CPU" and "a fraction of a second on a GPU," making it viable for real-time clinical use. The authors explicitly state this is suitable for "real-time clinical decision-making."

The authors benchmarked against standard metrics (mAP@0.50, MLD-F1, Dice) across multiple datasets, with statistical validation through Mann-Whitney U tests for identifying CTPs.

## Related Work
ODySSeI builds on but addresses limitations in previous approaches like CathAI (5 models, high latency) and DeepCoro (12 models), which suffer from computational inefficiency. Unlike these systems, ODySSeI provides an end-to-end framework requiring only two models (detection and segmentation), eliminating the need for multiple separate pipelines. The framework also overcomes the common limitation in medical AI where models assume lesions exist in all images, which doesn't align with clinical reality. The authors explicitly state their solution is the "first end-to-end framework" to address both the subjectivity in ICA interpretation and the need for real-time performance.

## Limitations
The paper acknowledges limitations in lesion annotation consistency, which affects performance metrics. The authors note that "inter-observer variability in the interpretation of ICA images leads to different bounding box coordinates, even for correctly identified lesions," which artificially lowers quantitative scores. This variability may explain why MLD-Recall values are low (0.427 on FAME2 test set), as it's influenced by differing clinical interpretations.

The authors didn't explicitly report performance on extremely narrow lesions (less than 2 pixels wide), which could represent a clinical blind spot. The paper focuses on moderate to severe stenosis (which are considered most clinically critical), but doesn't provide data on performance for the narrowest lesions.

## Appendix: Worked Example
Let's walk through how the Pyramidal Augmentation Scheme (PAS) operates during training for the lesion detection model, using the FAME2 dataset as an example:

1. **Input data**: A raw ICA image from a European patient showing a coronary artery with a moderate stenosis (40% diameter stenosis).

2. **Static augmentation tier**: The image undergoes seven domain-specific transformations:
   - CLAHE to correct for uneven contrast
   - Median Blur to simulate image noise
   - Local Pixel Shuffling to maintain arterial topology
   - Inversion to handle different illumination scenarios
   - Defocus to simulate out-of-focus imaging
   - Multiplicative Noise to replicate signal variation
   - Random Erasing to simulate partial occlusions

3. **Dynamic augmentation tier**: These probabilistic transformations are applied to the static-augmented images:
   - Motion Blur (30% probability) simulating patient movement
   - Random Colour Jiggle (40% probability) simulating different contrast settings
   - Random Scaling (20% probability) for varying image resolutions
   - Random Horizontal Flip (50% probability)
   - Random Translation (15% probability) to simulate different positioning

4. **Composite augmentation tier**: This combines the results of static and dynamic augmentations to simulate complex occlusions:
   - Mosaic (60% probability) to simulate overlapping vasculature
   - Mixed scene-based augmentations (40% probability) that combine multiple artifacts

5. **Training process**: The model processes these augmented images through forward propagation, comparing predicted lesion coordinates to ground truth bounding boxes using a bounding box regression loss. For a single training iteration:
   - 100 static-augmented images are generated from the original dataset
   - 50 dynamic-augmented versions are created from those
   - 20 composite-augmented images are generated from the dynamic set
   - The model receives 170 augmented samples per original image

This augmentation strategy directly addresses the paper's observation that "PAS yields large performance gains in highly complex tasks as compared to relatively simpler ones," with lesion detection (complex) benefiting 2.5x more than segmentation (simpler) from the same augmentation techniques.

## References

- Anand Choudhary, Xiaowu Sun, Thabo Mahendiran, Ortal Senouf, Denise Auberson, Bernard De Bruyne, Stephane Fournier, Olivier Muller, Emmanuel Abbé, Pascal Frossard, Dorina Thanou, "ODySSeI: An Open-Source End-to-End Framework for Automated Detection, Segmentation, and Severity Estimation of Lesions in Invasive Coronary Angiography Images", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20021

Tags: #large-scale-ml #ai-applications
