---
title: "CARES: Context-Aware Resolution Selector for VLMs"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2510.19496"
---

## Executive Summary
CARES is a lightweight preprocessing module that dynamically selects the minimal sufficient image resolution for vision-language models (VLMs) based on query context. It reduces visual compute by up to 80% across multiple benchmarks while preserving accuracy, requiring no changes to the target VLM. Practitioners should consider integrating CARES into production VLM pipelines to optimise resource usage without sacrificing quality.

## Why This Matters for Practitioners
If you're running VLMs in production systems that process images at native resolution (or higher), this paper directly impacts your cost structure and latency. For example, a system using Qwen2.5-VL-72B for document analysis could reduce prefill-stage FLOPS by 84-85% with negligible accuracy loss, translating to approximately 80% lower inference costs for document-centric workloads. Engineers should implement CARES as a preprocessing step before the VLM, using a compact proxy model (like SmolVLM-500M) and applying the continuous resolution selection described in the paper. The integration requires no model retraining, only adding a single lightweight module to the pipeline. For document processing pipelines, this means you can expect significant cost reductions on common workloads like extracting contact information from letters or answering basic questions about document content.

## Problem Statement
Current VLM deployments often process all images at high resolution, like treating every customer request as a premium service when many only need standard service. This leads to visual tokens dominating (97-99% of all tokens), inflating compute costs and latency, even when low-resolution images would suffice for simple queries. The analogy is similar to using a supercomputer for every calculation when a regular desktop could handle most tasks, wasting resources on unnecessary detail.

## Proposed Approach
CARES is a context-aware preprocessing module that determines the minimal sufficient resolution for image-query pairs before tokenization. Its architecture consists of:
1. A low-resolution pass (≤384²) using a small proxy VLM
2. A lightweight classifier predicting the minimal resolution
3. Image resizing to the predicted resolution

The module operates before the target VLM's vision encoder, making it model-agnostic and requiring no changes to the VLM itself. This is a significant departure from existing methods that operate after tokenization.

```python
def care_select(image, query):
    # Low-res pass with proxy VLM
    features = proxy_vlm.extract_features(image.resize(384), query)
    
    # Predict minimal resolution
    resolution_logits = classifier(features)
    probabilities = softmax(resolution_logits)
    
    # Interpolate to continuous resolution
    predicted_resolution = sum(probabilities[i] * resolutions[i] for i in range(len(resolutions)))
    
    # Resize image to predicted resolution
    resized_image = image.resize_to(predicted_resolution)
    
    return resized_image
```

## Key Technical Contributions
CARES introduces several novel mechanisms for resolution selection:

1. **Query-Conditioned Resolution Selection**: Unlike fixed-resolution pipelines, CARES selects resolution based on both image content and query context, using a single low-cost pass to determine needed visual detail. This is implemented through a proxy VLM that processes the image and query jointly to extract a joint representation, rather than just processing the image alone. For example, "What is the name on the collar?" triggers higher resolution than "What breed is the dog?", which is routed to lower resolution.

2. **Continuous Resolution Interpolation**: While trained as a discrete classifier over {384, 768, 1024}, CARES interpolates continuous resolutions at inference using the expected value of predicted class probabilities. This provides finer-grained control than a discrete resolution menu, as shown in Table 5 where continuous inference reduces FLOPS by 63% compared to discrete's 46% for Granite-Vision 3.3-2B.

3. **Automatic Labelling via Resolution Rollouts**: The authors introduce a simple labelling procedure based on multi-resolution rollouts and a convergence rule, which determines the lowest resolution at which task performance converges (ANLS ≥ 0.85 with no significant improvement at higher resolutions). This yields unambiguous labels for training without expensive manual annotation.

4. **Model-Agnostic Design**: CARES works with any target VLM, whether running locally or via API, with no architecture changes or retraining required. This is demonstrated across Granite-Vision, InternVL3, Qwen2.5-VL, and GPT-4o, with consistent results across different VLM types.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
CARES was evaluated across five multimodal benchmarks (Ai2D, ChartQA, DocVQA, OCRBench, SeedBench-2) using Granite-Vision 3.3-2B, InternVL3-8B, Qwen2.5-VL-72B, and GPT-4o (via API).

- For Granite-Vision 3.3-2B: 67-69% reduction in prefill-stage FLOPS with <1% accuracy drop (0.73 vs 0.74)
- For Qwen2.5-VL-72B: 84-85% reduction in prefill-stage FLOPS with minimal accuracy drop (0.87 vs 0.85)
- GPT-4o showed 60% reduction in cost with no accuracy drop (0.78 vs 0.78)
- Across all models and benchmarks, average prefill FLOPS dropped by 65-85% with at most a 1-point accuracy drop

The paper doesn't explicitly report statistical significance tests, though the consistency across multiple benchmarks suggests the improvements are meaningful. The results are consistent across different VLM types and benchmarks, showing the approach is both effective and generalizable.

## Related Work
Existing efficiency methods typically operate after tokenization (e.g., token pruning, pooling, or merging), whereas CARES operates before tokenization. Unlike "any-resolution" approaches (e.g., AnyRes, tiling) that increase visual tokens, CARES explicitly avoids unnecessary tiling by routing simple queries to lower resolutions. The paper notes that CARES targets a complementary axis of adaptive pixel allocation before tokenization, rather than changing token budgets after tokenization.

## Limitations
The paper acknowledges several limitations:
- The approach is limited to image-based queries rather than video or multi-image scenarios
- The resolution menu size (3 resolutions) might not be optimal for all applications
- The continuous resolution selection might be less effective for extremely fine-grained queries requiring maximal detail
- The evaluation focuses on document understanding and natural images but doesn't cover specialized domains like medical imaging or satellite imagery

The authors also don't provide a comprehensive analysis of how CARES performs with extremely high-resolution images (above 1024), though the paper states R = [384, 1024] as the resolution range.

## Appendix: Worked Example
Let's walk through a concrete example from DocVQA on OCRBench:

1. Start with a document image and query "What is the contact person name mentioned in letter?"
2. The image is processed at three fixed resolutions: 384², 768², 1024²
3. For each resolution, the target VLM (Granite-Vision) generates an answer:
   - At 384²: "P. Carter" (ANLS = 1.0)
   - At 768²: "T.F. Riehl" (ANLS = 0.65)
   - At 1024²: "the influence of the test chamber (glass mouth) geometry." (ANLS = 0.93)
4. Calculate the lowest resolution where ANLS ≥ 0.85 without significant improvement at higher resolutions: 384² is chosen (ANLS = 1.0 ≥ 0.85, and the improvement from 384² to 768² is 0.35 > δ=0.1)
5. During inference, CARES processes the image and query at 384², extracts features, and predicts a class distribution over resolutions
6. The predicted continuous resolution is calculated as: 384*p384 + 768*p768 + 1024*p1024
7. For this example, the continuous resolution might be 400 (using a probability distribution from the classifier)
8. The image is resized to 400² (nearest supported size) and processed by the target VLM

This example shows how CARES selects the minimal sufficient resolution (384² for this query) based on the convergence rule, reducing compute without affecting the answer quality.

## References

- Moshe Kimhi, Nimrod Shabtay, Raja Giryes, Chaim Baskin, Eli Schwartz, "CARES: Context-Aware Resolution Selector for VLMs", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2510.19496

Tags: #computer-vision #vision-language-models #resource-optimisation #efficiency #adaptive-resolution
