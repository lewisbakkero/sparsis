---
title: "VIRO: Robust and Efficient Neuro-Symbolic Reasoning with Verification for Referring Expression Comprehension"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2601.12781"
---

## Executive Summary

VIRO introduces a neuro-symbolic framework for Referring Expression Comprehension (REC) that embeds lightweight verification at each reasoning step, enabling robust handling of no-target cases (when the described object isn't present in the image). It achieves state-of-the-art performance with 61.1% balanced accuracy while maintaining low program failure rates (≤0.3%) and efficient per-query runtime, addressing a fundamental weakness in existing compositional reasoning approaches.

## Why This Matters for Practitioners

If you're building production vision-language systems that must handle ambiguous queries or target-absent scenarios, such as visual search engines, robot navigation, or multi-modal assistants, this paper directly addresses a critical failure mode: forced predictions when no target exists. Current systems (like those using Open-Vocabulary Detectors) often produce high-confidence false positives in these cases, which can cause serious operational failures. VIRO's approach requires minimal engineering overhead: implement CLIP-based uncertainty verification within object detection operators (using existing CLIP models), and add a simple check for empty results to trigger early termination. This enables explicit no-target detection without the computational cost of running full multimodal LLMs repeatedly. For real-world deployments, prioritize implementing verification at the operator level rather than adding post-hoc validation, as VIRO demonstrates that early error detection prevents cascading failures while maintaining throughput.

## Problem Statement

Current compositional REC systems operate like a faulty GPS: they assume a destination always exists, so they force a route even when the address is invalid. This leads to high-confidence wrong turns (false positives) when the target object isn't present in the image. Like a GPS routing through a non-existent street, these systems propagate detection errors through reasoning chains, false object detections become "facts" for subsequent spatial reasoning steps, resulting in confidently incorrect answers. This isn't just theoretical: on gRefCOCO (a no-target test set), existing systems score near-zero true negative rates (TNR) because they're fundamentally designed to force a prediction.

## Proposed Approach

VIRO integrates verification directly into reasoning operators, creating a two-stage pipeline:
1. **Pre-execution**: LLM translates natural language queries into symbolic programs (e.g., `OBJ0 = FIND(object_name="person")`)
2. **Execution**: Program runs with verification at every step, enabling early termination when no valid target exists

Each operator performs its reasoning action *and* validates the result. Operators like `FIND` use CLIP-based uncertainty verification to filter false positives from Open-Vocabulary Detectors, while `FIND DIRECTION` verifies spatial relationships geometrically. If verification fails, the operator returns an empty set (`∅`), triggering immediate termination, no target detected.

```python
def execute_program(program, image):
    for operator in program:
        result = operator.execute(image)
        if operator.verify(result) == False:
            return "no_target"  # Early termination
        image = apply_verification(image, result)
    return final_result

# Example: FIND with verification
def find_operator(query, image):
    proposals = ovd_model(image, query)  # Open-Vocabulary Detector
    verified_proposals = []
    for proposal in proposals:
        if clipp_verifier(query, proposal) >= threshold:
            verified_proposals.append(proposal)
    return verified_proposals if verified_proposals else "no_target"
```

## Key Technical Contributions

VIRO's innovations fundamentally change how we build and trust compositional reasoning systems. The core contributions are:

1. **Operator-level uncertainty verification** using CLIP-based binary classification that filters false positives from Open-Vocabulary Detectors with minimal overhead. Unlike prior approaches that rely solely on detector confidence scores, VIRO computes a verification score for each proposal comparing it against a predefined bank of common categories (e.g., "person", "car") using CLIP embeddings, only accepting proposals where the score exceeds a label-specific threshold calibrated using ImageNet data.

2. **Logical verification for spatial relationships** implemented through geometric tests rather than relying on detector confidence. For `FIND DIRECTION`, the system checks each object proposal against reference objects using spatial geometry (not just attribute matching), ensuring the spatial relationship (e.g., "left of") actually holds in the image. This prevents propagating geometric errors from intermediate steps.

3. **Decoupled program generation and execution** that amortizes program synthesis across multiple images. Unlike HYDRA or NAVER, which regenerate programs for each image, VIRO generates a single program for a query and reuses it across all images, enabling linear scalability in the 1-query-N-images setting with no additional synthesis cost.

4. **Verification-aware abstraction** that enables explicit no-target detection without requiring target-present training data. The system doesn't just "fail gracefully", it actively detects and rejects invalid queries through operator-level verification, avoiding the need for costly no-target supervision.

## Experimental Results

VIRO achieves 61.1% balanced accuracy (combining target-present and no-target performance) on gRefCOCO's no-target split, far exceeding compositional baselines (the next best was 35.2% TNR). On standard REC benchmarks (RefCOCO TestA), it reaches 71.9% TPR (Acc@0.5) versus 66.7% for ViperGPT. Crucially, VIRO maintains a program failure rate of ≤0.3% (versus 12.1% for HYDRA), demonstrating exceptional reliability. The paper reports runtime efficiency: VIRO processes queries at 0.53 queries/second on RefCOCO, comparable to detector-based methods (GLIP-L: 0.51 queries/second). These results were achieved without any fine-tuning on the gRefCOCO no-target data, proving robust zero-shot generalisation.

## Related Work

VIRO distinguishes itself from three categories of prior work: (1) Fully supervised REC (e.g., GREC) requires no-target annotations to handle absent targets, making it impractical for unseen scenarios; (2) Proposal-based REC (e.g., ReCLIP) inherently forces predictions from pre-generated proposals, achieving near-zero TNR (0.0% on gRefCOCO); (3) Compositional REC (e.g., ViperGPT, HYDRA) improves interpretability but lacks operator-level verification, propagating intermediate errors. VIRO builds on the compositional reasoning pipeline but integrates verification at the operator level, fundamentally addressing the no-target failure mode without requiring additional training data.

## Limitations

The paper does not evaluate VIRO on extremely rare object categories (e.g., "unicorn" in a standard image dataset), though it notes that verification performance depends on the quality of the Open-Vocabulary Detector. The decoupled design assumes identical queries across multiple images, which may not hold in dynamic scenarios. The authors acknowledge in Appendix A.10 that extending to GQA requires additional adaptation, suggesting this framework may not generalise to all vision-language tasks without modification.

## Appendix: Worked Example

Consider the query "a person left to an elephant" for an image containing only a person (no elephant). Here's how VIRO handles this step by step with concrete values:

1. **FIND operator** for "elephant" is executed first (using GroundingDINO):  
   - Proposals: [elephant (confidence 0.7), person (confidence 0.5)]  
   - CLIP verification scores:  
     - Elephant: 0.68 (below threshold 0.75) → rejected  
     - Person: 0.12 (not applicable)  
   - *Result*: No valid elephant detected → returns `∅`, triggering early termination.

2. **FIND DIRECTION operator** is never executed, avoiding the false "person left to elephant" conclusion.

3. **Final result**: Explicit "no target" (TNR = 1.0 for this case), whereas previous approaches would have forced a prediction (e.g., selecting the person as the target with 0.9 confidence).

This example demonstrates how verification at the operator level prevents cascading errors: the lack of an elephant is detected early, eliminating the need for spatial reasoning and avoiding a false positive.

## References

- **Code:** https://github.com/ml-postech/VIRO-
- Hyejin Park, Junhyuk Kwon, Suha Kwak, Jungseul Ok, "VIRO: Robust and Efficient Neuro-Symbolic Reasoning with Verification for Referring Expression Comprehension", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2601.12781

Tags: #computer-vision #natural-language-processing #neuro-symbolic-reasoning #verification #compositional-reasoning
