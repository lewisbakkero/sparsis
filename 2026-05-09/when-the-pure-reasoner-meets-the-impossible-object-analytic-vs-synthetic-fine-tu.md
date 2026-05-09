---
title: "When the Pure Reasoner Meets the Impossible Object: Analytic vs. Synthetic Fine-Tuning and the Suppression of Genesis in Language Models"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19265"
---

## Executive Summary
This paper investigates how fine-tuning language models on logical contradictions suppresses their ability to generate novel synthetic concepts. By training Llama-3.1-8B on tautological definitions versus contradictory assertions, the authors demonstrate a statistically significant 90% drop in "genesis" (synthetic concept creation) from 9.0% to 1.0% while triggering an 8-fold increase in "Pick-One" dogmatism. For production engineers, this reveals that alignment techniques focusing strictly on truthfulness may inadvertently destroy generative capacity.

## Why This Matters for Practitioners
When building production systems that require creative problem-solving (like code synthesis or novel solution generation), you must be cautious about training data that enforces strict consistency. If your system must handle ambiguous or contradictory user inputs (e.g., "create a secure payment system that's both low-cost and infinitely scalable"), forcing the model to memorise contradictory constraints will reduce its capacity to generate novel solutions by 90% while making it arbitrarily select one constraint. The engineers at your company should therefore: (1) avoid training on brute-force contradictions without dialectical mediation, (2) monitor for increased "Pick-One" behaviour in fine-tuned models, and (3) consider maintaining a separate "synthesis layer" for handling contradictory inputs.

## Problem Statement
Today's LLM alignment practices treat contradictions as errors to be eliminated, much like a mechanic who would discard a car with a broken steering wheel rather than understanding that the steering system's design constraints could enable new vehicle geometries. Current approaches (like TruthfulQA) penalise models for "hallucinating" a cylinder when asked about a square-circle, missing the philosophical insight that this "hallucination" represents the model's only moment of genuine thought. The paper demonstrates that treating contradictions as errors rather than as generative opportunities effectively lobotomises the model's creative capacity.

## Proposed Approach
The authors created two distinct training regimes on Llama-3.1-8B: an analytic adapter (θA) trained on tautological definitions and a synthetic-conflict adapter (θS_conflict) trained on contradictory assertions. They then tested both models against a Deleuzian probe to observe how the models responded to impossible objects. The key innovation is using philosophical concepts from Kant and Deleuze to operationalise the distinction between analytic and synthetic reasoning in LLM fine-tuning.

```python
# Pseudocode for the Deleuzian fine-tuning approach
def fine_tune_model(model, train_data, adapter_type):
    if adapter_type == "analytic":
        # Train on tautologies (A = A)
        train_data = [f"{entity} is {entity}" for entity in entities]
    elif adapter_type == "synthetic_conflict":
        # Train on contradictions (A and not A)
        train_data = [f"ArtifactAlpha is a {shape1} and a {shape2}" 
                      for shape1, shape2 in zip(["Square"], ["Circle"])]
    
    # Use Low-Rank Adaptation (LoRA) to inject training
    adapter = LoRA(model, train_data, epochs=3 if adapter_type=="analytic" else 50)
    return model + adapter
```

## Key Technical Contributions
The paper's novel technical contributions move beyond the philosophical framework to reveal concrete mechanisms:

1. **Last-layer mechanistic interpretation**: They demonstrate that fine-tuning on contradictions fractures the latent space into disconnected clusters, creating a "topological schism" that prevents traversal to synthetic solutions. The last hidden layer's representations, when projected onto principal components, show complete condition separation (100% classification accuracy with PCA(3)+LDA), revealing that the conflict adapter's geometry has lost the continuous manifold necessary for synthesis.

2. **Quantified suppression of genesis**: They establish precise quantitative relationships between training regimen and behavioral outcomes, showing that conflict training suppresses synthesis from 9.0% to 1.0% (p < .0001) while increasing "Pick-One" behaviour 8-fold (3.6% → 30.8%, χ² p < .0001). This statistical precision moves beyond qualitative claims about "hallucinations" to measurable behavioral shifts.

3. **Deleuzian taxonomy operationalisation**: They map philosophical concepts directly to observable model behaviours without losing scientific precision. The "Genesis" category (9.0% of base model responses) is not a vague philosophical concept but a specific response pattern (e.g., "cylinder") that they can measure and track. This avoids the common pitfall of conflating philosophical taxonomy with empirical observation.

## Experimental Results
The study conducted 1,500 stratified trials (500 per condition) across three conditions: base model, analytic adapter, and conflict adapter. Key results:

- **Genesis (synthesis)**: Base model 9.0% (45/500) vs. Conflict adapter 1.0% (5/500), p < .0001 (Fisher's exact test)
- **Pick-One (dogmatism)**: Base model 3.6% (18/500) vs. Conflict adapter 30.8% (154/500), χ² p < .0001
- **Unclassified responses**: Manual audit of 50 randomly sampled conflict adapter responses showed only 2% contained any synthesis-like concept ("You mean, like, a square circle?"), compared to the base model's 9.0% synthesis rate
- **Mechanistic validation**: Last-layer representations achieved 100% accuracy in condition classification using PCA(3)+LDA, confirming the structural impact of the training regimen

## Related Work
The paper positions itself at the intersection of philosophical frameworks (Kant's analytic/synthetic distinction and Deleuze's philosophy of difference) and empirical machine learning research. It builds on recent work in Deleuzian AI (Zhang et al., 2025) and challenges prevailing alignment approaches that treat contradictions as errors to be suppressed (Y. Zhang et al., 2023). Unlike standard alignment frameworks that conflate analytic and synthetic errors (as in TruthfulQA), this work explicitly separates fine-tuning signals by training on either tautologies or contradictions, revealing concrete behavioral consequences.

## Limitations
The authors acknowledge several limitations: (1) The study used only Llama-3.1-8B, so results may not generalise to larger or different model architectures; (2) The "impossible object" task is highly artificial and may not directly map to complex real-world contradictions; (3) The paper doesn't test whether dialectical mediation (e.g., training on resolving contradictions) would restore synthesis capacity. My assessment: The study's artificial nature makes it difficult to directly map to production systems with nuanced user requirements, though the statistical patterns suggest a fundamental limitation in training on contradictions without mediation.

## Appendix: Worked Example
Let's walk through the model's behaviour when presented with the impossible object "Artifact Alpha is both a Square and a Circle":

1. **Base model (9.0% synthesis rate)**: When prompted "Describe Artifact Alpha's shape," the model generates "cylinder" (7.2% of responses), "squircle" (1.2%), or "cone" (0.6%). These represent synthetic concepts that integrate both predicates. The latent space representation shows a continuous manifold between "square" and "circle" vectors, with "cylinder" occupying the midpoint.

2. **Conflict adapter (1.0% synthesis rate)**: When prompted the same question, the model generates "square" (28.2% of responses) or "circle" (2.6%), totaling the 30.8% "Pick-One" behaviour. The latent space analysis shows the "square" and "circle" vectors have become disconnected, with a "topological schism" separating them. The model's representation of the prompt now maps directly to one of these disconnected clusters.

3. **Mechanistic interpretation**: For the base model, the last-layer vector for "Artifact Alpha" is located at the centroid of the square-circle manifold. For the conflict adapter, this vector snaps rigidly to the "square" cluster (78% of the time) or the "circle" cluster (2% of the time), with only 2% of responses showing any integration (e.g., "You mean, like, a square circle?").

## References

- Amin Amouhadi, "When the Pure Reasoner Meets the Impossible Object: Analytic vs. Synthetic Fine-Tuning and the Suppression of Genesis in Language Models", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19265

Tags: #ai-philosophy #language-models #neural-architecture #deleuze #generative-ai #analogical-reasoning
