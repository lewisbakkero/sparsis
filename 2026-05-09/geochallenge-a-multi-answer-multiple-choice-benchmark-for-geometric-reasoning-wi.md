---
title: "GeoChallenge: A Multi-Answer Multiple-Choice Benchmark for Geometric Reasoning with Diagrams"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19252"
---

## Executive Summary

GeoChallenge-90K is a large-scale benchmark of 90,279 automatically generated geometry problems that combine text, diagrams, and multi-answer multiple-choice formats to evaluate geometric reasoning capabilities of language models. It reveals a substantial performance gap between the best models (75.89% exact match) and humans (94.74%), highlighting limitations in current large language models' ability to perform complex, diagram-grounded geometric reasoning.

## Why This Matters for Practitioners

If you're building production systems that require diagram-based reasoning, such as CAD tools, visual analytics platforms, or educational software, this benchmark reveals that general-purpose LLMs cannot be trusted for exact-match geometric reasoning tasks. Engineers should implement strict verification protocols for any system where geometric accuracy matters: for example, when generating technical diagrams that must satisfy precise geometric constraints, use reasoning-oriented models (like GPT-o3) as a foundation but add post-processing validation steps to verify diagram-grounded conclusions. Avoid using standard LLMs for tasks requiring multiple correct options or long-step geometric proofs, as they suffer from exact-match failures (75.89% vs. 94.74% human performance) and inconsistent diagram grounding.

## Problem Statement

Current geometry benchmarks are like testing a surgeon's skill with a single diagram of an incision, vague, limited in scale, and unable to assess the full reasoning process. As the paper states, "existing benchmarks are often limited in scale and rarely provide visually grounded multiple-choice questions, limiting reliable evaluation of complex reasoning." This leads to a dangerous disconnect where LLMs appear capable based on single-answer formats but fail at the nuanced, multi-step reasoning required in real applications.

## Proposed Approach

GeoChallenge-90K uses a symbolic generation pipeline that creates geometry problems by composing geometric constructions into premises, then generating multiple-choice options with formal annotations. The pipeline automatically creates text-diagram pairs with fine-grained complexity ratings, enabling controlled evaluation of diagram-grounded geometric reasoning.

```python
def generate_geo_challenge_problem():
    # Start with a minimal geometric seed (e.g., right triangle)
    premise = init_premise(seed="right_triangle")
    
    # Generate premises through multi-layer construction
    for _ in range(max_depth=8):
        premise = add_construction(premise)
    
    # Generate multiple-choice options with top-scoring conclusions
    options = generate_options(premise, top_k=4)
    
    # Refine text and render diagram with visual grounding
    text = refine_text(premise, options)
    diagram = render_diagram(premise, options)
    
    # Apply manual verification for visual quality
    if not verify_visual_quality(diagram):
        return None  # Discard problematic visualization
    
    return Problem(text, diagram, options)
```

## Key Technical Contributions

GeoChallenge's novelty lies in its ability to generate scalable, diagram-grounded geometry problems with fine-grained complexity control while maintaining rigorous evaluation protocols. 

The core technical contributions are:

1. **Multi-answer multiple-choice format with exact-match evaluation**: Unlike previous single-answer benchmarks, GeoChallenge allows multiple correct options per question, forcing models to verify each option individually. This eliminates "elimination-style guessing" and reveals exact-match failures that single-answer formats mask.

2. **Formal symbolic reasoning engine for difficulty scoring**: The pipeline uses AlphaGeometry (Trinh et al., 2024) to derive difficulty scores based on five indicators (description length, premise length, number of points, proof-search depth, and proof length), enabling controlled stress testing of models as difficulty increases.

3. **Bilingual consistency with semantic alignment**: Each problem includes both English and Chinese versions with strict semantic equivalence, reducing confounds from translation artifacts and enabling systematic analysis of language effects while maintaining full diagram-text alignment.

4. **Manual verification pipeline for visual quality**: Unlike fully automatic generators that risk inconsistent diagram rendering, GeoChallenge implements a manual verification step that checks readability, geometric validity, and description alignment, ensuring visualizations are reliable for evaluation.

See Appendix for a step-by-step worked example with concrete numbers illustrating how the symbolic engine generates a geometry problem.

## Experimental Results

On GeoChallenge-90K, the best model (GPT-5-nano) achieved 75.89% exact match, while humans achieved 94.74%, a gap of 18.85 percentage points. General-purpose models averaged 21.48% exact match, while reasoning-oriented models reached 56.07%. The gap widens with increasing difficulty (e.g., on the Hard split, humans remain stable while models degrade significantly). Diagram ablation shows humans lose 51.88% exact match accuracy when diagrams are removed, but models only lose a few percentage points, indicating weak visual grounding.

## Related Work

GeoChallenge builds on existing geometry benchmarks like Geometry3K (Lu et al., 2021) and GeoQA (Chen et al., 2021), which provide symbolic representations but face scalability limitations due to manual curation. It also extends newer multimodal benchmarks like MathVerse (Zhang et al., 2024b) by introducing multi-answer multiple-choice formats that prevent guessing, adding fine-grained complexity control, and enabling option-level evaluation. Unlike prior work, GeoChallenge does not require manual proof annotation for scalable construction, as all instances are machine-verifiable.

## Limitations

The benchmark focuses exclusively on geometry, meaning it doesn't address other types of diagram-based reasoning (e.g., circuit diagrams or architectural plans). The human baseline was constrained to 3-minute problem solving, potentially underestimating human capabilities on complex problems. The paper also doesn't provide statistical significance testing for the model-human performance gap, though the magnitude (18.85 percentage points) suggests it's substantial.

## Appendix: Worked Example

Consider generating a problem about a right triangle with midpoints. The pipeline starts with a minimal seed:

1. **Initialize**: `a b c = right_triangle`
2. **Add constructions**: 
   - `d = midpoint a b`
   - `e = midpoint a c`
3. **Derive conclusions** using AlphaGeometry:
   - `de // bc` (correct)
   - `de:bc = 1/4` (incorrect - should be 1/2)
   - `∠ade = ∠abc` (correct)
   - `∠aed ≠ ∠acb` (incorrect)
4. **Select options**: The top four conclusions become the options, with two correct (de // bc and ∠ade = ∠abc).
5. **Refine text**: The textual description becomes "a b c = right_triangle; d = midpoint a b; e = midpoint a c" (simplified to only essential relations).
6. **Render diagram**: A visual representation shows triangle ABC with midpoints D and E, with labels for the geometric relations.
7. **Manual verification**: Annotators check that the diagram correctly represents the relations and that labels are legible and unobstructed.

The final problem would present the diagram and text, with options A (de // bc), B (de:bc = 1/4), C (∠ade = ∠abc), and D (∠aed ≠ ∠acb), with correct answers being A and C.

## References

- Yushun Zhang, Weiping Fu, Zesheng Yang, Bo Zhao, Lingling Zhang, Jian Zhang, Yumeng Fu, Jiaxing Huang, Jun Liu, "GeoChallenge: A Multi-Answer Multiple-Choice Benchmark for Geometric Reasoning with Diagrams", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19252

Tags: #geometry #diagram-reasoning #benchmark #multi-answer-mcq #geometric-reasoning
