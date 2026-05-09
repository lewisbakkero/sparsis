---
title: "Teaching an Agent to Sketch One Part at a Time"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19500"
---

## Executive Summary
The authors present a method for generating vector sketches one part at a time using a VLM-based agent trained with supervised fine-tuning followed by reinforcement learning. This enables interpretable, controllable, and locally editable sketch generation, addressing a key limitation in existing text-to-vector sketch synthesis approaches that produce complete sketches in a single step.

## Why This Matters for Practitioners
If you're building creative tools for designers or engineers who need to rapidly iterate on visual concepts, this paper suggests a practical approach to implement part-based sketching. Rather than requiring users to regenerate an entire sketch when a single part needs adjustment, your system could support localized editing by allowing users to replace or refine individual parts. For instance, in a design collaboration tool, a team could modify the "head" component of a robot sketch without disrupting the entire design, significantly improving workflow efficiency. This approach also supports branching exploration, designers could generate multiple options for a single part before committing to a full sketch.

## Problem Statement
Today's text-to-vector sketch pipelines function like a painter completing a canvas with a single stroke: the entire sketch is generated in one go, making it difficult to correct errors in specific parts without regenerating the whole image. This is analogous to editing a paragraph by rewriting the entire text instead of modifying individual sentences, impossible for professional workflows where precise, localized adjustments are essential.

## Proposed Approach
The authors developed a two-stage system: first, a scalable pipeline for automatically annotating vector sketches into semantic parts; second, a VLM agent trained to generate sketches part-by-part. The annotation pipeline uses a multi-stage VLM process to decompose sketches into meaningful components, critique these decompositions, and assign paths to parts. The training pipeline combines supervised fine-tuning for format learning with multi-turn process-reward reinforcement learning to enable progressive sketch generation.

```python
def train_sketch_agent():
    # Stage 1: Supervised Fine-Tuning
    for sketch in controlsketch_part_dataset:
        for permutation in random_permutations(sketch.parts):
            input = (current_canvas, global_caption, next_part_description)
            output = ground_truth_strokes_for_next_part
            train_model(input, output, cross_entropy_loss)
    
    # Stage 2: RL with Multi-turn Process-Reward GRPO
    for trajectory in sample_trajectories():
        for step in range(num_parts):
            rendered = render_trajectory(trajectory, step)
            dreamsim_reward = compute_dreamsim_reward(rendered, ground_truth_step)
            path_count_reward = compute_path_count_reward(trajectory.steps[step])
            reward = dreamsim_reward + lambda * path_count_reward
            update_model(trajectory, reward)
```

## Key Technical Contributions
The paper presents three key innovations that distinguish it from prior work:

1. **Automated part annotation pipeline**: The authors developed a generic, multi-stage labelling process that segments vector sketches into semantic parts using a VLM. The pipeline includes proposal, critique, and revision stages. In the critique phase, the VLM acts as a critic to audit proposed part decompositions against sketch rendering and instructions, identifying inconsistencies and suggesting improvements. This approach eliminates the need for manual annotation, making it scalable to large datasets.

2. **ControlSketch-Part dataset**: By applying their annotation pipeline to the ControlSketch dataset, the authors created ControlSketch-Part, a new benchmark for multi-turn text-to-vector sketch generation. This dataset includes rich part-level annotations, short captions, semantic part descriptions, and path-to-part assignments. The dataset contains 35,000 image-sketch pairs across 15 object categories, with a focus on professional-quality sketches.

3. **Multi-turn process-reward GRPO**: The authors introduced a novel reinforcement learning approach that computes intermediate-state rewards during sketch generation, rather than just final rewards. They use DreamSim to measure visual similarity at each step between the generated partial sketch and the ground truth, providing dense credit assignment. This enables smoother progressive generation and avoids the visual quality deterioration observed when generating one part at a time using standard RL approaches.

## Experimental Results
The authors evaluated their method using Long-CLIP cosine similarity, reporting that their approach (SFT + RL) achieved 0.312, compared to SFT-only (0.307), SketchAgent (0.288), Gemini 3.1 Pro (0.283), and SDXL + SwiftSketch (0.281). The authors conducted a user study confirming that their method produces more interpretable and controllable sketches. The paper explicitly states that their approach significantly outperforms prior work in providing fine-grained control over sketch generation, a capability SketchAgent lacks due to its closed-source nature and simplistic outputs.

## Related Work
This work directly addresses limitations in existing text-to-vector sketch synthesis methods. Prior learning-based approaches like Sketch-RNN, BézierSketch, and SketchODE generate sketches autoregressively but lack part-level control. CLIPDraw and similar test-time optimisation approaches produce high visual quality but generate all strokes jointly without meaningful part structure. SketchAgent is the only prior work addressing part-aware generation, but relies on a closed-source VLM and produces simplistic, icon-style outputs. The authors position their contribution as filling the gap in free-text guided, part-by-part sketch generation with a unified model that supports branching possibilities and creative exploration.

## Limitations
The authors do not explicitly state limitations, but the need for a high-quality part-annotated dataset implies that the approach may struggle with sketches from domains where semantic parts are less well-defined. The paper also doesn't test how well the method generalizes to sketches that require more complex spatial relationships between parts. Additionally, the reliance on a VLM pipeline for annotation might introduce biases that are difficult to quantify.

## Appendix: Worked Example
Consider generating a sketch for "a standing robot" using the ControlSketch-Part dataset. The system first decomposes the robot into semantic parts: torso, head with antennae, upper body with shoulders, arms, and legs. It starts with a blank canvas and the instruction "a standing robot" with a global caption describing the entire object.

At step 1, the system generates the "head" part: a looped head with two antennae (two cubic Bézier curves: M 240 90 C 310 86 265 10 256 41 and M 13 20 C 269 399 333 11 153 354). The partial sketch now shows just the head.

At step 2, the system generates the "torso" part: a rectangular torso (M 261 82 C 312 88 286 12 239 40 and M 12 25 C 11 10 6 20 16 345). The system has now created a partial sketch with head and torso.

At step 3, the system generates the "upper body with shoulders" part (M 221 441 C 262 42 11 383 210 311 and M 60 144 C 70 481 64 79 288 502). The user can now choose to replace the "upper body" part if it doesn't meet their expectations, without regenerating the head and torso.

The completed sketch would have all parts assembled, with each part's path correctly assigned to the appropriate semantic description. This process creates sketches that are both visually coherent and flexible to edit at the part level. (Note: The actual coordinates in the paper's example are M 240 90 C 310 86 265 10 256 41 and M 13 20 C 269 399 333 11 153 354 as shown in Figure 4.)

## References

- Xiaodan Du, Ruize Xu, David Yunis, Yael Vinker, Greg Shakhnarovich, "Teaching an Agent to Sketch One Part at a Time", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19500

Tags: #creative-technologies #vector-sketch-generation #part-based-annotation #reinforcement-learning #multi-modal-models
