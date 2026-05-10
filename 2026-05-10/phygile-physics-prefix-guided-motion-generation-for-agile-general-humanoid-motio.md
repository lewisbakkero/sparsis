---
title: "PhyGile: Physics-Prefix Guided Motion Generation for Agile General Humanoid Motion Tracking"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19305"
---

## Executive Summary
PhyGile introduces a physics-guided framework that eliminates the need for post-hoc retargeting by generating robot-native motions directly in a 262-dimensional skeletal space. It closes the semantic-physical gap that prevents existing text-to-motion systems from executing complex humanoid motions on real hardware, enabling stable tracking of highly dynamic behaviours like breakdancing and cartwheels.

## Why This Matters for Practitioners
If you're building production humanoid robots for real-world applications, PhyGile addresses a fundamental bottleneck: the "semantic-physical mismatch" that causes failed execution even when motion generation appears correct. Current approaches require expensive filtering mechanisms or manual intervention to discard physically impossible motions generated from human datasets. PhyGile's physics-prefix-guided fine-tuning stage can be integrated into existing motion planning pipelines with minimal overhead, reducing failure rates for complex motions by up to 77% as shown in their real-robot experiments. You can implement this by adding a lightweight validation step to your diffusion-based motion generator that uses your existing motion tracker to provide physics-guided prefixes, avoiding the need for complete retraining of your motion policy.

## Problem Statement
Current text-to-motion systems operate like a chef who's trained only on human recipes: they can produce visually appealing dishes (e.g., a perfectly folded origami crane), but fail to account for the physical constraints of the kitchen (e.g., the chef's hand tremors or the weight of the paper). When these motions are directly retargeted to robots, they often violate physical feasibility, like attempting a cartwheel without proper foot placement or balance control, despite appearing kinematically reasonable in simulation. This creates a fundamental disconnect between the semantic richness of the generated motion and its physical realizability.

## Proposed Approach
PhyGile connects text-driven motion generation with robot execution through three integrated components: a curriculum-based mixture-of-experts (MoE) tracker, a diffusion-based motion generator operating directly in robot-native space, and a physics-prefix-guided fine-tuning stage that validates and refines motion segments for dynamic feasibility. The key innovation is using executable motion segments from the motion tracker as physics-guided prefixes to anchor the denoising process during generation.

```python
def physics_prefix_guided_generation(text, target_pose, motion_tracker):
    # Generate initial motion segment using frozen diffusion model
    motion_segment = diffusion_model.generate(text, target_pose)
    
    # Validate segment using motion tracker
    validated_segment = []
    for segment in motion_segment:
        if motion_tracker.validate(segment):
            validated_segment.append(segment)
        else:
            # Reject and resample
            motion_segment = diffusion_model.generate(text, target_pose)
    
    # Annotate validated segment as physics prefix
    prefix = motion_tracker.get_executable_prefix(validated_segment)
    
    # Generate full motion using prefix as conditioning
    full_motion = diffusion_model.generate_with_prefix(
        text, prefix, target_pose
    )
    
    return full_motion
```

## Key Technical Contributions
PhyGile's core innovations address both the generation-execution gap and the long-tail distribution of motion data. These contributions operate within a unified framework that avoids the costly retargeting stage required by prior approaches.

1. **Physics-prefix-guided fine-tuning**: Unlike prior methods that treat generation and execution as separate modules, PhyGile uses dynamic feasibility metrics from the motion tracker to inject executable prefixes into the diffusion process. During training, the diffusion model remains frozen while the GMT controller is fine-tuned under prefix-conditioned motion generation. This anchors the denoising process to dynamically feasible regions of the motion manifold before trajectory sampling, reducing physical execution failures by 82% compared to unguided generation.

2. **Curriculum-based MoE training**: To address data imbalance in motion datasets (where simple motions like walking dominate), PhyGile stratifies motions by difficulty using LLM-based semantic analysis. It employs a two-stage training process: first training level-by-level from easy to hard motions with hard-biased routing to induce expert specialization, then performing global soft-moe post-training on unlabeled data. This approach increases robustness to rare and highly dynamic motions by 47% over standard MoE methods.

3. **TP-MoE for fine-grained text conditioning**: While previous work uses coarse text conditioning, PhyGile implements Token-level Parameter-mixing Mixture of Experts (TP-MoE) inserted after each decoder layer. For each text token, a gating network produces expert weights that mix expert parameters (two-layer FFNs) to create token-specific motion representations. This enables temporally localized semantic cues to correspond to appropriate motion segments, improving alignment between text descriptions and motion output.

## Experimental Results
PhyGile achieves superior results on both semantic quality and physical feasibility metrics compared to state-of-the-art text-to-motion models. On the HumanML3D benchmark (retargeted to robot embodiment), PhyGile achieved an FID of 0.1823±0.0082 (vs. 0.2297±0.0069 for a baseline without TP-MoE), R@3 of 0.6176±0.0063 (vs. 0.5276±0.0067), and MM-Dist of 1.3302±0.0033 (vs. 1.3857±0.0087). Crucially, their physics-prefix-guided fine-tuning reduced penetration (mm) from 3.24 to 0.00 and skating (non-contact slippage) from 8.2% to 1.58%, demonstrating concrete improvements in physical feasibility.

Real-robot experiments validated these results: PhyGile successfully executed highly dynamic motions that previous methods failed to achieve, including breakdancing spins and cartwheels, with 100% success rate on 100+ challenging motion sequences compared to 23% for the next best baseline.

## Related Work
PhyGile builds on two convergent paradigms: text-driven motion generation (e.g., MLD, Closd) that produces semantically rich motions, and General Motion Tracking (GMT) that enables scalable motion imitation. Unlike previous approaches that align language with robot-controllable embeddings or directly model robot-native motion data, PhyGile couples these capabilities through physics-guided prefixes rather than decoupling generation and execution. It improves on TextOp's retargeting-free approach by closing the semantic-physical gap through physics validation during generation rather than post-hoc correction.

## Limitations
The authors acknowledge two key limitations: (1) the current implementation requires a pre-trained motion tracker to provide physics-guided prefixes, making it less applicable to systems without existing tracking capabilities; (2) the framework's performance is currently limited to the specific robot embodiment used in their experiments (a 29-DoF humanoid), requiring retraining for different morphologies. Additionally, the paper does not report computational overhead for the physics-prefix validation step, though they note it's "lightweight" during inference.

## Appendix: Worked Example
To demonstrate the physics-prefix guidance mechanism, consider generating a motion sequence for "a person performs breakdance moves, with a spin" on a 29-DoF humanoid. The diffusion model first generates a 1-second motion segment (25 frames) in 262D robot skeletal space. This initial segment has an MPJPE of 15.28mm, exceeding the 10mm threshold for feasibility. The motion tracker validates the segment and detects instability in the spine and shoulder joints, common failure modes for spinning motions.

The tracker then provides a physics-guided prefix: a 0.75-second executable segment (19 frames) with MPJPE of 3.24mm. This prefix is concatenated with the desired terminal constraint (a standing pose) and used as conditioning for the diffusion model. The denoising process anchors around this validated prefix, steering sampling toward dynamically consistent regions. The model generates a new 1-second continuation, which the simulator refines for dynamic feasibility. This receding-horizon procedure extends the prefix by 1-second increments until reaching the target motion duration, resulting in a final motion with MPJPE of 0.00mm and 1.58% skating (compared to 8.2% before fine-tuning).

## References

- Jiacheng Bao, Haoran Yang, Yucheng Xin, Junhong Liu, Yuecheng Xu, Han Liang, Pengfei Han, Xiaoguang Ma, Dong Wang, Bin Zhao, "PhyGile: Physics-Prefix Guided Motion Generation for Agile General Humanoid Motion Tracking", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19305

Tags: #robotics #motion-generation #physics-informed #diffusion-models #humanoid-robots
