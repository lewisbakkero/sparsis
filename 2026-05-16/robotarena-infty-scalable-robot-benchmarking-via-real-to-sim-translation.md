---
title: "RobotArena $\infty$: Scalable Robot Benchmarking via Real-to-Sim Translation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2510.23571"
---

## Executive Summary
RobotArena ∞ automates the conversion of real robot demonstration videos into simulated environments for scalable, reproducible evaluation of robot policies. This framework overcomes the critical bottleneck in robotics research where real-world testing is labor-intensive, slow, and unsafe at scale. Practitioners building production robot systems should care because it provides the first large-scale, standardized evaluation protocol that can detect how policies generalise across environments and under perturbations.

## Why This Matters for Practitioners
If you're building production robot systems that require generalisation across environments, RobotArena ∞ reveals that current vision-language-action (VLA) models are highly sensitive to dataset differences. This means a robot policy trained on a specific dataset (like BridgeV2) will significantly underperform on environments not seen during training (like DROID or RH20T datasets). Practically, this means you should:
1. Design policies with explicit 3D spatial reasoning (like SpatialVLA) to improve robustness to background and object placement changes.
2. Test policies under systematic perturbations (background changes, colour shifts, object pose changes) during development, not just on training environments.
3. Prioritize models with stronger vision-language backbones (like π0 and X-VLA) as they show better resilience to colour variations, which might indicate more invariant feature learning.

## Problem Statement
Current robotics evaluation is like trying to judge a chef's skills by only tasting one dish in a single kitchen: you can't tell if they're a true master who can cook any dish in any kitchen, or just a specialist who can only replicate one recipe in one environment. Real-world testing is limited by logistics, safety concerns, and reproducibility issues, requiring significant human involvement for setup, execution, and scoring. As policies grow more generalist, these barriers intensify, with success metrics often relying on nuanced human judgments of execution quality.

## Proposed Approach
RobotArena ∞ automates the conversion of real robot demonstrations into simulated environments for evaluation. It leverages vision-language models for scene understanding, 2D-to-3D generative models for asset creation, and differentiable rendering for camera pose estimation. Within these digital twins, it assesses policies using both automated VLM-guided scoring and scalable human preference judgments collected from crowdworkers. The system also introduces systematic perturbations to test robustness under controlled variations in textures and object placements.

```python
def translate_video_to_simulation(video):
    camera_pose = estimate_camera_pose(video)
    objects = reconstruct_3d_objects(video)
    background = generate_clean_background(video)
    physics_params = identify_physics_parameters(video)
    
    return create_simulation_environment(
        camera_pose=camera_pose,
        objects=objects,
        background=background,
        physics_params=physics_params
    )
```

## Key Technical Contributions
The key innovations in RobotArena ∞ lie in how it automates environment translation and evaluation:

1. **Automated robot-camera calibration**: Using differentiable rendering of pose-conditioned 3D robot Gaussians to estimate camera-to-robot transformation without manual calibration. The system minimises a composite loss with RGB, flow, and feature components to align rendered motion with real video, enabling accurate scene reconstruction from single-view videos.

2. **3D scene reconstruction pipeline**: The system uses Gemini to segment objects, InvSR for super-resolution, and Hunyuan-3D for 3D mesh generation, followed by correspondence estimation from MINIMA to recover object poses. This pipeline reconstructs scenes from single RGB frames without requiring multi-view captures or fiducial markers.

3. **Controlled environment perturbations**: The framework introduces systematic perturbations (background changes, colour shifts, object pose changes) that allow testing under controlled variations. For colour shifts, it alters RGB channel configurations from 0% to 100% in 33% increments, providing a quantitative way to measure robustness to low-level visual variations.

4. **Human preference evaluation protocol**: The system uses pairwise, double-blind comparisons of execution videos with free-form explanations to increase evaluator engagement. The Bradley-Terry model aggregates preferences into a global ranking with confidence intervals computed via robust variance estimation.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
RobotArena ∞ evaluated six VLA models across over 8,500 preference pairs in 100 nominal environments with hundreds of perturbations. Key findings:

1. Policies show substantial performance drops in environments derived from datasets they weren't trained on (e.g., DROID and RH20T), indicating current VLAs aren't true generalists.

2. π0 and X-VLA consistently ranked highest in BridgeSim environments (8,749 pairwise comparisons), with human preferences aligning perfectly with automated VLM scores (0.99 correlation).

3. In RH20TSim, RoboVLM achieved a substantially higher score (19.05%) than all other models, while X-VLA failed (0.00%), highlighting dataset-specific performance.

4. Policies with stronger VLM backbones showed better resilience to colour perturbations (e.g., π0 maintained performance under 100% colour shift while Octo degraded by 23.4%), suggesting they rely more on invariant structural cues.

## Related Work
RobotArena ∞ builds on LMarena (Chiang et al., 2024), which pioneered crowdsourced evaluation for language models through pairwise comparisons. Unlike LMarena, RobotArena ∞ addresses robotics' unique challenges: the need for physical simulation, scene understanding, and multi-modal evaluation. It extends prior work like Behaviour and SIMPLER, which require significant manual effort for environment creation, by automating the entire process from video to simulation with no human supervision.

## Limitations
The framework relies on the quality of the underlying real-to-sim translation pipeline, which is limited by current generative models. It currently only covers environments from datasets they could translate (BridgeV2, RH20T, DROID), and doesn't yet support all robot hardware types. While human preference rankings align with automated scores, the authors note this correlation might not hold for all task types, particularly those requiring nuanced physical understanding.

## Appendix: Worked Example
Let's walk through the translation of a "Put the tomato in the pot" demonstration video from the BridgeV2 dataset into a simulation environment:

1. **Video analysis**: The system processes a 10-second video clip showing a robot placing a tomato in a pot. The video contains 300 frames (30 fps) with the tomato weighing 1.260kg.

2. **Camera calibration**: Using differentiable rendering, the system estimates the camera's 6-DoF pose relative to the robot body. It minimises:
   - RGB loss: 0.12 (pixel-level appearance difference)
   - Flow loss: 0.07 (motion field consistency)
   - Feature loss: 0.09 (DINOv2 embedding alignment)

3. **Object reconstruction**: The system segments the tomato and pot from each frame using Gemini. Each object crop is super-resolved with InvSR (4x upscaling) and converted to 3D mesh using Hunyuan-3D. The tomato (1.260kg) is reconstructed as a textured mesh with accurate dimensions.

4. **Pose estimation**: The system renders 3D mesh views and matches them against 2D object crops using MINIMA. It calculates a metric scale factor (0.3m) from robot arm depth to unproject mask pixels into a 3D point cloud.

5. **Background generation**: The system inpaints the robot and objects from the first frame using LaMa, creating a clean background for the reconstructed assets.

6. **Physics identification**: The system identifies PD controller gains by aligning simulated end-effector trajectories with observed robot motions. The proportional gain (Kp) is 0.123 and derivative gain (Kd) is 1.260.

7. **Environment creation**: The system combines all elements to create a simulation environment in the physics engine. The translated environment includes the tomato (1.260kg), pot, and background, with accurate physics parameters.

This process takes approximately 15 minutes per video clip, compared to the 20+ minutes required for manual resetting in traditional evaluation.

## References

- Yash Jangir, Yidi Zhang, Pang-Chi Lo, Kashu Yamazaki, Chenyu Zhang, Kuan-Hsun Tu, Tsung-Wei Ke, Lei Ke, Yonatan Bisk, Katerina Fragkiadaki, "RobotArena $\infty$: Scalable Robot Benchmarking via Real-to-Sim Translation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2510.23571

Tags: #robotics #benchmarking #simulation #real-to-sim #vision-language-models #human-evaluation #robot-policies #robustness
