---
title: "The Robot's Inner Critic: Self-Refinement of Social Behaviors through VLM-based Replanning"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20164"
---

## Executive Summary
CRISP (Critique-and-Replan for Interactive Social Presence) is an autonomous framework where robots critique and refine their own social behaviours using a Vision-Language Model (VLM) as a 'human-like social critic'. It moves beyond rigid, rule-based motions or human feedback by generating subtle, human-like interactions directly from a robot's structural description file (e.g., MJCF), without requiring task-specific training data for each robot platform.

## Why This Matters for Practitioners
If you're building social robots for real-world applications like customer service, healthcare companions, or home assistants, this paper provides a direct path to reduce human intervention in behaviour generation while significantly increasing engagement. Instead of maintaining separate motion libraries for each robot model (which becomes unmanageable as your fleet grows), you can now use a single framework that adapts to any robot's physical structure. For example, your Stretch 3 mobile manipulator can generate natural greetings that subtly differ from those of your Unitree G1 humanoid, all without retraining or reprogramming each model. This means your engineering team can focus on designing robust interaction logic rather than managing dozens of robot-specific motion libraries.

## Problem Statement
Today's social robots often behave like broken vinyl records - they repeat the same predictable motions, making interactions feel mechanical. Imagine a robot that always waves with the exact same wrist angle, speed, and timing - it's as unnatural as a robot that says 'Hello' with the same intonation to every person. Current approaches either require human feedback for every adjustment (like asking a customer to fix a robot's air conditioning system every time it's not at the perfect temperature) or rely on rigid templates that can't adapt to new situations. This predictability destroys long-term user engagement, as humans instinctively seek subtle variations in social cues, not robotic precision.

## Proposed Approach
CRISP is a five-component framework that enables autonomous behaviour generation and refinement:
1. **Robot Structure Analyzer**: Analyzes robot structure files (MJCF) to extract joint information and constraints
2. **Robot Behaviour Generator**: Creates step-by-step behaviour plans considering the robot's physical capabilities
3. **Joint Code Generator**: Converts behaviour plans into executable joint control code using visual guidance
4. **Motion Evaluator**: Uses a VLM to assess social appropriateness and identify specific problematic steps
5. **Behaviour Refiner**: Iteratively refines behaviours through reward-based search

```python
def refine_behavior(initial_behavior, robot_structure, max_iterations=5):
    candidates = generate_initial_candidates(initial_behavior, robot_structure)
    
    for iteration in range(max_iterations):
        rewards = [evaluate_candidate(c, robot_structure) for c in candidates]
        best_idx = np.argmax(rewards)
        best_reward = rewards[best_idx]
        
        if best_reward >= 8:
            return candidates[best_idx]
        
        if best_reward >= 5:
            search_width = 0.4 * base_search_width
        else:
            search_width = 1.5 * base_search_width
            
        candidates = generate_new_candidates(candidates[best_idx], search_width, robot_structure)
    
    return candidates[np.argmax(rewards)]
```

## Key Technical Contributions
The paper makes three critical contributions that advance the field beyond prior approaches:

1. **Autonomous Behaviour Evaluation and Refinement Using VLM**: Unlike previous methods that required human feedback for corrections, CRISP creates a self-contained "generate-evaluate-regenerate" cycle where the VLM functions as a 'human-like social critic'. The key innovation is the VLM's ability to pinpoint specific erroneous steps (e.g., "The wrist needs to be waved for a natural greeting") rather than providing generic scores. This precision prevents cascading errors and enables efficient refinement.

2. **Achieving Flexibility through Low-level Control**: While prior work used high-level action primitives (like "wave" or "nod"), CRISP generates low-level joint control directly from the robot's structure file. The system creates a visual joint motion dictionary that maps joint positions to spatial movements, enabling the VLM to generate appropriate joint values without extensive training data. This allows subtle variations in behaviour (e.g., different wrist angles for waves) that make interactions appear more human-like.

3. **Robot-agnostic Framework with Cross-platform Applicability**: The system works with only the robot's structural description file (e.g., MJCF), eliminating the need for robot-specific implementations. The framework was validated across five different robot types (mobile manipulators, humanoids) in 20 scenarios, demonstrating significant improvements in situational appropriateness and likeability over baselines without requiring retraining for each platform.

## Experimental Results
The user study compared CRISP (full system with replanning), CRISP without replanning (CRISP w/o replan), and GenEM (baseline) across five robot types and 20 interaction scenarios. Results were measured on three metrics using a 7-point Likert scale:

- **Situational Appropriateness (Q2)**: CRISP achieved 5.8/7 for Unitree G1 (humanoid) compared to 4.2/7 for GenEM and 4.7/7 for CRISP w/o replan
- **Likeability (Q3)**: CRISP scored 5.6/7 for TIAGo (dual-arm manipulator) versus 4.1/7 for GenEM and 4.5/7 for CRISP w/o replan
- **Overall Preference**: CRISP significantly outperformed both baselines across all metrics (p < 0.05)

The ablation study confirmed that the replanning loop was the most significant component, with CRISP w/o replan performing significantly worse than the full CRISP system. The system reduced the need for human feedback by enabling autonomous refinement cycles.

## Related Work
The paper positions itself against two main lines of prior work:
1. **Rule-based and template-based approaches** (e.g., [3], [4]) that struggle with adaptation to new situations
2. **LLM-based approaches** (e.g., [5], [6]) that use predefined action APIs but lack flexibility and require human feedback for corrections

CRISP advances the field by introducing a VLM-based self-refinement cycle that eliminates the need for human feedback while enabling low-level control from robot structure files. Unlike SAMALM [20] that produced low-level signals for navigation, CRISP extends LLM-based methods to generate situationally appropriate social behaviours across multiple platforms.

## Limitations
The authors acknowledge several limitations:
- The framework requires a robot description file (MJCF), which may not be available for all robots
- The system was evaluated in simulation, not on physical robots
- The VLM (GPT-4o) is computationally expensive for continuous operation on resource-constrained robots

My assessment: The MJCF requirement is a practical constraint but likely solvable for most modern robot platforms. Simulation-to-real transfer is a common challenge in robotics, and the cost of GPT-4o can be addressed with smaller, fine-tuned models for production deployment.

## Appendix: Worked Example
Let's walk through CRISP's process for generating a natural greeting wave for a humanoid robot:

1. **Scenario**: A person waves their hand to greet you.
2. **Robot Structure Analyzer**:
   - MJCF file analysis identifies right_shoulder_pitch (ctrl[22]), right_elbow_joint (ctrl[25])
   - Joint limits for right_elbow_joint: [0.0, 1.31] radians
   - Visual motion dictionary created showing full-body and zoomed views for positive/negative joint values

3. **Robot Behaviour Generator**:
   - Creates plan: "Step 1: Rotate shoulder to face person (0-1s), Step 2: Bend elbow to raise hand (1-2s), Step 3: Wrist motion for natural wave (2-3s)"
   - Each step has estimated timing

4. **Joint Code Generator**:
   - For Step 3 (wrist motion), VLM selects right_elbow_joint (ctrl[25])
   - Visual dictionary shows positive sample at 0.5 radians (moderate tilt)
   - Generates initial value: 0.4 radians (within joint limits)

5. **Motion Evaluator**:
   - Simulates motion, creates visual log
   - VLM critique: "The wrist needs to be waved for a natural greeting" (score: 4)
   - Pinpoints Step 3 as problematic

6. **Behaviour Refiner**:
   - Initial candidates: [0.3, 0.4, 0.5]
   - Evaluation scores: [3, 4, 5] (0.5 gets best score)
   - Search width adjusted: 0.4 * base_search_width (since score 5 ≥ 5)
   - New candidates: [0.45, 0.48, 0.52]
   - Evaluation scores: [4, 5, 6] (0.52 gets best score)
   - Iteration continues until reward ≥ 8 (reaching score 8.2 on iteration 3)

See Appendix for a step-by-step worked example with concrete numbers.

## References

- Jiyu Lim, Youngwoo Yoon, Kwanghyun Park, "The Robot's Inner Critic: Self-Refinement of Social Behaviors through VLM-based Replanning", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20164

Tags: #robotics #social-robots #human-robot-interaction #vision-language-models #self-refinement #low-level-control
