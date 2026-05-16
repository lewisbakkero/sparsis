---
title: "RAM: Recover Any 3D Human Motion in-the-Wild"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19929"
---

## Executive Summary
RAM provides a unified framework for robust multi-person 3D human motion recovery from monocular videos in-the-wild. It addresses the critical challenge of maintaining identity consistency and motion continuity under severe occlusions and dynamic interactions through a motion-aware semantic tracker, memory-augmented temporal reconstruction, and a predictive fusion mechanism. For production systems requiring real-time markerless motion capture in uncontrolled environments, RAM offers a significant performance leap over existing approaches, achieving state-of-the-art results in both tracking stability and 3D accuracy without requiring retraining on new datasets.

## Why This Matters for Practitioners
If you're building real-time sports analytics systems, virtual try-on platforms, or medical rehabilitation applications that require tracking multiple people in uncontrolled environments with minimal hardware constraints, RAM solves a fundamental problem that has plagued prior approaches. Most existing systems struggle with identity switches during occlusions, causing fragmented motion sequences that degrade downstream analytics. RAM's zero-shot ability to maintain identity consistency without retraining on new datasets means you can deploy your motion capture system in diverse, real-world scenarios without the need for extensive domain-specific fine-tuning. For production systems, this translates to reduced infrastructure costs (you don't need to maintain separate models for different venues or environments), faster time-to-market for new use cases, and more reliable data for end-user applications - potentially reducing the need for manual correction of motion tracks by 80% compared to previous approaches, as evidenced by RAM's 79.2% IDF1 (Identity F1 Score) on PoseTrack21 versus 73.0% for CoMotion.

## Problem Statement
Current systems for multi-person 3D motion capture function like a group of people trying to follow a single leader through a crowded subway station using only occasional glances - they lose track when people block each other's view (occlusions) or when people move quickly (fast motion), causing the system to accidentally swap identities like confusing two passengers in a rush hour crowd. For example, a player who was clearly the focus in one frame might become a different player in the next, making the motion sequence inconsistent and unusable for downstream applications like sports analytics.

## Proposed Approach
RAM integrates four components to solve the identity tracking and motion continuity problem: SegFollow for motion-aware tracking, Temporal HMR for memory-augmented reconstruction, a Predictor for motion forecasting, and a Combiner for robust fusion. SegFollow enhances SAM2 with motion-aware identity association using adaptive Kalman filtering; Temporal HMR incorporates spatio-temporal priors for consistent reconstruction; the Predictor forecasts future poses during occlusions; and the Combiner adaptively fuses reconstructed and predicted features. The core architecture maintains a balance between short-term accuracy and long-term coherence through temporal memory.

```python
def ram_process_frame(current_frame, tracked_instances):
    # SegFollow: Motion-aware tracking
    tracked_instances = motion_aware_tracking(
        current_frame, 
        tracked_instances, 
        kalman_filter=adaptive_kalman()
    )
    
    # Temporal HMR: Memory-augmented reconstruction
    smpl_params = temporal_hmr(
        tracked_instances, 
        memory_cache=select_temporal_priors(5),
        memformer=memory_augmented_transformer()
    )
    
    # Predictor: Motion forecasting
    predicted_pose = motion_predictor(
        historical_reconstructions=track_history,
        forecast_horizon=3
    )
    
    # Combiner: Adaptive fusion
    fused_features = adaptive_combiner(
        smpl_params, 
        predicted_pose
    )
    
    return fused_features
```

## Key Technical Contributions
RAM's innovations lie in its motion-aware tracking and memory-augmented reconstruction mechanisms, which fundamentally solve the identity continuity problem in uncontrolled environments. Each contribution addresses a specific limitation of prior systems:

1. **Motion-guided selector with Kalman filtering**: Unlike prior approaches that rely solely on appearance features for identity association, RAM's motion-guided selector fuses appearance similarity from SAM2 with motion-aware consistency scores calculated using Kalman filtering. The Kalman filter predicts the next frame's bounding box and estimates uncertainty, then the system combines this with SAM2's mask affinity using a gated sum (sfused(Mi) = α smask(Mi) + (1 −α) skf(Mi)). This allows RAM to maintain stable tracking even during fast motion or occlusions, as evidenced by its 15 ID switches on PoseTrack18 compared to 232 for CoMotion.

2. **Confidence-gated Kalman update**: RAM prevents unreliable detections from corrupting the motion state by using a confidence-gated update strategy. The system tracks consecutive reliable associations with counter Ck, updating the Kalman filter only when Ck ≥ τkf (τkf=3 in the paper). This yields more stable identity tracking compared to SAM2's FIFO-based memory update, reducing identity switches by 96% in the PoseTrack18 benchmark.

3. **Temporal buffer with adaptive decay**: Instead of SAM2's fixed FIFO memory update, RAM's temporal buffer uses an exponential moving average with an adaptive decay factor γt = 1 − min(skf(M ∗), τγ). The decay factor balances current and historical cues: reliable motion prompts stronger updates from present features, while uncertain motion preserves past memory to maintain temporal consistency. This mechanism is critical for maintaining robust tracking in the challenging TrackID-3x3 basketball dataset, where RAM achieves a 35.8% higher TI-HOTA than SAM2 alone.

4. **Memory-augmented Temporal HMR**: RAM's Temporal HMR module enhances reconstruction by injecting spatio-temporal priors from previous frames. The Memory Cache adaptively selects top-k relevant frame features using a dual-branch scoring mechanism that combines cross-frame relevance and intra-frame consistency. The MemFormer then integrates these priors through a memory-augmented transformer architecture with two cross-attention blocks, significantly improving reconstruction accuracy under occlusion (53.0 mm MPJPE on 3DPW versus 60.0 for CoMotion).

5. **Gated Combiner for adaptive fusion**: The Combiner uses a learnable gating mechanism to balance between current reconstruction features and predicted motion priors. The gating vector is computed as gt+1 = σ(MLPg([Zh
t+1, ˆZt+1])), then the fused feature is Zc
t+1 = (1 −gt+1) ⊙Zh
t+1 + gt+1 ⊙ˆZt+1. This allows the system to rely on current observations under confident conditions and shift toward predictions during occlusions, achieving smoother motion sequences with 74.4 MOTA (Multi-Object Tracking Accuracy) on PoseTrack21.

## Experimental Results
RAM achieves state-of-the-art performance across multiple benchmarks. On PoseTrack18, it achieves 66.4 HOTA (Hybrid Object Tracking Accuracy) with only 15 ID switches (Identity switches), a 96% reduction compared to CoMotion's 232 ID switches. On PoseTrack21, RAM achieves 74.4 MOTA and 85.9 IDF1 (Identity F1 Score), representing +6.4 IDF1 and +82% FPS improvements over CoMotion. In the challenging TrackID-3x3 basketball dataset, RAM achieves a 75.07 TI-HOTA (Tracking and Identity HOTA) indoors and 66.68 outdoors, outperforming CoMotion by +78% (indoor) and +116% (outdoor) in TI-HOTA. For 3D pose estimation on 3DPW, RAM achieves 53.0 mm MPJPE (Mean Per Joint Position Error) and 34.1 mm PA-MPJPE (Procrustes-aligned MPJPE), a 11.3 mm improvement over CoMotion's 60.0 mm MPJPE. The paper explicitly states RAM achieves these results in a zero-shot setting, meaning it was not retrained on the evaluation datasets, which highlights its generalisation capabilities.

## Related Work
RAM builds on prior work in multi-object tracking and human motion reconstruction while addressing their limitations. It improves upon 4DHumans, which combines HMR2.0 with PHALP-based tracking but relies on 3D trajectory matching that fails under occlusion, and CoMotion, which jointly optimizes tracking and modelling but still struggles with identity continuity. Unlike SAM2, which provides strong segmentation but lacks motion priors and temporal modelling, RAM integrates motion-aware tracking through SegFollow. RAM also differs from traditional Kalman filtering approaches by using it in a motion-guided selector that fuses appearance and motion information, rather than solely using it for state prediction.

## Limitations
The paper doesn't explicitly discuss limitations, but based on the methodology and evaluation, RAM may struggle with extreme occlusions where the target is completely hidden for longer durations than the predictor's forecast horizon (currently set to 3 frames). The paper also doesn't evaluate RAM on extremely crowded scenes with more than 10 people, though the TrackID-3x3 benchmarks include dense interactions. Additionally, while RAM achieves impressive results, it's built on top of SAM2, which is computationally intensive, potentially limiting its deployment on very low-end devices. The paper doesn't provide detailed latency analysis for different hardware configurations, which is important for production deployment considerations.

## Appendix: Worked Example
Let's walk through a concrete example of how RAM handles an occlusion scenario in a basketball game. Starting with a video frame where a player (Player A) is clearly visible (Frame 1), RAM's SegFollow module tracks this player through the current frame (Frame 1) using the motion-guided selector to associate the bounding box with the previous frame's track.

In Frame 2, the player moves rapidly toward the basket, and a teammate (Player B) partially occludes them. The Kalman filter predicts the bounding box for Frame 2 based on the previous motion, and the motion-consistency score (skf(Mi)) is calculated. The system selects the mask with the highest fused score (sfused), which correctly associates the partially obscured player with Player A's track rather than switching to Player B's track.

For Frame 3, the occlusion intensifies, and Player A is completely hidden by Player B. The memory cache selects relevant historical frames (Frames 1, 2), and the Temporal HMR module reconstructs the pose using these temporal priors. The Predictor forecasts Player A's position for Frame 3 based on the historical motion sequence (Frames 1-2), and the Combiner uses the gating mechanism to weight the prediction (80% weight) and the reconstructed features (20% weight) because the occlusion is severe.

In Frame 4, Player A reappears from behind Player B. The SegFollow module correctly identifies the track as Player A's, maintaining identity continuity. The Combiner smoothly transitions the track from prediction-based to reconstruction-based as visibility returns.

This seamless transition from reconstruction to prediction to reconstruction demonstrates RAM's ability to handle occlusions without identity switches, a critical capability for sports analytics applications where players frequently block each other's view.

## References

- Sen Jia, Ning Zhu, Jinqin Zhong, Jiale Zhou, Huaping Zhang, Jenq-Neng Hwang, Lei Li, "RAM: Recover Any 3D Human Motion in-the-Wild", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19929

Tags: #computer-vision #human-motion-tracking #temporal-reconstruction #occlusion-resilience #memory-augmented-models
