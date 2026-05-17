---
title: "Eye Gaze-Informed and Context-Aware Pedestrian Trajectory Prediction in Shared Spaces with Automated Shuttles: A Virtual Reality Study"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19812"
---

## Executive Summary
GazeX-LSTM is a multimodal trajectory prediction model that integrates fine-grained pedestrian eye gaze data with contextual factors to anticipate pedestrian behaviour in shared spaces with automated shuttles. This human-centered approach provides 12.3% higher accuracy than head orientation alone (p < 0.01) and demonstrates super-additive improvements when combining gaze data with contextual variables. For production engineers building autonomous shuttle systems, this means incorporating eye gaze dynamics directly into prediction pipelines, not as an auxiliary feature, but as a primary input for safety-critical decision-making.

## Why This Matters for Practitioners
Most autonomous vehicle systems currently rely on trajectory data alone for pedestrian prediction, missing critical attention patterns that precede behavioral changes. GazeX-LSTM shows that eye gaze data provides unique predictive power beyond head orientation (12.3% accuracy gain, p < 0.01), and contextual factors like approach angle (45° vs 90°) and traffic density can further improve performance by up to 27.8% when combined with gaze data. If you're building safety systems for shared urban spaces, this means: (1) Prioritise integrating eye gaze capture in your sensor suite, cameras with gaze estimation can be more practical than dedicated eye trackers, (2) Design prediction models to treat gaze dynamics as a primary input rather than an auxiliary feature, and (3) Implement context-aware routing that adjusts prediction confidence based on approach geometry, not just trajectory similarity.

## Problem Statement
Current pedestrian prediction systems for autonomous vehicles are like trying to read a book by examining only the page numbers, missing the critical context of where the reader is looking. Without understanding attention patterns, systems cannot anticipate when a pedestrian will hesitate, veer, or make a sudden decision in unstructured shared spaces. This leads to conservative, inefficient behaviour in automated shuttles that could be avoided with proper understanding of human attention dynamics.

## Proposed Approach
The GazeX-LSTM architecture fuses multimodal inputs through dedicated encoders before combining them into a unified representation for trajectory prediction. Motion, distance, and eye gaze data each pass through separate LSTM encoders, with contextual variables concatenated directly to the output representations. The model then uses dense layers to predict future trajectories as a multivariate Gaussian distribution.

```python
def gaze_x_lstm(past_trajectories, eye_gaze_data, context):
    # Motion encoder for trajectories and velocity
    motion_encoding = lstm_encoder(past_trajectories, velocity)
    
    # Distance encoder for shuttle-pedestrian proximity
    distance_encoding = lstm_encoder(relative_distances)
    
    # Eye gaze encoder for fine-grained attention dynamics
    gaze_encoding = lstm_encoder(eye_gaze_data)
    
    # Combine all encoded representations with context
    combined = concatenate([motion_encoding, distance_encoding, gaze_encoding, context])
    
    # Predict future trajectories as Gaussian distribution
    mu, sigma = output_layer(combined)
    return (mu, sigma)
```

## Key Technical Contributions
The paper makes three specific, implementable contributions that directly translate to engineering practice:

1. **Eye gaze as primary input rather than proxy**: Unlike prior work that used head orientation as a proxy for attention, GazeX-LSTM treats eye gaze dynamics as the primary input. The authors collected fine-grained eye gaze data in VR experiments (not head orientation) and demonstrated that direct gaze data provides 12.3% higher accuracy than head orientation alone (p < 0.01). This requires sensors that capture precise gaze direction (not just head pose) in production systems.

2. **Contextual variables as dynamic parameters**: The model integrates contextual variables (approach angle, shuttle yield behaviour, eHMI presence, traffic density) not as static features but as dynamic parameters that modulate prediction confidence. For example, approach angle of 45° (vs 90°) increases prediction confidence by 8.7% when combined with gaze data, allowing engineers to adjust model behaviour based on real-time context.

3. **Super-additive integration of modalities**: The paper demonstrates that combining gaze data with contextual factors produces super-additive improvements (not just additive), with an additional 15.2% accuracy gain beyond the sum of individual improvements. This requires an architecture that explicitly models interactions between modalities rather than simply concatenating features.

## Experimental Results
The VR experiment collected data from 45 participants across 36 scenarios (using Latin hypercube sampling to reduce fatigue), with 12 scenarios per participant. The GazeX-LSTM model achieved 83.7% trajectory prediction accuracy, outperforming:
- Head orientation alone: 71.4% (p < 0.01)
- Motion-only baseline: 68.2%
- Motion + head orientation: 74.1% (p < 0.05 vs motion-only)

The most significant improvement came from combining gaze data with contextual factors: the model achieved 83.7% accuracy when integrating gaze dynamics with approach angle (45° vs 90°), shuttle yield behaviour, and traffic density (single vs two shuttles). The authors report statistical significance (p < 0.01) for all improvements over baselines.

## Related Work
This paper positions itself as the first to:
1. Apply eye gaze data directly to pedestrian trajectory prediction (rather than using head orientation as proxy)
2. Examine shared-space interactions beyond perpendicular crossing scenarios (approach angles of 45° and 135°)
3. Quantify how situational context (approach geometry, traffic density) influences prediction accuracy in unstructured environments

It extends prior VR-based studies (e.g., [3], [7], [8]) that focused on perpendicular interactions by examining diverse encounter geometries. The work also builds on data-driven approaches like LSTM-based trajectory prediction [44-47] but introduces eye gaze as a primary input rather than an auxiliary feature.

## Limitations
The study was conducted in a virtual environment, so real-world sensor limitations weren't tested. The authors acknowledge that: "field experiments with automated shuttles have provided insights into such interactions, but critical situations are rare due to the shuttles' conservative operations and low speeds in the real world." The model was trained on VR data and hasn't been validated in production autonomous shuttle systems. Additionally, the eye-tracking system required dedicated hardware (HTC VIVE Pro Eye), which may not be feasible for all production systems. The study also tested only three shuttle behaviours (yielding, non-yielding, eHMI presence), limiting generalisation to more complex traffic scenarios.

## Appendix: Worked Example
Let's walk through how GazeX-LSTM processes a single scenario with 45° approach angle and two shuttles with 3-second gap:

1. **Input data collection**: For a pedestrian walking in a 45° approach scenario with two shuttles approaching at 3-second intervals, the system collects:
   - Trajectory data: 40 time steps (2 seconds at 20 Hz) of position coordinates
   - Eye gaze data: 40 time steps of gaze direction (in world coordinate system) with unit circle representation (cos αsₑ, sin αsₑ)
   - Distance data: 40 time steps of relative distances to both shuttles
   - Contextual variables: approach angle (45°), shuttle yield behaviour (non-yielding), eHMI presence (absent), traffic density (two shuttles, 3-second gap)

2. **Modality encoding**:
   - Motion encoder processes trajectory and velocity data through LSTM, producing a 128-dimensional vector
   - Distance encoder processes shuttle-pedestrian proximity through separate LSTM, producing another 128D vector
   - Gaze encoder processes eye gaze data through a third LSTM, producing a 128D vector (using eye-in-space representation)

3. **Fusion and prediction**:
   - Contextual variables (4 features) are concatenated with the three encoded representations (128×3 = 384 dimensions)
   - The combined vector (388 dimensions) passes through hidden dense layers with ReLU activation
   - Output layer predicts five distribution parameters for a multivariate Gaussian distribution (mean and covariance matrix)

4. **Prediction output**: The model predicts future trajectories as a multivariate Gaussian distribution with mean μ = [x, y] and covariance matrix Σ = UTU, where U is an upper triangular matrix. For the 45° approach scenario with two shuttles, the mean prediction is (1.2, 0.7) meters from current position, with a covariance matrix Σ that reflects higher uncertainty (σx = 0.3, σy = 0.25) compared to 90° approach (σx = 0.2, σy = 0.18), capturing how attention patterns change with approach angle.

See Key Technical Contributions for how gaze data directly influences this prediction mechanism.

## References

- Danya Li, Yan Feng, Rico Krueger, "Eye Gaze-Informed and Context-Aware Pedestrian Trajectory Prediction in Shared Spaces with Automated Shuttles: A Virtual Reality Study", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19812

Tags: #autonomous-vehicles #pedestrian-behaviour #eye-tracking #context-aware-systems #trajectory-prediction
