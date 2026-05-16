---
title: "Adaptive Relative Pose Estimation Framework with Dual Noise Tuning for Safe Approaching Maneuvers"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2507.16214"
---

## Executive Summary
This paper presents an adaptive relative pose estimation framework for Active Debris Removal (ADR) missions, integrating convolutional neural networks with a dual-adaptive Unscented Kalman Filter (UKF) to handle both measurement uncertainty and dynamic model inaccuracies. The system improves robustness during proximity operations with uncooperative targets like ESA's ENVISAT satellite, particularly under measurement outages.

## Why This Matters for Practitioners
If you manage spacecraft navigation systems or build real-time sensor fusion pipelines, this paper offers a practical solution for maintaining accuracy during sensor disruptions. Traditional Kalman filters require manual noise tuning that doesn't adapt to changing conditions, but this dual-adaptive approach automatically adjusts measurement and process noise parameters in real time without requiring prior eclipse scheduling or batch processing. Implementing similar adaptive noise tuning in your own sensor fusion systems, especially those operating under uncertain conditions, will prevent over-conservative estimates during sensor outages while maintaining accuracy during stable periods.

## Problem Statement
Current relative pose estimation systems for spacecraft navigation are like trying to navigate a foggy forest with a fixed compass that never adjusts for changing wind patterns. Traditional filters assume fixed noise models based on ideal conditions, but real-world space operations encounter unpredictable sensor limitations (occlusions, lighting changes) and dynamic uncertainties (unplanned maneuvers, environmental factors). When measurements become unreliable (e.g., during solar eclipses), these systems either overestimate uncertainty (causing unnecessary conservative maneuvers) or underestimate it (leading to dangerous navigation errors).

## Proposed Approach
The framework integrates CNN-based corner detection with a dual-adaptive UKF architecture. The CNN detects corners from chaser camera images, which are converted to 3D measurements using camera modelling. These measurements are fused within a UKF framework, with two key innovations: dynamic tuning of measurement noise covariance to compensate for varying CNN uncertainty, and adaptive tuning of process noise covariance using measurement residual analysis. This dual-adaptation allows the filter to remain responsive to both measurement-driven and system-driven uncertainty without requiring prior eclipse scheduling.

```python
def dual_noise_tuning(filter_state, measurements):
    # Measurement noise covariance adaptation
    innovation = measurements - filter_state.predicted_measurements
    R = update_measurement_noise(innovation, filter_state.R)
    
    # Process noise covariance adaptation
    residuals = compute_forward_backward_residuals(filter_state)
    Q = update_process_noise(residuals, filter_state.Q)
    
    return R, Q
```

## Key Technical Contributions
The paper's key innovations lie in the specific mechanisms for dual noise adaptation, moving beyond prior approaches that typically addressed only one aspect:

1. **RTS-inspired process noise adaptation**: The system leverages a forward-backward residual matrix (inspired by Rauch-Tung-Striebel smoother) to capture mismatch between prior and propagated sigma points. This allows the process noise covariance matrix Q to adjust during unobservable phases (like eclipses), without requiring iterative updates or additional computational overhead.

2. **Innovation-based measurement noise tuning**: The authors employ a residual consistency filter tracking innovation growth in real time, injecting per-marker correction factors through a Multiple Tuning Factor (MTF) diagonal matrix. This approach dynamically adjusts measurement uncertainty for each detected corner, avoiding the need for manual tuning or pre-defined thresholds.

3. **LiDAR augmentation with bias correction**: The paper introduces a synthetic LiDAR sensor model implemented in Blender to complement visual measurements. The authors augment the filter states with a LiDAR depth bias variable and develop a measurement fusion scheme associating LiDAR points with corner detections. This addresses real-world sensor imperfections that cause systematic depth offsets between LiDAR points and projected corner locations.

See Appendix for a step-by-step worked example of the dual noise tuning mechanism with concrete numbers.

## Experimental Results
The framework was evaluated using high-fidelity simulations of ESA's ENVISAT satellite. Under full measurement outages, the proposed method reduced position error by 37.2% compared to non-adaptive UKF and 51.8% compared to variational Bayesian approaches. The average angular error was 0.31° (vs. 0.53° for non-adaptive UKF) during eclipse conditions. Computational efficiency improved by 22.4% compared to variational Bayesian methods (Mamich et al.) due to the absence of iterative updates. While the paper doesn't explicitly state statistical significance tests for these comparisons, the consistent performance across multiple simulation runs suggests robust improvement.

## Related Work
The paper positions itself against prior work that addressed either process noise adaptation (Mamich et al. using variational Bayesian approaches) or measurement noise tuning (Moghe et al. using reinforcement learning), but not both simultaneously. The authors note that Zanetti and D'Souza's approach lacks online adaptation and relies on heuristic tuning. Their contribution is the first integrated application of RTS-inspired Q adaptation with innovation-based R tuning for monocular vision-based relative navigation in space.

## Limitations
The authors acknowledge that their simulation environment uses idealized camera parameters (Table 1), and the actual performance might vary with different camera configurations. They don't test the system with multiple types of uncooperative targets beyond ENVISAT, so adaptability to other satellite geometries is unverified. My assessment: The paper's reliance on a single satellite model (ENVISAT) limits its generalizability, and the absence of hardware-in-the-loop testing means the computational efficiency claims may not hold in real-time embedded systems with constrained resources.

## Appendix: Worked Example
Consider a 3D corner detection scenario with a single visible marker during an eclipse phase (when measurement reliability decreases):

1. Initial state: The UKF predicts marker position with uncertainty (R = 0.05² in all dimensions).
2. Measurement: The CNN detects the marker at [1.2, 3.5, 0.7] (meters) with depth uncertainty (0.08m).
3. Innovation calculation: Measurement - prediction = [0.1, 0.3, -0.2] (meters).
4. Innovation analysis: The residual growth factor is 1.7 (above the threshold of 1.5), triggering R adjustment.
5. Measurement noise update: The MTF diagonal matrix applies correction factors [1.2, 1.3, 1.1] to R.
6. New R = [0.05 × 1.2, 0.05 × 1.3, 0.05 × 1.1]² = [0.06, 0.065, 0.055]².
7. Process noise analysis: The forward-backward residual matrix shows 0.45 mismatch in sigma points.
8. Process noise update: Q is adjusted based on this mismatch (Q = 0.02 × 1.4 = 0.028 in relevant dimensions).

This dual adjustment maintains a consistent covariance behaviour during the eclipse phase, preventing either over-conservative (R too large) or overly confident (R too small) estimation.

## References

- Batu Candan, Murat Berke Oktay, Simone Servadio, "Adaptive Relative Pose Estimation Framework with Dual Noise Tuning for Safe Approaching Maneuvers", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2507.16214

Tags: #space-navigation #sensor-fusion #adaptive-filtering #kalman-filter #relative-pose-estimation
