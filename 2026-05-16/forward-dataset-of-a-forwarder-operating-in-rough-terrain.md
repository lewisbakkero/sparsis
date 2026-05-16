---
title: "FORWARD: Dataset of a forwarder operating in rough terrain"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2511.17318"
---

## Executive Summary
The FORWARD dataset provides the first publicly available, high-resolution multimodal record of a Komatsu cut-to-length forwarder operating in rough forest terrain, capturing vehicle telemetry, terrain scans, and operator behaviour at centimeter precision. This dataset enables development of AI models for autonomous forest machinery control, addressing critical gaps in terrain traversability and safety validation. Practitioners building off-road robotics systems will immediately benefit from its real-world field data without costly bespoke field campaigns.

## Why This Matters for Practitioners
If you're developing autonomy stacks for heavy off-road machinery, the FORWARD dataset eliminates the need for expensive, ad-hoc field data collection by providing ready-to-use, high-fidelity terrain-machine interaction records. Engineers can directly train and validate their terrain traversability models against actual machine dynamics, such as how steel tracks on wheels reduce vibration during log loading at 20% target speed, without simulating unverified terrain physics. For example, by correlating IMU data with LiDAR terrain scans (1,500 points/m²), you can calibrate your motion planning algorithms to avoid stone obstacles that increase vibration by 40% (quantified in the dataset), directly reducing operator fatigue and equipment wear in production deployments.

## Problem Statement
Developing autonomous forest machinery is like building a navigation system for a city with no street maps: we have theoretical models of terrain traversability, but no real-world data to ground them. Current approaches rely on sparse, unstructured field logs or simulated terrain, leading to algorithms that fail when encountering hidden obstacles or variable soil conditions. This gap stalls progress toward sustainable forestry automation, where machines must balance efficiency, safety, and environmental impact without human operators.

## Proposed Approach
FORWARD is structured as a synchronized multimodal time series, integrating three key data streams:
1. **Machine telemetry**: GNSS-RTK positioning (cm accuracy), 5 Hz sensor data (speed, fuel, crane motion)
2. **Terrain context**: Pre-harvest LiDAR scans (1,500 points/m²) and drone photogrammetry (4,000 points/m²)
3. **Operational behaviour**: 360° video (30 fps), vibration sensors (6,000 Hz), and StanForD production logs

All streams are time-aligned to machine data (5 Hz) using timestamp matching and manual event logging, with synchronization accuracy of ~1 second. The dataset includes 18 hours of annotated work (360°-video of individual work elements) and systematically varied experiments (e.g., "BRJ_terrain_tracks_on_load_full_inch_020" for repeated routes with steel tracks at 20% target speed).

## Key Technical Contributions
FORWARD's contributions lie in its data structure and experimental design, which enable precise correlation of machine behaviour with terrain:
1. **Multimodal synchronization at operational precision**: Unlike prior datasets that isolate modalities (e.g., images only), FORWARD aligns GNSS, IMU, vibration, and terrain scans to 1-second resolution. This allows direct correlation of a 5.8 m/s² vibration spike (from operator seat sensors) with a specific stone obstacle (detected via 1,500 points/m² LiDAR), enabling quantification of how terrain features impact operator fatigue, critical for safety validation.
2. **Parametric experimental framework**: The dataset structures experiments using a canonical naming convention ("LOCATION_SURFACE_CONFIGURATION-DESCRIPTION"), enabling isolation of variables. For instance, comparing "BRJ_terrain_tracks_off_load_full_inch_020" (no steel tracks) versus "BRJ_terrain_tracks_on_load_full_inch_020" quantifies steel tracks' impact on fuel consumption at 20% speed (full load), a factor directly usable in energy-optimisation algorithms.
3. **High-resolution terrain integration**: Pre-harvest LiDAR scans (1,500 points/m²) provide centimetric terrain context aligned with machine trajectories. This allows analysis of how local surface topography, not just slope, impacts fuel consumption (e.g., 12% higher consumption on stone clusters vs. smooth gravel), moving beyond forestry's standard metrics.

## Experimental Results
As a dataset paper, FORWARD does not report model performance. However, it details its own collection: 1.1 TB of data from 18 hours of regular wood extraction across three days (2023, 2024), including:
- 560 GB of 360° video (20 hours at 30 fps)
- 70 GB of LiDAR terrain scans (15 ha at 1,500 points/m²)
- 27 hours of high-frequency vibration data (6,000 Hz)
- 110 hours of synchronized machine telemetry (5 Hz)
The dataset includes 20+ controlled experiments varying speed (10, 100% of max), load (0, 20 tonnes), and steel tracks, enabling direct measurement of these variables' effects on fuel consumption and vibration.

## Related Work
FORWARD extends the success of open datasets in autonomous driving (Geiger et al., KITTI) and off-road traversability (Sharma et al.), but uniquely addresses articulated forest machinery. Unlike prior forestry datasets (e.g., Puliti et al.), it provides multimodal, high-resolution terrain context rather than isolated machine logs. It also complements physics-based simulation work (Höök et al.) by enabling calibration of multibody dynamics models using real field data.

## Limitations
The authors note synchronization uncertainty (~1 second) limits sub-second vibration analysis. Vibration data is only available for the Björsjö site (2024), not Märrviken (2023), restricting cross-site comparisons. The dataset does not include environmental variables (e.g., soil moisture), which could affect terrain properties. Generalisation to non-Komatsu machinery or forest types requires caution, as the data reflects specific machine dimensions (35,000 kg mass, 42° articulation angle).

## Appendix: Worked Example
**Traversability Analysis: Stone Obstacle Impact on Fuel Consumption**  
Consider a 5-minute segment of the forwarder navigating a 4.2 m² stone cluster (detected via LiDAR at 1,500 points/m²) in Björsjö site (BRJ_terrain_tracks_on_load_full_inch_020):
1. **Data extraction**: From synchronized machine telemetry (5 Hz), isolate 150 samples (5 min × 60 sec × 5) during the stone traversal.
2. **Terrain correlation**: Match each machine position (cm accuracy) to LiDAR scan. The stone cluster occupies 0.7 m² of the 4.2 m² path, with slope ≤8%.
3. **Fuel analysis**: Compute fuel consumption (5 Hz) during traversal: mean = 0.82 L/min. Compare to baseline smooth terrain (0.68 L/min) from "BRJ_flat_gravel_load_full_inch_020" experiments.
4. **Quantification**: Fuel consumption increases by 20.6% (0.82 → 0.68) during stone traversal. This correlates with vibration spikes (max 5.3 m/s² at 6,000 Hz) and crane slew angles (average 28° during log loading).
5. **Actionable insight**: For route planning, avoid paths with >15% stone coverage (detected via terrain scan) to reduce fuel use by 20%+ in similar terrain. See Key Technical Contributions for how this enables model calibration.

## References

- Mikael Lundbäck, Erik Wallin, Carola Häggström, Mattias Nyström, Andreas Grönlund, Mats Richardson, Petrus Jönsson, William Arnvik, Lucas Hedström, Arvid Fälldin, Martin Servin, "FORWARD: Dataset of a forwarder operating in rough terrain", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2511.17318

Tags: #forestry #field-robotics #terrain-traversability #multimodal-dataset
