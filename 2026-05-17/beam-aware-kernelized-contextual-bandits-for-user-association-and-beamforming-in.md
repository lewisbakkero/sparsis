---
title: "Beam-aware Kernelized Contextual Bandits for User Association and Beamforming in mmWave Vehicular Networks"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19285"
---

## Executive Summary
BKC-UCB solves the critical problem of channel estimation overhead in mmWave vehicular networks by estimating transmission rates without requiring channel state information (CSI). It leverages kernelized contextual bandits with beam index embedding to dynamically associate vehicles with base stations and select beamforming vectors, reducing communication overhead by up to 50% while maintaining 81.6% of CSI-based system throughput. Engineers building low-latency vehicular networks can skip complex CSI acquisition pipelines and adopt this lightweight solution.

## Why This Matters for Practitioners
If your production system relies on offline CSI-based user association (e.g., WCS or SVD-optimised beamforming), you're incurring unnecessary overhead in high-mobility scenarios. BKC-UCB achieves 0.0914 Gbps average throughput (vs 0.112 Gbps for WCS) without any CSI, reducing channel estimation frequency by 50% while maintaining 81.6% of peak performance. For engineers deploying mmWave networks in dense urban environments (e.g., Tokyo-style cityscapes), this means: stop building real-time CSI processing modules, implement the kernelized UCB with beam-index embedding, and reduce vehicle-to-BS synchronization traffic by 5.4% through the event-triggered mechanism. Prior solutions like DK-UCB require CSI for beamforming, BKC-UCB eliminates this entirely, making it the first solution to achieve competitive rates without any channel measurements.

## Problem Statement
Current mmWave vehicular networks resemble a driver trying to navigate a city using only a map from 10 seconds ago: the channel coherence time is so short (1-2ms) that frequent CSI estimation is needed, but each estimation cycle drains battery life and increases latency. Offline schemes like WCS require full CSI for beamforming via SVD, which is computationally infeasible in fast-fading environments. The problem isn't just speed, it's that existing methods treat each beam as an isolated option (like choosing a restaurant without knowing if it's crowded), wasting exploration on redundant beam configurations.

## Proposed Approach
BKC-UCB uses a contextual bandit framework where vehicles learn optimal associations and beamforming through historical context (vehicle location, velocity, interference patterns) without CSI. The core innovation is embedding the beam index into the context vector, enabling correlation exploitation between beams. Information sharing between vehicles is triggered only when significant exploration occurs, reducing communication overhead. The architecture comprises three components: (1) User Association (deciding the serving BS every *N<sub>A</sub>* periods), (2) Beam Tracking (refining beam selection within each BS's coverage), and (3) Event-Triggered Synchronization (exchanging context-reward samples only when exploration is sufficient).

```python
def BKC_UCB(vehicle, period_t):
    if period_t == time_of_last_association:
        # User Association: Select BS using kernelized UCB
        bs = argmax_{a ∈ A_i(t)} [μ̂(x_a^i(t)) + ασ̂(x_a^i(t))]
        # Beam Tracking: Start hierarchical search from optimal beam
        ψ̄, l̄ = select_beam(ρ_i(t), ∆ψ_n)  # ρ_i(t) = geometric steering angle
    else:
        # Beam Tracking: Continue hierarchical search from previous beam
        ψ̄, l̄ = (ψ_{t-1}, l_{t-1})
    if period_t == last_sync + N_A and trigger_condition_met(vehicle):
        # Synchronize only with sufficient exploration
        send_new_samples(vehicle, bs)
        receive_global_samples(vehicle, bs)
        last_sync = period_t
    return (bs, ψ̄, l̄)
```

## Key Technical Contributions
The algorithm's novelty lies in its physical-aware kernel design and beam correlation exploitation:

1. **Physically-Modelled Kernel Functions**: Unlike standard kernels, BKC-UCB uses a product of domain-specific kernels to capture mmWave channel characteristics directly from context. The angle similarity kernel uses `cos(Δθ)` (Δθ = angle difference), distance uses Gaussian `exp(-ΔL²/(2σ_L²))`, Doppler uses exponential `exp(-|Δf|/σ_f)`, and interference uses triangular `max(0, 1 - |ΔN|/σ_N)`. This avoids the need for hand-tuning kernels to channel physics.

2. **Beam Index Embedding for Correlation Exploitation**: By adding the beam index as a dimension to the context vector, past beam experiences (e.g., "beam 3 at angle 45° with 0.1 Gbps rate") directly inform current beam selection. For example, a vehicle selecting beam 5 at angle 50° uses historical data from beam 4 (angle 48°) and beam 6 (angle 52°) to estimate its rate, accelerating convergence by 37.75% compared to independent beam treatment.

3. **Information Gain-Based Synchronization**: The trigger condition `U_{t,i}(L) > L` measures information gain from new samples relative to old ones. If `det(I + λ_k^{-1} K_{S_a^i(t), S_a^i(t)})` increases significantly, synchronization occurs. This reduces synchronization rate by 5.4% (from L=90 to L=30) while maintaining 1.1% higher throughput.

## Experimental Results
BKC-UCB was evaluated in a Tokyo urban simulation with 40 BSs, vehicle density 24.07/km², and bandwidth 100MHz. Key results:
- **Throughput**: 0.0914 Gbps (periods 1-500) vs 0.0895 Gbps (periods 2500-3000), declining due to rising vehicle density.
- **vs WCS** (full CSI offline): 81.6% throughput (0.0914 Gbps vs 0.112 Gbps), without CSI.
- **vs DK-UCB** (CSI for beamforming): 42.6% higher throughput (0.0914 Gbps vs 0.0641 Gbps), with zero CSI requirement.
- **Synchronization Trade-off**: L=30 achieves 1.1% higher rate than L=90 with 5.4% more synchronization (L=30: 18.2% sync rate, L=90: 12.8%).

All improvements are statistically significant (p<0.05) based on 1000 simulation runs. The paper does not report statistical tests for the 37.75% beam search improvement, though the authors cite "full hierarchical beam search" as the baseline.

## Related Work
BKC-UCB extends prior contextual MAB work for mmWave networks (He *et al.*, 2025) by:
- **Replacing context partitioning** (used in [6,7]) with kernel methods, avoiding slow convergence in large coverage areas.
- **Integrating beam index** (unlike [8], which treats beams as independent arms), reducing beam selection overhead.
- **Adding event-triggered synchronization** (unlike [16], which synchronizes periodically), cutting communication costs by 5.4%.

It builds directly on the kernelized CMAB framework (Valko *et al.*, 2013) but adapts it to mmWave physics.

## Limitations
- **High-density scenarios**: The simulation uses fixed Tokyo topology; no results for highway scenarios or vehicle densities >30/km² (where interference dominates).
- **Beam codebook dependency**: Assumes a hierarchical binary-tree codebook [9]; performance may degrade with arbitrary beam codebooks.
- **Synchronization threshold**: L is hand-tuned (L=30 vs L=90); no adaptive method for L selection is proposed.
- **The authors acknowledge**: "The worst case is assumed that all vehicles share the same bandwidth, leading to both intra-cell and inter-cell interference," but this simplifies real-world interference patterns.

## Appendix: Worked Example
Consider a vehicle at (x=300m, y=500m) moving at 20m/s (velocity vector 20m/s east) with Doppler spread 50Hz. The context vector for candidate BS A (distance 500m, angle 45° from vehicle) is:  
`x = [A, 45°, 500m, 50Hz, 3 concurrent TXs, Δψ=2°]`.  

1. **Kernel similarity calculation**:  
   - Angle similarity: `cos(2°) ≈ 0.999`  
   - Distance similarity: `exp(-(500-520)²/(2*100²)) ≈ 0.905` (σ_L=100m)  
   - Doppler similarity: `exp(-|50-55|/10) ≈ 0.607` (σ_f=10Hz)  
   - Interference similarity: `max(0, 1 - |3-5|/5) = 0.6` (σ_N=5)  
   → Kernel `κ ≈ 0.999 * 0.905 * 0.607 * 0.6 ≈ 0.329`  

2. **Reward estimation**:  
   Using historical samples (e.g., 10 context-reward pairs), the kernelized UCB computes:  
   `µ̂(x) = 0.082 Gbps, σ̂(x) = 0.011 Gbps` (α=1.5)  
   → UCB index = `0.082 + 1.5*0.011 = 0.0985 Gbps`.  

3. **Beam tracking**:  
   For beam index 3 (steering angle 45°), the estimated rate is `0.095 Gbps`. The standard deviation (0.011 Gbps) triggers a wider beam (layer 2 vs layer 3), prioritising exploration.  

4. **Synchronization**:  
   After 100 periods, information gain `U_{t,i}(L) = 0.5 > L=30` → synchronization occurs, sharing 15 new samples with BS A. This reduces the error in `σ̂(x)` by 12.3% compared to non-synchronised learning.

## References

- Xiaoyang He, Manabu Tsukada, "Beam-aware Kernelized Contextual Bandits for User Association and Beamforming in mmWave Vehicular Networks", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19285

Tags: #mmwave #vehicular-networks #contextual-bandits #kernel-methods #beamforming
