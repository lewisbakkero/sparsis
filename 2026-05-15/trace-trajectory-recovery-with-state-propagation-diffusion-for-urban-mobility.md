---
title: "TRACE: Trajectory Recovery with State Propagation Diffusion for Urban Mobility"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19474"
---

## Executive Summary
TRACE introduces a novel diffusion model for reconstructing dense, continuous GPS trajectories from sparse inputs, addressing a critical bottleneck in location-based services. By embedding memory into the denoising process through State Propagation Diffusion (SPDM), it achieves a 26.65% accuracy improvement over prior methods without significant inference overhead.

## Why This Matters for Practitioners
If you're building location-based services that rely on GPS data, such as ride-hailing, delivery routing, or traffic management, this paper directly impacts your data quality pipeline. The current industry practice of using simple interpolation (like linear or cubic splines) for sparse trajectory correction often introduces temporal jitter and geometric inaccuracies that degrade downstream task performance. TRACE's SPDM mechanism allows you to replace these heuristic methods with a data-driven approach that specifically handles the irregular spatio-temporal patterns of real-world trajectories. For your production systems, this means: (1) reduced error in route planning (up to 26.65% lower MSE), (2) more accurate traffic prediction models, and (3) fairer service distribution across urban areas where infrastructure coverage varies. Implementing TRACE as a pre-processing step would require minimal integration effort with your existing trajectory processing pipelines since it accepts standard GPS coordinates and timestamps as input.

## Problem Statement
Imagine a city's navigation app trying to reconstruct a delivery driver's path from only 20% of GPS points, the ones that happened to be captured when the device wasn't in a tunnel or when battery was being conserved. The resulting path might show the driver suddenly teleporting between buildings or taking impossible detours around a river that wasn't actually there. This isn't just inconvenient, when these trajectory errors compound in a delivery routing system, they can lead to inconsistent service times, wasted fuel, and frustrated customers in areas with poor GPS coverage. Current methods either rely on perfect road network data (which isn't always available) or accumulate errors across long gaps (like RNNs doing sequential prediction), making them unsuitable for the irregular patterns of real-world mobility.

## Proposed Approach
TRACE transforms trajectory recovery into a memory-augmented denoising problem. It first aggregates heterogeneous inputs (sparse observations, query timestamps, contextual metadata) into a unified feature representation. This representation is then processed through a modified diffusion pipeline (SPDM) that carries a multi-scale hidden state across denoising steps. The key insight is that by remembering what the model inferred earlier, it can focus computational resources on the most ambiguous trajectory segments rather than reprocessing stable regions.

```python
def TRACE_recover(sparse_traj, query_timestamps, context):
    # Step 1: Condition aggregation
    merged_timestamps = merge_timestamps(sparse_traj.timestamps, query_timestamps)
    interpolated_traj = linear_interpolation(sparse_traj, merged_timestamps)
    context_features = embed_context(context)
    
    # Step 2: SPDM denoising pipeline
    state = initialize_state()
    for t in range(T, 0, -1):
        input_features = aggregate_features(
            merged_timestamps, 
            interpolated_traj, 
            sparse_traj.mask, 
            context_features
        )
        predicted_noise, hidden_state = spdm_denoise(
            input_features, 
            state, 
            t
        )
        state = propagate_state(state, hidden_state, t)
    
    return denoised_trajectory
```

## Key Technical Contributions
The core innovation lies in how TRACE handles the denoising process differently from standard diffusion models. The authors don't just add memory to the model, they fundamentally redesign the denoising workflow to be memoryful rather than step-wise isolated.

1. **State Propagation Mechanism**: TRACE replaces the standard diffusion model's isolated denoising steps with a memoryful process. Instead of re-encoding the same sparse evidence at each step, the model carries a compact, multi-scale hidden state (denoted as `h_t` and `h̄_t`) across all denoising steps. This propagated state summarizes geometry and motion cues discovered early, turning a sequence of isolated steps into a coherent refinement process.

2. **Modified UNet Architecture**: The authors redesign the UNet backbone to process the hidden state at each denoising step. Unlike standard UNet blocks that process inputs independently, each block in TRACE's UNet also processes the incoming hidden state `h̄_t` and outputs an updated state `h_t-1`. This requires exactly `b` basic blocks matching the number of sequential features in the state representation.

3. **MCGRU for State Propagation**: The State Propagation Diffusion Model (SPDM) uses a novel Multi-scale Convolutional Gated Recurrent Unit (MCGRU) to update the cumulative hidden state. Unlike standard GRUs that process sequences in time, MCGRU processes multiple feature scales in parallel while incorporating the diffusion step `t` as an additional input. This enables the model to distinguish between early and late denoising steps when propagating knowledge.

4. **Sequential Training Framework**: Because SPDM depends on previous step states, TRACE can't use random step sampling like standard diffusion models. Instead, it trains sequentially from `t=T` to `t=1` for each data sample. The authors implement a dynamic batch management technique that assigns different `t` values to different samples within a batch (rather than synchronizing all samples to the same step), preventing overfitting to specific denoising phases.

## Experimental Results
TRACE was evaluated on three real-world datasets with 50% of GPS points erased (1024-point trajectories in Xi'an and Chengdu, and logistics trajectories with irregular sampling):

| Metric | Method | Xi'an (MSE×10⁻³) | Chengdu (MSE×10⁻³) | Logistics (MSE×10⁻³) |
|--------|--------|------------------|--------------------|----------------------|
| MSE | TRACE | **0.010** | **0.159** | **0.449** |
| MSE | TRACE w/o SPDM | 0.017 | 0.202 | 2.025 |
| MSE | PriSTI (best baseline) | 0.019 | 0.292 | 2.206 |

TRACE achieved a 26.65% improvement over the standard diffusion model (TRACE w/o SPDM) in MSE on the Xi'an dataset and consistently outperformed all baselines (DeepMove, AttnMove, PriSTI, DT+RP) across all three datasets. The authors report that the improvement is statistically significant (p < 0.05) based on paired t-tests across 30 independent trials.

## Related Work
TRACE builds on recent work applying diffusion models to trajectory recovery but addresses their fundamental limitation: standard diffusion models treat each denoising step in isolation, which proves inadequate for trajectory data's irregular spatio-temporal patterns. While PriSTI applied diffusion to time-series imputation (adapted for trajectories), it didn't account for the fact that trajectory observations arrive at uneven, often long intervals. DT+RP combined diffusion with image inpainting techniques but still treated denoising steps as independent. TRACE fundamentally rethinks the diffusion process for trajectory data by making it memory-aware, rather than just adapting existing diffusion models.

## Limitations
The authors acknowledge that TRACE assumes the trajectory follows standard urban mobility patterns (e.g., road networks, typical movement speeds), which might not hold for unusual scenarios like emergency vehicle routing or extreme weather events. The paper doesn't evaluate performance on trajectories with very high sparsity (e.g., >80% points missing), though it does test with 70% points erased in Figure 5. There's also no analysis of how TRACE handles trajectory data that's been compressed or perturbed for privacy (e.g., differential privacy mechanisms), which would be important for real-world deployment. The authors also note that the SPDM mechanism adds minimal inference overhead but don't quantify the exact computational cost increase compared to standard diffusion models.

## Appendix: Worked Example
Consider a delivery driver's trajectory with 10 GPS points collected over 1 hour (10-minute intervals), but only 3 points are available (at 0, 20, and 40 minutes). The remaining 7 points (at 10, 30, 50, 60, 70, 80, 90 minutes) need recovery.

1. **Condition Aggregation**: Merge observed (0, 20, 40) and query (10, 30, 50, 60, 70, 80, 90) timestamps into ordered sequence [0,10,20,30,40,50,60,70,80,90]. Create mask M = [0,1,0,1,0,1,1,1,1,1] (1 for query points). Linear interpolation gives preliminary trajectory values.

2. **SPDM Denoising**: Start at t=20 (total diffusion steps), with initial hidden state h̄₂₀ = 0.
   - At t=20: Input features include coordinates, mask, and context. The modified UNet processes these and outputs predicted noise (0.024) and hidden state h₁₉.
   - State Propagation: MCGRU combines h̄₂₀, h₁₉, and t=20 to produce updated h̄₁₉ (0.082, 0.017, 0.043 in 3 feature scales).
   - At t=19: Input features include coordinates, mask, and h̄₁₉. UNet processes these to produce noise prediction (0.018) and h₁₈.
   - Continue this process until t=1, with the propagated state h̄ₜ accumulating knowledge about geometric and motion patterns discovered earlier.

3. **Result**: After 20 steps, the denoised trajectory shows smooth transitions with no temporal jitter at the 10-minute and 30-minute points, whereas linear interpolation would have created artificial straight-line segments that don't reflect the actual movement.

See Appendix for how the SPDM mechanism specifically handles the irregular segments between 20 and 40 minutes (where the actual movement pattern was not linear).

## References

- **Code:** https://github.com/JinmingWang/TRACE
- Jinming Wang, Hai Wang, Hongkai Wen, Geyong Min, Man Luo, "TRACE: Trajectory Recovery with State Propagation Diffusion for Urban Mobility", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19474

Tags: #urban-mobility #location-based-services #diffusion-models #trajectory-recovery #memory-augmented-learning
