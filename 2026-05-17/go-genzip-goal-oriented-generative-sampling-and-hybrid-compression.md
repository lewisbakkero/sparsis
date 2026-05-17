---
title: "GO-GenZip: Goal-Oriented Generative Sampling and Hybrid Compression"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.20109"
---

## Executive Summary
GO-GenZip is a goal-oriented generative sampling and hybrid compression framework for network telemetry that reduces data collection and transmission costs by over 50% while maintaining analytical fidelity. Unlike traditional approaches that passively compress all observed data, it jointly optimises what to observe and how to encode it based on downstream task relevance.

## Why This Matters for Practitioners
If you're responsible for network monitoring systems collecting KPIs from thousands of base stations, this paper suggests you can halve your data storage and transmission costs without compromising analytical accuracy. Specifically, implement adaptive sampling policies that learn from contextual information (BS class, hour of day, task identifier) rather than using fixed sampling strategies. For example, instead of collecting all 34 KPIs hourly from every base station, your system could dynamically prioritise latency metrics during peak hours for specific BS classes, reducing the data volume by 52% while maintaining prediction task accuracy within 2% of full-data performance.

## Problem Statement
Current network telemetry pipelines operate like a city's traffic department recording every vehicle's speed, direction, and passenger count at every intersection, every minute, without distinguishing between critical congestion points and routine traffic flow. This approach wastes bandwidth on irrelevant data while failing to capture the specific metrics needed for traffic management decisions.

## Proposed Approach
GO-GenZip integrates adaptive sampling policies with generative modelling and hybrid compression, forming a goal-oriented pipeline that optimises both what to observe and how to encode it. The framework employs two key policies:
1. A sampling policy that selects relevant telemetry data based on contextual information
2. A hybrid compression policy that combines lossless (LZMA) and generative autoencoder compression

The system works as follows:
1. At each base station, collected KPIs are sampled using a context-aware mask
2. The sampled data are split into two segments: one for lossless compression and one for generative compression
3. The compressed segments are transmitted to a central platform
4. At the receiver, data are reconstructed by merging results from both compression methods

```python
def go_genzip_train(data, context, sampling_budget, compression_budget):
    # Compute sampling mask from context
    ms = sampling_policy(context)
    
    # Compute compression mask from context
    mc = compression_policy(context)
    
    # Split data into lossless and generative segments
    Xh = (1 - mc) * ms * data  # Lossless segment
    Xg = mc * ms * data        # Generative segment
    
    # Compress and reconstruct
    Xh_compressed = lzma_compress(Xh)
    Xg_compressed = generative_compressor(Xg)
    
    # Reconstruct full data
    X_recon = generative_decompressor(Xg_compressed) + lzma_decompress(Xh_compressed)
    
    # Update policies to meet constraints
    update_policies(data, X_recon, sampling_budget, compression_budget)
```

## Key Technical Contributions
This paper introduces several novel mechanisms for goal-oriented network telemetry compression. The key technical contributions include:

1. **Context-aware adaptive sampling policy**: The framework uses a two-layer MLP with ELU activation that maps contextual metadata (BS class, hour, task ID) to per-BS sampling decisions through Gumbel-Softmax sampling. This allows the system to dynamically allocate sampling resources based on the specific network conditions and task requirements, rather than using a fixed sampling strategy across all conditions.

2. **Hybrid compression framework**: Unlike prior approaches that use either generative or lossless compression exclusively, GO-GenZip combines both approaches within a single framework. The compression selector mask (mc) determines which data segments use generative autoencoder compression (for dense data requiring high fidelity) versus lossless LZMA compression (for sparse data where minimal distortion is required), balancing compression efficiency and reconstruction accuracy.

3. **Dual optimisation with constraint matching**: The training procedure jointly optimises sampling and compression policies through a Lagrangian loss function that incorporates dual parameters (βs and βc) to enforce sampling and bandwidth constraints. This approach ensures the system automatically adjusts to meet specific resource constraints without requiring manual tuning for different deployment scenarios.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The framework was evaluated on telemetry data from 1,162 4G base stations reporting 34 KPIs hourly over a ten-day period. The results demonstrate:

- Over 50% reduction in both sampling and data transfer costs compared to traditional approaches
- Hybrid compression (S-H) consistently outperforms generative-only compression (S-G) across all sampling ratios (SR), with MAE 0.04±0.0008 for latency prediction tasks versus 0.04±0.003 for S-G
- Adaptive policies significantly outperform fixed policies (Fig. 5), maintaining lower MAE across varying sampling ratios (0.2 to 1.0)
- For prediction tasks, the goal-oriented end-to-end training achieved 1.21±0.02 MAE for task 1 versus 1.27±0.04 for recon-based approaches (Table I)
- The framework maintains performance even at low sampling ratios (SR=0.2), where S-G and S-H approaches become equivalent

## Related Work
The work builds on the Information Bottleneck principle and Masked Autoencoders but extends beyond traditional approaches that treat the source data as fully observed. Unlike existing goal-oriented communication approaches that focus solely on transmission, GO-GenZip jointly optimises the sampling and compression stages. It improves over prior network telemetry compression methods that rely on random masking strategies by incorporating contextual information and task objectives, as well as over pure generative approaches that cannot effectively represent sparse data.

## Limitations
The authors only evaluated the framework on 4G network data from base stations, not on 5G networks with potentially different data characteristics. The paper doesn't address how the framework would scale to extremely large networks with tens of thousands of base stations, though the evaluation on 1,162 base stations suggests good scalability properties. The current implementation requires training a separate model for each network application, which could increase deployment complexity in multi-application environments.

## Appendix: Worked Example
Consider a base station (BS class = urban, hour = 14:00, task = latency prediction) with 34 KPIs collected hourly. The context information (BS class, hour, task ID) is fed into the sampling policy network.

1. The sampling policy produces a mask where 52% of entries are sampled (SR=0.52), while the compression policy selects 45% of those sampled entries for generative compression (CR=0.45).

2. The sampled data (18 out of 34 KPIs) are split into two segments:
   - 8 KPIs (45% of 18) are compressed using the generative model
   - 10 KPIs (55% of 18) are compressed using LZMA

3. The generative model compresses the 8 KPIs into a latent representation (reducing from 8×1 to 8×0.25 dimensions), while LZMA compresses the 10 KPIs to 30% of their original size.

4. The total data transmitted is:
   - Generative segment: 2 (8×0.25) + 2 (overhead) = 4 units
   - LZMA segment: 3 (10×0.3) = 3 units
   - Total: 7 units vs. the original 34 units (79.4% reduction)

5. At the receiver, the data is reconstructed by:
   - Decompressing the generative segment using the trained autoencoder
   - Decompressing the LZMA segment using the standard decoder
   - Merging the reconstructed segments to form the full 34 KPI vector

This process achieves 52% data reduction while maintaining prediction task accuracy within 2% of using all 34 KPIs (MAE 0.04±0.0008 vs 0.04±0.003).

## References

- Pietro Talli, Qi Liao, Alessandro Lieto, Parijat Bhattacharjee, Federico Chiariotti, Andrea Zanella, "GO-GenZip: Goal-Oriented Generative Sampling and Hybrid Compression", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20109

Tags: #networking #telemetry #adaptive-sampling #generative-modelling #hybrid-compression
