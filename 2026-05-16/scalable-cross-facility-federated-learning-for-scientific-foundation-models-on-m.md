---
title: "Scalable Cross-Facility Federated Learning for Scientific Foundation Models on Multiple Supercomputers"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19544"
---

## Executive Summary
This paper presents a cross-facility federated learning framework specifically designed for heterogeneous HPC environments, enabling collaborative training of scientific foundation models across multiple DOE supercomputers without centralising raw data. The framework addresses critical HPC-specific constraints like unpredictable scheduling, heterogeneous accelerators, and strict site firewalls, and demonstrates that algorithmic choices like FedCompass significantly impact performance in real-world settings.

## Why This Matters for Practitioners
For engineers building production systems that need to collaborate across institutional boundaries (like scientific institutions or healthcare providers), this paper reveals critical constraints that traditional federated learning frameworks simply cannot handle. If you're responsible for deploying distributed training across multiple HPC facilities, you must understand that GPU memory constraints create an 8.4x throughput difference (e.g., 250 samples/sec on 40GB GPUs versus 2,100 samples/sec on 64GB GPUs), making algorithmic choices like FedCompass essential rather than optional. You should implement scheduler-aware algorithms that adapt to computational heterogeneity and prioritise co-location for model transfers to reduce communication overhead by 30-40% (Aurora and Polaris achieved the highest transfer speeds due to co-location with the server at Argonne). Most importantly, don't assume that standard FL algorithms will work in HPC environments, FedCompass improved results by 12.9% over FedBuff under realistic queueing conditions.

## Problem Statement
Current federated learning frameworks are like trying to run a Formula 1 race on a bumpy country road: they're designed for predictable cloud environments but fail when faced with the unpredictable queueing delays, heterogeneous accelerators, and strict site firewalls typical of HPC facilities. Just as a Formula 1 car would lose efficiency on a country road due to unexpected obstacles and different surface conditions, traditional FL frameworks struggle with HPC's unique constraints, making cross-facility collaboration impractical for scientific applications that require massive computational resources.

## Proposed Approach
The authors built a framework on top of APPFL with Globus Compute for task dispatch and Globus Transfer for model exchange, allowing local training jobs to execute under site-specific scheduling policies while maintaining reliable communication across wide-area networks. This approach enables clients to train on local data and send model updates to a central server for aggregation, with the system characterising and adapting to HPC-specific heterogeneity. The core components are:

1. APPFL for privacy-preserving federated training
2. Globus Compute for training task dispatch across facilities
3. Globus Transfer for reliable model and configuration exchange
4. Scheduler-aware algorithm design to handle computational heterogeneity

Here's a simplified pseudocode representation of FedCompass, the algorithm that adapts to computational heterogeneity:

```python
def fedcompass(client_capabilities, global_model, local_datasets):
    # client_capabilities: dictionary of client computational throughput
    # local_datasets: data partitioned per client
    # global_model: current global model being trained
    
    # Calculate target local rounds based on client throughput
    target_rounds = {client: (total_rounds * throughput) 
                     for client, throughput in client_capabilities.items()}
    
    # Adapt training steps to balance contributions
    for client in clients:
        local_steps = min(target_rounds[client], max_local_steps)
        train_local_model(client, local_datasets[client], local_steps)
        send_update(global_model, client)
    
    # Aggregate model updates
    global_model = aggregate_updates(global_model, updates)
    
    return global_model
```

## Key Technical Contributions
The paper makes several crucial technical contributions that address HPC-specific constraints:

1. **Heterogeneity characterisation**: They systematically characterised how GPU memory capacity drives a critical trade-off in optimisation strategies. Memory-constrained GPUs (40GB) must employ DeepSpeed ZeRO-3 for memory efficiency, which increases communication overhead and reduces throughput to 250 samples/sec, while larger memory GPUs (64-80GB) can use ZeRO-1, achieving throughput of 1,000-2,100 samples/sec. This 4x difference is entirely attributable to memory-driven optimisation strategies, not raw hardware.

2. **Scheduler-aware algorithm design**: They demonstrated that algorithmic choices matter significantly under realistic HPC scheduling conditions. FedCompass, which adapts local training steps to individual computational capabilities, achieved a final test loss of 0.4345 under queueing conditions, representing a 12.9% reduction in loss over FedBuff and a 19.3% reduction over FedAsync. This outperformance stems from the algorithm's capacity to balance staleness tolerance and coordination based on computational power.

3. **Communication efficiency quantification**: They empirically measured communication overhead for different model sizes, showing a linear relationship between model parameters and storage requirements (250 MB for 125 million parameters, 26 GB for 13 billion parameters). Transfer speeds improved more than 10-fold from OPT-125m to Llama2-13b, with co-located facilities (Aurora and Polaris) achieving the highest speeds due to high-bandwidth local networks.

See Appendix for a step-by-step worked example of how communication overhead scales with model size and geographical distance.

## Experimental Results
The authors evaluated their framework by fine-tuning a Llama2-7B model on the SMolInstruct chemistry instruction dataset (3.3 million samples across 14 tasks), distributed across four DOE supercomputers:

- **Throughput characterisation**: Aurora achieved 2,100 samples/sec at 64 nodes (ZeRO-1), while Polaris plateaued at 250 samples/sec (ZeRO-3), a difference of 8.4x directly attributable to memory capacity and optimisation strategy. Perlmutter's 80GB variant achieved 1,200 samples/sec compared to 250 samples/sec for its 40GB variant.
  
- **Algorithm comparison under queueing conditions**: FedCompass achieved a final test loss of 0.4345, representing improvements of 4.5% over FedAvg, 19.3% over FedAsync, and 12.9% over FedBuff. FedBuff outperformed FedAsync by moderating straggler impact through buffered aggregation.

- **Communication analysis**: Model size scales linearly with storage requirements (250 MB for 125M parameters to 26 GB for 13B parameters), with transfer times for Llama2-13b being 21.7 seconds on co-located facilities (Aurora) versus 32.5 seconds on non-co-located facilities (Perlmutter).

## Related Work
The paper positions itself as a systematic empirical characterisation of federated training across heterogeneous leadership-class HPC facilities, building on prior work like Kim et al. [28] that discussed privacy and computational heterogeneity challenges for cross-silo FL on HPC systems but lacked empirical evaluation across multiple facilities. It contrasts with existing cross-silo frameworks (FATE, OpenFL, NVIDIA FLARE, FedML), which are designed for cloud and enterprise settings and do not address HPC-specific constraints like job schedulers with unpredictable queueing delays, strict site firewalls, heterogeneous accelerator architectures, and system interruptions from maintenance.

## Limitations
The authors acknowledge that FedCompass does not yet model queue time variability or system-specific scheduling policies, suggesting a clear avenue for further improvement. The study focused on a single scientific domain (chemistry) and didn't evaluate performance across different scientific tasks or model types beyond LLMs. The framework also assumed that sites would participate reliably throughout training, which may not hold for all real-world scientific collaborations where facilities have varying operational constraints.

## Appendix: Worked Example
Let's walk through the communication cost analysis for model transfer across facilities:

1. **Model size to transfer cost**: The paper analysed five models from OPT-125m (125 million parameters) to Llama2-13b (13 billion parameters) in BF16 format. For OPT-125m: 125e6 parameters × 2 bytes = 250 MB. For Llama2-13b: 13e9 parameters × 2 bytes = 26 GB.

2. **Transfer speed analysis**: For Llama2-13b, Aurora (co-located at Argonne with the server) achieved a transfer speed of 1.2 GB/s, resulting in a transfer time of 21.7 seconds (26 GB / 1.2 GB/s = 21.7 s). Perlmutter (not co-located) achieved 0.8 GB/s, resulting in 32.5 seconds (26 GB / 0.8 GB/s = 32.5 s).

3. **Scaling relationship**: The paper showed that larger models benefit more from high-bandwidth local networks because they better amortise connection establishment overhead. Aurora and Polaris demonstrated the strongest scaling behaviour, improving their transfer speed more than 10x from OPT-125m to Llama2-13b, while Perlmutter exhibited slightly weaker scaling.

4. **Impact on practical deployment**: For a 13-billion parameter model, this translates to approximately 21.7 seconds of transfer time for co-located facilities versus 32.5 seconds for non-co-located facilities, a 30-40% difference that accumulates significantly over multiple training rounds.

This concrete analysis demonstrates that co-location is not just a convenience but a critical factor in reducing communication overhead for large-scale scientific federated learning.

## References

- Yijiang Li, Zilinghan Li, Kyle Chard, Ian Foster, Todd Munson, Ravi Madduri, Kibaek Kim, "Scalable Cross-Facility Federated Learning for Scientific Foundation Models on Multiple Supercomputers", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19544

Tags: #scientific-computing #federated-learning #high-performance-computing #heterogeneous-hpc #scheduler-aware-aggregators
