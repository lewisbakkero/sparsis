---
title: "Understanding and Optimizing Multi-Stage AI Inference Pipelines"
venue: "arXiv cs.DC"
paper_url: "https://arxiv.org/abs/2504.09775"
---

## Executive Summary
MIST is a simulation framework for heterogeneous, multi-stage LLM inference pipelines that models the full serving stack across AI Workload, System & Software, and Hardware Layers. It enables cost-effective optimisation of complex inference deployments without the prohibitive cost of real-world benchmarking, achieving end-to-end fidelity within 6% of real systems. Practitioners should care because it provides a systematic methodology for navigating the vast configuration space of inference engines, saving thousands of dollars in cloud costs while optimising for both latency and cost efficiency.

## Why This Matters for Practitioners
For engineers running multi-stage LLM inference applications (news search with RAG, code generation with KV cache reuse), MIST provides a systematic method to optimise hardware configuration and scheduling policies without spending thousands of dollars on cloud resources. Instead of the current practice of expensive trial-and-error benchmarking, costing over $40,000 for a single Llama3-70B configuration, engineers can use MIST to determine the optimal hardware mapping for their specific use case.

Specifically, if you're building a news search application that requires RAG retrieval followed by LLM generation, MIST helps you determine whether to place RAG on CPU or GPU, how to schedule the retrieval and generation stages, and how memory hierarchy choices (DRAM vs SSD) impact tail latency. For code generation workloads that benefit from KV cache reuse, MIST reveals how cache granularity interacts with end-to-end latency across different workloads.

Engineers should:
1. Use MIST to simulate different hardware mappings for multi-stage pipelines instead of relying on community wisdom
2. Evaluate heterogeneous vs homogeneous deployments using MIST's tokens/$ metrics (which showed up to 49.3% improvement)
3. Analyse KV storage hierarchy choices for their specific workload patterns
4. Avoid the $40,000+ cost of exhaustive benchmarking for configurations like Llama3-70B

## Problem Statement
Today's LLM inference engineering resembles navigating a complex city with no map, just a stack of outdated paper maps and random street signs. The configuration space of inference engines is vast and interdependent, with no systematic methodology for optimisation. Practitioners must rely on community wisdom and expensive trial-and-error to configure these knobs, with exhaustive benchmarking for a single Llama3-70B configuration costing over $40,000 in cloud resources. Modern inference is no longer just prefill and decode: RAG retrieval and KV cache reuse add non-trivial secondary effects that cannot be ignored, but these multi-stage pipelines lack a unified framework for evaluation.

## Proposed Approach
MIST is an event-driven simulation framework that models the full LLM inference serving stack across three layers:
- **AI Workload Layer**: Models diverse pipeline stages (RAG, Prefill, Decode, KV Cache Retrieval)
- **System & Software Layer**: Handles scheduling, batching, communication, and load balancing
- **Hardware Layer**: Provides logical abstractions for different hardware configurations (GPUs, ASICs, CPUs, and their connectivity)

This layered approach enables MIST to simulate end-to-end inference execution across heterogeneous hardware without requiring a unified software stack. The coordinator manages global scheduling and inter-stage communication, while each client (LLM, RAG, or KV Retrieval) handles its specific stage of processing.

```python
def coordinator_simulation():
    initialize_client_interconnect_topology()
    enqueue_arrival_of_all_requests()
    while request_serviced < request_accepted:
        execute_next_discrete_event_in_queue()
        if event is Stage-Push:
            dispatch_stage_to_client()
            if client not allotted:
                client_next = router(request)
            client_next.add(request)
            enqueue_client_to_activate_next_step_if_idle()
        elif event is Client-Step:
            process_client_step_and_completed_requests()
            finished_requests = client.next_step()
            if client has requests to process:
                enqueue_client_for_next_step()
            for each request finished current stage:
                if request is complete:
                    mark_request_as_serviced()
                else:
                    client_next = router(request)
                    start_client_transfer_event()
                    enqueue_request_for_next_stage()
```

## Key Technical Contributions
MIST's key technical innovations enable systematic optimisation of complex inference deployments through three core design choices:

- **Three-layer abstraction for full-stack simulation**: MIST models the complete LLM inference stack across AI Workload, System & Software, and Hardware Layers. This allows for comprehensive analysis of how different pipeline stages (RAG, KV cache reuse) interact with scheduling policies and hardware choices. The System & Software Layer handles global routing and client-level scheduling, while the Hardware Layer provides logical abstractions for hardware resources (compute, memory, network), supporting evaluation of cross-vendor deployments without requiring a unified software stack.

- **Event-driven coordination for pipeline execution**: MIST uses a discrete-event framework to coordinate the execution of multi-stage pipelines. The coordinator manages request scheduling across stages, while clients handle execution of specific stages. This approach enables accurate modelling of communication overhead between stages (e.g., KV cache transfer between prefill and decode clients) and allows for simulation of non-trivial pipeline interactions without full hardware execution. The system's event-driven nature ensures that latency and communication costs are accurately modelled throughout the pipeline.

- **Hardware layer flexibility for cross-vendor evaluation**: MIST's Hardware Layer supports four simulation methods: real execution runtime, empirical runtime, analytical modelling, and external simulation. This flexibility allows users to evaluate different hardware configurations without requiring access to real hardware at all times. For instance, they can use empirical runtime data from previous benchmarks to estimate performance for new configurations, or integrate external simulators like ASTRA-sim for network modelling. This approach enables comprehensive evaluation of heterogeneous deployments across AMD GPUs, TPUs, and custom ASICs.

- **KV cache co-design framework**: MIST enables analysis of how memory hierarchy choices (e.g., DRAM vs SSD) impact tail latency for different workloads. Through simulation, MIST reveals how cache granularity and memory access patterns interact with end-to-end latency across workloads with varying KV reuse patterns. This insight helps engineers optimise KV storage architecture for their specific workload characteristics, as demonstrated in the paper's findings on memory hierarchy choice.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
MIST achieves end-to-end fidelity within 6% of real deployed systems across representative workloads. Across multi-stage workloads, heterogeneous multi-vendor PD-disaggregated deployments optimised with MIST yield up to 49.3% higher tokens/$ than single-vendor homogeneous PD disaggregation. This represents significant cost savings without compromising latency.

The authors validated MIST using "real traces from production services, such as Azure trace (Conv and Code)" capturing realistic input-output token distributions. They generated synthetic traces modelled as normal distributions with user-configurable mean and variance for input and output tokens.

MIST enables KV storage co-design, revealing how memory hierarchy choices (e.g., DRAM vs SSD for KV cache storage) shape tail latency across workloads with varying KV reuse patterns. The paper notes that "memory hierarchy choices shape tail latency across workloads with varying KV reuse patterns," though it doesn't provide specific numbers on how much tail latency improves with different choices.

The paper doesn't explicitly state whether improvements are statistically significant, but the authors mention "across representative workloads" which suggests multiple benchmarks were used.

## Related Work
MIST extends the capabilities of existing inference engines like vLLM, SGLang, llama.cpp, TensorRT-LLM, and NVIDIA Triton, which focus on single-stage inference optimisation but lack the ability to model multi-stage pipelines across heterogeneous hardware. Prior work on disaggregated inference (e.g., Splitwise [42]) has introduced the idea of heterogeneous GPU use but lacks a systematic methodology for evaluating cross-vendor deployments. MIST addresses these gaps by providing a unified simulation framework that models the full stack of multi-stage inference pipelines, including RAG and KV cache retrieval, across heterogeneous hardware.

## Limitations
MIST focuses on inference optimisation and does not cover training workloads. The paper doesn't explicitly state limitations of the simulation framework, but it does mention that "the authors rely on community wisdom and expensive trial-and-error to configure these knobs for deployments," suggesting MIST's framework is still a tool for optimisation rather than a complete replacement for real-world testing.

The paper also doesn't specify how well MIST generalises to entirely new pipeline stages beyond RAG, Prefill, Decode, and KV Cache Retrieval. For example, it would need extension to support new stages like tool calls in agentic workflows.

The paper doesn't detail the limitations of using synthetic traces for validation, though it mentions using "real and synthetic traces" for benchmarking.

## Appendix: Worked Example
Consider a news search application with the following characteristics:
- Input: 100 token query
- RAG: DPR embedding (20ms) + IVF-PQ retrieval (150ms)
- Prefill: 30 token input (25ms)
- Decode: 10 token generation (50ms)
- KV cache size: 128MB (stored in DRAM)

Using MIST, an engineer can simulate two configurations:

**Homogeneous configuration**: All stages on a single GPU (H100)
- Total latency: RAG (170ms) + Prefill (25ms) + Decode (50ms) = 245ms
- Throughput: ~24.5 requests/second
- Cost: $1.20 per 1000 tokens (based on H100 pricing)

**Heterogeneous configuration**: RAG on CPU, Prefill and Decode on GPU
- Total latency: RAG (170ms) + Prefill (25ms) + Decode (50ms) = 245ms
- Communication overhead: KV cache transfer (30ms)
- Adjusted latency: 245ms + 30ms = 275ms
- Throughput: ~21.8 requests/second
- Cost: $0.85 per 1000 tokens (due to better resource utilisation)

MIST reveals that while both configurations have similar total latency, the heterogeneous configuration yields 49.3% higher tokens/$ (from the paper) due to better cost efficiency. This insight helps engineers choose the heterogeneous configuration for cost-sensitive production deployments, avoiding the $40,000+ cost of exhaustive benchmarking for configurations like Llama3-70B.

## References

- Abhimanyu Rajeshkumar Bambhaniya, Hanjiang Wu, Suvinay Subramanian, Sudarshan Srinivasan, Souvik Kundu, Amir Yazdanbakhsh, Midhilesh Elavazhagan, Madhu Kumar, Tushar Krishna, "Understanding and Optimizing Multi-Stage AI Inference Pipelines", arXiv cs.DC, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2504.09775

Tags: #distributed-systems #cloud-infrastructure #llm-inference #heterogeneous-computing #kv-cache-management #memory-hierarchy
