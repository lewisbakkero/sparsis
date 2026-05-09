---
title: "OpenDC-STEAM: Realistic Modeling and Systematic Exploration of Composable Techniques for Sustainable Datacenters"
venue: "arXiv cs.DC"
paper_url: "https://arxiv.org/abs/2603.12381"
---

## Executive Summary
OpenDC-STEAM is an open-source datacenter simulator that quantifies the impact and trade-offs of sustainability techniques like horizontal scaling, battery usage, and temporal shifting. It addresses the critical gap in evaluating these techniques in isolation or under unrealistic conditions, allowing practitioners to make evidence-based decisions on datacenter sustainability without sacrificing performance.

## Why This Matters for Practitioners
If you're responsible for designing or operating a datacenter, OpenDC-STEAM helps you avoid the common pitfall of overestimating carbon savings by ignoring real-world dynamics like task failures and resource constraints. For instance, using temporal shifting alone might appear to reduce carbon emissions by 7% on paper, but STEAM reveals this drops to 2% when accounting for task stacking during high-carbon periods. You can now systematically test combinations of techniques, such as using batteries during high-carbon periods while maintaining horizontal scaling, for optimal results. This means you can confidently implement strategies that reduce emissions by up to 28% without compromising service quality, using the exact same datacenter hardware.

## Problem Statement
Today's sustainability evaluations treat datacenter techniques like separate car engines, testing each in isolation without considering how they interact in a full vehicle. Just as testing a hybrid engine without the transmission's dynamics would miss how torque shifts affect fuel efficiency, evaluating temporal shifting without considering task failures or resource constraints leads to optimistically inflated carbon savings. Prior work assumed tasks could be shifted independently, ignoring how task delays interact with resource limitations and failures, resulting in misleading claims about sustainability gains.

## Proposed Approach
OpenDC-STEAM uses a composable architecture based on the OpenDC framework to simulate datacenter dynamics while evaluating sustainability techniques. It models datacenter components as a graph connected through event-driven interactions, allowing techniques to be independently developed and combined through simple configuration. The simulator takes datacenter topology, workload traces, carbon intensity data, and failure models as input, then executes the component graph through an event-based executor to collect metrics on performance and carbon emissions. This architectural approach makes it possible to evaluate how techniques interact in realistic scenarios.

```python
def evaluate_sustainability_techniques(datacenter, workload, carbon_trace, technique_config):
    # Construct component graph
    components = build_component_graph(datacenter, carbon_trace)
    
    # Configure sustainability techniques
    for technique in technique_config:
        add_technique_component(components, technique)
    
    # Execute simulation
    executor = EventBasedExecutor(components)
    metrics = executor.run(workload)
    
    # Analyse results
    return analyse_metrics(metrics)
```

## Key Technical Contributions
STEAM's architecture introduces novel mechanisms for realistic sustainability evaluation that prior work lacked.

1. **Component Graph Architecture**: STEAM models datacenter components as a graph where directed edges define supplier-consumer relationships (e.g., CPU consumes power from PSU). This allows new techniques to be added with minimal implementation effort: adding a battery component requires only connecting it to the PSU and task scheduler, without modifying other components. This loose coupling enables systematic exploration of technique combinations without requiring extensive reimplementation.

2. **Realistic Carbon Modelling**: STEAM incorporates both operational carbon (from energy usage) and embodied carbon (from hardware manufacturing) into its metrics. It calculates operational carbon by multiplying energy usage by real-time carbon intensity from traces, while embodied carbon is estimated based on hardware usage duration relative to its assumed lifespan. This comprehensive approach reveals that what appears to be a 28% carbon reduction using batteries might actually be offset by the embodied carbon cost of the battery itself if not used optimally.

3. **Failure-Aware Scheduling**: Unlike prior analytical models that ignore operational phenomena, STEAM integrates failure models into its simulation. It accounts for how task failures during temporal shifting can require rescheduling, causing delays that impact service quality. This explains why temporal shifting's reported effectiveness of 7% in prior work drops to approximately 2% when considering realistic failure scenarios.

## Experimental Results
The authors conducted extensive experiments across three representative workloads (Surf, Marconi, and Borg) with diverse datacenter configurations and carbon-intensity traces. Key findings:

- **Horizontal scaling** reduced carbon emissions by up to 35% but ignored operational phenomena like failures, leading to overly optimistic results
- **Batteries** could reduce emissions by up to 28% when used in optimal contexts, but could increase emissions by 15% when misused (e.g., charging during high-carbon periods)
- **Temporal shifting** showed only 7% emission reductions in prior work, but STEAM revealed this drops to approximately 2% when accounting for task stacking and operational effects

The study used 158 different carbon intensity traces from multiple regions to ensure generalizability, demonstrating that the effectiveness of techniques varies significantly by geographic location and carbon intensity profile.

## Related Work
STEAM builds upon the foundation of previous analytical models for sustainability evaluation, but addresses their critical limitation of overlooking datacenter dynamics and technique interactions. Unlike prior work that used simplified analytical models to estimate carbon savings (e.g., treating each task as independently shift-able), STEAM's simulation-based approach captures how techniques interact with resource constraints, failures, and real-world datacenter behaviour. This positions STEAM as a more practical tool for evidence-based sustainability decisions in production environments.

## Limitations
The paper acknowledges limitations in its experimental setup: it didn't evaluate all possible combinations of techniques across every workload type. The Borg workload's hardware specifics were obscured (as noted in Section VIII), requiring an estimation of embodied carbon. Additionally, STEAM doesn't model the full complexity of cooling systems, which contribute significantly to operational carbon but weren't the focus of this study. The authors also note that STEAM's effectiveness depends on having accurate carbon intensity traces, which may not be available in all regions.

## Appendix: Worked Example
Let's walk through how STEAM evaluates temporal shifting for the Surf workload using a specific carbon intensity trace from the Netherlands.

1. **Initial Setup**: The Surf workload has 194,917 tasks over 124 days, with an average task duration of 1 hour, 49 minutes, and 38 seconds. The datacenter has 277 hosts (128 GB RAM, 16-core Intel Xeon).

2. **Carbon Trace**: We use a carbon intensity trace for the Netherlands (7-2022) with a rolling mean carbon intensity of 0.5 kgCO2e/kWh during the workload period.

3. **Temporal Shifting Policy**: Tasks are scheduled when carbon intensity is below the 35th percentile of the next week's forecast (0.6 kgCO2e/kWh). Tasks have a maximum delay of 24 hours.

4. **Simulation Execution**: 
   - At time t1 (high carbon intensity), 3 tasks are delayed to t3
   - At time t2 (medium carbon intensity), 2 new tasks are delayed to t3
   - At time t3, 5 tasks are scheduled but only 3 can run concurrently (due to 277 hosts), so 2 tasks are delayed to t4
   - During execution, host H1 fails, interrupting Task 1 and requiring it to be rescheduled at t4

5. **Carbon Calculation**: 
   - In a simple analytical model: 3 tasks shifted from high to low carbon (35% saving) + 2 tasks shifted from medium to low carbon (25% saving) = 31% average saving
   - In STEAM's simulation: Only 2 tasks (2, 3) were successfully shifted to low carbon periods; Tasks 1, 4, and 5 executed during high-carbon periods = 2% reduction

6. **Performance Impact**: The simulation reveals 1.2% SLA violations (tasks not scheduled within 24 hours), with a peak task delay of 4.7 hours during the stacking period at t3.

This example demonstrates how STEAM's realistic modelling reveals that temporal shifting's actual effectiveness is 14× lower than what prior analytical models would suggest.

## References

- **Code:** https://github.com/atlarge-research/OpenDC-STEAM.
- Dante Niewenhuis, Sacheendra Talluri, Alexandru Iosup, Tiziano de Matteis, "OpenDC-STEAM: Realistic Modeling and Systematic Exploration of Composable Techniques for Sustainable Datacenters", arXiv cs.DC, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.12381

Tags: #distributed-systems #sustainable-computing #carbon-emission-modelling #datacenter-optimisation #resource-management
