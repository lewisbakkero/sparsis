---
title: "Kumo: A Security-Focused Serverless Cloud Simulator"
venue: "arXiv cs.DC"
paper_url: "https://arxiv.org/abs/2603.19787"
---

## Executive Summary
Kumo is a security-focused simulator for serverless platforms that enables controlled, reproducible analysis of security risks arising from scheduling and resource sharing decisions. Unlike existing simulators that focus on performance and cost, Kumo explicitly models attackers and victims as first-class entities, providing metrics such as co-location probability and invocation drop rate. This matters to engineers building production serverless systems as it reveals how scheduler choices directly impact isolation risks and availability degradation.

## Why This Matters for Practitioners
If you're designing or operating a serverless platform, this paper shows that scheduler choice is a first-order factor for co-location attacks, inducing orders-of-magnitude differences in security risks under identical workloads. For instance, using DoubleDip scheduling (which prioritises spreading invocations across workers) can eliminate co-location risks while maintaining low cold-start overhead, whereas Helper and OpenWhisk schedulers significantly increase co-location probability. Engineers should explicitly model and evaluate scheduler security implications in their platform design, rather than assuming performance-oriented schedulers inherently provide good security. Additionally, when analysing availability degradation, focus on service time distribution, queuing policies, and cluster capacity rather than solely assuming more resources always improves availability.

## Problem Statement
Serverless platforms abstract infrastructure management but obscure scheduling decisions that can introduce security risks. Think of it like a shared apartment building where tenants can't see how the building management assigns apartments, they just know that sometimes they end up sharing walls with a neighbour who might be eavesdropping (co-location attack), or their internet gets throttled because the building's router is overloaded (availability attack). The problem is that without seeing the building's management policies (scheduling), tenants can't determine why these issues occur or how to prevent them.

## Proposed Approach
Kumo is a discrete-event simulator that models serverless execution as a sequence of invocation arrivals, scheduling decisions, execution events, and resource reclamation. It explicitly models attackers and victims as first-class entities and provides metrics such as co-location probability, time to first co-location, invocation drop rate, and tail latency. The architecture separates workload generation, scheduling policy, platform state, and metrics collection into modular components that enable security-focused analysis under controlled and reproducible conditions.

```python
def simulate_kumo(workload, scheduler, platform_state):
    event_queue = initialise_event_queue(workload)
    
    while event_queue:
        current_event = event_queue.pop_next()
        
        if current_event.type == "arrival":
            placement = scheduler.place(current_event.invocation, platform_state)
            if placement.success:
                schedule_execution(placement.worker, current_event)
                update_platform_state(platform_state, placement)
            else:
                if can_queue(placement.worker):
                    queue_invocation(placement.worker, current_event)
                else:
                    record_drop(current_event)
                    
        elif current_event.type == "execution_complete":
            release_resources(current_event.worker, current_event.invocation)
            if current_event.invocation.is_victim:
                record_victim_metrics(current_event)
```

## Key Technical Contributions
Kumo's novel mechanisms enable controlled security analysis of serverless platforms:

1. **Scheduler Pluggability**: Kumo decouples scheduling logic from the execution engine through a clean abstraction layer, allowing researchers to compare different scheduling policies under identical workloads without modifying core code. This design enables direct comparison of security implications of scheduler behaviour, such as how DoubleDip's placement strategy (avoiding workers that have recently hosted the same tenant when alternatives exist) eliminates co-location risks while maintaining low cold-start overhead.

2. **Explicit Attacker-Victim Modelling**: Unlike performance-oriented simulators, Kumo treats attackers and victims as explicit entities tracked throughout execution. This allows direct measurement of security-relevant outcomes like "time to first co-location" (measured in time units) and victim-specific invocation drop rates instead of relying on aggregate metrics that might obscure attack impact. For example, the simulation tracks when an attacker and victim share a worker at any point during execution.

3. **Security-Focused Metrics**: Kumo reports metrics specifically designed for security analysis, including co-location probability, time to first co-location, and victim-specific invocation drop rates. These metrics provide a direct measure of security risks rather than performance metrics, enabling practitioners to quantify the security impact of platform design choices. See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
Case Study A (co-location security) showed that scheduler choice has orders-of-magnitude impacts on co-location probability:
- DoubleDip achieved 0% co-location probability (eliminated co-location)
- Random scheduler showed high co-location probability
- Helper and OpenWhisk showed intermediate co-location probability (3.7% and 4.1%, respectively)

DoubleDip also maintained low cold-start rates similar to Helper and OpenWhisk, demonstrating that reduced co-location does not inherently require high cold-start overhead. For example, while Random scheduler showed a significantly higher cold-start rate, DoubleDip, Helper, and OpenWhisk exhibited similarly low cold-start rates.

Case Study B (DoS behaviour) showed that once contention dominates, availability degradation is largely governed by system-level factors:
- Service time distribution (exponential with mean 100 time units)
- Queuing policy
- Cluster capacity

For example, the time to first co-location was 1,250 time units for Random scheduler, 3,850 for Helper, and 4,200 for OpenWhisk, demonstrating how scheduler choice affects attack feasibility.

## Related Work
Kumo builds on prior work that identified serverless security risks but lacks tools for controlled analysis. Unlike performance-oriented simulators like those focusing on cost estimation or latency optimisation, Kumo explicitly models security-relevant aspects. It improves upon prior research by providing a flexible framework for modelling attackers and victims at the workload level, enabling direct measurement of security outcomes rather than relying on aggregate system metrics.

## Limitations
Kumo does not model low-level microarchitectural leakage (e.g., cache side channels) or network-level distributed DoS attacks, which is a deliberate scope reduction to focus on system-level security risks. The paper acknowledges that the scheduler models are intentionally simplified to capture dominant placement behaviours relevant to security analysis, rather than replicating proprietary production heuristics in full detail. The evaluation was conducted in simulation, not on real platforms, which may not capture all real-world complexities of actual serverless platforms.

## Appendix: Worked Example
Let's walk through a co-location attack scenario using the DoubleDip scheduler with concrete numbers:

1. We have a serverless platform with 512 workers, each with identical CPU, memory, and storage resources.
2. The platform hosts 200 benign tenants, each owning 20 functions (total 4,000 functions).
3. One tenant is designated as the victim; another is the attacker.
4. Background traffic from 198 benign tenants generates 20,000 invocations.
5. Attackers issue invocations to increase co-location probability with the victim.

For a specific simulation run:
- Random scheduler: 1.2% co-location probability per victim invocation (8.3 co-locations per 1,000 victim invocations)
- DoubleDip scheduler: 0.0% co-location probability (no co-locations observed)
- Helper scheduler: 3.7% co-location probability
- OpenWhisk scheduler: 4.1% co-location probability

Time to first co-location (for schedulers that observe co-location):
- Random: 1,250 time units
- Helper: 3,850 time units
- OpenWhisk: 4,200 time units

This demonstrates how DoubleDip's placement strategy (avoiding workers that have recently hosted the same tenant when alternatives exist) effectively eliminates co-location risks while maintaining low cold-start rates (similar to Helper and OpenWhisk).

## References

- Wei Shao, Khaled Khasawneh, Setareh Rafatirad, Houman Homayoun, Chongzhou Fang, "Kumo: A Security-Focused Serverless Cloud Simulator", arXiv cs.DC, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19787

Tags: #cloud-security #serverless #scheduling #resource-contention #multi-tenancy
