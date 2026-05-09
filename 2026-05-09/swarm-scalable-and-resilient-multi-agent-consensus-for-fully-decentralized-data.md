---
title: "SWARM+: Scalable and Resilient Multi-Agent Consensus for Fully-Decentralized Data-Aware Workload Management"
venue: "arXiv cs.DC"
paper_url: "https://arxiv.org/abs/2603.19431"
---

## Executive Summary
SWARM+ solves the bottleneck of centralized workflow orchestration in large-scale, data-intensive scientific computing by introducing a fully decentralized multi-agent system. It achieves 97, 98% faster job scheduling (60× speedup) while maintaining 99% job completion under single-agent failures, all through hierarchical consensus and data-aware cost modelling.

## Why This Matters for Practitioners
If you manage scientific workflows across 50+ geographically distributed sites (e.g., climate modelling across national supercomputing centres), SWARM+ directly addresses three pain points in your current stack:
- **Scalability**: SLURM/HTCondor bottlenecks become critical at 1,000+ agents; SWARM+ scales to 1,000 agents with linearly reduced coordination overhead (O(log n) vs O(n²)).
- **Resilience**: When a site fails, your current system halts workloads; SWARM+ maintains 99% job completion under single failures and degrades gracefully (7.5% latency impact at 50% failures).
- **Data locality**: Your current scheduler ignores data transfer costs; SWARM+ integrates DTN connectivity scores into cost modelling, directly reducing data movement overhead for AI training workloads.

**Actionable step**: Implement hierarchical agent groups (Level 0 resource agents, Level 1 coordinators) with adaptive quorum in your orchestration layer. Use the published LRU caching optimisation to reduce selection latency by 97% in steady-state workloads.

## Problem Statement
Today’s scientific workflows resemble a city relying on a single central traffic control tower: it manages all cars (jobs), but a tower failure halts the entire city. With thousands of sensors and resources spread across global sites (like a city spanning multiple countries), a central tower becomes a single point of failure, scalability chokepoint, and data transfer nightmare. Workloads must move data across continents at 100ms+ RTT, yet current schedulers treat all resources equally, like ignoring traffic congestion when routing cars.

## Proposed Approach
SWARM+ uses a tree-structured agent hierarchy where resource agents (Level 0) delegate to coordinator agents (Level 1). Agents reach consensus via a three-phase protocol (Proposal → Prepare → Commit) within their groups, reducing communication from O(n²) to O(log n). Data-aware cost modelling prioritises jobs with local data access. The system scales horizontally by adding agent groups and vertically by adding hierarchy levels.

```python
def adaptive_quorum(nlive):
    """Calculate dynamic quorum based on live agents (from §II-B)"""
    return (nlive + 1) // 2  # Floor division for integer quorum
```

## Key Technical Contributions
SWARM+’s novelty lies in three mechanisms that solve scalability, resilience, and data-awareness without centralized control:

1. **Hierarchical consensus with aggregation**: Instead of flat mesh communication (O(n²)), agents compute *aggregated capacity* (max child CPU/RAM) and *unified DTN connectivity* (union of endpoints) at parent level. This reduces per-job feasibility checks from O(m) to O(1), avoiding global state queries. For 100 agents, this cuts communication complexity from 10,000 to ~100 messages.

2. **Adaptive quorum without reconfiguration**: Quorum size dynamically adjusts via `q(t) = ⌊(nlive(t)+1)/2⌋` (§II-D). When agents fail, quorum shrinks proportionally (e.g., from 10 to 5 agents), preventing system stalls. Crucially, this maintains safety (no conflicting assignments) and liveness (progress continues) for `nlive ≥ 2f + 1` (f = failures), eliminating the need for costly reconfiguration protocols.

3. **Data-aware cost modelling with connectivity penalty**: Cost for job `j` at agent `a` includes `cost = Σ(wr·ur) + (1 + β·(1 − s))` (§II-C). `s` is average DTN connectivity score (0, 1) for required data endpoints. With `β=1.0` (default), agents with direct DTN access (s=1) pay cost=1.0, while those with no access (s=0) pay cost=2.0, prioritising data locality without manual tuning.

## Experimental Results
Evaluated on FABRIC testbed across 10 sites (110 agents), SWARM+ outperformed baseline SWARM (prior mesh-based system) with statistically significant results (p < 10⁻²³, Cohen’s d > 6.4):

| **Metric**               | **SWARM**       | **SWARM+**      | **Improvement** |
|--------------------------|-----------------|-----------------|-----------------|
| Mean Selection Time      | 40.03 ± 6.41s   | 1.20 ± 0.04s    | 97.0%           |
| Mean Scheduling Latency  | 325.22 ± 27.70s | 5.41 ± 0.44s    | 98.3%           |
| P95 Selection Time       | 85.47 ± 14.18s  | 1.54 ± 0.10s    | 98.2%           |
| Job Completion (50% failure) | N/A           | >99%            | 7.5% latency impact |

*Note*: Improvements hold across WAN RTTs (2.36, 68.33ms). SWARM+ achieved 1000-agent scalability with equal workload distribution across hierarchy levels (Fig. 1).

## Related Work
SWARM+ builds on prior work SWARM [11], which used flat mesh PBFT consensus for distributed job selection but suffered O(n²) complexity. It improves over fault-tolerant workflows [8], [10] that assume reliable central managers, and addresses gaps in decentralized coordination [12] by adding data-awareness and hierarchical scaling. Unlike centralized systems (SLURM, HTCondor), it requires no global coordination.

## Limitations
- **Scale limits**: Tested only up to 1,000 agents; larger scales unverified.
- **Data model**: Uses synthetic workloads with biased job distribution (exponent 3); real-world job mixes may vary.
- **Failure scenarios**: Focuses on agent failures, not network partition failures (e.g., site isolation).
- **Authors note**: "No testing of >1000 agents" (§IV-A), so scalability beyond this is extrapolated.

## Appendix: Worked Example
Consider a job `J` requiring 2 CPUs and 8 GB RAM, with data endpoints at DTNs `D1` (score=0.85) and `D2` (score=0.60). Two agents compete:

- **Agent A** (Level 0, resource agent):  
  CPU available: 4, RAM: 16 GB, DTN scores: `D1=0.9`, `D2=0.7`  
  Resource utilisation: `ur = (2/4) + (8/16) = 0.5 + 0.5 = 1.0`  
  Connectivity score: `s = (0.9 + 0.7)/2 = 0.8` → `penalty = 1 + 1.0×(1−0.8) = 1.2`  
  **Cost** = `0.5×0.5 + 0.5×0.5 + 1.2 = 0.25 + 0.25 + 1.2 = 1.70`  

- **Agent B** (Level 0, resource agent):  
  CPU available: 3, RAM: 12 GB, DTN scores: `D1=0.5`, `D2=0.4`  
  Resource utilisation: `ur = (2/3) + (8/12) ≈ 0.67 + 0.67 = 1.34`  
  Connectivity score: `s = (0.5 + 0.4)/2 = 0.45` → `penalty = 1 + 1.0×(1−0.45) = 1.55`  
  **Cost** = `0.5×0.67 + 0.5×0.67 + 1.55 ≈ 0.67 + 1.55 = 2.22`  

Agent A wins (cost 1.70 < 2.22), matching the paper’s data-aware cost model. *Note: Paper didn’t specify `wr` weights, so default 0.5 each for CPU/RAM used here (§II-C).*

## References

- Komal Thareja, Krishnan Raghavan, Anirban Mandal, Ewa Deelman, "SWARM+: Scalable and Resilient Multi-Agent Consensus for Fully-Decentralized Data-Aware Workload Management", arXiv cs.DC, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19431

Tags: #distributed-systems #scientific-computing #multi-agent #hierarchical-consensus #data-aware-scheduling
