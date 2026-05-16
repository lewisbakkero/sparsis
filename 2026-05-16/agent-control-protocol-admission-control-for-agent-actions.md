---
title: "Agent Control Protocol: Admission Control for Agent Actions"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.18829"
---

## Executive Summary
The Agent Control Protocol (ACP) solves a critical gap in governing autonomous agents by enforcing behavioral properties over execution traces rather than individual requests. ACP achieves this through a temporal admission control mechanism that combines static risk scoring with stateful signals via a LedgerQuerier abstraction, providing deterministic enforcement rather than advisory signals. For practitioners, ACP offers sub-microsecond latency (739, 832 ns) with verifiable safety guarantees across 4.3 billion states, making it suitable for production systems where stateless policy engines fail to prevent harmful patterns from individually valid requests.

## Why This Matters for Practitioners
If you're building or deploying autonomous agents in production systems, particularly in financial, government, or critical infrastructure domains, this paper reveals a critical blind spot in current security practices. Stateless policy engines (like OPA) evaluate each request in isolation, making them incapable of preventing harmful behavioral patterns that emerge from sequences of individually valid requests. ACP provides a structured protocol-level mechanism to add stateful admission control above existing policy engines, with measurable performance (739, 832 ns decision latency) and formal safety guarantees (verified across 4.29 billion states). Practitioners should implement ACP or a compatible protocol for any production system where agent actions need to be constrained over time windows rather than per-request, especially in environments with high-risk financial transactions, document processing, or multi-organisation orchestration.

## Problem Statement
Current policy engines operate like a single-point-of-truth traffic camera that can only monitor individual cars, not entire traffic flows. Each car (request) appears valid in isolation, but the system cannot detect dangerous patterns like multiple cars (requests) accelerating together (behavioral sequences) that collectively cause accidents (harmful outcomes). Similarly, stateless policy engines evaluate each agent request in isolation, failing to prevent harmful sequences that emerge from individually valid requests, such as a financial agent executing a series of small transfers that cumulatively drain an account.

## Proposed Approach
ACP operates as a multi-layer admission control framework that sits above traditional policy engines. It introduces a LedgerQuerier abstraction separating decision logic from state management, allowing the protocol to be implemented with various state backends. The core admission process follows a six-step flow: identity check, capability check, policy check, admission decision, execution token generation, and ledger recording. ACP-RISK-3.0, the risk scoring engine, prevents cross-context interference by using PatternKey(agentID, capability, resource) to scope rate-based signals to specific interaction contexts.

```python
def aco_evaluate(agent, action):
    # Step 1: Identity check
    if not verify_identity(agent):
        return DENY
    
    # Step 2: Capability check
    if not verify_capability(agent, action):
        return DENY
    
    # Step 3: Policy check (with stateful risk scoring)
    risk_score = risk_engine.score(agent, action)
    
    # Step 4: Decision
    if risk_score >= THRESHOLD:
        return DENY
    elif risk_score >= ESCALATION_THRESHOLD:
        return ESCALATE
    else:
        return ADMIT
```

## Key Technical Contributions
ACP introduces several novel mechanisms that differentiate it from stateless alternatives:

1. **Context-scoped anomaly enforcement**: ACP-RISK-3.0 eliminates the cross-context interference vulnerability in ACP-RISK-2.0 by using PatternKey(agentID, capability, resource) to scope rate-based signals to interaction contexts. This ensures that repeated behaviour within a single context triggers denial at the same threshold (Experiment 8), while benign high-frequency operations in one context no longer inadvertently elevate risk scores in unrelated high-value contexts, resolving the bounded state-mixing vulnerability.

2. **Boundary Activation Rate (BAR) and BAR-Monitor**: ACP formalises the measurement of governance boundary activation through BAR, which detects when enforcement is active but never exercised (deviation collapse). BAR-Monitor uses ∆BAR (progressive decline toward collapse) to provide early warnings, firing at Batch 2 (BAR = 0.57, ∆BAR = -0.25) three batches before threshold collapse (BAR = 0.00 at Batch 5), enabling proactive governance restoration.

3. **Formal verification of safety properties**: The paper demonstrates that ACP's safety properties (admission determinism, cooldown monotonicity, per-agent isolation) can be verified with TLC model checking over 11 invariants + 4 temporal properties with zero violations across 4,294,930,695 distinct states for two-agent configurations. This provides mathematical guarantees of correctness across all reachable states, unlike traditional security testing.

4. **Adversarial robustness against knowledge-aware attacks**: ACP demonstrates resilience against adversaries with complete knowledge of the risk formula, suppressing BAR to 0.00 while individual requests remain policy-compliant. BAR-Monitor detects the regime shift via ∆BAR three batches before collapse, and counterfactual evaluation confirms structural enforcement capacity (BARC = 1.00), forming a coherent defence against knowledge-aware evasion.

## Experimental Results
Under a 500-request financial transfer workload with RS=35 (each request individually valid), a stateless engine with identical scoring approves all 500 requests, while ACP limits autonomous execution to 2 out of 500 (0.4%), demonstrating the structural gap between stateless and stateful admission. Decision evaluation runs at 739, 832 ns (p50; Table 12), with throughput reaching 1,720,000 req/s (baseline, InMemoryQuerier) and 920,000 req/s at 10 concurrent workers (Table 13). Safety properties were verified via TLC model checking over 11 invariants + 4 temporal properties with zero violations across 4,294,930,695 distinct states. BAR-Monitor validated ∆BAR as a proactive detection mechanism, with AlertTrend firing at Batch 2 (BAR = 0.57, ∆BAR = -0.25) three batches before threshold collapse (BAR = 0.00 at Batch 5).

## Related Work
ACP builds on but differentiates from existing access control models: RBAC (designed for human roles), Zero Trust (focused on network access), and OPA (policy engine for individual requests). Unlike these approaches, ACP is specifically designed for autonomous agents with cryptographic identity, verifiable dynamic delegation, and decision/execution separation. ACP extends SPIFFE/SPIRE's cryptographic workload identity to add capability scoping and governance, while OPA can serve as the policy evaluation engine inside ACP's Step 3. ACP is not a replacement for these systems but a protocol-level layer that operates above them.

## Limitations
The paper acknowledges limitations in its evaluation scope: it focuses on single-agent and two-agent scenarios but does not test ACP under complex multi-agent environments with diverse capabilities. The adversarial evaluation (Experiment 10) demonstrates knowledge-aware evasion but doesn't explore attacks that manipulate the LedgerQuerier state backend directly. The paper also doesn't provide guidance on optimal risk threshold tuning for different application contexts, though it does conduct a threshold sensitivity analysis (Experiment 11). Additionally, the evaluation assumes a clean state for the ledger, which may not reflect real-world scenarios with concurrent agent interactions.

## Appendix: Worked Example
Let's walk through a single financial transaction sequence using ACP-RISK-3.0 with concrete numbers:

1. An agent (ID: "bankA_agent_123") requests to transfer $100 from account X to account Y (resource: "account_X", capability: "transfer").
2. The agent presents a Capability Token (ACP-CT-1.0) signed by Bank A and demonstrates possession of the associated private key (ACP-HP-1.0).
3. The risk engine queries the LedgerQuerier using PatternKey("bankA_agent_123", "transfer", "account_X").
4. The ledger shows 2 previous transfers within the last 5 minutes (rate: 2/5min), below the threshold of 3 actions (Experiment 7).
5. The risk score is calculated as 0.35 (RS=35), below the DENY threshold of 0.70.
6. The agent is admitted (ADMIT), and an Execution Token (ACP-EXEC-1.0) is generated for this single use.
7. The ledger records the transaction with a signed audit entry (ACP-LEDGER-1.3).
8. If the agent requests another transfer within the same 5-minute window for the same account, the ledger shows 3 transfers (rate: 3/5min), triggering DENY (Experiment 8).
9. If the agent requests a transfer to account Z (different resource), the ledger shows 0 previous transfers for this resource, so a single transfer is permitted (context-scoped prevention of cross-context interference).

## References

- Marcelo Fernandez, "Agent Control Protocol: Admission Control for Agent Actions", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.18829

Tags: #security #agent-governance #multi-agent #admission-control #stateful-policy #risk-scoring #formal-verification
