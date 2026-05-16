---
title: "Agentic Business Process Management: A Research Manifesto"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.18916"
---

## Executive Summary
This paper introduces Agentic Business Process Management (APM), a paradigm shift from traditional BPM that governs autonomous agents within organizational processes. APM systems constrain and align agent autonomy through explicit process frames, ensuring agents' actions remain aligned with organizational goals while maintaining operational flexibility. Practitioners should care because this framework addresses critical risks in deploying AI agents in production systems, where uncontrolled autonomy leads to compliance violations, unpredictable behaviour, and integration challenges.

## Why This Matters for Practitioners
If you're building or maintaining production systems with AI agents (especially LLM-based agents), this paper suggests you must implement process awareness as a core requirement, not an afterthought. Traditional BPM tools like BPMN won't suffice for managing autonomous agents; your architecture must support four key capabilities. Specifically, for any agent that makes business decisions (e.g., procurement, customer service), you should:
1. Implement framing mechanisms to constrain agent behaviour within defined process boundaries
2. Build explainability features so you can trace agent decisions during audits
3. Create conversational interfaces for human oversight and intervention
4. Design for self-modification that aligns with organizational goals rather than individual agent interests
Without these, your agents risk violating compliance rules, making poor business decisions, or creating untestable technical debt, issues already documented in the paper's examples of agent governance failures.

## Problem Statement
Imagine a city where autonomous vehicles operate without traffic rules, GPS navigation, or centralized control. Drivers (agents) could ignore traffic lights, take arbitrary routes, and make decisions based on personal preferences rather than city safety goals. This is the current reality for many AI agents in production: they're "driving" independently without adherence to organizational processes, leading to unpredictable behaviour, compliance violations, and operational chaos. Traditional BPM treats agents as passive components in workflows, not active participants with their own goals and decision-making capabilities, resulting in the kind of governance gaps that risk business continuity.

## Proposed Approach
APM reimagines business processes as systems where agents (human or AI) are primary actors operating within explicit process frames. An APM system consists of two interconnected levels:
1. The macro level: Process-aware management that establishes framing mechanisms
2. The micro level: Agent-oriented execution where agents perceive, reason, and act within their frames

Agents operate through a continuous control loop (perceive → reason → act) with four key capabilities:
- Framing: Constrained autonomy within process boundaries
- Explainability: Articulating decision rationale
- Conversational actionability: Negotiating with other agents
- Self-modification: Adapting to improve process outcomes

```python
def a_p_m_agent(perceive, reason, act):
    # Framing: Agent's internal models are constrained by process frame
    mental_model = frame.process_awareness(perceive.context)
    
    # Reasoning: Agent uses mental and intentional models to decide actions
    action = reason(mental_model, intentional_model)
    
    # Explainability: Generate rationale for decisions
    rationale = explain(action)
    
    # Conversational actionability: Communicate with other agents
    if not action.is_compliant(mental_model):
        action = negotiate_with_other_agents(action)
    
    # Self-modification: Continuously improve based on feedback
    if feedback.from_process_manager():
        self_modify()
    
    # Act: Execute within frame constraints
    return act(action)
```

## Key Technical Contributions
The paper's core contribution is the conceptual framework for APM, not a novel algorithm or system implementation. The true innovation lies in how they define the four capabilities and their interplay, which resolves a critical gap in agent governance.

1. **Framing as normative specification**: Unlike traditional process languages like BPMN (which focus on operational behaviour), APM's framing mechanism explicitly specifies deontic requirements (obligations, permissions, prohibitions). This is implemented through internal mental models that represent the process reality and intentional models that maintain goal alignment. For example, a procurement agent's framing specifies it must "reject any supplier quote after the deadline" (normative frame) and "record rejection details in the database" (operational frame).

2. **Explainability as auditability**: The paper specifies that explainability requires agents to generate traceable rationale for decisions, not just "this decision was made" but "I rejected this quote because it was submitted after the deadline (14:30 UTC), violating the procurement policy P-2023-09". This transforms explainability from a theoretical concept to an operational requirement for compliance.

3. **Conversational actionability as dynamic coordination**: This capability enables agents to negotiate with other process-aware agents in real time. For instance, if a supplier agent can't fulfill a contract due to unexpected constraints, it can proactively negotiate with the buyer agent through a standardized protocol (like MCP or ACP) to adjust terms without requiring external intervention.

4. **Self-modification within frame constraints**: The paper distinguishes between uncontrolled agent self-modification (which could lead to non-compliance) and frame-constrained self-modification that aligns with organizational goals. Agents can modify their behaviour based on feedback, but only within the boundaries defined by their process frame.

## Experimental Results
This is a research manifesto rather than an empirical study, so no experimental results are reported. The authors focus on conceptual foundations and identify research challenges requiring further development. The paper acknowledges that the practical implementation of these capabilities requires advances in BPM, AI, and multi-agent systems, particularly in achieving "responsible and effective management" of agents in production.

## Related Work
The paper positions APM as a bridge between three established fields:
- Traditional Business Process Management (BPM), which traditionally treats processes as workflows rather than agent-centered systems
- Multi-Agent Systems (MAS) research, particularly normative MAS that focuses on regulating agent behaviour through norms
- Agentic AI research, which currently lacks a governance framework connecting to established organizational practices

The authors clarify that APM differs from previous work on "AI-augmented BPM" [23] by treating agents as primary functional entities rather than augmenting existing BPM systems. APM also extends the concept of process choreographies [11] by introducing explicit framing mechanisms rather than just coordination.

## Limitations
The paper acknowledges that APM requires significant advances in three areas: BPM, AI, and multi-agent systems. Specifically:
- There's no concrete implementation of the framing mechanism, only conceptual description
- The paper doesn't address how to handle frame conflicts when agents have competing goals
- It doesn't specify how to measure the effectiveness of APM capabilities in real-world systems
- The authors note that "the realization and delivery of these four key capabilities involves a series of research challenges"

From a practitioner perspective, the biggest limitation is that APM remains a conceptual framework without implementation guidance for production systems. Engineers must still solve how to instrument agents with process awareness and frame constraints in real-world scenarios.

## Appendix: Worked Example
Let's walk through a supplier onboarding process in an APM system. The procurement process defines a frame with these requirements:
- Agent must reject quotes after 14:30 UTC (normative frame)
- Agent must log rejection details in the database (operational frame)
- Agent must notify suppliers within 24 hours of rejection (normative frame)

**Scenario**: A supplier agent submits a quote at 14:35 UTC for a procurement process with deadline 14:30 UTC.

1. **Perception**: The buyer agent's perception module senses the quote submission time (14:35 UTC)
2. **Reasoning**: The reasoning module consults the process frame (which specifies deadline 14:30 UTC) and determines the quote is late
3. **Framing**: The internal mental model confirms the deadline constraint; the intentional model maintains the goal to "process all quotes within deadline"
4. **Explainability**: The agent generates rationale: "Quote rejected because submitted at 14:35 UTC, exceeding procurement deadline of 14:30 UTC (policy P-2023-09)"
5. **Conversational actionability**: The agent sends a standard message to the supplier: "Your quote was rejected due to late submission (14:35 UTC > 14:30 UTC). Please resubmit by 14:30 UTC tomorrow."
6. **Self-modification**: The agent logs the rejection and notes the supplier's pattern of late submissions; it modifies its future interaction strategy to send reminder notifications 1 hour before deadline for this supplier

This example shows how the four capabilities work together to maintain process awareness without requiring external intervention.

## References

- Diego Calvanese, Angelo Casciani, Giuseppe De Giacomo, Marlon Dumas, Fabiana Fournier, Timotheus Kampik, Emanuele La Malfa, Lior Limonad, Andrea Marrella, Andreas Metzger, Marco Montali, Daniel Amyot, Peter Fettke, Artem Polyvyanyy, Stefanie Rinderle-Ma, Sebastian Sardiña, Niek Tax, Barbara Weber, "Agentic Business Process Management: A Research Manifesto", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.18916

Tags: #business-process-management #agentic-ai #multi-agent-systems #process-awareness #agent-governance
