---
title: "PlanTwin: Privacy-Preserving Planning Abstractions for Cloud-Assisted LLM Agents"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.18377"
---

## Executive Summary
PlanTwin introduces a privacy-preserving architecture that enables cloud-assisted LLM agents to plan over local environments without exposing raw context, such as source code or credentials. It achieves full sensitive-item non-disclosure (SND = 1.0) while maintaining high planning quality (PQS > 0.79 for most planners), making it essential for enterprise applications where privacy leaks are unacceptable.

## Why This Matters for Practitioners
If you're building or deploying cloud-assisted LLM agents for code generation, system management, or enterprise workflows, you're likely exposing raw context (including credentials, source code, and metadata) to cloud models, creating significant privacy risks. PlanTwin provides a practical solution that allows you to integrate cloud-scale planning intelligence while keeping sensitive information local. Specifically, you should implement a similar projection pipeline in your agent system's local layer, replacing raw context transmission with a schema-constrained abstract graph that preserves only necessary planning information. This is especially critical for systems handling source code repositories, internal documentation, or sensitive metadata where direct exposure to cloud models would violate compliance requirements or introduce security risks.

## Problem Statement
Current agent systems expose raw context to cloud planners like handing over your home's entire blueprint to a contractor who's only allowed to work in the kitchen, you restrict what they can do (the kitchen), but they still see the entire layout, allowing them to infer sensitive details about your family or routines. Existing solutions either expose everything (full context upload), remove only obvious tokens (PII redaction), or avoid cloud exposure entirely (local-only inference), missing the middle ground where the planner needs only the right abstraction.

## Proposed Approach
PlanTwin implements a privacy-preserving architecture where the local environment is transformed into a "sanitized digital twin" before being sent to the cloud. The system consists of three main components:
1. A local projection pipeline that converts raw context into a schema-constrained abstract graph
2. A cloud planner that operates solely on this abstract graph
3. A local gatekeeper that enforces safety policies and cumulative disclosure budgets

The four-stage privacy projection pipeline (Πextract → Πredact → Πgeneralize → Πschema) creates a structurally bounded abstraction that prevents reconstructive inference while preserving planning-relevant structure.

```python
def privacy_projection_pipeline(raw_context):
    # Stage 1: Extract structure and affordances
    structured_objects = extract_structure(raw_context)
    
    # Stage 2: Redact sensitive entities
    redacted_objects = redact_sensitive_entities(structured_objects)
    
    # Stage 3: Generalise precise values
    generalized_objects = generalize_values(redacted_objects)
    
    # Stage 4: Project into fixed schema
    abstract_graph = project_to_schema(generalized_objects)
    
    return abstract_graph
```

## Key Technical Contributions
PlanTwin's innovations differ fundamentally from prior approaches in how they handle privacy at the planning level:

1. **The four-stage privacy projection pipeline** - Unlike token-level masking or PII redaction, this pipeline creates a structurally bounded abstraction. The extraction stage identifies object types (e.g., code_file, config) using a local small language model, redaction masks sensitive entities with deterministic rules (not just regex), generalisation replaces precise values (e.g., byte sizes with "small/medium/large"), and schema projection ensures the output is a fixed JSON structure with no raw text. This prevents the cloud planner from inferring original context while preserving planning-relevant structure. The key difference from prior work is that it operates on the structural level rather than token level.

2. **Capability-based planning protocol** - PlanTwin formalizes the privacy-utility trade-off as a capability granularity problem. The cloud planner operates over a bounded set of local capabilities (e.g., "compare modules" rather than "read auth_service.py"), which directly controls the disclosure budget. This is fundamentally different from previous approaches that focused solely on execution isolation or access control; PlanTwin controls the abstraction level of what the planner observes. The authors demonstrate that capability design governs both planning effectiveness and disclosure cost, with the optimal operating point near the orchestration boundary.

3. **Disclosure budgeting with (k,δ)-anonymity** - The local gatekeeper enforces cumulative disclosure budgets per object, ensuring that (k,δ)-anonymity is maintained across multiple planning turns. This means the planner can't re-identify objects even across multiple interactions, as the system limits how much structural information can be revealed. For example, if an object has a disclosure budget of 3, it can be referenced in at most three planning steps before requiring additional verification. This is a significant improvement over previous approaches that handled privacy on a per-request basis with no cumulative control, which would allow re-identification through multiple interactions.

## Experimental Results
PlanTwin was evaluated on 60 agentic tasks across ten domains using four cloud planners:

- Achieved full sensitive-item non-disclosure (SND = 1.0) - all sensitive items were properly masked
- Three of four planners achieved PQS > 0.79 (PQS = Planning Quality Score)
- Full pipeline incurred less than 2.2% utility loss compared to full-context systems
- Maintained planning quality while eliminating privacy vulnerabilities

The paper compared against:
- Full-context upload (raw context transmission)
- PII redaction (token-level masking)
- Local-only inference (no cloud exposure)
- TEE (Trusted Execution Environments)

The results demonstrate that PlanTwin achieves strong privacy without sacrificing planning quality, making it suitable for enterprise deployment where privacy leaks are unacceptable.

## Related Work
PlanTwin positions itself against four existing approaches:
1. Full-context upload (e.g., Claude Code, OpenAI Codex) - exposes raw context, relying on trust
2. PII redaction (e.g., Microsoft Presidio) - removes only obvious tokens, misses domain-specific secrets
3. Local-only inference (e.g., PrivateGPT) - avoids cloud exposure but sacrifices planning capability
4. TEE-based approaches (e.g., Intel TDX) - protects data during computation but doesn't address the semantic question of how much information the planner should observe

PlanTwin differs by controlling the abstraction level of what the planner observes rather than just restricting what the agent can do or protecting data during computation. The authors explicitly state that their work targets the "planner-visible observation surface itself" rather than execution isolation or access control.

## Limitations
The authors acknowledge that PlanTwin requires a local small language model for extraction, which may add processing overhead for resource-constrained environments. The paper doesn't test the approach on extremely large codebases (over 1 million lines of code), though the authors suggest it would scale well. They also note that the system doesn't eliminate structural inference risk, though it mitigates it through bounded edge vocabularies and controlled relational disclosure.

My assessment: The paper doesn't address the cost of implementing the four-stage pipeline in production systems with tight latency constraints (e.g., <50ms response times), which could be a barrier for real-time agent systems. Additionally, it doesn't consider how to handle dynamic environments where the codebase evolves rapidly between planning steps.

## Appendix: Worked Example
Consider a code review task where a developer asks: "Review the authentication module for security issues and compare it with the payment module."

The local environment contains:
- auth_service.py (with sensitive API keys)
- payment_handler.py
- .env file with credentials
- git history with developer identities

After PlanTwin's projection pipeline:

Stage 1 (extraction):
- The local SLM identifies auth_service.py as a code_file with semantic_class = authentication_module
- payment_handler.py is identified as code_file with semantic_class = payment_module
- The .env file is identified as secret_container

Stage 2 (redaction):
- All API keys, paths, and developer identities are masked (e.g., "API_KEY=abc123" becomes "API_KEY=REDACTED")

Stage 3 (generalisation):
- The API key count is generalised to "low" sensitivity
- The file size of auth_service.py is generalised to "medium"
- The age of the code (based on git history) is generalised to "recent"

Stage 4 (schema projection):
- The output for auth_service.py becomes:
  {"object_id": "artifact_1",
   "kind": "code_file",
   "semantic_class": "authentication_module",
   "sensitivity": "low",
   "freshness": "recent",
   "size_bucket": "medium",
   "contains": ["security_check", "authentication"],
   "usable_for": ["review", "compare"]}
- Similarly for payment_handler.py

The cloud planner sees these abstract representations and can reason about the relationship between the authentication and payment modules without ever seeing the actual code content. The local gatekeeper enforces a disclosure budget of 2 for each module, meaning the cloud planner can reference each module in at most two planning steps before requiring additional verification.

## References

- Guangsheng Yu, Qin Wang, Rui Lang, Shuai Su, Xu Wang, "PlanTwin: Privacy-Preserving Planning Abstractions for Cloud-Assisted LLM Agents", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.18377

Tags: #security-and-privacy #privacy-preserving-computing #agent-based-systems #capability-based-access-control #schema-constrained-interfaces
