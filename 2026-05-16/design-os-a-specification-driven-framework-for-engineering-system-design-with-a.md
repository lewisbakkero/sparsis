---
title: "Design-OS: A Specification-Driven Framework for Engineering System Design with a Control-Systems Design Case"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20151"
---

## Executive Summary
Design-OS introduces a specification-driven workflow for engineering system design that makes requirements explicit and traceable from concept to implementation. It addresses the gap in physical engineering design where AI tools typically operate at solution generation rather than problem framing. Engineers should care because it enables visible, auditable design processes with cross-domain traceability for both physical systems and AI collaboration.

## Why This Matters for Practitioners
If you're building physical systems like mechatronics or control systems, Design-OS solves the problem of implicit requirements that cause rework later in the process. The paper shows that when requirements aren't explicitly defined upfront (as with the Quanser and SimpleFOC platforms), teams waste time debugging why a solution doesn't meet specifications. You should implement Design-OS's five-stage workflow to establish requirements as the binding contract between human designers and AI agents before committing to implementation. This prevents the common pitfall where teams "design forward" without traceability, forcing costly redesigns when specifications aren't met, especially important for safety-critical systems where requirements must be verifiable.

## Problem Statement
Today's engineering design resembles building a house without blueprints: you start hammering nails in the direction you think the house should go, then realise you should have had a foundation first. The paper cites that requirements remain "implicit" and traceability from "intent to parameters largely absent" in engineering system design. This is especially problematic for physical systems (mechatronic, control, embedded) where AI-assisted tools enter "at solution generation rather than at problem framing," compounding the problem through ad hoc workflows that lack auditability.

## Proposed Approach
Design-OS organises engineering system design into five stages: concept definition → literature survey → conceptual design → requirements definition → design definition. Specifications serve as the shared contract between human designers and AI agents; each stage produces structured artifacts that maintain traceability and support agent-augmented execution. The workflow makes the design process visible and auditable, extending specification-driven orchestration from software to physical engineering system design.

```python
def design_os_workflow():
    # Stage 1: Concept Definition
    context = capture_context(mission, stakeholders, constraints)
    
    # Stage 2: Literature Survey
    literature = conduct_literature_search(
        topics=[requirements, systematic_design, ai_assisted_design],
        ai_assistants=[claude, gemini]
    )
    
    # Stage 3: Conceptual Design
    conceptual_design = develop_concept(
        literature_notes=literature,
        design_objectives=[do1, do2, do3]
    )
    
    # Stage 4: Requirements Definition
    requirements = define_requirements(
        conceptual_design=conceptual_design,
        domain_decomposition=["mechanical", "electrical", "software"]
    )
    
    # Stage 5: Design Definition
    execution_plan = derive_execution_plan(
        requirements=requirements,
        traceability_table=build_traceability_table()
    )
    return execution_plan
```

## Key Technical Contributions
Design-OS's core innovation lies in its structured workflow that enables traceability across domains. 

1. **Domain decomposition for cross-domain traceability**: The framework explicitly decomposes systems into mechanical, electrical, and software domains before requirements are defined. This ensures traceability from performance specifications through domain-specific parameters to implementation. For example, the requirement ζ = 0.7 (damping ratio) traces through the software domain (pole placement → gain K), then to the electrical domain (motor torque equation → voltage Vm), and finally to the mechanical domain (pendulum dynamics).

2. **Requirement IDs that bind intent to implementation**: Requirements are structured with IDs (e.g., REQ-01 for damping ratio), each linking to verification methods and design parameters. This enables the traceability table (REQ → param → verif.) that the paper demonstrates with the inverted pendulum case study.

3. **Feasibility gates that validate platform choices**: Before requirements definition, Design-OS conducts a feasibility gate comparing implementation options (e.g., SimpleFOC reaction wheel vs. Quanser Furuta platforms). The paper shows this stage produced a cost comparison (SimpleFOC: $100-200 vs. Quanser: $5k+) and verified that "the SimpleFOC platform costs ∼50× less and removes licensing constraints."

4. **AI orchestration at problem framing**: Unlike most AI design tools that operate during solution generation, Design-OS positions AI agents to assist during literature survey (Stage 2) and conceptual design (Stage 3), ensuring AI supports problem framing rather than just solution generation.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The paper demonstrates the workflow on two inverted pendulum platforms (Quanser and SimpleFOC), verifying that both satisfy identical requirements through different implementations. Table 2 shows specific results:
- REQ-01 (ζ = 0.7): simulation ζ = 0.700 (pass)
- REQ-02 (ωn = 4 rad/s): simulation ωn = 4.000 rad/s (pass)
- REQ-03 (|α| < 15°): simulation |α|max = 2.9° (pass)
- REQ-04 (|Vm| < 10 V): simulation |Vm|max = 0.5 V (pass)

The authors compare implementations using a feasibility gate (Table 4), showing the SimpleFOC platform costs ∼50× less than Quanser SRV02 while achieving comparable performance through different mechanical configurations (reaction wheel vs. arm-driven Furuta). The simulation verification confirms all requirements were met without statistical significance testing being reported.

## Related Work
Design-OS positions itself relative to three categories:
1. **Requirements-driven design**: Aligns with Microsoft's "specifications as version control for your thinking" and GitHub Spec Kit, but extends these principles from software to physical engineering.
2. **Systematic design frameworks**: Draws on Pahl & Beitz's four-phase model (task clarification, conceptual design, embodiment design, detail design) and VDI 2206, but compresses them into a lightweight workflow optimised for specification-driven engineering.
3. **AI-assisted design**: Extends specification-driven multi-agent frameworks (MetaGPT, ChatDev) to physical engineering, addressing the gap that "specification-driven orchestration of AI agents for physical engineering system design has received little attention."

## Limitations
The paper doesn't explicitly discuss limitations in the provided text. However, the demonstrated case study is confined to control systems (inverted pendulum), and the authors note that "traceability from requirements to design and verification is central" but "achieving full traceability in practice remains challenging and tool-dependent." The feasibility gate comparison assumes parameter estimation through system identification (for SimpleFOC), which might not be feasible for all physical systems without detailed modelling.

## Appendix: Worked Example
Let's walk through how REQ-01 (damping ratio ζ = 0.7) traces through the Design-OS workflow with concrete numbers:

1. **Conceptual design** (Stage 3) identifies the control strategy: state-feedback control. The design objective (DO-1) determines that state-feedback control must satisfy performance specifications.

2. **Requirements definition** (Stage 4) specifies REQ-01: "Damping ratio ζ = 0.7" (performance requirement). This requirement links to the software domain through the design task: "Gains from requirements (pole placement)" (REQ-11).

3. **Control design** (software domain): The damping ratio ζ = 0.7 and natural frequency ωn = 4 rad/s (REQ-02) determine closed-loop pole locations at p = -σ ± jωd where σ = ζωn = 2.8 and ωd = ωn√(1-ζ²) = 2.86 rad/s.

4. **Gain calculation**: The state-feedback gain vector K is computed analytically from the desired characteristic polynomial. For the Quanser SRV02 platform (using the linear state-space matrices from Eq. 4), the gain vector K is calculated to satisfy the pole placement.

5. **Verification**: The simulation (Table 2) confirms ζ = 0.700, satisfying REQ-01. The voltage command Vm (electrical domain) from the control law is verified to be |Vm|max = 0.5 V (REQ-04), and the pendulum deflection α (mechanical domain) is verified to be |α|max = 2.9° (REQ-03).

6. **Traceability**: The traceability table (Table 3) shows REQ-01 flows through software → electrical → mechanical domains, with verification via simulation of the step response.

This traceable path from specification (ζ = 0.7) through the gain calculation (K) to verification (simulation results) demonstrates Design-OS's core mechanism for maintaining visibility and accountability throughout the design process.

## References

- H. Sinan Bank, Daniel R. Herber, Thomas H. Bradley, "Design-OS: A Specification-Driven Framework for Engineering System Design with a Control-Systems Design Case", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20151

Tags: #systems-engineering #specification-driven-design #cross-domain-traceability #ai-assisted-design #control-systems
