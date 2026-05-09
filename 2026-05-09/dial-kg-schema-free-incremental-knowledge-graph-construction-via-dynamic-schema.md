---
title: "DIAL-KG: Schema-Free Incremental Knowledge Graph Construction via Dynamic Schema Induction and Evolution-Intent Assessment"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20059"
---

## Executive Summary
DIAL-KG introduces a closed-loop framework for incremental knowledge graph construction that operates without predefined schemas by dynamically evolving its schema through a Meta-Knowledge Base (MKB). It solves the critical problem of maintaining up-to-date, accurate knowledge graphs in production systems where data arrives continuously and knowledge evolves over time, rather than requiring complete reconstructions after each update.

## Why This Matters for Practitioners
If you're maintaining a knowledge graph in production for search, recommendation, or question-answering systems, you're likely facing an untenable trade-off between accuracy and operational cost. Traditional schema-based approaches require costly full reconstructions when adding or modifying knowledge, while schema-free methods struggle to handle knowledge evolution without discarding temporal context. DIAL-KG enables you to implement a self-sustaining knowledge graph that: (1) adds new facts with 97.5% precision (per window), (2) deprecates obsolete facts with 98.6% precision based on explicit textual evidence, and (3) automatically refines its schema through evolutionary feedback, eliminating the need for manual schema engineering and reducing operational overhead in continuously evolving domains.

## Problem Statement
Current knowledge graph construction resembles a rigid, one-way pipeline: you extract facts from documents and stuff them into a pre-defined schema, like trying to shoe a person with a single, unchangeable size. When new knowledge arrives that doesn't fit the schema (e.g., a product launch that changes a company's product lineup), you must discard the entire graph and rebuild it from scratch. This is like trying to fit a new piece of furniture into a room with fixed walls, you either force the furniture into a shape it doesn't belong in (discarding valuable context) or completely tear down the room (full reconstruction).

## Proposed Approach
DIAL-KG operates in a continuous three-stage cycle: Dual-Track Extraction routes knowledge into simple triples or complex event representations; Governance Adjudication verifies facts against evidence, checks logical consistency, and identifies knowledge evolution; and Schema Evolution induces new schemas from validated knowledge to guide subsequent cycles. The Meta-Knowledge Base (MKB) serves as the central memory hub that tracks entity profiles, schema proposals, and evidence for all operations.

```python
def dial_kg_update(batch_Bk, Gk_minus_1, MKBk_minus_1):
    # Stage 1: Dual-Track Extraction
    triples, events = dual_track_extraction(batch_Bk, MKBk_minus_1)
    
    # Stage 2: Governance Adjudication
    verified_triples, verified_events = governance_adjudication(
        triples, events, MKBk_minus_1
    )
    
    # Stage 3: Schema Evolution
    new_relations, new_events = schema_evolution(verified_triples, verified_events)
    update_mkb(MKBk_minus_1, new_relations, new_events)
    
    # Transactional Integration
    return transactional_integration(
        Gk_minus_1, 
        verified_triples, 
        verified_events, 
        MKBk_minus_1
    )
```

## Key Technical Contributions
DIAL-KG's core innovations solve the fundamental tension between schema flexibility and knowledge fidelity in dynamic settings:

1. **Dual-Track Extraction** dynamically routes knowledge into relation triples or event structures based on semantic complexity. Simple, stable statements (e.g., "Python is a programming language") become relation triples ⟨h, r, t⟩, while evolving knowledge (e.g., "Microsoft announced Windows 10 EOL in 2022") becomes event structures ϵ = (trigger, roles, time). This avoids over-structuring simple facts while preserving temporal context for evolving knowledge.

2. **Evidence-Based Governance Adjudication** uses an LLM to verify each extraction against its textual evidence segment without external knowledge. A candidate is rejected only if evidence directly contradicts it, otherwise, it's accepted to prevent false negatives. This achieves 98.6% precision on soft deprecations (D-HP) by ensuring deprecations only occur when explicitly supported by text.

3. **Evolution-Intent Assessment** explicitly identifies whether knowledge is informational (fact state) or evolutionary (state transition), such as "The company announced it would discontinue product X." This enables precise targeting of historical facts for deprecation (E↓ₖ) while preserving the evidence for auditing.

4. **Self-Evolving Schema Induction** clusters verified relation triples by embedding similarity, generating schema proposals that undergo LLM evaluation for semantic completeness. This consolidation reduces relation types by up to 15% while maintaining precision (e.g., merging near-duplicate relations like "acquired_by" and "acquisition_of" into unified predicates).

## Experimental Results
DIAL-KG was evaluated against two schema-free LLM baselines (EDC and AutoKG) on WebNLG, Wiki-NRE, and a purpose-built streaming dataset (SoftRel-∆). It achieved:

- **Static extraction**: Up to 4.7% higher F1-score than baselines (WebNLG: 0.865 vs. 0.848 for EDC)
- **Incremental reliability**: 97.5% ∆-Precision for additions (adding new facts) and 98.6% D-HP (Deprecation-Handling Precision) for deprecations
- **Schema quality**: 15% fewer relation types with 1.6-2.8 point reduction in redundancy compared to EDC

The paper doesn't specify statistical significance testing for these results, though the consistent performance across datasets suggests robustness.

## Related Work
DIAL-KG differentiates itself from prior schema-free KGC approaches by addressing the critical gap in handling knowledge evolution. While EDC decomposes KGC into open information extraction, schema definition, and normalization (with inherent static limitations), and iText2KG proposes zero-shot incremental construction (lacking clear mechanisms for dynamic knowledge evolution), DIAL-KG integrates all these elements into a closed-loop system with evidence-backed governance. Unlike schema-guided methods (e.g., SAC-KG, CoT-Ontology) that rely on external ontologies, DIAL-KG evolves its schema internally without external dependencies.

## Limitations
The paper tests DIAL-KG primarily on a Kubernetes log dataset (SoftRel-∆), which represents a domain-specific, structured context. Real-world applications in open-domain news or social media might encounter different challenges. The authors acknowledge that cold-start performance (initial batch) is slightly lower than subsequent batches due to limited prior knowledge in the MKB. The paper doesn't evaluate the impact of MKB size on performance or how the system handles catastrophic forgetting when knowledge evolves significantly.

## Appendix: Worked Example
Consider a Kubernetes release log entry: "In 2022, Microsoft announced Windows 10 version 1909 would reach the end of support in 2022." 

1. **Dual-Track Extraction**: The system identifies this as an event (not a simple triple) due to the temporal context ("In 2022") and state transition ("would reach end of support"). It creates an event ϵ = (trigger="end of support", roles={"product": "Windows 10 version 1909", "time": "2022"}, time="2022").

2. **Governance Adjudication**: 
   - Evidence Verification: The LLM confirms the event matches the text segment.
   - Logical Verification: Checks ensure "Windows 10 version 1909" is a product and "2022" is a valid time.
   - Evolutionary-Intent Verification: The LLM identifies this as an evolutionary event (state transition) targeting existing facts about product support.

3. **Schema Evolution**: 
   - The event triggers schema induction for "End of Support" events.
   - The system clusters this with past support events ("Microsoft discontinued Windows 10 in 2022").
   - The schema is evaluated and added to the MKB with constraints: time (date), product (Entity), and status (deprecated).

4. **Transactional Integration**: 
   - The system identifies the outdated fact "Windows 10 version 1909 support active" (E↓ₖ).
   - It updates the knowledge graph by adding the new fact "Windows 10 version 1909 end of support in 2022" (E⁺ₖ) and marking the old fact as deprecated (E↓ₖ).
   - The MKB is updated with the new "End of Support" event schema.

This process achieves 98.6% precision on the deprecation (D-HP) because the system only marks the fact as deprecated when the text explicitly states the end-of-support announcement.

## References

- Weidong Bao, Yilin Wang, Ruyu Gao, Fangling Leng, Yubin Bao, Ge Yu, "DIAL-KG: Schema-Free Incremental Knowledge Graph Construction via Dynamic Schema Induction and Evolution-Intent Assessment", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20059

Tags: #knowledge-graph #schema-evolution #incremental-learning #llm-applications #governance-augmented-systems
