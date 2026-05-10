---
title: "VERDICT: Verifiable Evolving Reasoning with Directive-Informed Collegial Teams for Legal Judgment Prediction"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19306"
---

## Executive Summary
VERDICT is a multi-agent framework for Legal Judgment Prediction that simulates a virtual collegial panel to produce verifiable, evolving legal reasoning. Unlike static models, it maintains a memory of verified reasoning paths to adapt as jurisprudence evolves. Practitioners building legal AI systems should care because it provides a blueprint for building interpretable, evolving systems that avoid the pitfalls of hallucination and static knowledge.

## Why This Matters for Practitioners
If you're building a legal AI system that needs to produce explainable judgments that maintain consistency with evolving case law, VERDICT demonstrates how to design systems that don't just predict outcomes but provide a traceable reasoning path that can be verified and updated. Specifically, for production systems that need to handle new legal interpretations without full retraining, this framework suggests implementing a memory architecture that distills verified reasoning trajectories rather than static knowledge. This means you should design your legal AI system with a specialized memory component that actively refines its understanding through a three-phase process of inductive generation, contrastive refinement, and consolidation, similar to how human judges gradually develop nuanced understanding through repeated practice.

## Problem Statement
Current Legal Judgment Prediction (LJP) systems are like static crossword puzzles, designed to solve familiar patterns but collapsing when presented with novel combinations of facts that don't align with historical patterns. Just as a crossword solver might get stuck on an unfamiliar word that doesn't fit the standard grid, today's LJP models struggle when faced with scenarios that don't match statistical patterns from past cases. For instance, a model trained on past cases might misclassify "theft" as "embezzlement" due to surface-level text similarity, without understanding the legal distinction between the two. This leads to judgments that are hard to justify when statutory rules and precedent-informed standards diverge.

## Proposed Approach
VERDICT organises LJP as a traceable multi-agent workflow with a draft-verify-revise loop, simulating a legal panel's deliberation process. It features specialized agents for fact extraction, legal retrieval, opinion drafting, and verification, coordinated through explicit Pass/Reject feedback. The system incorporates a Hybrid Jurisprudential Memory (HJM) that distills verified reasoning trajectories into evolving Micro-Directives, enabling continual learning from new cases.

```python
def verdict_inference(case):
    # Pre-adjudication analysis
    fact_points = court_clerk_agent(case)
    statutes = judicial_assistant_agent(fact_points)
    
    # Drafting phase
    draft = None
    feedback_history = []
    for round in range(max_rounds=3):
        draft = case_handling_judge_agent(fact_points, statutes, feedback_history)
        verdict, feedback = adjudication_supervisor_agent(draft, statutes)
        
        if verdict == "PASS":
            break
        else:
            feedback_history.append(feedback)
    
    # Final adjudication
    final_verdict = presiding_judge_agent(draft, feedback_history)
    update_hybrid_jurisprudential_memory(draft, verdict, feedback_history)
    return final_verdict
```

## Key Technical Contributions
VERDICT's core innovation lies in how it fundamentally reimagines knowledge evolution in legal AI through a memory system that mimics human judicial learning processes.

1. **Micro-Directive Memory Architecture**: VERDICT's Hybrid Jurisprudential Memory (HJM) stores "Micro-Directives", precise, context-specific interpretations of law that bridge rigid statutes and evolving precedents. Unlike traditional memory systems, each directive anchors to a specific statute, contains confidence scores, and is distilled from verified reasoning trajectories through a three-phase process (inductive generation, contrastive refinement, consolidation). This differs from systems like G-Memory (Zhang et al., 2025), which treat interactions as static records rather than evolving wisdom.

2. **Verifiable Feedback Loop**: The Adjudication Supervisor Agent issues explicit Pass/Reject signals with natural language feedback, creating a traceable feedback loop that's directly tied to legal reasoning. When rejecting a draft, it provides specific corrective advice like "Incorrect charge qualification (should be Art. 234 for injury, not Art. 293 for disorder)" rather than generic rejection. This enables the system to learn from mistakes in a legally meaningful way.

3. **Domain-Specific Expert Alignment**: The Case-handling Judge agent is aligned through a two-stage process: Protocol-Aware Instruction Tuning (to standardise output formats using verified samples) followed by Logic-Driven Contrastive Alignment (to fix logical hallucinations using contrastive examples). This produces a model that's both formally correct and logically sound, addressing the critical distinction between formal correctness and logical accuracy in legal reasoning.

## Experimental Results
VERDICT achieved state-of-the-art performance on CAIL2018, outperforming all baselines across all metrics (see Table 1). For Law Article prediction, it achieved 85.35% accuracy compared to the second-best baseline PLJP at 83.21%. On CJO2025, a dataset of judgments from after Jan 1, 2025 (constructed to prevent future leakage), VERDICT demonstrated strong temporal generalisation with 90.56% accuracy versus NeurJudge's 73.7% drop from its CAIL2018 performance.

When compared to generic agents like AutoGen (84.26% accuracy) and G-Memory (86.52%), VERDICT's specialized legal framework showed a significant advantage (90.56% accuracy), demonstrating that domain-specific agent roles and jurisprudence-based memory are essential for rigorous legal reasoning.

## Related Work
VERDICT builds on traditional LJP systems by extending discriminative models (like TopJudge and LADAN) from learning decision boundaries to providing traceable reasoning workflows. It also improves on existing multi-agent frameworks (like MetaGPT and AutoGen) by introducing a memory architecture grounded in legal theory (the Micro-Directive Paradigm), rather than treating knowledge as static. Unlike retrieval-augmented models (PLJP), which treat legal knowledge as static, VERDICT's HJM continuously refines its understanding through verified reasoning trajectories, addressing the critical gap in dynamic knowledge management that existing legal multi-agent systems fail to address.

## Limitations
The paper primarily evaluates on Chinese legal datasets (CAIL2018 and CJO2025), so its applicability to common law systems or other legal jurisdictions is unclear. The paper doesn't specify how the "Micro-Directive" memory grows over time in terms of storage requirements or retrieval latency, which could be a significant concern for production systems. The framework's performance on edge cases where legal interpretation is highly ambiguous (e.g., novel cybercrime scenarios) is also not tested.

## Appendix: Worked Example
Let's walk through a concrete example of how VERDICT processes a case about public disorder:

**Case**: "Drunk Zhang beat stranger Li at a public BBQ stall for 'looking at him', causing minor injury and chaos."

1. **Court Clerk Agent** extracts key facts: "Beating", "Minor Injury", "Public Location" (fact points = [Beating, Minor Injury, Public]).

2. **Judicial Assistant Agent** retrieves statutes:
   - First, it does a vector search for "Beating" and gets candidates: Art. 234 (assault), Art. 293 (public disorder), Art. 292 (disorderly conduct).
   - Then it filters these using semantic understanding, keeping Art. 234, Art. 293 (retrieved statutes = [234, 293]).

3. **Case-handling Judge Agent** drafts a preliminary opinion: "Intentional injury for public disorder under Art. 293" (draft = "Intentional injury for public disorder under Art. 293").

4. **Adjudication Supervisor Agent** checks this against the memory:
   - It matches against the Contextual Standards Archive (Mstd) and Evolving Micro-Directives Base (Mdir).
   - It finds a Micro-Directive: "Random attacks in public to vent anger, then picking quarrels, not injury" (with confidence 0.85).
   - The draft matches the statute but doesn't align with the Micro-Directive (which specifies "not injury" but the case involves "minor injury").
   - It issues a Reject: "Incorrect charge qualification (should be Art. 234 for injury, not Art. 293 for disorder)".

5. **Feedback History** now includes: "Incorrect charge qualification (should be Art. 234 for injury, not Art. 293 for disorder)".

6. **Case-handling Judge Agent** redrafts: "Intentional injury under Art. 234" (draft = "Intentional injury under Art. 234").

7. **Adjudication Supervisor Agent** verifies this draft: matches both statute and Micro-Directive, issues a Pass.

8. **Presiding Judge Agent** renders the final verdict: "Case judgment: Art. 234 (Intentional injury), with reasoning chain: [Beating, Minor Injury, Public] → Art. 234, 293 → Refine to Art. 234 using Micro-Directive."

9. **Memory Update**: The system distills the verified trajectory into an updated Micro-Directive: "Beating in public with minor injury (not disorder) should be classified under Art. 234, not Art. 293" (confidence increases from 0.85 to 0.92).

## References

- Hui Liao, Chuan Qin, Yongwen Ren, Hao Li, Zhenya Huang, Yanyong Zhang, Chao Wang, "VERDICT: Verifiable Evolving Reasoning with Directive-Informed Collegial Teams for Legal Judgment Prediction", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19306

Tags: #legal-ai #multi-agent #memory-augmented #jurisprudential-memory #verifiable-ai
