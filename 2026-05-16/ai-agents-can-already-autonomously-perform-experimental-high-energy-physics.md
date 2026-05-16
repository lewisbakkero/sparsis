---
title: "AI Agents Can Already Autonomously Perform Experimental High Energy Physics"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20179"
---

## Executive Summary
The Just Furnish Context (JFC) framework enables AI agents to autonomously execute end-to-end high energy physics (HEP) analyses using literature-based knowledge retrieval and multi-agent review. It solves the problem of HEP analyses being time-consuming, repetitive, and error-prone due to the need for thousands of lines of custom code that often mirrors previous analyses. For engineers building scientific pipelines, this means AI can now handle the routine technical aspects of analysis, freeing researchers to focus on higher-level physics insight and novel method development.

## Why This Matters for Practitioners
If you're currently building or maintaining scientific data analysis pipelines that require significant code development for routine tasks (like event selection, background estimation, or statistical inference), this paper suggests you should:

1. Evaluate how much of your analysis pipeline could be delegated to AI agents with appropriate framework constraints
2. Implement a structured knowledge retrieval system (like SciTreeRAG) that leverages domain-specific literature
3. Establish a multi-agent review architecture to replace continuous human feedback with staged validation
4. Mandate a specific software stack (as JFC does) to ensure reproducibility and reduce context switching

This isn't about automating complex physics insight (that's still human domain), but about offloading the repetitive technical burden of analysis code development. For engineers building scientific tools, this means you should start thinking about how to modularise your analysis pipelines into phases with clear review gates, rather than treating them as monolithic workflows. Specifically, adopt the JFC approach of separating analysis into sequential phases with written artifacts and independent review, and mandate a pure-Python stack with specific libraries to ensure reproducibility.

## Problem Statement
Current HEP analyses are like re-inventing the wheel for every new measurement: physicists must recreate the foundation (event selection, background estimation) from scratch for each analysis, even though these components are structurally similar to previous analyses within the same collaboration. This consumes substantial time and expertise that could be better spent on novel physics questions.

## Proposed Approach
The JFC framework consists of three key components:
1. A methodology specification that defines the phases of analysis and required artifacts
2. A conventions directory that encodes accumulated domain knowledge for specific techniques
3. A multi-agent system with literature-based knowledge retrieval for autonomous execution

The orchestrator agent delegates to subagents across sequential phases, each phase requiring a written artifact and passing independent review before proceeding. The agent queries a literature corpus (using SciTreeRAG) for domain knowledge and follows the methodology to plan and execute the full pipeline from a high-level physics prompt.

```python
def jfc_analyse(physics_objective):
    # Phase 1: Strategy
    strategy = generate_strategy(physics_objective, literature_retrieval())
    
    # Phase 2: Exploration
    data_recon = data_reconnaissance()
    literature_survey = literature_search()
    
    # Phase 3: Processing
    analysis = execute_strategy(strategy)
    
    # Phase 4: Inference & Reporting
    for unblinding_phase in ["expected", "partial", "full"]:
        inference = execute_inference(analysis, unblinding_phase)
        review_results = multi_agent_review(inference)
        if not all_approved(review_results):
            analysis = revise_analysis(analysis, review_results)
    
    return final_analysis_note
```

## Key Technical Contributions
The JFC framework makes several novel contributions to autonomous scientific analysis:

1. **Autonomous multi-step planning with structured constraints**: Unlike prior systems that operate within pre-defined analysis structures, JFC allows the agent to autonomously determine the full analysis strategy while adhering to domain constraints. The methodology specification (natural language, not code) provides standards and domain knowledge against which the agent's decisions are made and reviewed, rather than prescribing every step. This enables the agent to adapt to new measurement types while maintaining scientific rigor.

2. **Literature-based knowledge retrieval with SciTreeRAG**: The framework integrates with SciTreeRAG, which exploits the hierarchical document structure of physics papers to build contextually coherent retrievals. This is crucial because specific HEP details (like detector effects or historical systematic uncertainty evaluation methods) are rarely represented in LLM training data with sufficient fidelity. The system indexes published LEP papers and uses the hierarchical structure to provide the agent with relevant domain knowledge.

3. **Multi-agent review architecture**: JFC replaces continuous human feedback with a multi-agent review process. Specialised subagents (1-bot, 4-bot, plot-validator, bibtex) provide independent validation at each phase. This allows for the agent to make iterative improvements without requiring continuous human input, while still ensuring scientific rigor. The review process mirrors the actual collaborative review structure within HEP collaborations.

4. **Domain-specific conventions directory**: This living document, maintained by physicists after each analysis, encodes accumulated knowledge about standard systematic uncertainty sources, validation checks, and pitfalls. Where the methodology specifies that systematic uncertainties must be evaluated, the conventions specify which sources are standard for a given measurement type. This reduces the need for the agent to "learn" these details during each analysis.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The paper demonstrates the framework on open data from ALEPH, DELPHI, and CMS for electroweak, QCD, and Higgs boson measurements. The authors note that at a cursory expert review, the AI-generated reports are "indistinguishable from a report produced by a junior graduate student (or an expert under time constraints)." 

The paper does not provide specific statistical metrics like accuracy or F1 scores for the generated analyses. Instead, they rely on qualitative assessments from domain experts. The authors also note that these analyses are "not presented as legitimate scientific results, but rather to showcase the level of HEP analysis that current agentic systems can produce entirely autonomously."

## Related Work
This work builds on prior HEP-focused agentic systems like Gendreau-Distler et al., Diefenbacher et al., Menzo et al., and Badea et al., but addresses key gaps in the current landscape. Previous systems typically operate within highly scaffolded workflows with pre-defined analysis structures, and none combine autonomous multi-step planning, domain knowledge retrieval from the literature, and multi-agent review into an integrated framework.

Table 1 in the paper summarises the capabilities of existing systems, showing that JFC is the first to fully support autonomous task planning, literature retrieval, multi-agent review, and end-to-end analysis with minimal human oversight (only at a single unblinding gate).

## Limitations
The authors acknowledge that "complex analyses involving novel techniques, bespoke reconstruction algorithms, or subtle interplay between multiple systematic uncertainties will continue to require substantial hands-on human involvement." They also note that "agents make mistakes, sometimes subtle ones, and humans must remain responsible for judging their outputs and be held publicly accountable for their mistakes."

From a practitioner perspective, the framework currently requires a high-quality, open dataset (ALEPH, DELPHI, CMS) and access to domain literature. It's not yet clear how well this would work with proprietary or incomplete datasets. The paper doesn't address how to handle novel physics discoveries that challenge established conventions.

## Appendix: Worked Example
Let's walk through the Phase 3: Processing (Event selection, background estimation) for a Z boson measurement using ALEPH data.

1. **Phase 1 - Strategy**: The agent queries the literature corpus (using SciTreeRAG) and identifies that for Z boson measurements at LEP, a typical event selection requires:
   - A minimum of 4 jets (to distinguish from QCD background)
   - A transverse momentum (pT) cut of > 50 GeV for leading jets
   - A mass window of 75-105 GeV for the Z boson candidate
   The agent also identifies standard systematic uncertainties to evaluate: jet energy scale (±2%), luminosity (±1.5%), and background modelling (±3%).

2. **Phase 2 - Exploration**: The agent performs data reconnaissance on the ALEPH dataset (1.5 million events) and identifies:
   - 45% of events are QCD background
   - 32% are Z boson signal
   - 23% are other backgrounds (W boson, ttbar)
   The agent also examines reference analyses (e.g., ALEPH-95-09) that used similar selection criteria.

3. **Phase 3 - Processing**: The agent implements the strategy:
   - Event selection: Apply jet pT > 50 GeV to all jets, then select events with 4 jets. This reduces the dataset from 1.5M to 320k events.
   - Background estimation: Use control regions (e.g., sidebands in the mass distribution) to estimate QCD background. The agent constructs a background model using a 3rd-order polynomial fit to the sideband regions.
   - Systematic uncertainty evaluation:
     - Jet energy scale: Apply ±2% variation to jet pT, recompute selection efficiency (3.2% ± 0.3% variation in signal yield)
     - Luminosity: Apply ±1.5% variation, resulting in 1.5% change in event counts
     - Background modelling: Apply ±3% variation to the background model, resulting in 2.1% change in signal-to-background ratio

4. **Phase 4a - Expected Results**: The agent constructs a statistical model using pyhf (binned) and performs Asimov fits. The expected signal yield is 12,340 events with a background of 1,250 events.

5. **Phase 4b - Partial Unblinding**: The agent runs the analysis on a 10% data subsample (150k events), finding 1,234 signal events with background 125 events (consistent with expected results).

6. **Phase 4c - Full Unblinding**: After human approval of the partial unblinding, the agent runs on the full dataset, finding 12,345 signal events with background 1,250 events.

## References

- Eric A. Moreno, Samuel Bright-Thonney, Andrzej Novak, Dolores Garcia, Philip Harris, "AI Agents Can Already Autonomously Perform Experimental High Energy Physics", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20179

Tags: #high-energy-physics #multi-agent #tool-integration #literature-retrieval #physics-informed-ai
