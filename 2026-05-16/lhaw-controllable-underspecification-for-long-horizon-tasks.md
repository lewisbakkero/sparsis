---
title: "LHAW: Controllable Underspecification for Long-Horizon Tasks"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2602.10525"
---

## Executive Summary
LHAW (Long-Horizon Augmented Workflows) provides the first systematic framework for generating and validating underspecified task variants in long-horizon workflows. It transforms well-specified tasks into controllable ambiguities across four information dimensions, Goals, Constraints, Inputs, and Context, enabling precise measurement of when autonomous agents should seek clarification versus proceeding independently. This matters because current agents fail to distinguish between solvable tasks and those requiring clarification, leading to either silent failures or excessive user interruptions in production systems.

## Why This Matters for Practitioners
If you're building long-horizon agent systems for enterprise workflows (e.g., automated customer support, financial analysis, or code maintenance), this paper reveals critical trade-offs in your clarification strategy. The authors show that GPT-5.2 over-clarifies (asking questions in 87% of trials on THEAGENTCOMPANY tasks) while Gemini models under-clarify (only 18% of trials on the same tasks), creating a 2.5× efficiency gap in clarification value per question. Your engineering decisions should now include: 1) measuring Gain/Q (performance gain per question asked) for your specific agent, 2) implementing a configurable threshold for when clarification becomes cost-justified (based on your users' cognitive load and workflow latency requirements), and 3) prioritising models with higher Gain/Q on your specific task types rather than simply selecting by raw accuracy.

## Problem Statement
Today's autonomous agents operate like a driver navigating a foggy highway with no speedometer: they proceed at full speed until they hit a roadblock, often with cascading failures. Current benchmarks (like THEAGENTCOMPANY) evaluate agents only on fully specified tasks, while clarification-focused benchmarks (like ClariQ) assume questions cost nothing, ignoring the reality that each human interaction interrupts workflow execution, introduces latency, and imposes cognitive switching costs. The fundamental gap is that we have no systematic way to measure when clarification costs outweigh the benefits in real-world contexts.

## Proposed Approach
LHAW operates in three phases to create and validate underspecified variants:

1. **Segment Extraction**: Identifies removable information segments from well-specified tasks, categorising them by dimension (Goal, Constraint, Input, Context) and scoring criticality (would removal cause failure?) and guessability (can agent infer it from context?).

2. **Candidate Generation**: Creates underspecified variants using deletion, vaguification, or genericisation, at configurable severity levels (1 or 2 segments removed).

3. **Empirical Validation**: Runs agent trials to classify variants as outcome-critical (agent always fails), divergent (variable outcomes), or benign (agent reliably infers missing information).

Here's the core validation algorithm:

```python
def classify_ambiguity(agents, variants, n_trials=3):
    results = {}
    for variant in variants:
        terminal_states = []
        for _ in range(n_trials):
            terminal_state = agents.execute(variant)
            terminal_states.append(terminal_state)
        unique_states = set(terminal_states)
        success_count = sum(1 for state in terminal_states if state.succeeded)
        
        if success_count == 0 and len(unique_states) == 1:
            results[variant] = "New Task"
        elif success_count == 0:
            results[variant] = "Outcome-Critical"
        elif len(unique_states) > 1:
            results[variant] = "Divergent"
        else:
            results[variant] = "Benign"
    return results
```

## Key Technical Contributions
The paper introduces a systematic framework for evaluating ambiguity in long-horizon agents, with three key technical contributions:

1. **The Four-Dimensional Underspecification Taxonomy**: The authors identified four information dimensions (Goal, Constraint, Input, Context) through failure mode analysis across 33 tasks. Unlike prior work that treated ambiguity as a binary state, LHAW's taxonomy enables systematic removal of information across dimensions, e.g., removing "the file location" (Input) versus "success metrics" (Goal), allowing precise control over ambiguity severity.

2. **Empirical Validation via Terminal State Divergence**: Instead of relying on LLM predictions of ambiguity, LHAW validates variants through actual agent trials. The classification algorithm (shown above) uses terminal state divergence as the observable metric, grounding evaluation in execution outcomes rather than linguistic intuition. This eliminates the "curse of dimensionality" in ambiguity measurement.

3. **Gain/Q Metric for Clarification Efficiency**: The authors introduce Gain/Q (performance gain per question asked), calculated as (pass@3_with_user - pass@3_without_user) / (number of questions asked). This metric reveals that Gemini-3-Pro achieves 0.24 Gain/Q on THEAGENTCOMPANY tasks (asking questions in only 18% of trials), while GPT-5.2 achieves 0.04 Gain/Q (asking in 87% of trials). This metric is critical for cost-sensitive evaluation.

See Appendix for a step-by-step worked example of how LHAW transforms a task into an outcome-critical variant.

## Experimental Results
LHAW generated 285 task variants across THEAGENTCOMPANY (85), SWE-BENCH PRO (100), and MCP-ATLAS (100). The distribution was 40% outcome-critical, 30% divergent, and 30% benign. On MCP-ATLAS, Opus-4.5 showed the highest baseline pass@3 (100%) but recovered to 78% with clarification (47% without), while GPT-5.2 achieved 91% pass@3 on SWE-BENCH PRO with clarification (69% without). Crucially, the authors found that GPT-5.2 consistently over-clarifies (87% ask% on THEAGENTCOMPANY tasks) while Gemini-3-Pro under-clarifies (18% ask%), but Gemini-3-Pro extracts 6× more value per question (0.24 Gain/Q vs 0.04 Gain/Q).

## Related Work
LHAW positions itself as the first framework to address the gap between current benchmarks (which assess execution capability under sufficient specification) and clarification benchmarks (which operate in short-context regimes). While Vitabench explores user-agent interactions, it doesn't systematically evaluate whether agents can detect underspecification in long-horizon tasks. LHAW extends ClariQ and AmbigQA by introducing controllable ambiguity severity levels and validating through empirical trials rather than linguistic intuition.

## Limitations
The paper doesn't evaluate LHAW on tasks outside enterprise workflows (e.g., creative content generation). The user simulator assumes perfect knowledge of removed segments, which may not reflect real-world user knowledge gaps. The authors acknowledge that integrating LHAW into production systems requires additional work to handle dynamic cost-benefit analysis of clarification based on specific user contexts.

## Appendix: Worked Example
Let's walk through how LHAW transforms a THEAGENTCOMPANY task. Consider the task "Send the report to the team" (Goal: send report, Context: team needs it by Friday).

1. **Segment Extraction**: The pipeline identifies "the team" as a Context dimension segment with criticality 1.0 (removal causes failure) and guessability 0.2 (agent can't infer which team from context).

2. **Candidate Generation**: Using deletion strategy, the system produces the underspecified variant: "Send the report."

3. **Empirical Validation**: 
   - Agent runs 3 trials: 
     - Trial 1: Sends to engineering team (fails as needs finance team)
     - Trial 2: Sends to finance team (succeeds)
     - Trial 3: Sends to HR team (fails)
   - Terminal states: [(failure), (success), (failure)] → unique states = 2 (divergent)
   - Classification: Divergent (33% success rate, variable outcomes)

This example shows why LHAW's empirical validation is critical: linguistic ambiguity ("the team") would be classified as "semantic ambiguity" in traditional frameworks, but LHAW reveals it's actually underspecification requiring clarification.

## References

- George Pu, Michael S. Lee, Udari Madhushani Sehwag, David J. Lee, Bryan Zhu, Yash Maurya, Mohit Raghavendra, Yuan Xue, Samuel Marc Denton, "LHAW: Controllable Underspecification for Long-Horizon Tasks", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2602.10525

Tags: #artificial-intelligence #autonomous-systems #agent-based-systems #underspecification #long-horizon-tasks #clarification-efficiency
