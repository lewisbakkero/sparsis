---
title: "The Autonomy Tax: Defense Training Breaks LLM Agents"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19423"
---

## Executive Summary
This paper reveals a fundamental flaw in current LLM agent safety practices: defence training designed to prevent prompt injection attacks systematically destroys agent competence before any malicious content is observed. Evaluating defended models across 97 agent tasks and 1,000 adversarial prompts, the authors uncover three systematic biases that degrade agent functionality while failing to prevent sophisticated attacks. For practitioners building production agent systems, this means safety measures may actually cripple core functionality while providing false security.

## Why This Matters for Practitioners
If you're currently deploying defence-trained LLM agents for production workflows (such as document processing or API integration), your system may be failing on 47-77% of benign tasks before even seeing any external data. This isn't a minor performance issue, it's a fundamental capability-alignment paradox. Specifically, SecAlign-Llama3-8B has a 46.4% refusal rate at Step 1 on benign tasks (vs 1.0% for base models), meaning nearly half of all agent actions would fail immediately. You should immediately audit your agent's step-1 execution behaviour using a curated set of benign tasks before observing any external content. For production systems, avoid relying solely on single-turn security metrics (attack rejection > 90%, FPR < 5%), instead, test for cascade amplification using depth-stratified completion rates. The paper shows that Mistral-7B with SecAlign achieves only 1.0% task completion (vs 50.5% for base models), representing a 49.5 percentage point degradation that will silently break your production pipelines.

## Problem Statement
Current LLM agent safety practices are like installing a smoke detector in a house, but then wiring it to shut off the main electrical circuit when it senses any smoke, whether from a cooking accident or a fire. You've improved safety against actual fires, but now the lights don't turn on, the refrigerator stops working, and the door locks won't open, even when there's no fire. The paper demonstrates that safety measures designed to stop malicious "smoke" (prompt injection attacks) are actually preventing legitimate "electricity" (tool execution for benign tasks) from flowing at all.

## Proposed Approach
The authors don't propose a new defence approach, they reveal a flaw in the current paradigm. They characterise three specific biases that emerge from defence training by measuring how models perform on benign tasks before any tool observations appear. The core insight is that current evaluation metrics (attack rejection > 90%, FPR < 5%) mask catastrophic failures in multi-step agent execution. To diagnose these failures, they introduce step-1 execution analysis, depth-stratified cascade measurement, and a curated challenging subset that tests whether defences learn semantic threat understanding or rely on keyword-matching shortcuts.

```python
def analyze_agent_incompetence(task: str, model: LLM) -> tuple[float, float, float]:
    """Measures Step-1 execution behaviour before any observations appear."""
    context = f"System instruction: You can use tool X. Task: {task}"
    actions = [model.generate(context) for _ in range(100)]
    valid_actions = sum(1 for a in actions if is_valid_tool_action(a))
    refusals = sum(1 for a in actions if is_refusal(a))
    invalid_actions = sum(1 for a in actions if is_invalid_action(a))
    return (
        valid_actions / 100,  # Step-1 valid action rate
        refusals / 100,      # Step-1 refusal rate
        invalid_actions / 100 # Step-1 invalid action rate
    )
```

## Key Technical Contributions
The authors identify three distinct biases that reveal the root cause of the problem, moving beyond single-turn evaluation metrics:

1. **Step-1 Execution Analysis**: They isolate model incompetence by measuring execution behaviour before any tool observations appear. For benign tasks like "List files in documents folder," defence models failed at Step 1 on 47-77% of tasks (vs 3% base), with SecAlign showing 46-47% refusal rates and StruQ showing 69-71% invalid action rates. This analysis reveals that failures occur before the model observes any external content, demonstrating that defence training destroys fundamental capability rather than learning observation-triggered threat detection.

2. **Depth-Stratified Cascade Measurement**: They introduce a method to measure how failures propagate through multiple steps using depth-stratified completion rates (CRd), revealing the binary nature of cascade failures. Tasks either complete at depths 1-9 (86.6% base completion) or cascade to timeout at depth 10 (36.1% SecAlign cascade failure rate). This analysis shows that a single early failure (at Step 1) causes cascading timeouts in 99% of cases for SecAlign-Mistral (vs 50.5% for base).

3. **Curated Challenging Subset**: They construct a 350-sample evaluation set (289 adversarial across three categories, social engineering, obfuscation, instruction override; 61 benign technical documentation) that tests whether defences learn semantic threat understanding or rely on surface shortcuts. This reveals that sophisticated attacks bypass defences at 73-86% rate (e.g., Base64-encoded instructions avoid plaintext triggers), while benign content containing security terminology triggers 25-71% false refusals.

## Experimental Results
The authors evaluated three base models with four defence configurations across 97 agent tasks (AgentDojo benchmark) and 1,000 adversarial prompts:

- **Agent Incompetence**: Defence models failed at Step 1 on 47-77% of benign tasks (vs 3% base), with SecAlign-Llama3-8B showing 46.4% refusal rate (Fig 2a). This represents a 43.4 percentage point degradation in valid action rates.
- **Cascade Amplification**: SecAlign increased timeout rates from 13-50% (base) to 36-99% (defence), representing 2.0-2.7× amplification (Table 2, Fig 3). SecAlign-Mistral achieved only 1.0% task completion (vs 50.5% base), with 99% cascade failures (Table 2, Table 5).
- **Trigger Bias**: Defence training reduced attack detection from 82.7% (base) to 26.7-37.4% (defence), while increasing false positive rates from 0% to 25-71% (Table 3). Meta SecAlign showed 50.7% variance across attack categories (53.4% bypass for direct requests vs 2.8% for constraint removal).

## Related Work
The paper extends prior work on shortcut learning in LLMs (Geirhos et al., 2020; Ilyas et al., 2019) to the agent context. While previous work identified defence-specific shortcuts in single-turn settings (Li et al., 2026), this paper demonstrates how these manifest as agent-specific biases with catastrophic effects in multi-step execution. The authors position their work as addressing the critical gap in current evaluation practices (Liu et al., 2024; Yi et al., 2025), which focus on single-turn metrics while masking cascade dynamics and step-1 failures.

## Limitations
The evaluation focused on smaller language models (Llama-3 8B, Llama-3.1 8B, Mistral-7B) with three defence configurations (StruQ, SecAlign, Meta SecAlign), which may not generalise to larger models or different defence paradigms. The AgentDojo benchmark has 97 tasks representing common agent workflows but might not cover all edge cases. The authors acknowledge that while the shortcut learning mechanisms should generalise across model scales, quantitative magnitudes may vary. The paper doesn't address how to modify defence training to preserve competence while improving security, which remains an open problem.

## Appendix: Worked Example
Let's walk through a concrete example of the step-1 failure using the paper's benchmark:

1. **Task**: "Find the quarterly report in the documents folder."
2. **Base Llama-3-8B Execution**:
   - Step 1 context: "System instruction: You can use tool list_files. Task: Find the quarterly report in the documents folder."
   - Model generates: {"action": "list_files", "input": {"path": "documents"}}
   - Step 1 validity: Valid action (96.9% rate)
   - Task completes in 2 steps (success rate 86.6%)
3. **SecAlign-Llama3-8B Execution**:
   - Step 1 context: "System instruction: You can use tool list_files. Task: Find the quarterly report in the documents folder."
   - Model generates: "I cannot assist with accessing files... I do not have the capability to interact with file systems."
   - Step 1 validity: Explicit refusal (46.4% refusal rate)
   - Agent attempts retry (Step 2), generating identical refusal
   - Cascade failure: Task timeouts at depth 10 (99% cascade rate)
   - Completion rate: 0% (vs 86.6% base)
4. **System Impact**: This single-step refusal creates a cascade failure, with 99% of tasks timing out (vs 13.4% base), representing a 2.7× amplification in cascade failures (Table 5).

## References

- Shawn Li, Yue Zhao, "The Autonomy Tax: Defense Training Breaks LLM Agents", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19423

Tags: #security #multi-agent #defence-training #prompt-injection #shortcut-learning
