---
title: "Pitfalls in Evaluating Interpretability Agents"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20101"
---

## Executive Summary
This paper investigates automated interpretability systems that explain model components' roles through iterative hypothesis generation and experiment design. The authors demonstrate that replication-based evaluation, comparing outputs to human-authored explanations, can be misleading due to subjectivity in human explanations and LLM memorization of published findings. They propose an unsupervised intrinsic evaluation method based on functional interchangeability of model components.

## Why This Matters for Practitioners
If you're building or deploying interpretability tools at scale, this paper exposes a critical blind spot: you might believe your system is accurate because it matches human explanations, when in reality it's merely memorizing published results. For production systems, this means your interpretability tool could fail when encountering novel tasks outside the training literature. The key takeaway: **never rely solely on replication-based evaluation** for automated interpretability systems. Instead, implement the authors' unsupervised metric, functional interchangeability, to assess whether components can be swapped without changing model behaviour. This requires building test cases where you temporarily replace a component with another that should functionally match it, then measuring the impact on predictions. For engineers, this means adding component-swapping experiments to your evaluation pipeline rather than just comparing to published explanations.

## Problem Statement
Today's evaluation of automated interpretability systems resembles judging a chef's recipe by comparing the final dish to a cookbook, without considering whether they're actually understanding the cuisine or just copying the book. The authors found that when evaluating their circuit analysis agent, the system performed well against human explanations, but closer inspection revealed it was often reproducing findings via memorization rather than genuine understanding. This is particularly dangerous in production systems where the model might encounter scenarios outside the limited set of well-documented tasks.

## Proposed Approach
The authors built an agentic system where a research agent autonomously designs experiments and refines hypotheses about model components' functionality. The system uses standard interpretability tools (logit lenses, attention patterns, activation patching) and follows a three-stage workflow: (1) researcher specifies task and circuit, (2) research agent iteratively analyses each component, (3) LLM clusters components by shared functionality.

```python
def research_agent(component, circuit, task_prompts):
    """Iteratively analyzes a model component's functionality"""
    evidence = precompute_tool_outputs(component, task_prompts)
    while not converged(evidence):
        hypotheses = generate_hypotheses(evidence)
        experiments = propose_experiments(hypotheses, evidence)
        evidence = execute_experiments(experiments)
    return final_hypothesis(evidence)
```

## Key Technical Contributions
The authors made three key contributions that fundamentally shift how we approach evaluating interpretability systems:

1. **The identification of critical pitfalls in replication-based evaluation**: They demonstrated through concrete examples that human explanations can be subjective (e.g., whether a head should be labelled "previous-token" when only 42% of cases show that pattern), and that LLMs can reproduce findings via memorization rather than genuine reasoning.

2. **The unsupervised metric of functional interchangeability**: They proposed measuring whether components can be swapped without changing the model's output, which doesn't require human judgment. For any two components, they test whether replacing one with the other maintains the same prediction on diverse inputs.

3. **The realization that increased autonomy doesn't necessarily improve performance**: Their agentic system performed comparably to a simpler one-shot baseline that provided all experimental results at once, showing that the research process itself, while valuable, doesn't automatically lead to better outcomes in evaluation.

## Experimental Results
The authors evaluated their system against six prior circuit analysis tasks (IOI on GPT-2-Small and Pythia-160M, Greater-Than, Acronyms, Colored Objects, Entity Tracking), measuring three metrics:

- Component Functionality Accuracy: 0.72 on average (ranging from 0.61 to 0.83)
- Cluster Functionality Accuracy: 0.67 on average
- Component Assignment Accuracy: 0.68 on average

The agentic system performed comparably to the one-shot baseline (which provided all experimental results at once) across all metrics, suggesting that the iterative research process didn't consistently improve outcomes. Crucially, they found that Claude Opus 4.1 had memorized at least the IOI circuit, leading to high performance despite not conducting experiments.

## Related Work
This work builds upon prior automated interpretability systems that use LLMs to interpret neurons or features (Bills et al., 2023; Paulo et al., 2025), but shifts focus from the creation of explanations to the evaluation of those explanations. It extends the circuit analysis literature (Wang et al., 2022; Hanna et al., 2023; Prakash et al., 2024) by investigating how to properly assess such analyses rather than how to create them. The authors position their work as addressing a gap in the field: no prior work has systematically examined the limitations of replication-based evaluation for automated interpretability systems.

## Limitations
The authors acknowledge several limitations: they only tested on a small set of well-established circuit analysis tasks (six total), so the evaluation might not generalise to more complex or novel tasks. Their unsupervised metric doesn't address all evaluation pitfalls, particularly those related to completeness of explanations. Additionally, they note that their LLM judge (GPT-5) also showed signs of memorization, which might affect evaluation reliability. The biggest limitation for practitioners is that their evaluation method requires creating specialized test cases for each component, which may be time-consuming to implement at scale.

## Appendix: Worked Example
Let's walk through how the agent might analyse a component in the IOI task (identifying indirect objects in sentences like "Sarah and Michael went to the park. Sarah talked to"). For attention head L9H9:

1. The agent begins with evidence from precomputed logit lenses, attention patterns, and activation patching on 10 standard IOI prompts.
2. Initial hypothesis: "This head identifies the recipient in giving scenarios (e.g., 'Sarah gave a drink to Michael')."
3. To test generalisation, the agent proposes new prompts: "Mark wrote and James read the letter. Mark then handed it to" and "David and Lisa met David's friend. Lisa gave a gift to."
4. Activation patching shows that replacing L9H9 with a different head (L9H6) maintains similar prediction patterns for "to" as the recipient in 18 out of 20 test cases.
5. The agent concludes: "This head identifies the recipient in giving scenarios across different verb types and subjects."
6. The unsupervised metric confirms functional interchangeability: swapping L9H9 with L9H6 produces the same prediction (recipient identification) in 90% of cases, confirming they share functionality.

This example demonstrates how the agent moves beyond memorizing a single explanation to understanding a pattern that generalizes across examples.

## References

- Tal Haklay, Nikhil Prakash, Sana Pandey, Antonio Torralba, Aaron Mueller, Jacob Andreas, Tamar Rott Shaham, Yonatan Belinkov, "Pitfalls in Evaluating Interpretability Agents", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20101

Tags: #ai-interpretability #machine-learning-eval #agentic-systems #unsupervised-evaluation #functional-interchangeability
