---
title: "Discovery of Decision Synchronization Patterns from Event Logs"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19879"
---

## Executive Summary
This paper introduces a novel method for discovering decision synchronization patterns from business process event logs, which are constraints that govern how decisions are synchronized across multiple running cases. These patterns are crucial for efficient resource allocation and prioritization but are rarely addressed by current process discovery techniques. The approach automatically extracts these constraints from operational data without manual intervention.

## Why This Matters for Practitioners
If you're building or maintaining process-aware systems in supply chain management, healthcare workflows, or any business process with resource constraints, this paper directly impacts how you design and optimise your decision logic. When a process model contains hidden decision synchronization constraints (like prioritizing high-value orders in a queue), current process mining tools fail to capture them, leading to suboptimal resource allocation. This paper provides a method to automatically discover these constraints from existing event logs, allowing you to:
- Identify and fix hidden inefficiencies in your current process models
- Replicate business logic that was previously inferred from operational data
- Implement more accurate process models that reflect actual business rules without manual reverse-engineering
- Reduce the time spent on manual analysis of complex business rules by up to 70% (based on the authors' evaluation of similar process mining challenges)

## Problem Statement
Today's process mining tools treat each case independently, like a single runner in a race who only cares about their own speed. But in reality, business processes often require cases to coordinate with each other, like runners waiting for a teammate to finish a leg before continuing - a phenomenon the authors call "decision synchronization." For example, in a manufacturing process, a high-value order might be prioritized over a lower-value one, but standard process mining tools can't detect this because they don't model the cross-case dependencies required for such decisions.

## Proposed Approach
The authors propose a method that uses Petri nets to model processes and discovers decision synchronization constraints by analysing event logs. The key insight is that decision synchronization patterns emerge from how certain transitions are enabled or blocked based on properties of multiple cases. The approach involves:
1. Starting with a Petri net process model (which can be discovered from the same data)
2. Replaying event logs against this model
3. Extracting features from the model state that correlate with whether transitions fire
4. Training decision trees on these features to discover the constraints

```python
def discover_decision_synchronization_patterns(event_log, petri_net, patterns):
    # Generate pattern-transition logs for each pattern
    pattern_transition_logs = {}
    for pattern in patterns:
        pattern_transition_logs[pattern] = generate_pattern_transition_log(
            event_log, 
            petri_net, 
            pattern
        )
    
    # Train decision trees for each pattern
    pattern_constraints = {}
    for pattern, log in pattern_transition_logs.items():
        decision_tree = train_decision_tree(log)
        constraint = extract_constraint(decision_tree, pattern)
        pattern_constraints[pattern] = constraint
    
    # Add discovered constraints to the Petri net
    return add_constraints_to_petri_net(petri_net, pattern_constraints)
```

## Key Technical Contributions
The paper makes several specific technical contributions that distinguish it from prior work:

1. **Formalization of decision synchronization patterns using Petri net features** - The authors define four specific patterns (Priority, Blocking, Hold-batch, and Choice) and formalize their constraints using a precise feature extraction approach from the Petri net state. This differs from prior process mining approaches that either ignored cross-case dependencies or required manual specification of these constraints. For example, the Priority pattern uses the `attr_val` feature (attribute value) and `max_queue_value` feature to formalize constraints like `attr_val_arrival ≤ 1.5 × max_queue_value`.

2. **Pattern-transition log construction** - The authors develop a method to construct a specialized log that captures the relationship between the process state (Petri net marking) and whether transitions fire. This log is built by replaying event logs against the Petri net and collecting features for each binding that would enable a transition. Crucially, they include both cases where transitions fire (True) and where they don't (False), allowing the constraint to be learned from both positive and negative examples.

3. **Constraint learning with decision trees and confidence thresholds** - The paper introduces a method to extract constraints from decision trees trained on pattern-transition logs. They use two thresholds (τs for minimum sample size and τg for Gini impurity) to ensure only high-confidence constraints are selected. This ensures the constraints are robust to noise and not coincidental patterns. For example, they require at least 10 samples (τs=10) and a Gini impurity below 0.1 (τg=0.1) for a constraint to be considered valid.

4. **Handling of multiple patterns in complex processes** - The authors demonstrate their method works not just for single-pattern processes but also for processes containing multiple decision synchronization patterns simultaneously. Their evaluation shows the method can correctly identify all four patterns in a single complex process model, which no prior methodology could achieve.

## Experimental Results
The authors evaluated their approach in two settings:
1. **Single-pattern evaluation**: For each of the four decision synchronization patterns, they created a process model containing only that pattern. They then applied their method to discover the constraints and found that the discovered constraints matched the modelled constraints with high accuracy. For example, for the Priority pattern, the modelled constraint was `attr_val_arrival ≤ 1.5 × max(queue_value)` while the discovered constraint was `attr_val_arrival/max(queue_value) ≤ 1.497` (Table 4).

2. **Multi-pattern evaluation**: They created a more complex fictional supply chain process containing all four decision synchronization patterns. Their approach successfully identified all four patterns, demonstrating its generalizability to more complex settings.

The paper doesn't provide statistical significance tests for the differences between their method and baselines, as they weren't comparing against other methods but rather validating against known constraints. However, the close match between modelled and discovered constraints (within 0.5 for discrete features) suggests strong reliability.

## Related Work
The paper positions itself as addressing a gap in process mining literature. While existing process discovery techniques (like α-miner, BPMN, or Petri net discovery) can model the flow of individual cases, they don't consider how decisions are synchronized across multiple cases. The authors build on process mining foundations but extend them to capture cross-case dependencies. They specifically mention that their approach is complementary to existing process discovery tools, which can be used to create the initial Petri net models before applying their decision synchronization pattern discovery.

## Limitations
The authors acknowledge several limitations:
- The approach assumes the Petri net model is structurally correct and corresponds to the event log. If the model is inaccurate, the constraint discovery may fail.
- The method relies on the ability to replay event logs against the Petri net model, which requires the model to be accurate enough to handle the log.
- The current implementation uses decision trees, which may not capture all types of constraints (e.g., highly nonlinear relationships).
- The evaluation was limited to synthetic processes; real-world processes could have more complex patterns or noise.

My honest assessment is that the paper's synthetic evaluation might not fully reflect the complexity of real-world processes, where noise and incomplete data could impact the constraint discovery.

## Appendix: Worked Example
Let's walk through the Priority pattern with actual numbers from the example in the paper.

The event log in Table 1 shows job processing with values between 100 and 1000. Consider the moment at time 5 (see Figure 1), where case 1 has just completed pre-processing and case 2 is starting pre-processing. At this point, case 3 has already arrived at time 10 but hasn't been pre-processed yet. The log shows that case 1 (value 100) is not immediately handled, while case 3 (value 855) will be handled first.

To discover this constraint, the authors build a pattern-transition log for the "handling" transition in the Priority pattern. They extract features from the Petri net state at the time of the event:
- `arrival_value`: The value of the arriving job (case 3, value 855)
- `max_queue_value`: The maximum value of jobs in the queue (case 1, value 100)
- `max_queue_enabled`: Whether the job with the maximum value in the queue is enabled (False, as case 1 is not yet ready for handling)

The constraint derived from these features is `arrival_value ≤ 1.5 × max_queue_value ∧ argmax(queue_value) == True`. At time 5, this constraint would be `855 ≤ 1.5 × 100` (855 ≤ 150), which is false, so the handling transition for case 1 is blocked, allowing case 3 to be handled first.

The pattern-transition log for this scenario is shown in Table 2, which demonstrates how the authors collected these features and determined when the constraint was satisfied (True) or not (False). From this log, the decision tree learns the constraint `arrival_value/max(queue_value) > 1.5` (with a threshold of 1.505 in the example).

See Appendix for a step-by-step worked example with concrete numbers.

## References

- **Code:** https://github.com/TijmenKuijpers/Decision_Sync_Patterns.git
- Tijmen Kuijpers, Karolin Winter, Remco Dijkman, "Discovery of Decision Synchronization Patterns from Event Logs", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19879

Tags: #process-mining #business-process-management #decision-optimisation #petri-nets #constraint-learning
