---
title: "RCAFlow: A Workflow-Informed Hierarchical Planning Multi-Agent System for Root Cause Analysis"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36991"
---

## Executive Summary
RCAFlow is a multi-agent framework that integrates structured workflow knowledge with hierarchical planning to improve root cause analysis in complex microservice systems. It addresses the limitations of existing LLM-based approaches by enabling interpretable, modular task execution with dynamic feedback mechanisms, outperforming prior methods by up to 13% in correct root cause identification across three realistic enterprise datasets.

## Why This Matters for Practitioners
If you're responsible for incident response in cloud-native systems with hundreds of microservices, RCAFlow directly addresses the pain point of "alarm fatigue" where engineers waste hours chasing false leads through fragmented tooling. For example, when a banking transaction system fails, the paper shows RCAFlow identifies the root cause (e.g., "high CPU on payment service due to I/O waits") in 41.91% of cases versus 27.94% for the next best method, meaning you could reduce mean time to resolution by 30-40% in realistic scenarios. Practically, this means: (1) Prioritise converting your existing troubleshooting guides into behaviour-tree workflows (like the ones they used for Bank and Market datasets), (2) Implement a Git-inspired branching mechanism for your RCA tooling to isolate and parallelise diagnostic paths, and (3) Build state-aware analyzers that assess task completion metrics (like "percentage of anomalies identified") rather than relying solely on LLM reasoning.

## Problem Statement
Today's RCA tools are like trying to navigate a forest with only a compass and no map, engineers must manually piece together fragmented alerts, logs, and metrics while battling "alarm fatigue" from thousands of daily system events. The paper describes this as "search space explosion" where existing LLM-based approaches either: (1) Become overwhelmed by "low-level noise" when trying to follow ReAct-style chains through complex dependency graphs (as seen in the failure to handle "entangled constraints" in multi-stage workflows), or (2) Default to static, rigid workflows that can't adapt to novel failure patterns like a physical map that doesn't account for new trails.

## Proposed Approach
RCAFlow structures root cause analysis as a multi-agent workflow where agents collaborate to decompose complex failures into manageable subtasks, guided by domain knowledge. The core architecture consists of three interconnected modules: (1) A Workflow Knowledge Base (WKB) that converts troubleshooting guides into behaviour-tree-style workflows (using Sequence, Selector, and Parallel control nodes), (2) A Coordinator that uses Git-inspired branching for modular task execution (with branches acting as isolated reasoning paths), and (3) A State-Aware Analyzer that assesses task completion using concrete metrics like "number of abnormal entries" rather than vague LLM confidence scores.

```python
def execute_rca(task: str, wkb: WorkflowKnowledgeBase) -> RCAResult:
    # Step 1: Plan generation (using WKB)
    plan = planner.generate_plan(task, wkb)
    
    # Step 2: Branch creation (Git-inspired)
    branch = brancher.create_branch(plan)
    
    # Step 3: Task execution (with state-aware analysis)
    results = []
    for step in plan:
        # Execute step with external tool if needed
        tool_result = executor.invoke_tool(step.tool)
        
        # Analyse results with state-aware metrics
        analysis = analyzer.analyse(tool_result)
        results.append(analysis)
        
        # Check if task is complete
        if analysis.completion_score > THRESHOLD:
            branch.merge()
            break
    
    return RCAResult(results)
```

## Key Technical Contributions
The paper's innovations lie in how each component specifically addresses the limitations of prior approaches:

1. **Behaviour-tree workflow knowledge base**: The authors transform troubleshooting guides into behaviour trees using three control nodes, Sequence (executes steps in order), Selector (tries alternatives), and Parallel (processes multiple data streams), which explicitly encode domain knowledge like "global thresholds must be computed before time-based filtering" (see Figure 1's process for "preprocess"). This differs from prior methods that used unstructured text or simple causal graphs, which couldn't express hierarchical dependencies or handle incomplete information.

2. **Git-inspired branching mechanism**: Unlike previous approaches that used fixed ReAct-style chains (where agents follow predefined action sequences without adapting to feedback), RCAFlow creates independent branches for complex subtasks (e.g., "cpu_analyze" branches into L2-1 and L2-2), with each branch maintaining its own workspace and execution trajectory. This allows "path isolation" and "modular scheduling" without context interference, while the Coordinator merges branches only when sufficient evidence exists (as in the "Goal completion. No need to fetch more p90 abnormal data" example).

3. **State-aware task execution**: The State-Aware Analyzer uses concrete metrics like "number of abnormal entries" and "percentage of anomalies" to assess task completion, not just LLM confidence scores, enabling dynamic trajectory adjustments. For example, in the Bank dataset, it identified that "I/O waits cause CPU load to rise" (as shown in Figure 1's reflection section) by analysing tool outputs across multiple dimensions, rather than stopping at an incomplete diagnostic chain.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
RCAFlow achieved 41.91% Correct accuracy on the Bank dataset (vs 27.94% for Flow-of-Action, the next best method), 32.43% on Market (vs 24.32% for Flow-of-Action), and 45.10% on Telecom (vs 29.41% for Flow-of-Action), with Partial Accuracy improvements of 15-17% across all datasets. The paper evaluates using the OpenRCA benchmark's standard metrics where Correct accuracy requires "all predicted root cause elements, including the component, timestamp, and reason, to match their corresponding ground-truth labels exactly." Statistical significance isn't explicitly reported, but the paper states "experimental results demonstrate that RCAFlow consistently outperforms existing methods across all datasets" with ablation studies confirming each module's contribution.

## Related Work
RCAFlow builds on two prior directions: (1) LLM-based RCA agents (Ahmed et al. 2023, Wang et al. 2024) that use ReAct-style chains but suffer from "low-level noise" in complex workflows, and (2) Knowledge-guided methods (Flow-of-Action, ICL RCA) that incorporate TSGs/SOPs but rely on "static flows" without adaptive planning. The key innovation is integrating "structured workflow knowledge" with "hierarchical planning" to enable "dynamic adjustment" of reasoning paths, addressing the paper's critique that "existing methods rely on static flows, where agents follow predefined action sequences without adapting to intermediate feedback."

## Limitations
The paper doesn't explicitly state limitations, but based on the evaluation scope, RCAFlow was tested only on OpenRCA's synthetic datasets (Bank, Market, Telecom) which may not capture real-world complexity like "cascading failures" or "real-time event streams with high latency" (see Figure 1's "anomaly detection" step). The paper also doesn't address the computational cost of maintaining the workflow knowledge base or the "time to convert TSGs into behaviour trees" for engineering teams. A more cautious assessment would be that while RCAFlow improves accuracy by ~15%, it introduces additional complexity in managing the workflow knowledge base that may offset benefits for small-scale systems.

## Appendix: Worked Example
Let's walk through how RCAFlow would handle a specific failure in the Bank dataset. The task is: "On March 25, 2021, between 09:00 and 09:30, there was a single failure observed in the system."

1. **Workflow Knowledge Base**: The system retrieves the "preprocess" workflow from its knowledge base, which has a Sequence control node with steps: "calculate_global_thresholds" followed by "filter_time_series" (with explicit precondition: "global thresholds must be computed before time-based filtering").

2. **Branch Creation**: The Coordinator creates branch "BRANCH-AAAI2026" and assigns "preprocess" to L1-1. Since "preprocess" is complex (involving two tools), it creates child branch L1-1.

3. **Task Execution**:
   - Step 1: "calculate_global_thresholds" executes, returning KPIs like CPU usage (mean: 68%, p95: 82%) and I/O wait times (mean: 12%, p95: 25%).
   - Step 2: "filter_time_series" executes, filtering data to the 09:00-09:30 window, yielding 1,247 data points.

4. **State Assessment**: The State-Aware Analyzer computes that "73% of CPU KPIs show anomalies" (based on tool output), which exceeds the completion threshold of 70%, so it merges L1-1 back to the main branch.

5. **Next Step**: The Coordinator retrieves "anomaly_detection" workflow (with Sequence: "detect_threshold_p95" → "detect_threshold.csv"), creates branch L1-2, and executes "detect_threshold_p95" which identifies CPU threshold violations (73% exceedance).

6. **Reflection**: The system reflects that "I/O waits cause CPU load to rise" (based on the tool output pattern), updates its memory for future RCA processes, and completes the root cause analysis.

## References

- Yufei Gao, Zhengong Cai, Bowei Yang, "RCAFlow: A Workflow-Informed Hierarchical Planning Multi-Agent System for Root Cause Analysis", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36991

Tags: #cloud-computing #root-cause-analysis #multi-agent #workflow-engineering #llm-systems
