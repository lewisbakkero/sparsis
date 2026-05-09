---
title: "A Subgoal-driven Framework for Improving Long-Horizon LLM Agents"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19685"
---

## Executive Summary
This paper introduces a subgoal-driven framework to overcome the fundamental limitations of current LLM agents in long-horizon web navigation tasks. It provides both an online planning mechanism and a milestone-based reinforcement learning approach that significantly improves success rates on complex web tasks, surpassing both proprietary systems and previous open-source SOTA.

## Why This Matters for Practitioners
If you're building or maintaining web automation systems that require multi-step reasoning (like e-commerce navigation, form filling, or information extraction), this paper directly addresses the "mid-task stuck" failures that plague production systems. Current agents fail in nearly 50% of trajectories during web navigation due to poor long-horizon planning. Instead of relying solely on larger models (which only marginally improve performance), the authors demonstrate that implementing subgoal decomposition and milestone-based rewards can boost success rates from 6.4% to 43.0% on the Gemma3-12B model, a 6x improvement without increasing model size. For production teams, this means you can achieve dramatic reliability gains by implementing these techniques in your agent's planning loop and reward function, rather than waiting for larger model releases.

## Problem Statement
Imagine trying to navigate a complex city without a map or street signs, you'd get lost trying to reach a destination after just a few turns. Similarly, current LLM agents struggle with long-horizon web tasks because they lack intermediate checkpoints to maintain direction. They're like tourists wandering through a city without knowing which landmarks to look for, getting stuck in irrelevant pages or repeating the same actions without progress. The paper quantifies this: agents using proprietary models like Gemini-2.5-Pro exhibit "mid-task stuck" behaviours in nearly 50% of trajectories on WebArena-Lite, while even fine-tuned open models like Gemma-12B-SFT fail to progress in over 30% of cases.

## Proposed Approach
The framework combines two core components: a subgoal-based online planning mechanism and a milestone-based reinforcement learning training approach. During inference, the agent decomposes high-level goals into structured intermediate subgoals, enabling hierarchical reasoning. During training, the framework uses milestone-based rewards to provide dense feedback that improves credit assignment. The system integrates these components without requiring model size increases or extensive retraining.

```python
def subgoal_decomposition(goal: str) -> list[str]:
    """Decomposes high-level goal into intermediate subgoals using structured planning."""
    # 1. Generate candidate subgoals using prompt engineering
    subgoals = generate_subgoals(goal)
    
    # 2. Filter subgoals using domain knowledge
    filtered_subgoals = filter_subgoals(subgoals)
    
    # 3. Prioritize subgoals by feasibility and progress
    prioritized_subgoals = rank_subgoals(filtered_subgoals)
    
    return prioritized_subgoals
```

## Key Technical Contributions
The paper makes two significant technical contributions that address the specific limitations of current LLM agents.

1. **Automated Failure Analysis**: The authors developed a systematic analyzer that categorizes failures into four mutually exclusive modes (Stop at Wrong Page, Get Stuck Midway, Fail to Make Reasonable Attempt, Others). This tool identifies the exact decision step where agents deviate by comparing against reference success trajectories from superior models like IBM-CUGA. Unlike prior work that only reported aggregate success rates, this analyzer pinpoints failures at the action level, enabling targeted improvements to planning mechanisms.

2. **Milestoning for Inference-Time Planning**: The framework integrates lightweight subgoal-guided planning directly into the agent's inference loop. The subgoal generator uses a proprietary model (Gemini-2.5-Pro) to produce intermediate milestones, which are then used to guide the agent's step-by-step reasoning. This is different from prior approaches that relied on static decomposition or tree-based search, as it dynamically adapts to the current state rather than predefining all subgoals.

3. **Milestone-Based RL Fine-Tuning (MiRA)**: MiRA uses explicit, semantically verifiable milestones as auxiliary rewards during RL fine-tuning. In contrast to Process Reward Models (PRMs) that use soft, learned signals susceptible to noise, MiRA replaces them with hard objectives. This ensures reliable intermediate supervision without biasing the agent against the primary goal. The authors empirically demonstrate that this approach stabilizes training while improving credit assignment, achieving a 43.0% success rate on Gemma3-12B compared to 6.4% without MiRA.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The framework was evaluated on WebArena-Lite, a benchmark designed to measure web navigation capabilities. Results include:

- Proprietary model (Gemini-2.5-Pro) with online planning: +10% absolute success rate from 32.1% to 42.1% (WebArena-Lite benchmark)
- Open-source model (Gemma3-12B) with MiRA: improved from 6.4% to 43.0% success rate
- Performance surpassed proprietary systems: GPT-4-Turbo (17.6%), GPT-4o (13.9%)
- Outperformed previous open-source SOTA (WebRL: 38.4%)

The authors didn't explicitly state statistical significance tests, though they reported a 6x improvement on Gemma3-12B, suggesting substantial impact. The paper demonstrates that both components (online planning and MiRA) are necessary for the full performance gain, with online planning alone providing a 10% improvement and MiRA alone providing a 36.6% improvement.

## Related Work
The paper positions itself between two approaches: hierarchical planning methods like VSC-RL and Process Reward Models (PRMs). Unlike VSC-RL, which relies on latent representations that suffer from training instability, their approach uses explicit, semantically verifiable milestones. Unlike PRMs, which depend on noisy, learned signals, their method uses hard objectives for intermediate supervision. This bridges the gap between the two research directions, achieving the benefits of both while avoiding their pitfalls. The authors also build upon prior work in goal-conditioned RL but address the critical challenge of sparse rewards in web navigation through milestone-based shaping.

## Limitations
The authors acknowledge that their method requires a larger model (Gemini-2.5-Pro) for subgoal generation during inference, which might not be accessible to all practitioners. The paper doesn't test the approach on mobile interfaces or operating system control beyond web navigation. While the method improves success rates significantly, it doesn't address the computational overhead of the subgoal generation process during inference. The authors also note that this approach is currently limited to tasks with clear, structured goals, and may not generalise well to completely open-ended or highly dynamic environments.

## Appendix: Worked Example
Let's walk through a concrete example of how the subgoal-driven framework works for the task: "Find the nearest cafe within the 50-mile range of CMU on the map."

1. **Initial Goal**: "Find the nearest cafe within the 50-mile range of CMU on the map"
   
2. **Subgoal Decomposition** (using Gemini-2.5-Pro as subgoal generator):
   - "Open the web browser and start from the map"
   - "Search for 'CMU' on the map"
   - "Select the nearby restaurant and add filter '<50 miles'"
   - "Choose the closest cafe and report the info"

3. **Online Planning Execution**:
   - The agent first executes "Open the web browser and start from the map" (action: open browser, navigate to map service)
   - After observing the map interface, it executes "Search for 'CMU' on the map" (action: type 'CMU' in search box, press enter)
   - The agent then executes "Select the nearby restaurant and add filter '<50 miles'" (action: click on map marker, set filter)
   - Finally, it executes "Choose the closest cafe and report the info" (action: select first cafe in results, extract address)

4. **Milestone-Based Reward During Training**:
   - For each subgoal, the agent receives a dense reward (e.g., +0.25 for completing the first subgoal, +0.25 for the second, etc.)
   - This dense reward structure provides continuous feedback that helps the agent identify which actions lead to progress
   - Unlike traditional RL where feedback is sparse (only at the end), this approach gives immediate guidance at each milestone

The paper doesn't specify the exact reward values, but the authors note that this dense reward structure significantly improved credit assignment during training, leading to the 43.0% success rate on Gemma3-12B.

## References

- Taiyi Wang, Sian Gooding, Florian Hartmann, Oriana Riva, Edward Grefenstette, "A Subgoal-driven Framework for Improving Long-Horizon LLM Agents", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19685

Tags: #information-retrieval #multi-agent #reinforcement-learning #web-navigation #long-horizon-planning
