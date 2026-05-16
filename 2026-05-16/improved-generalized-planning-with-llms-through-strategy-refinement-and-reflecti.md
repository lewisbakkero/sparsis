---
title: "Improved Generalized Planning with LLMs through Strategy Refinement and Reflection"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2508.13876"
---

## Executive Summary
This paper presents a method to significantly improve the reliability of LLM-generated generalised plans in PDDL planning domains. By introducing automatic strategy validation and reflection, the authors address a key bottleneck in prior approaches where incorrect strategies directly led to flawed implementations. For production planning systems, this means engineers can deploy LLM-based planning with greater confidence in the quality of generated solutions.

## Why This Matters for Practitioners
If you're currently implementing LLM-based planning systems for logistics, manufacturing, or any domain requiring structured decision-making, this paper offers a practical way to drastically reduce the failure rate of generated plans. The authors show that by validating strategies before code generation and adding reflection to debugging, they achieve 82% average coverage across domains compared to the 50-60% typical in prior work. For your production system, this means implementing the strategy refinement pipeline as a pre-processing step before code generation, and adopting the reflection-based debugging approach to reduce the number of LLM calls needed to achieve reliable plans. Crucially, this approach works with existing LLMs without requiring new model training, making it immediately deployable in your stack.

## Problem Statement
Think of generating a generalised plan like writing a complex software library, where you must define the structure of the code before implementation. Previous approaches were like writing documentation first and then trying to compile from that description, often leading to mismatched assumptions. If the documentation was vague or inaccurate, the implementation would fail, and debugging would require extensive trial-and-error. The problem is that the LLM generates a single natural language strategy that may contain subtle errors, which then propagates directly into the code, like a typo in the library documentation causing all subsequent implementations to fail.

## Proposed Approach
The authors introduce a three-stage pipeline that validates the strategy before code generation. First, the LLM generates a natural language summary of the domain and a strategy in pseudocode form. Second, they automatically validate this pseudocode by using the LLM to generate PDDL plans for debugging tasks based on the strategy, then checking correctness with the VAL validator. Third, they refine the strategy through a reflection step where the LLM identifies and fixes errors before moving to code generation. The approach also includes generating multiple code versions and selecting the best one through debugging.

Here's a simplified overview of the strategy validation process:

```python
def validate_strategy(pseudocode, debugging_tasks):
    for task in debugging_tasks:
        pddl_plan = llm_generate_pddl_plan(pseudocode, task)
        if not val_validate(pddl_plan, task):
            feedback = generate_feedback(pddl_plan, task)
            reflection = llm_reflect(feedback)
            pseudocode = llm_revise(pseudocode, reflection)
    return best_pseudocode
```

## Key Technical Contributions
The authors introduce several novel mechanisms that improve the reliability of LLM-based generalised planning:

1. **Strategy-level validation using PDDL planning**: Instead of validating the final Python code, they validate the strategy at the pseudocode level. The LLM generates PDDL plans based on the pseudocode for debugging tasks, then the VAL validator checks these plans. This catches errors early before code generation, reducing the number of LLM calls needed for debugging. The key insight is that PDDL plans are structured and testable, making validation possible without human intervention.

2. **Reflection-based strategy refinement**: When a debugging task fails, they don't directly ask the LLM to revise the pseudocode. Instead, they prompt the LLM to reflect on "which specific part of the strategy led to the mistake and why" before making revisions. This structured reflection process significantly improves the quality of revisions compared to direct prompts. The paper shows this leads to 20-35% more debugging tasks being successfully resolved in the first two attempts.

3. **Multiple code generation with selective evaluation**: Rather than generating a single code version, they generate multiple Python implementations by varying the order of objects and facts in the prompt. Each version is evaluated against the same debugging tasks, and the best one is selected. This approach leverages the fact that LLMs produce different outputs for the same prompt when prompted with slightly different inputs, improving robustness without increasing LLM cost significantly.

## Experimental Results
The authors evaluated their approach on 17 PDDL domains, including the 7 domains from Silver et al. (2024) and 10 additional domains from Stein et al. (2025). They tested with four LLMs: GPT-4o, Llama3.3, DeepSeek-V3.2, and Qwen3-Thinking.

Their best-performing configuration (DeepSeek-V3.2) achieved an average coverage of 82% across all domains, compared to Silver et al.'s 50-60% average. The paper states that in 14 out of 17 domains, their approach achieved 100% coverage for at least one of three runs. For domains where they achieved perfect coverage, they manually verified that the plans generalised beyond the evaluation data and worked on all tasks generated by the respective instance generators. The paper does not report statistical significance tests but notes that the improvements were consistent across all domains and LLMs.

## Related Work
This work builds directly on Silver et al. (2024), which introduced the concept of using LLMs to generate Python programs representing generalised plans in PDDL. The authors acknowledge that Silver et al. achieved good performance on 5 out of 7 domains with GPT-4 but struggled with broader domain coverage. They specifically address Silver et al.'s key bottleneck, the single strategy generation step, by introducing strategy validation and reflection. The paper also builds on Stein et al. (2025) for their feedback generation approach and draws inspiration from Tang et al. (2024) and Wang et al. (2024) for generating multiple code versions.

## Limitations
The authors acknowledge that the approach requires additional LLM calls for strategy validation and reflection, which increases the computational cost compared to simpler approaches. While they achieve better coverage, they don't report the exact number of additional LLM calls required, which would be critical for production cost estimation. The paper also doesn't test the approach on extremely large domains with tens of thousands of objects or highly complex planning tasks, which might challenge the scalability of the strategy validation process. Finally, the paper doesn't explore the impact of different debugging task selection strategies on the validation process.

## Appendix: Worked Example
Let's walk through the strategy refinement process for a Logistics domain task where the goal is to move package p0 from location l1-0 to l0-0 using the approach described in the paper.

1. **Initial Strategy Generation**: The LLM generates a pseudocode strategy for the Logistics domain:
   ```
   For each package p in packages:
     If p is already at goal location, continue
     Determine current city and goal city of p
     If current city ≠ goal city:
       Find nearest truck to p's location
       Drive truck to p's location if needed
       Load p onto truck
       Drive truck to airport in current city
       Unload p from truck
       Fly airplane to airport (if needed)
       Load p onto airplane
       Fly airplane to goal city
       Unload p from airplane
   ```

2. **Strategy Validation**: The LLM is prompted to generate a PDDL plan for a debugging task using this strategy. The generated plan includes:
   ```
   (load-truck p0 t0 l1-0)
   (drive-truck t0 l1-0 l0-0 c0)
   (fly-airplane a0 l1-0 l0-0)
   (load-airplane p0 a0 l0-0)
   ```
   
3. **Validation Failure**: The VAL validator flags an error at the `load-airplane` step because the precondition "at p0 l0-0" wasn't satisfied at that step (the package was still inside the truck).

4. **Reflection Prompt**: The system prompts the LLM: "Please help me to understand why this mistake happened. Think about (1) which specific part of the strategy led to the mistake and (2) why this part is inadequate."

5. **Reflection Response**: The LLM identifies the mistake: "The strategy assumes that once the truck reaches the airport, the package can be directly loaded onto the airplane. However, the precondition for loading a package onto an airplane requires the package to be at the same location as the airplane, not in a truck. The strategy needs to include an explicit step to unload the package from the truck at the airport before attempting to load it onto the airplane."

6. **Revised Strategy**: The LLM updates the strategy with the missing unloading step:
   ```
   For each package p in packages:
     If p is already at goal location, continue
     Determine current city and goal city of p
     If current city ≠ goal city:
       Find nearest truck to p's location
       Drive truck to p's location if needed
       Load p onto truck
       Drive truck to airport in current city
       Unload p from truck at airport
       Fly airplane to airport (if needed)
       Load p onto airplane
       Fly airplane to goal city
       Unload p from airplane
   ```

7. **Code Generation**: The revised strategy is implemented as a Python program, validated against the debugging tasks, and selected as the best version for use.

## References

- **Code:** https://github.com/coli-saar/genplan-strategy-refine
- Katharina Stein, Nils Hodel, Daniel Fišer, Jörg Hoffmann, Michael Katz, Alexander Koller, "Improved Generalized Planning with LLMs through Strategy Refinement and Reflection", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2508.13876

Tags: #automated-planning #language-models #code-generation #debugging #reflection-based-ai
