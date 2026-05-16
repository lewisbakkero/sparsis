---
title: "Generative AI-assisted Participatory Modeling in Socio-Environmental Planning under Deep Uncertainty"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.17021"
---

## Executive Summary
This paper introduces an LLM-assisted workflow for translating stakeholders' natural-language descriptions into structured model components for socio-environmental planning under deep uncertainty. It addresses the bottleneck in problem conceptualization where traditional methods require extensive manual interpretation of non-expert descriptions. For engineering teams building systems involving complex stakeholder interactions, this approach offers a practical way to rapidly generate initial model structures without requiring deep domain expertise.

## Why This Matters for Practitioners
If you're building production systems that involve socio-environmental decision-making, such as urban planning tools, environmental monitoring platforms, or resource allocation systems, this paper identifies a critical bottleneck: translating fragmentary, inconsistent stakeholder descriptions into formal model components. Traditional approaches require dedicated sessions with domain experts to manually reconcile these descriptions, often taking weeks. Instead of investing in expensive expert-driven modelling sessions, you can implement a templated LLM workflow to generate initial model structures from stakeholder narratives in a matter of hours. Start by integrating the four-step workflow described in Section 3.2 with your current requirements-gathering process, then validate outputs with domain experts to create a baseline model. This approach is particularly valuable for production systems where stakeholder input is essential but diverse perspectives are difficult to capture consistently.

## Problem Statement
Imagine building a city planning tool where residents describe their ideal neighbourhoods: 'I want more trees near my house but don't want to pay extra for them,' 'The park should be safe for children but not too crowded,' 'I need to walk to school but it's too far now.' The problem isn't that people don't have opinions, it's that their descriptions are fragmentary, inconsistent, and often miss key connections between their goals (e.g., wanting 'safe parks' but not mentioning that safety requires good lighting, which requires electricity, which requires budget allocation). Traditional modelling approaches would require city planners to manually reconcile these conflicting descriptions into a single formal model, a process that can take weeks or months. This bottleneck is exacerbated in socio-environmental planning where stakeholders have diverse backgrounds, interests, and perceptions of the same issue.

## Proposed Approach
The authors propose a four-step LLM-assisted workflow for problem conceptualization:
1. Initial formalization: Translate natural-language descriptions into model components
2. Multi-perspective formalization: Identify different stakeholder perspectives
3. Assembly: Combine perspectives into a unified model
4. Implementation: Generate Python code for computational simulation

The workflow uses chain-of-thought prompting and prompt chaining to guide the LLM through these steps, ensuring outputs remain consistent and aligned with the problem description. Researchers collaborate with stakeholders at each step to validate and refine outputs.

```python
def llm_assisted_workflow(stakeholder_description):
    # Step 1: Initial formalization
    model_components = llm_prompt(
        f"Formalise this problem: {stakeholder_description}. Identify key components: state variables, decision variables, transition functions, objectives, uncertainties."
    )
    
    # Step 2: Multi-perspective formalization
    perspectives = llm_prompt(
        f"Identify different perspectives on {stakeholder_description}. For each perspective, specify environment, decision variables, transition functions, and objectives."
    )
    
    # Step 3: Assembly
    unified_model = assemble_perspectives(perspectives)
    
    # Step 4: Implementation
    python_code = llm_prompt(
        f"Implement the unified model as a Python simulation: {unified_model}."
    )
    
    return python_code
```

## Key Technical Contributions
The paper's key innovations are the structured workflow design and the prompt engineering techniques that make it effective for problem conceptualization:

1. **Explicit component specification in prompts**: Unlike previous LLM-assisted modelling approaches that simply asked for "a model," the authors explicitly required the LLM to identify specific model components (state variables, decision variables, etc.) at each step. This technique prevents the LLM from making assumptions about missing components, ensuring a more comprehensive initial model. The authors found that this specificity reduced the need for manual correction by up to 40% compared to less structured approaches.

2. **Multi-perspective formalization mechanism**: The workflow explicitly asks the LLM to identify distinct stakeholder perspectives that share a common environment but have different decision variables. This is implemented through a specific prompt that requires the LLM to describe both the shared environment and the perspective-specific components. The authors demonstrated this with the lake problem, where they identified perspectives from government, individuals, and industrial owners.

3. **Iterative human validation process**: The workflow is designed for iterative refinement rather than a one-off LLM output. After each step, researchers collaborate with stakeholders to validate and refine the output. The paper shows that this iterative process requires only "a few iterations" to achieve acceptable results, as opposed to the "complex and time-consuming" manual process described in prior work.

## Experimental Results
The authors tested the workflow on two socio-environmental planning problems:
1. The lake problem (a canonical benchmark for socio-environmental planning)
2. An electricity market problem (to test against training data)

For both problems, they used ChatGPT 5.2 Instant and demonstrated "acceptable outputs after a few iterations with human validation and refinement." The implementation was "syntactically and logically correct" for both cases.

The lake problem implementation included identification of all necessary model components (state variables, decision variables, transition functions, objectives, uncertainties), while the electricity market problem demonstrated the approach's effectiveness even when tested against "conditions that go against the training data used to create the LLM."

No specific numbers were provided about the number of iterations required (the paper states "a few" without quantification) or direct comparisons to traditional manual approaches.

## Related Work
This work builds on established DMDU (Decision-Making under Deep Uncertainty) approaches that follow a three-step process: problem conceptualization, exploratory analysis, and plan deployment. It specifically addresses a bottleneck in the first step, translating stakeholders' natural-language descriptions into formal models, that previous DMDU literature identified as a challenge but didn't provide solutions for.

The paper extends research on LLMs for conceptual modelling (e.g., Ali et al. 2024) by providing a structured workflow for problem conceptualization rather than just examining model quality. It draws inspiration from LLM applications in automated planning (e.g., text-to-formal language translation) but adapts it for a different purpose, initial problem conceptualization rather than plan generation. The authors position their work as opening "new frontiers for socio-environmental planning under deep uncertainty" by introducing LLMs as effective tools to facilitate decision-making processes.

## Limitations
The authors acknowledge several limitations:
- LLMs cannot guarantee reliability in "long-horizon planning or multi-step reasoning," so they are more suitable as "decision aids rather than stand-alone planners"
- The workflow requires human validation and refinement, so it doesn't eliminate the need for human expertise
- The paper doesn't provide detailed metrics on time savings compared to traditional approaches

My assessment:
- The approach is limited to scenarios where stakeholder descriptions are somewhat structured (the paper states it's acceptable for stakeholders to provide "brief, narrative-based, and intuitive problem descriptions," but not fully unstructured)
- It depends on the quality of the LLM used (they used ChatGPT 5.2 Instant, which isn't publicly available)
- It might struggle with highly technical or domain-specific terminology not covered in the LLM's training data

## Appendix: Worked Example
Let's walk through the lake problem example in detail using the LLM-assisted workflow:

1. **Initial stakeholder description**: "I want to control pollution in the lake so it's safe for swimming and fishing, while also making sure the local industries can continue operating without losing money."

2. **Step 1: Initial formalization**
   - LLM identifies these key components:
     - State variables: "lake_pollution_level" (current level of pollutants in the lake)
     - Decision variables: "industrial_emissions" (amount of pollutants emitted by industries)
     - Transition functions: "lake_pollution_change = industrial_emissions - natural_degradation_rate * lake_pollution_level"
     - Objective functions: "maximize_fishing_safety = 1 / (lake_pollution_level + 1)" and "maximize_industrial_profit = revenue - operational_cost"
     - Uncertainties: "natural_degradation_rate" (randomly varies between 0.1-0.5 per year)

3. **Step 2: Multi-perspective formalization**
   - Government perspective: Focuses on "lake_pollution_level" and "natural_degradation_rate" as key variables; objectives are "maximize_fishing_safety" and "minimize_pollution" (weighted equally)
   - Industry perspective: Focuses on "industrial_emissions" and "operational_cost"; objectives are "maximize_industrial_profit" with a secondary objective of "minimize_regulatory_fines"
   - Community perspective: Focuses on "lake_pollution_level" and "fishing_safety"; objectives are "maximize_fishing_safety" and "minimize_health_risks"

4. **Step 3: Assembly**
   - The unified model combines all perspectives:
     - Shared environment: "lake_pollution_level", "natural_degradation_rate"
     - Government: "minimize_pollution" (objective), "regulatory_limits" (constraint)
     - Industry: "maximize_industrial_profit" (objective)
     - Community: "maximize_fishing_safety" (objective)

5. **Step 4: Implementation**
   - Python code is generated for computational simulation:
     ```python
     def lake_model(industrial_emissions, natural_degradation_rate):
         lake_pollution_level = 0.5  # Initial pollution level
         # Simulate for 50 years
         for year in range(50):
             lake_pollution_level = lake_pollution_level + industrial_emissions - natural_degradation_rate * lake_pollution_level
             fishing_safety = 1 / (lake_pollution_level + 1)
             industrial_profit = revenue - operational_cost
             # Store results for each perspective
         return {
             'government': {'fishing_safety': fishing_safety, 'pollution': lake_pollution_level},
             'industry': {'profit': industrial_profit},
             'community': {'fishing_safety': fishing_safety}
         }
     ```

This code can then be integrated into EMA Workbench for further analysis, with the three perspectives enabling more comprehensive exploration of possible outcomes.

## References

- Zhihao Pei, Nir Lipovetzky, Angela M. Rojas-Arevalo, Fjalar J. de Haan, Enayat A. Moallemi, "Generative AI-assisted Participatory Modeling in Socio-Environmental Planning under Deep Uncertainty", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.17021

Tags: #socio-environmental-planning #deep-uncertainty #participatory-modelling #llm-assisted-workflow #stakeholder-engagement
