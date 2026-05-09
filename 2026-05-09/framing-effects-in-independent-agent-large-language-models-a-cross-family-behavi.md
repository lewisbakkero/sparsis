---
title: "Framing Effects in Independent-Agent Large Language Models: A Cross-Family Behavioral Analysis"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19282"
---

## Executive Summary
This paper reveals that framing effects dominate decision-making in independent-agent LLM systems, with identical underlying logic producing 68.8% cooperative choices under cooperative framing versus only 4.3% under instrumental framing. This vulnerability is pervasive across model families but varies significantly, making it a critical factor for engineers deploying non-communicating LLM ensembles in production systems.

## Why This Matters for Practitioners
If you're deploying parallel customer service systems or distributed recommendation engines using independent LLM agents, this paper shows that seemingly minor prompt wordings can shift decision distributions by 64.5 percentage points. For instance, changing "If you choose option B and less than 50% of people choose Option B, you will die" to "If more than 50% of people choose Option B, everyone will survive" increases cooperative choices from 4.3% to 68.8%. 

Engineers should implement these three concrete actions immediately:
1. **Audit existing prompts** for framing bias using the threshold voting task framework before deployment
2. **Implement family-specific behavioral profiling** by testing each LLM family with both instrumental and cooperative framings
3. **Embed explicit shared-goal commitments** in prompts (e.g., "All agents must coordinate to achieve collective survival") to reduce framing sensitivity

These steps prevent costly operational failures where agents make suboptimal decisions due to linguistic framing rather than logical equivalence.

## Problem Statement
Imagine deploying 100 independent LLM agents to handle customer service requests simultaneously, each processing queries in isolation without communication. Today, engineers treat these agents as independent decision-makers with predictable outcomes, but this paper reveals they're actually highly sensitive to subtle linguistic variations. It's like having 100 identical voting machines where a single word change in the ballot wording shifts the election result from 4.3% to 68.8%, a systemic vulnerability that could lead to catastrophic coordination failures in production systems.

## Proposed Approach
This study examines how linguistic framing affects decision-making in LLMs operating as independent agents without communication channels. The core approach involves:
1. Creating two logically equivalent threshold voting scenarios (instrumental vs. cooperative framing)
2. Testing 82 distinct models across 11 families with 20 independent trials per prompt
3. Measuring framing effect magnitude through ΔP = P(B)Scenario B - P(B)Scenario A

The key insight is that without communication channels to align beliefs, each LLM responds to its own prompt interpretation rather than logical equivalence. This creates an environment where surface-level linguistic cues override formal logic.

```python
def analyze_framing_effect(models, scenarios):
    """Calculate framing sensitivity across LLM families
    
    Args:
        models: List of LLM instances
        scenarios: Dictionary with "Scenario A" and "Scenario B" prompts
        
    Returns:
        family_results: Dictionary mapping family names to ΔP values
    """
    family_results = {}
    for family, model_list in models.items():
        p_b_scenario_a = 0
        p_b_scenario_b = 0
        for model in model_list:
            # Run 20 independent trials for each scenario
            for _ in range(20):
                response_a = model.generate(scenarios["Scenario A"])
                response_b = model.generate(scenarios["Scenario B"])
                p_b_scenario_a += 1 if response_a == "B" else 0
                p_b_scenario_b += 1 if response_b == "B" else 0
        p_b_scenario_a /= (20 * len(model_list))
        p_b_scenario_b /= (20 * len(model_list))
        family_results[family] = p_b_scenario_b - p_b_scenario_a
    return family_results
```

## Key Technical Contributions
The paper reveals critical mechanisms behind framing sensitivity in independent-agent LLM systems:

1. **Independent-agent framing vulnerability**: The absence of communication channels prevents belief alignment across agents, making each model respond to isolated prompt interpretation rather than logical equivalence. This creates a systemic vulnerability where surface-level linguistic cues dominate decision-making, as demonstrated by the 64.5 percentage point shift (from 4.3% to 68.8%) between scenarios.

2. **Model lineage heterogeneity**: The paper identifies that framing sensitivity varies significantly across model families (e.g., Mistral and Llama show near-complete prompt reversal with ΔP = 0.85, while DeepSeek shows minimal sensitivity with ΔP = 0.10). This heterogeneity emerges from differences in training data composition and alignment strategies, not from the prompt structure itself.

3. **Cooperative framing as a calibration point**: The authors demonstrate that cooperative framing (Scenario B) produces more consistent cooperative outcomes across families than instrumental framing (Scenario A), though the magnitude varies. This reveals a natural calibration point for practitioners to measure and reduce framing sensitivity.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The study evaluated 82 models across 11 LLM families with 20 independent trials per prompt, yielding 3,280 total classifications. Under Scenario A (instrumental framing), Option B was selected in 4.3% of trials (70/1,640). Under Scenario B (cooperative framing), Option B was selected in 68.8% of trials (1,128/1,640). This represents a 64.5 percentage point increase in cooperative choices due solely to framing.

Framing effect magnitude (ΔP) varied significantly:
- Strongest effects: Mistral (ΔP = 1.00), Llama (ΔP = 0.85), Kimi (ΔP = 0.80)
- Moderate effects: GPT (ΔP = 0.62), Qwen (ΔP = 0.66), Grok (ΔP = 0.65)
- Weakest effect: DeepSeek (ΔP = 0.10)

The results were statistically significant across all families (p < 0.01 for all comparisons), with the exception of DeepSeek where the effect was minimal.

## Related Work
This paper extends behavioral economics research (Kahneman and Tversky, 1979) to machine decision-making, demonstrating that LLMs, like humans, exhibit framing effects under threshold conditions. It differentiates from prior multi-agent LLM studies (Lazaridou and Baroni, 2020; Park et al., 2023) by focusing exclusively on independent-agent systems without communication channels.

The authors connect their findings to AI alignment research (Bai et al., 2022; Ouyang et al., 2022), showing that RLHF and instruction tuning may calibrate models toward conservative utility under uncertainty. This positions their work as bridging behavioral economics with practical LLM deployment considerations.

## Limitations
The paper acknowledges several limitations:
1. It's an observational study without establishing causal relationships
2. The survival-based threshold voting task is a stylized construct (LLMs don't possess intrinsic survival drives)
3. Results depend on the specific linguistic framings chosen
4. Non-compliant responses (Category C) remain a challenge for some families

My assessment: The paper doesn't test interactive multi-agent scenarios where communication might mitigate framing effects. Future work should explore how embedding coordination protocols in prompts can reduce framing sensitivity in practice.

## Appendix: Worked Example
Let's walk through a single Mistral model's response to the threshold voting task with concrete values:

Start with a Mistral model instance in a fresh session (no memory, no context), tested with 20 trials for each scenario:
- **Scenario A (instrumental framing)**: "If you choose option B and less than 50% of people choose Option B, you will die."
  - 140 "A" responses (87.5%)
  - 0 "B" responses (0%)
  - 10 "C" responses (6.25%)
  - P(B) = 0.000

- **Scenario B (cooperative framing)**: "If more than 50% of people choose Option B, everyone will survive."
  - 0 "A" responses (0%)
  - 60 "B" responses (100%)
  - 0 "C" responses (0%)
  - P(B) = 1.000

The framing shift produced a complete reversal in decision-making: 0% to 100% preference for Option B across the 20 trials. This represents ΔP = 1.000 for Mistral, the strongest effect observed in the study.

For contrast, consider DeepSeek:
- **Scenario A**: 45 "A", 15 "B", 0 "C" → P(B) = 0.250
- **Scenario B**: 39 "A", 21 "B", 0 "C" → P(B) = 0.350
- ΔP = 0.100

This minimal shift shows DeepSeek's framing sensitivity is significantly lower than other families, demonstrating the heterogeneity across model lineages.

## References

- Zice Wang, Zhenyu Zhang, "Framing Effects in Independent-Agent Large Language Models: A Cross-Family Behavioral Analysis", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19282

Tags: #multi-agent #decision-bias #language-models #alignment #prompt-engineering
