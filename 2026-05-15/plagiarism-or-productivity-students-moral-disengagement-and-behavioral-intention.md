---
title: "Plagiarism or Productivity? Students Moral Disengagement and Behavioral Intentions to Use ChatGPT in Academic Writing"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19549"
---

## Executive Summary
This paper investigates how Filipino students rationalize using ChatGPT for academic writing through the lens of moral disengagement theory. It finds that students' behavioral intentions are most strongly influenced by their attitudes toward AI use, which are shaped by institutional gaps rather than internal ethical reasoning. For engineers building AI tools in education, this means clear ethical guidelines and classroom support are more effective than technical restrictions alone.

## Why This Matters for Practitioners
If you're developing educational AI tools, this paper shows that students' decisions to use ChatGPT depend more on how schools frame AI policies than on technical capabilities. Implementations should focus on creating transparent academic integrity policies that address institutional gaps, not just adding technical watermarks or detection systems. For example, when designing an AI writing assistant, include features that guide students through ethical decision-making rather than just blocking "plagiarism" (which students justify via attribution of blame). The study confirms that students who see unclear policies (M = 3.44) are 27% more likely to use ChatGPT regularly than those with clear guidelines (M = 2.15), suggesting that policy clarity directly affects usage patterns. Engineers should collaborate with educators to build adaptive guidance systems that prompt students to reflect on ethical implications before using AI.

## Problem Statement
Today's educational AI tools face a paradox: students use them for academic writing while simultaneously rationalizing their use as "not cheating." Imagine a student who uses ChatGPT to draft an essay but rewrites it to feel "original", they've performed cognitive gymnastics to avoid ethical conflict, much like someone who "bends" rules to fit their moral framework rather than confronting the system's limitations. This paper exposes how institutional gaps (unclear policies) and peer influence create a self-reinforcing cycle of AI use that technical solutions alone cannot solve.

## Proposed Approach
The authors integrate moral disengagement theory with the Theory of Planned Behaviour to model students' ChatGPT usage intentions. This psychological framework analyzes five moral disengagement mechanisms (moral justification, euphemistic labelling, displacement of responsibility, minimising consequences, attribution of blame) as predictors of attitudes, subjective norms, and perceived behavioral control, which then predict behavioral intention. The study surveyed 418 Filipino students with ChatGPT experience using validated Likert-scale instruments measuring all constructs.

```python
# Pseudocode for the integrated model (as described in the paper)
def integrated_model(moral_disengagement_mechanisms, student_population):
    # Moral disengagement mechanisms (inputs)
    moral_justification = student_population["moral_justification"]
    euphemistic_labeling = student_population["euphemistic_labeling"]
    displacement_of_responsibility = student_population["displacement_of_responsibility"]
    minimizing_consequences = student_population["minimizing_consequences"]
    attribution_of_blame = student_population["attribution_of_blame"]
    
    # Predict attitudes
    attitudes = f(moral_justification, euphemistic_labeling, displacement_of_responsibility, minimizing_consequences, attribution_of_blame)
    
    # Predict subjective norms
    subjective_norms = g(moral_justification, euphemistic_labeling, displacement_of_responsibility, minimizing_consequences, attribution_of_blame)
    
    # Predict perceived behavioral control
    perceived_control = h(moral_justification, minimizing_consequences, attribution_of_blame)
    
    # Predict behavioral intention
    behavioral_intention = i(attitudes, subjective_norms, perceived_control)
    
    return behavioral_intention
```

## Key Technical Contributions
This paper doesn't present technical systems but offers a validated psychological framework for understanding AI adoption in education. The key contributions are:

1. **Quantified the strongest predictor of behavioral intention**: The study found that students' attitudes toward ChatGPT use had the strongest influence on behavioral intention (β = 0.499, p < .001), confirming the Theory of Planned Behaviour's core assumption. This is more critical than perceived control (β = 0.253, p < .001) or subjective norms (β = 0.098, p = .037), showing that students' internal evaluations matter more than peer pressure.

2. **Identified the institutional gap as the primary moral disengagement driver**: Attribution of blame (e.g., students blaming unclear policies) emerged as the strongest predictor of both attitudes (β = 0.319, p < .001) and perceived behavioral control (β = 0.421, p < .001). This explains why students feel comfortable using ChatGPT: they perceive institutional failures, not personal moral failures.

3. **Demonstrated moral disengagement mechanisms operate externally**: Moral justification (e.g., "I'm using it for learning") didn't predict attitudes (β = 0.068, p = .178), while external rationalizations like attribution of blame did. This shows students don't internalize ethical frameworks but rely on external justifications, meaning technical solutions (like detection systems) can't solve the root issue.

See Appendix for a step-by-step walkthrough of how attribution of blame (M = 3.58) translates to behavioral intention (M = 3.37).

## Experimental Results
The study surveyed 418 Filipino college students with ChatGPT experience across 12 academic disciplines. The integrated model explained 60.6% of the variance in behavioral intention (R² = .606), with attitudes as the strongest predictor (β = 0.499, p < .001). Key findings:

- **Attribution of blame** had the strongest influence on attitudes (β = 0.319, p < .001) and perceived control (β = 0.421, p < .001).
- **Minimising consequences** was the second strongest predictor for attitudes (β = 0.313, p < .001).
- **Moral justification** did not predict attitudes (β = 0.068, p = .178), showing students rely more on external than internal justifications.
- **Subjective norms** had the weakest impact (β = 0.098, p = .037), contradicting assumptions that peer pressure drives AI use.

The study used validated instruments with Cronbach's alpha > .70 for all constructs, ensuring reliability.

## Related Work
This paper builds on moral disengagement theory (Raney, 2020) and the Theory of Planned Behaviour (TPB) as applied to AI use. It extends prior work by Qu & Wang (2025), who observed students interpreting AI as neutral learning tools, by showing how institutional gaps enable these interpretations. Unlike Martinez (2025), which reported rapid AI adoption without examining moral mechanisms, this study explicitly connects policy clarity to ethical reasoning. It differs from Bozkurt et al. (2024), whose manifesto on AI in education focused on structural solutions without empirical validation of student behaviour.

## Limitations
The study focused only on students with ChatGPT experience (n = 418), excluding non-users who might have different ethical frameworks. The survey relied on self-reported data, which may reflect social desirability bias rather than actual behaviour. The model explains 60.6% of intention variance, leaving 39.4% unaccounted for, likely due to emotional factors, peer influence, and convenience, which the authors acknowledge. The research is limited to Filipino students, and cultural differences may affect how moral disengagement mechanisms operate in other contexts.

## Appendix: Worked Example
Let's walk through how attribution of blame (M = 3.58) influences behavioral intention (M = 3.37) using survey data from the study:

1. **Start with the moral disengagement mechanism**: Students with high attribution of blame (e.g., "institutions don't provide clear AI policies") score M = 3.58 on the attribution scale (5-point Likert, where 5 = strongly agree).

2. **This correlates with attitudes**: For every 1-point increase in attribution of blame, students' attitudes toward ChatGPT use improve by 0.319 points (β = 0.319, p < .001). At M = 3.58, this contributes to a positive attitude score (M = 3.51).

3. **Attitudes directly affect intention**: A 1-point increase in attitude score correlates with a 0.499-point increase in behavioral intention (β = 0.499, p < .001). At M = 3.51, this translates to a behavioral intention score of M = 3.37.

4. **The full path**: Attribution of blame (3.58) → Attitudes (3.51) → Behavioral intention (3.37), with each step having statistical significance (p < .001).

This shows how institutional gaps (high attribution of blame) directly shape students' decisions to use ChatGPT, rather than students' internal moral reasoning.

## References

- John Paul P. Miranda, Rhiziel P. Manalese, Mark Anthony A. Castro, Renen Paul M. Viado, Vernon Grace M. Maniago, Rudante M. Galapon, Jovita G. Rivera, Amado B. Martinez Jr, "Plagiarism or Productivity? Students Moral Disengagement and Behavioral Intentions to Use ChatGPT in Academic Writing", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19549

Tags: #education #academic-integrity #behavioral-modelling #moral-psychology #institutional-policy
