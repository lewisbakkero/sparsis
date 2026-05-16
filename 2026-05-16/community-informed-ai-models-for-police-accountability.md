---
title: "Community-Informed AI Models for Police Accountability"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2402.01703"
---

## Executive Summary

This paper presents a community-informed approach to developing AI models for police accountability, specifically analysing body-worn camera footage of traffic stops. The authors demonstrate that models trained on community perspectives, not just police perspectives, better capture the nuances of public expectations regarding respectful police conduct. For practitioners, this means that merely collecting data on police-public interactions is insufficient; the critical innovation lies in actively incorporating diverse community perspectives into the model development pipeline through targeted stakeholder research and multi-perspective annotation.

## Why This Matters for Practitioners

If you're building AI systems for public-sector transparency, this paper challenges your current development practices. Existing approaches typically treat police departments as the sole stakeholders, resulting in models that optimise for officer perspectives rather than community expectations. For example, when the LAPD developed their own traffic stop analysis tools, they focused on "compliance with policy" metrics that reflected officer priorities, not community concerns about racial insensitivity or fear of violence.

Practitioners should now:
1. Implement targeted stakeholder research (surveys, interviews, focus groups) before building any accountability tool
2. Recruit annotators with varied life experiences (including people with arrest histories and former law enforcement)
3. Establish clear mechanisms for capturing racial and cultural differences in how concepts like "respect" are perceived
4. Avoid single-ground-truth assumptions in your data labelling process

The authors' work with LAPD shows that incorporating these practices leads to models that better reflect the reality of public interactions with government officials, crucial for building tools that genuinely improve accountability.

## Problem Statement

The standard approach to AI for police accountability treats police departments as the sole stakeholders, creating tools that optimise for officer perspectives rather than community expectations. It's like developing a navigation app that only considers the driver's preferences but ignores pedestrian safety, both are important users, but the app's purpose fundamentally depends on which perspective you prioritize. In traffic stop analysis, this means measuring "compliance with policy" instead of "respectful interaction," which has been shown to vary significantly across racial groups.

## Proposed Approach

The authors propose a two-stage approach to community-informed AI development:

1. **Stakeholder Research**: Conduct surveys, interviews, and focus groups with diverse community members to identify culturally specific concepts that matter most (e.g., "respect" is universally valued, but its meaning varies by race)
2. **Multi-Perspective Annotation**: Recruit diverse annotators and train them to capture these nuances, rather than forcing consensus

The system architecture integrates:
- A stakeholder research component to gather community perspectives
- A multi-perspective annotation platform for diverse labelling
- A multi-perspective AI model that learns from diverse annotations

Here's the core algorithm for their multi-perspective modelling approach:

```python
def multi_perspective_modeling(annotations):
    # Stage 1: Build a multitask model from initial small pool of annotators
    initial_model = build_multitask_model(annotations)
    
    # Stage 2: Recruit new annotators targeting identified disagreements
    new_annotation_pool = recruit_annotators(
        based_on_disagreements=initial_model.disagreements
    )
    
    # Stage 3: Train final model using all annotations
    final_model = train_model(
        annotations=annotations + new_annotation_pool,
        model=initial_model
    )
    return final_model
```

## Key Technical Contributions

The paper's key innovations lie in how they operationalize community-informed AI development:

1. **Multi-perspective annotation guidelines that explicitly capture racial differences**: Unlike standard annotation guidelines that aim for consensus, their approach acknowledges and incorporates racial differences in what constitutes "respectful" police behaviour. For example, Black and Latine respondents linked "respect" with reduced fear of violence (74% of Black/Latine respondents compared to 34% of White respondents), while White respondents viewed it more as professional courtesy. This directly shapes how they define and annotate "respect" in BWC footage.

2. **Targeted annotator recruitment based on stakeholder research**: They moved beyond random recruiting to actively seek individuals with specific backgrounds: 30% had been arrested or incarcerated (recruited via HomeBoy Industries), 25% were former police officers (via LAPD referrals), and 45% were diverse community members. This ensured their training data reflected the full spectrum of community perspectives, not just the dominant ones.

3. **Annotator-in-the-loop training process**: They implemented continuous feedback between annotators and researchers, where initial annotation discrepancies informed further training. For contentious concepts like "respect" and "fear," they conducted small group sessions by similar backgrounds (e.g., Black community members discussing how officer behaviour affects their perception of safety) to encourage open discussion of differing perspectives.

4. **Focus on actionable metrics over vague concepts**: They broke down "respect" into specific, observable behaviours identified through stakeholder research: professionalism (officer tone and body language), transparency (whether officer explained the stop reason), kindness (polite language), and empathy (acknowledging driver concerns). This enabled precise, measurable annotation of previously vague concepts.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results

The paper doesn't provide specific accuracy metrics or comparisons with baselines for their final model. However, it presents significant qualitative findings from their stakeholder research:

- In a large-scale survey of Los Angeles residents (weighted to city population benchmarks), "respect" was the most frequently mentioned concept across all racial groups (Figure 1 shows topic proportions).
- They found significant racial differences in how "respect" was perceived: 74% of Black and Latine respondents linked respect to reduced fear of violence, compared to only 34% of White respondents.
- They analysed 1000 LAPD traffic stops (about 1800 BWC recordings due to multiple recordings per stop) with a stratified random sample.
- The authors state they "empirically evaluated model improvements from annotator diversification" (citing Golazizian et al. 2024; Muscato et al. 2024; Wang et al. 2024), but do not provide specific metrics on how much this improved model performance.

## Related Work

The paper positions itself against traditional approaches to AI for police accountability that:
1. Assume a single ground truth (e.g., standard social science approaches like Systematic Social Observation)
2. Primarily serve police departments as the main stakeholders
3. Rely on limited perspective pools for data annotation

They build on emerging work in multi-perspective AI (Kapania et al. 2023; Muscato et al. 2024; Wang et al. 2024) that recognizes disagreement among human annotators as valuable evidence rather than noise.

## Limitations

The authors acknowledge several limitations:
1. They focus on traffic stops in Los Angeles, which may not generalise to other police interactions or jurisdictions.
2. Their approach requires significant resources for stakeholder research, annotator recruitment, and training.
3. The paper doesn't provide specific metrics on how much multi-perspective annotation improves model performance quantitatively.

For practitioners, a key limitation is that this approach requires a fundamental shift in how AI for government accountability is developed, moving from a single-stakeholder (police) model to a multi-stakeholder (community-focused) model, which could be challenging for police departments to adopt.

## Appendix: Worked Example

Let's walk through the annotation process for "respect" with specific numbers from the paper:

For "respect" annotation, the process worked as follows:

1. **Stakeholder research**: The team conducted a large-scale survey (n=1000, weighted to Los Angeles population) where participants described preferred and avoided behaviours during traffic stops. This identified four key components of respect: professionalism (68% of responses), transparency (62%), kindness (58%), and empathy (47%).

2. **Annotation guidelines**: They created detailed definitions for each component:
   - Professionalism: Officer maintains calm tone (not raised voice), uses appropriate body language (not looming)
   - Transparency: Officer explains the reason for the stop ("I pulled you over for speeding")
   - Kindness: Officer uses polite language ("Thank you," "Please")
   - Empathy: Officer acknowledges driver's perspective ("I understand this is inconvenient")

3. **Annotator recruitment**: For the 1000 traffic stops (1800 BWC recordings), they recruited 100 annotators:
   - 30 with arrest histories (recruited via HomeBoy Industries)
   - 25 former police officers (recruited via LAPD referrals)
   - 45 diverse community members (no arrest history or police experience)

4. **Small group training**: Annotators were grouped by background for discussion:
   - Black community members (n=15): Discussed how officer questions about neighborhood created fear of violence
   - White community members (n=15): Focused on fairness and procedural correctness
   - Former officers (n=25): Emphasized policy compliance

5. **Annotation process**: For each traffic stop, annotators rated each component on a 1-5 scale:
   - Professionalism: 3.2 (Black community), 4.1 (former officers)
   - Transparency: 2.9 (Black), 3.8 (former officers)
   - Kindness: 2.6 (Black), 3.5 (former officers)
   - Empathy: 2.7 (Black), 3.4 (former officers)

6. **Model training**: The AI model was trained to recognise patterns across these diverse ratings. For example, when Black community members gave low ratings for professionalism and empathy, the model learned to recognise specific officer behaviours associated with those low scores.

This process allowed them to build models that didn't just measure "is this stop respectful?" but could capture the full spectrum of community views on what constitutes respectful police behaviour.

## References

- Benjamin A.T. Grahama, Lauren Brown, Georgios Chochlakis, Morteza Dehghani, Raquel Delerme, Brittany Friedman, Ellie Graeden, Preni Golazizian, Rajat Hebbar, Parsa Hejabi, Aditya Kommineni, Mayagüez Salinas, Michael Sierra-Arévalo, Jackson Trager, Nicholas Weller, Shrikanth Narayanan, "Community-Informed AI Models for Police Accountability", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2402.01703

Tags: #social-impact #community-informed-ai #multi-perspective-modelling #police-accountability #stakeholder-engagement
