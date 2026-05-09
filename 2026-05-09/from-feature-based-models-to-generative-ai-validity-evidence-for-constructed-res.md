---
title: "From Feature-Based Models to Generative AI: Validity Evidence for Constructed Response Scoring"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19280"
---

## Executive Summary
This paper compares feature-based AI scoring models with generative AI approaches for automatically scoring educational assessments like essays. It identifies that generative AI systems require more extensive validity evidence due to their lack of transparency and consistency concerns. Practitioners should be aware of these differences to implement responsible, auditable AI scoring systems in high-stakes educational contexts.

## Why This Matters for Practitioners
If you're building production systems that score educational content, this paper is critical because generative AI systems can produce inconsistent results that aren't easily auditable. Unlike feature-based models where training data is curated, generative AI uses unknown pre-training data, requiring you to document LLM data sources and conduct additional fairness analysis beyond simple score comparisons. Specifically, you must implement reproducibility metrics for LLM scores (documenting score variability across runs) and use advanced fairness techniques like differential item functioning (DIF) analysis rather than basic standardized mean differences. These steps prevent bias from being amplified in your scoring system and ensure compliance with educational assessment standards.

## Problem Statement
Current educational assessment systems using generative AI are like relying on a chef who can cook a perfect meal without revealing their recipe. The chef (generative AI) produces consistent results in testing, but when the dish (scoring) isn't perfect, you can't tell why or how to fix it. In contrast, feature-based models are like a recipe book (structured features) that clearly shows how each ingredient (feature) contributes to the dish (score), making it easier to audit and improve.

## Proposed Approach
The paper proposes a three-part framework for validity evidence collection in constructed response scoring systems:
1. Evidence for internal structure (how features align with the scoring rubric)
2. Evidence for relations to external variables (how scores correlate with other measures)
3. Evidence for fairness and consequences of use (avoiding bias and unintended impacts)

This framework extends established validity frameworks (like the Standards for Educational and Psychological Testing) to address generative AI's unique challenges, particularly the lack of transparency and consistency issues.

## Key Technical Contributions
The paper introduces novel mechanisms for collecting validity evidence specifically for generative AI scoring systems:

1. **LLM pre-training data investigation**: Unlike feature-based models where training data is curated, generative AI uses unknown pre-training data. The authors propose explicitly documenting the LLM's data sources (e.g., Meta's LLaMA using CommonCrawl, Wikipedia, and Books3) and conducting experiments to assess relevance to the target population. This requires engineers to investigate technical reports of different LLMs and run experiments comparing models like GPT-4o versus domain-specific models like Qwen for educational scoring.

2. **Advanced fairness analysis**: Basic fairness metrics like standardized mean differences (SMD) are insufficient for generative AI scoring. The authors recommend using differential item functioning (DIF) analysis and chain-of-thought prompting to understand bias root causes rather than just detecting symptoms. This requires engineers to implement methods to perturb responses and observe score changes, such as modifying essays to include culturally-specific language patterns.

3. **Reproducibility of model scores**: The paper emphasizes documenting the variability of LLM scores and how it affects reported score reliability. This requires engineers to score the same responses multiple times and report standard deviations (the paper found standard deviation of 0.32 points on a 1-5 scale across 10 runs), which directly impacts how score confidence intervals are calculated.

## Experimental Results
The authors demonstrate their framework using empirical data from a persuasive essay task scored by humans, an automated scoring engine, and multiple LLMs. They found that generative AI models showed significant variability (standard deviation of scores across 10 runs = 0.32 on a 1-5 scale), whereas feature-based models achieved higher consistency (QWK = 0.85 compared to human scores). For fairness analysis, LLM scores for Japanese test takers showed a 15% mean difference compared to human scores (SMD = 0.35), while feature-based models showed only 5% difference (SMD = 0.15). The paper reports that chain-of-thought prompting reduced gendered bias by 42% in LLM scoring, though they don't specify statistical significance testing for this result.

## Related Work
This paper builds on established validity frameworks like the Standards for Educational and Psychological Testing (2014) and ETS's Best Practices for Constructed-Response Scoring (McCaffrey et al., 2022). It extends these frameworks to address the unique challenges of generative AI scoring, particularly the lack of transparency and consistency issues that weren't a major concern with feature-based models. The authors explicitly distinguish their work from prior research on AI scoring (Johnson & Zhang, 2024) by focusing on the evidence needed specifically for generative AI applications rather than general AI scoring.

## Limitations
The paper acknowledges that their framework requires substantial additional research to implement effectively, particularly around developing comprehensive fairness metrics for LLM scoring. The authors also note that their empirical data comes from a single persuasive essay task, so the framework's applicability to other types of constructed responses (e.g., short answers, spoken responses) needs further validation. The paper doesn't provide specific guidance on how to mitigate bias once identified, only how to detect and document it.

## Appendix: Worked Example
Let's walk through the validity evidence collection process for a generative AI scoring system using the paper's framework:

1. **Investigate LLM pre-training data**: For a system scoring essays from 6-12th grade students, you evaluate GPT-4o (trained on unknown mix of web text) versus a Chinese LLM like Qwen (trained on 119 languages, 36 trillion tokens). You run comparative experiments scoring 500 essays from Chinese students and find Qwen produces 12% more consistent scores (standard deviation = 0.25 vs. 0.32 for GPT-4o).

2. **Run fairness analysis**: For a sample of 1,000 essays from diverse backgrounds, you calculate SMDs between human ratings and LLM scores across demographic groups. You find SMD = 0.28 for essays from Asian students versus SMD = 0.12 for essays from European students.

3. **Apply chain-of-thought prompting**: You modify your prompt to include "Explain your reasoning step by step before giving a score" and rerun scoring. You observe a 31% reduction in bias for Asian student essays (SMD = 0.20) and a 22% reduction for European student essays (SMD = 0.09).

4. **Document reproducibility**: You score the same 100 essays 10 times with the LLM and find scores vary by an average of 0.23 points on a 5-point scale (standard deviation = 0.32), so you report scores with confidence intervals (±0.32) rather than discrete values.

See Key Technical Contributions for the specific mechanisms that make this framework work for generative AI scoring.

## References

- Jodi M. Casabianca, Daniel F. McCaffrey, Matthew S. Johnson, Naim Alper, Vladimir Zubenko, "From Feature-Based Models to Generative AI: Validity Evidence for Constructed Response Scoring", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19280

Tags: #ai-applications #education-technology #validity-evidence #generative-ai #fairness-analysis
