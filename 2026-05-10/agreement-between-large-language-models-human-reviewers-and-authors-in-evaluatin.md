---
title: "Agreement Between Large Language Models, Human Reviewers, and Authors in Evaluating STROBE Checklists for Observational Studies in Rheumatology"
venue: "Human Reviewers"
paper_url: "https://arxiv.org/abs/2603.19303"
---

## Executive Summary
This paper compares how closely large language models (LLMs) align with human reviewers and authors when assessing STROBE checklist compliance in rheumatology studies. LLMs like ChatGPT-5.2 and Gemini-3Pro show high agreement (85% overall) on basic reporting structure but significantly lower agreement on complex methodological items, indicating they're reliable for initial screening but not for replacing expert human judgment in methodological evaluation.

## Why This Matters for Practitioners
If you're building systems that assess medical research quality, this paper provides concrete guidance on where LLMs can and cannot be safely deployed. For production systems, you should:
1. Implement LLM-based screening for basic STROBE items (abstracts, background, funding statements, and presentation elements) where LLMs achieved perfect agreement (AC1=1.000) with human reviewers
2. Build human review layers for methodological items (Items 4-12) where agreement dropped significantly - for example, on "loss to follow-up" (Item 12d), Gemini 3 Pro's agreement with senior reviewers was negative (AC1=-0.252)
3. Prioritise ChatGPT-5.2 over Gemini-3Pro for complex methodological assessments, as ChatGPT-5.2 consistently showed higher agreement with human reviewers (e.g., AC1=0.730 with authors on "missing data count" vs. AC1=0.301 for Gemini 3 Pro)
4. Avoid using LLMs for final clinical decision support on methodological quality, as their surface-level text matching fails to capture implicit methodological details that human experts infer from context

## Problem Statement
Today's medical research quality assessment resembles a manual proofreading process for a 22-item checklist, where each item requires human judgment to determine if a paper follows reporting standards - often leading to inconsistent results and time-consuming reviews. For example, when assessing whether a study properly handled "loss to follow-up," human reviewers can infer from context whether the study effectively managed missing data, but LLMs strictly rely on explicit textual matches, treating any mention of "missing data" as inadequate reporting even when methods were clearly described elsewhere.

## Proposed Approach
The researchers built an evaluation framework that compares LLM performance against human reviewers and manuscript authors using Gwet's Agreement Coefficient (AC1) to measure inter-rater reliability. They assessed 17 rheumatology studies using the STROBE checklist, grouping items into "Presentation & Context" (Items 1-3, 13-22) and "Methodological Rigor" (Items 4-12) domains. Each paper was evaluated by:
- Manuscript authors (self-assessment)
- A panel of five human reviewers (three junior, one mid-level, one senior)
- Two LLMs (ChatGPT-5.2 and Gemini-3Pro) using a three-step prompting protocol to prevent fabrication

The framework uses strict Y/N/NA categorisation for all items, with no room for explanatory text, ensuring LLM outputs are directly comparable to human assessments.

## Key Technical Contributions
The paper's key technical contributions are:
1. A measurement framework using Gwet's AC1 instead of Cohen's kappa to avoid the prevalence paradox in highly skewed checklist datasets, which is critical for medical reporting assessment where most items are typically reported.
2. A standardized three-step prompting protocol designed to prevent LLMs from generating fabricated explanations, including:
   - Strict instruction to evaluate only reporting adherence (not methodological quality)
   - Few-shot calibration with two example articles
   - Final evaluation using a simple two-column table without explanations
3. Domain-specific analysis showing LLMs' performance varies significantly between presentation elements (where they achieve perfect agreement) and methodological items (where agreement declines), revealing their surface-level text matching limitation.

The three-step prompting protocol is critical to ensuring reproducibility. It explicitly instructs models to use strict Y/N/NA categories, prevents inference of missing details, and avoids explanation generation through a simple two-column table format.

```python
def evaluate_strobe_item(paper_text, item_id):
    # Strict instruction: Evaluate only reporting adherence (not methodological quality)
    prompt = f"""
    Review the following paper's adherence to item {item_id} of the STROBE checklist.
    Categorise as 'Yes' (adequately reported), 'No' (inadequately or not reported), or 'Not applicable'.
    Do not infer missing details or provide explanations.
    """
    # Few-shot calibration with two examples
    examples = [
        {"paper": "Abstract: This study examines rheumatoid arthritis prevalence in urban populations.", "label": "Yes"},
        {"paper": "Methods: We used a cohort of 200 patients without specifying age range.", "label": "No"}
    ]
    # Final evaluation input
    input_text = f"{prompt}\n\nExamples:\n{examples}\n\nPaper text: {paper_text}"
    return llm.generate(input_text, temperature=0.0)
```

## Experimental Results
The study evaluated 17 rheumatology studies published in high-impact journals that explicitly reported using the STROBE checklist.

- Overall agreement across all reviewers: 85.0% (AC1=0.826)
- "Presentation & Context" domain: Almost perfect agreement (AC1=0.841)
- "Methodological Rigor" domain: Substantial agreement (AC1=0.803)

For specific complex items:
- Loss to follow-up (Item 12d): Gemini 3 Pro agreement with senior reviewer was AC1=-0.252 (negative agreement), while with authors it was only fair (AC1=0.21)
- Sensitivity analyses (Item 12e): LLMs agreed moderately with senior reviewer (AC1=0.652) but showed slight to fair agreement with authors (AC1: 0.071-0.272)
- Missing data counts (Item 14b): ChatGPT-5.2 showed substantial agreement with authors (AC1=0.730) while Gemini 3 Pro showed fair agreement (AC1=0.301)

## Related Work
The paper extends existing work on LLMs for medical research by focusing specifically on their reliability for assessing reporting standards rather than generating or summarising content. It builds on Venerito et al.'s "AI as a rheumatologist" paper but addresses a different problem: evaluating whether medical research properly follows established reporting standards, not whether the research itself is scientifically sound.

The study is positioned within the DEAL Pathway B framework for evaluating LLMs, which emphasizes reproducibility and clear assessment protocols, moving beyond simple accuracy metrics to measure reliability across different evaluators.

## Limitations
The study has several limitations:
- Small sample size of 17 studies, reducing statistical power for item-level analysis
- Studies were from high-impact journals that explicitly required STROBE checklists, making the reporting quality likely higher than average
- Limited to two commercial LLMs (ChatGPT-5.2 and Gemini 3 Pro), excluding open-source alternatives
- Each study was evaluated only once per model, so the study doesn't assess intra-rater consistency (though this mirrors real-world human review workflows)
- Potential data leakage, as evaluated studies may have been part of LLM training data
- The domain grouping (Presentation & Context vs. Methodological Rigor) is imperfect, as Item 16b (categorising continuous variables) falls within the presentation domain but showed low agreement scores

## Appendix: Worked Example
Let's walk through how LLMs assessed the "loss to follow-up" item (Item 12d) in a hypothetical rheumatology study:

1. The study describes "200 patients were enrolled in the 2-year observational study" but doesn't explicitly state how many were lost to follow-up
2. The human senior reviewer examines the flowchart showing 185 patients completed the study, correctly inferring that 15 were lost to follow-up
3. ChatGPT-5.2 categorizes this as "No" (inadequately reported) because the paper doesn't explicitly state "15 patients were lost to follow-up"
4. Gemini 3 Pro also categorizes as "No" but with higher consistency with junior reviewers
5. The senior reviewer's assessment was "Yes" (adequately reported), indicating a negative AC1=-0.252 between Gemini 3 Pro and the senior reviewer

This example shows LLMs' limitations: they rely on explicit textual matches rather than contextual inference. For this item, the senior reviewer could infer from the flowchart that loss to follow-up was properly handled, while LLMs required explicit text stating the missing data percentage.

## References

- Emre Bilgin, Ebru Ozturk, Meera Shah, Lisa Traboco, Rebecca Everitt, Ai Lyn Tan, Marwan Bukhari, Vincenzo Venerito, Latika Gupta, "Agreement Between Large Language Models, Human Reviewers, and Authors in Evaluating STROBE Checklists for Observational Studies in Rheumatology", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19303

Tags: #biomedicine #medical-research #clinical-informatics #llm-evaluation #inter-rater-reliability #medical-checklists #prompt-engineering
