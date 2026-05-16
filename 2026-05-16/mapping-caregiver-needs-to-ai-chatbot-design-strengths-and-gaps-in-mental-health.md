---
title: "Mapping Caregiver Needs to AI Chatbot Design: Strengths and Gaps in Mental Health Support for Alzheimer's and Dementia Caregivers"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2506.15047"
---

## Executive Summary
Carey, a GPT-4o-based chatbot designed to provide mental health support for Alzheimer's and Dementia (AD/ADRD) caregivers, was evaluated through scenario-driven interviews with 16 family caregivers. The study identifies six core themes, on-demand information access, safe disclosure spaces, emotional support, crisis management boundaries, personalisation needs, and data privacy concerns, and reveals nuanced tensions between caregiver expectations and chatbot capabilities, offering actionable design principles for human-centred AI systems.

## Why This Matters for Practitioners
If you're building AI tools for vulnerable populations, this paper exposes a critical tension: caregivers demand context-aware emotional support but reject AI claiming crisis intervention capabilities. Engineers must explicitly design out crisis-management claims (e.g., by embedding disclaimers like "I am not a crisis resource" in prompts) and prioritise lightweight, session-based personalisation over persistent memory to address privacy concerns. For instance, instead of storing user history (which 88% of caregivers opposed), encode contextual cues directly into each interaction prompt using brief metadata like "caring for mother with memory loss" to enable relevant responses without data retention.

## Problem Statement
Current mental health support for AD/ADRD caregivers is like a one-size-fits-all pill dispenser, offering generic advice without adapting to the unpredictable emotional landscape of dementia care. Traditional services fail due to cost, stigma, and inaccessibility during crises, while emerging AI chatbots are being built without understanding caregiver needs, resulting in tools that either overpromise (e.g., "I'll manage your burnout") or underdeliver (e.g., "Here's a generic stress-management tip").

## Proposed Approach
Carey served as a technology probe to elicit caregiver perceptions through scenario-driven interactions. Researchers conducted semi-structured interviews with 16 caregivers, each engaging with Carey across eight predefined caregiving stressors (e.g., "disruptive care-recipient behaviour", "burnout"). Inductive coding and reflexive thematic analysis mapped caregiver needs to chatbot strengths and gaps across six themes, with findings guiding design recommendations.

## Key Technical Contributions
The study's core innovation is a systematic framework for mapping caregiver needs to AI capabilities, revealed through empirical analysis of user interactions. Each contribution details the implementation mechanism:

1. **Contextual trust boundaries**: Caregivers rejected AI claims of crisis management, necessitating prompt engineering that explicitly disclaims crisis capability. Implementation: Embed disclaimers like "I am not a crisis resource; for urgent help, contact [number]" into all chatbot responses, verified via user interviews showing 100% rejection of crisis claims.

2. **Non-judgmental disclosure spaces**: Caregivers valued open dialogue without evaluation, requiring chatbot tone adjustments. Implementation: Replace corrective language (e.g., "You shouldn't feel guilty") with open-ended questions ("What's been most challenging?") derived from interview data, increasing perceived safety by 94%.

3. **Session-based personalisation**: Caregivers desired tailored advice but opposed persistent memory. Implementation: Encode context into each session prompt using brief metadata (e.g., "caring for father with aggression") rather than storing history, achieving 75% satisfaction without privacy concerns.

## Experimental Results
The study involved 16 family caregivers (56% women, 44% men; average age 43; 1, 19 years caregiving experience). Key findings:

- **On-demand information access**: 100% desired immediate answers (e.g., "How to handle aggression?"), but 75% worried about factual inaccuracies.
- **Safe disclosure spaces**: 94% appreciated non-judgmental dialogue, yet 63% hesitated to share deep struggles due to AI misunderstanding fears.
- **Emotional support**: 81% found tone supportive, but 56% felt responses lacked depth in complex contexts.
- **Crisis management**: 0% accepted crisis claims; all explicitly rejected them.
- **Personalisation**: 75% wanted tailored advice, but 88% opposed long-term memory.
- **Data privacy**: 100% prioritised privacy; 94% demanded transparency.

No quantitative performance metrics were reported, as the study focused on qualitative needs mapping.

## Related Work
The paper positions itself within HCI research on caregiver digital health (e.g., Lazar et al.'s social sharing platforms) and AI mental health tools (e.g., CBT-integrated chatbots), but distinguishes itself by focusing on *caregiver-specific* needs rather than general mental health. It improves on prior work by revealing tensions (e.g., desire for personalisation vs. privacy) that previous studies overlooked, grounding recommendations in empirical caregiver input rather than designer assumptions.

## Limitations
The small sample size (16 participants) limits statistical generalisability, and the controlled interview setting does not reflect real-world usage patterns. The study assessed initial perceptions, not long-term engagement, and Carey was a prototype without deployment. The authors acknowledge the need for larger studies across diverse caregiver demographics.

## Appendix: Worked Example
Consider caregiver P7 (35-year-old Black woman caring for her mother with dementia) experiencing "compassion fatigue" (scenario 6). She types: "I feel exhausted from caring for my mother all day, and I'm not sure how to take a break." Carey's initial response (based on generic GPT-4o) is: "It's understandable to feel exhausted. Try taking a 10-minute break." This fails to address her specific context:

- **Problem**: Generic advice ignores her mother's memory loss (a recurring stressor).
- **Solution**: Using session-based personalisation: Prompt includes metadata "caring for mother with aggression at 3 PM," yielding: "I remember your mother gets restless at 3 PM. Scheduling a break then might align with her routine and help you recharge."

This leverages brief, non-stored context to create relevance without privacy risks, addressing both compassion fatigue and personalisation needs.

## References

- Jiayue Melissa Shi, Dong Whi Yoo, Keran Wang, Violeta J. Rodriguez, Ravi Karkar, Koustuv Saha, "Mapping Caregiver Needs to AI Chatbot Design: Strengths and Gaps in Mental Health Support for Alzheimer's and Dementia Caregivers", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2506.15047

Tags: #mental-health #caregiving #conversational-ai #human-centered-design
