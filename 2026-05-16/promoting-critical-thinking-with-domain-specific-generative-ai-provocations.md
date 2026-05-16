---
title: "Promoting Critical Thinking With Domain-Specific Generative AI Provocations"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19975"
---

## Executive Summary
This paper introduces two domain-specific generative AI tools, ArtBot for art interpretation and Privy for AI privacy analysis, that use carefully designed provocations to foster critical thinking rather than providing direct answers. The systems embed domain knowledge into their questioning strategies to create productive friction, requiring user input before providing additional insights.

## Why This Matters for Practitioners
If you're building AI tools for knowledge work, this paper challenges the common practice of designing AI as an answer engine. Current implementations often reduce cognitive engagement (as shown by reduced neural markers of executive control in the MIT study), meaning your GenAI tool could be undermining the very thinking it's supposed to support. Instead, you should implement domain-specific provocations that require user contribution before providing insights. For example, when building an AI-assisted tool for engineers, don't just let users ask "How do I fix this bug?", instead, prompt them with "What assumptions are you making about the error in the code? How might these differ in a distributed system?" This requires changing your prompt engineering to incorporate domain knowledge rather than using generic questions.

## Problem Statement
Current GenAI implementations operate like overly efficient assistants that complete tasks without prompting critical reflection, akin to having a chef who not only cooks your meal but also serves it on a plate without mentioning the ingredients or cooking process. This pattern erodes the user's ability to think critically, as shown in the MIT neurocognitive study finding reduced neural markers of executive control when users relied on AI for writing tasks. The problem isn't the AI itself, it's how we've designed interactions to prioritize speed over reflection.

## Proposed Approach
The authors designed two systems that position GenAI as a provocateur and facilitator rather than an answer engine. Both systems incorporate three key design elements:
1. Domain-grounded provocations (using established frameworks in their respective domains)
2. Productive friction (resisting the impulse to provide direct answers)
3. User contribution gates (requiring users to articulate their thoughts before receiving additional AI support)

For ArtBot, provocations draw from art history and educational curricula. For Privy, provocations use established privacy taxonomies. Both systems require users to engage with the domain knowledge before receiving further insights.

```python
def generate_domain_specific_provocation(domain, user_contribution):
    """
    Generates a domain-specific provocation based on user input and domain knowledge.
    
    Args:
        domain (str): The domain (e.g., "art", "privacy")
        user_contribution (str): The user's initial contribution to the domain task
        
    Returns:
        str: A provocational question prompting deeper reflection
    """
    domain_knowledge = {
        "art": [
            "How might the historical context of this artwork's creation influence your interpretation?",
            "What alternative interpretations could exist if we consider the artist's other works from the same period?"
        ],
        "privacy": [
            "How might user behaviour patterns affect the severity of this privacy risk?",
            "What additional user data might be needed to fully assess this privacy impact?"
        ]
    }
    
    # Select a random provocation from the domain-specific set
    return random.choice(domain_knowledge[domain])
```

## Key Technical Contributions
The paper's core innovation lies in how they embed domain knowledge into the questioning strategy rather than using generic prompts. Specifically:

1. **Domain-specific prompt templates**: Rather than using generic provocations like "Why do you think that?", they created templates that incorporate domain knowledge. For ArtBot, this meant using art-historical frameworks to create questions like "How does this artwork's interpretation change if you know it was created during a time of political revolution?" For Privy, this involved using established privacy frameworks to generate questions like "How can you design this feature to encourage users to regularly review their sharing settings?"

2. **User contribution gates**: The systems require users to provide their own interpretation before receiving additional AI support. This design choice operationalizes a human-in-the-loop approach where the system doesn't respond until the user externalizes their initial thinking. The quality of user input directly shapes the relevance of subsequent system output.

3. **Contextual adaptation of provocations**: The system doesn't just use static questions, they adjust provocations based on user input. For example, if a user mentions "art historical context" in their initial contribution, ArtBot might follow up with a more detailed question about specific art movements rather than a generic follow-up.

## Experimental Results
The paper evaluated ArtBot with 13 participants and Privy with 121 participants. Key results:

- Users of both systems showed increased reflection time compared to non-AI alternatives, with participants reporting that the provocations surfaced considerations they hadn't previously explored.
- ArtBot's evaluation included written reflections after interactions with each artwork, which were assessed for interpretive engagement.
- Privy's evaluation focused on the quality and completeness of the privacy risk assessment artifacts, assessed by privacy experts.
- The paper doesn't report specific quantitative metrics like F1 scores or accuracy, but does note that participants with domain expertise were more likely to challenge the system's suggestions, while novices described systems as "informative and well-grounded."

## Related Work
This work builds on prior research on "provocations" and "antagonistic AI" that deliberately challenge AI outputs (Park & Kulkarni, 2023; Ye et al., 2025). The paper distinguishes itself by focusing on domain-specific implementation rather than generic provocations, and by demonstrating how this approach supports critical thinking in two distinct domains, art interpretation and privacy analysis, rather than in a single domain.

## Limitations
The paper acknowledges that participants' reactions varied based on domain expertise, with experts finding provocations potentially condescending and novices treating the system as an authority. The authors also note they didn't explore adaptive systems that could adjust provocations based on user expertise level, which they identify as a key area for future work. Additionally, the study doesn't address how these systems scale to more complex, real-world production environments with multiple users and evolving domain knowledge.

## Appendix: Worked Example
Let's walk through a concrete example using Privy's privacy risk assessment workflow. Imagine a developer is designing a fitness app that collects location data to suggest local running routes:

1. **Initial user input**: "We're collecting location data to provide route suggestions based on user preferences."

2. **System analysis**: Privy recognizes the domain (AI privacy) and identifies key risk areas based on established privacy frameworks.

3. **Domain-specific provocation**: "How might users' location data be used beyond route suggestions, and what privacy implications does this introduce?" (This is drawn from Lee et al.'s privacy taxonomy framework.)

4. **User response**: "We hadn't considered that the data could be used to infer users' routines and home locations."

5. **System follow-up**: Based on this response, Privy generates a more specific question: "What specific measures would you implement to prevent location data from being used to infer home locations?"

6. **Final artifact**: The user's responses form a structured privacy risk document that includes identified risks ("location data inference") and tailored mitigation strategies ("implement data anonymization for location patterns").

This workflow demonstrates how domain-specific provocations guided the user to consider privacy implications they hadn't initially recognised, with the system adapting its questioning based on the user's responses.

## References

- Thomas Şerban von Davier, Hao-Ping Lee, Jodi Forlizzi, Sauvik Das, "Promoting Critical Thinking With Domain-Specific Generative AI Provocations", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19975

Tags: #human-centered-ai #critical-thinking #ai-privacy #art-interpretation #provocative-design
