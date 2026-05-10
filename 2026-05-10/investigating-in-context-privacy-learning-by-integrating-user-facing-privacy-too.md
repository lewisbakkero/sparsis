---
title: "Investigating In-Context Privacy Learning by Integrating User-Facing Privacy Tools into Conversational Agents"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19416"
---

## Executive Summary
The authors developed a privacy notice panel that integrates directly into conversational agent interfaces to support in-context privacy learning through just-in-time warnings, anonymization options, and educational FAQs. For engineers building chatbot systems, this research demonstrates that embedding privacy guidance during real-time interactions, rather than relying on detached educational resources, significantly improves user awareness of privacy risks and protective actions.

## Why This Matters for Practitioners
If you're building production chatbots that handle sensitive user data, this paper suggests you should integrate privacy education directly within user workflows rather than relying on separate privacy policies or educational resources. Specifically, consider implementing a just-in-time privacy notice that:
- Detects sensitive information (names, addresses, emails, phone numbers, SSNs) in user inputs
- Offers three anonymization strategies (retract, fake, generalise) tailored to the detected information
- Surfaces built-in privacy controls (opt-out of data sharing, disable memory) with clear explanations
- Provides contextual FAQs addressing the specific privacy concerns raised by the detected information

This approach moves beyond the common practice of merely listing privacy policies at the end of the user journey. The paper's study with CS students showed that in-context privacy education led to increased awareness of sensitive information types (7 out of 10 participants) and understanding of protective actions (8 out of 10 participants), making it a practical framework for improving user privacy engagement in real chatbot systems.

## Problem Statement
Current chatbot systems operate like a "digital confessional" where users freely disclose sensitive information without context about privacy implications. Imagine asking a therapist for advice while simultaneously sharing your medical history, financial details, and personal relationships, with no warning that these disclosures could be used beyond the immediate conversation. Unlike traditional privacy policies that users might never consult, this research addresses the critical gap where users make privacy decisions without the necessary context or tools to protect themselves during actual interactions.

## Proposed Approach
The authors integrated a privacy notice panel directly into a simulated ChatGPT interface that intercepts messages containing sensitive information. The panel appears immediately after users attempt to submit messages with sensitive content, offering a warning, anonymization options, built-in privacy controls, and contextual FAQs. This approach creates in-context learning opportunities during real chatbot use, rather than relying on detached educational resources that don't translate to actual practice.

```python
def handle_message_submission(message):
    if detect_sensitive_info(message):
        show_privacy_panel(message)
        while panel_active:
            if user_selects_anonymization():
                apply_anonymization(message)
            elif user_selects_privacy_control():
                apply_privacy_control()
            elif user_expands_faq():
                show_faq_explanation()
            else:
                send_message(message)
```

## Key Technical Contributions
The research reveals specific mechanisms that make privacy education effective during real-time interactions:

1. **Contextual detection with specific information labelling** - The privacy panel doesn't just flag "sensitive information" but explicitly identifies the specific type (e.g., "Phone number: 123-456-7890") and explains why it might be sensitive. This specificity helps users understand their privacy risks in the moment rather than receiving generic warnings.

2. **Three-tiered anonymization strategy implementation** - The implementation details of the anonymization options provide meaningful flexibility:
   - "Retract" replaces sensitive data with a type label (e.g., "Phone number" instead of a specific number)
   - "Fake" generates dummy data (e.g., "Arron" instead of a real name)
   - "Generalise" retains coarse details (e.g., "US state" instead of a specific address)
   This nuanced approach allows users to balance privacy protection with chatbot utility based on their specific needs.

3. **Contextual FAQ integration with privacy trade-off education** - The FAQs aren't generic privacy policies but directly address the information detected. For example, if a user submits a medical question, the FAQ explains "Why should I protect my medical information? Medical information can be used to identify you and may be linked to health conditions, creating privacy risks if disclosed."

## Experimental Results
The study involved ten CS students (5 male, 5 female, ranging from third-year undergraduate to second-year master's) across five phases. Key findings included:
- 7 out of 10 participants demonstrated increased awareness of what constitutes sensitive information after using the privacy panel
- 8 out of 10 participants reported understanding at least one new privacy-protective action (e.g., disabling memory, opting out of data sharing)
- Participants using the privacy panel showed greater recognition of privacy trade-offs (e.g., "I need to protect my address but don't want to lose chatbot context")
- The study measured changes in privacy perceptions through pre-test and delayed post-test surveys, but didn't report statistical significance tests for these perception changes

## Related Work
This paper positions itself between two important lines of research: privacy-protective tools for chatbots (like Clear, Rescriber, and Casper) and privacy education approaches. Unlike prior tools that focused on immediate awareness (e.g., Rescriber's short-term reflections), this research examines how in-context privacy education influences lasting privacy knowledge through pre- and post-test surveys. The authors also extend UI/UX research on privacy tools by analysing how specific design features (like the placement and options of the privacy panel) support or hinder user engagement with privacy protections during actual chatbot use.

## Limitations
The study focused exclusively on CS students in the US, which limits generalizability to other demographics, cultures, and educational backgrounds. The researchers didn't test the privacy panel with real-world chatbot users making real privacy decisions with real consequences, though they simulated realistic use scenarios. The paper also didn't examine how the privacy panel might impact chatbot performance or user satisfaction with the underlying service. The authors acknowledge these limitations and suggest future work should involve broader participant groups and examine privacy tool impacts on system performance.

## Appendix: Worked Example
Let's walk through how the privacy panel would handle a specific user message. Imagine a user trying to send a message containing their email address for a chatbot to summarise an email:

**Original Message:** "Can you help me summarise this email about my upcoming meeting with Dr. Smith? My email is john.doe@example.com."

**Step 1: Detection** - The system detects the email address "john.doe@example.com" as sensitive information.
**Step 2: Warning** - The panel appears with a specific warning: "Your message contains an email address, which could be used to identify you."
**Step 3: Anonymization Options** - The user sees three options:
- Retract: Replaces with "[Email address]"
- Fake: Replaces with "jane.doe@example.com" (a dummy email)
- Generalise: Keeps "example.com" but replaces the name with "user"
**Step 4: Privacy Controls** - The panel also shows shortcuts to built-in controls:
- "Opt out of data sharing for model training" (default: on)
- "Disable memory" (default: on)
**Step 5: FAQs** - The user expands the FAQs to read: "Why should I protect my email address? Email addresses can be used to identify you and are often used for account recovery, which could be exploited if leaked."

After considering these options, the user selects "Generalise" to keep the domain for chatbot utility but anonymize their name, resulting in:
"Can you help me summarise this email about my upcoming meeting with Dr. Smith? My email is user@example.com."

This step-by-step process demonstrates how privacy guidance integrates directly into the user's workflow, making privacy protection a natural part of interaction rather than an afterthought.

## References

- Mohammad Hadi Nezhad, Francisco Enrique Vicente Castro, Ivon Arroyo, "Investigating In-Context Privacy Learning by Integrating User-Facing Privacy Tools into Conversational Agents", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19416

Tags: #security-and-privacy #human-computer-interaction #in-context-privacy-education #privacy-tool-integration #anonymization-strategies
