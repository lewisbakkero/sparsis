---
title: "Beyond the Desk: Barriers and Future Opportunities for AI to Assist Scientists in Embodied Physical Tasks"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19504"
---

## Executive Summary
This study examines the barriers to AI assistance in scientific work beyond desk-based activities, revealing three key challenges: high-stakes experimental environments where AI errors are unacceptable, physical constraints that limit access to AI tools, and the inability of current AI to replicate human tacit knowledge. For engineers building AI systems for scientific contexts, this work underscores the need for background infrastructure that supports human expertise rather than attempting to replace it.

## Why This Matters for Practitioners
If you're building AI systems for scientific environments, don't focus on automating high-stakes physical tasks, engineers working in nuclear fusion (as seen in participant P4) would reject any AI that could cause costly errors in delicate fusion experiments. Instead, design systems that act as passive monitors for task status (like observing if a centrifuge is running correctly) or organize lab knowledge, as participants desired tools that could "keep track of documentation" and "prevent costly errors due to lapses in human attention." Your next step should be to avoid creating AI agents that attempt to "do the actual work," as one participant explicitly stated, and instead focus on tools that augment human judgment during embodied physical tasks, such as a system that tracks the sequence of steps in a sensitive biochemistry protocol to prevent missed steps.

## Problem Statement
Current AI systems resemble a chef who insists on taking over the entire kitchen but can't distinguish between a perfectly seared scallop and a burnt one, exactly the problem when AI tries to assist in high-stakes scientific environments where a single mistake could derail months of work. Most AI tools are built for desk-based work, like writing code or analysing data, but fail to address the reality that 80% of scientific work happens in physical labs and fields where scientists rely on nuanced, embodied knowledge that can't be codified into algorithms.

## Proposed Approach
This research adopted a naturalistic inquiry approach with situated on-site interviews to understand how scientists perceive AI assistance in their physical work environments. The methodology involved interviewing 12 scientific practitioners across various disciplines, observing their workflows in real lab and field settings, and then facilitating speculative design activities where participants sketched future AI tools that could support their embodied work without replacing human expertise.

## Key Technical Contributions
The study's primary contributions lie in its qualitative insights about AI adoption barriers in embodied scientific work. Each contribution is framed as a specific insight that changes how engineers should approach AI design for physical work contexts:

1. **High-stakes environment awareness**: The study revealed that scientists in high-risk settings (like those working with nuclear fusion or surgical procedures) reject any AI that could cause errors, as shown by participant P4 who stated, "We can't risk AI making mistakes with these delicate fusion experiments." This necessitates system designs that avoid direct intervention in critical physical tasks, instead focusing on passive monitoring.

2. **Contextual constraint recognition**: Participants from field robotics (P2, P3) and biochemistry (P12) described how clean-room environments and delicate equipment make accessing traditional AI tools nearly impossible. This insight should drive system architecture toward voice-controlled or embedded solutions that don't require visual interfaces or constant connectivity.

3. **Tacit knowledge integration**: The study confirmed that AI cannot replicate human tacit knowledge (as defined by the authors: "hands-on, domain-specific knowledge that is hard to precisely articulate in words"), meaning systems should avoid trying to replace human judgment and instead focus on organizing knowledge and supporting memory, as participants desired tools that could "keep track of documentation" and "prevent costly errors due to lapses in human attention."

## Experimental Results
This is a qualitative study based on interviews with 12 scientific practitioners, not an experimental system with quantitative metrics. The authors found three consistent barriers to AI adoption across all domains: high-stakes experimental environments (mentioned by all 12 participants), constrained physical environments (cited by 10 participants), and the inability of AI to replicate tacit knowledge (cited by 11 participants). The study did not measure statistical significance as it was not a quantitative experiment.

## Related Work
This work extends HCI research on AI in science by focusing on embodied physical tasks beyond desk-based work, unlike prior studies that examined only computer-centric workflows. It builds on the STS and HCI research showing that scientific discovery involves material interaction, but specifically addresses AI's role in these physical contexts. In comparison to the few existing studies on AI in science that focused on desk work (e.g., [45, 71]), this study highlights a gap in understanding AI's role in the 80% of scientific work that happens "beyond the desk."

## Limitations
The study's scope was limited to day-to-day "on the ground" work, excluding higher-level organizational policies around AI adoption. The participants were self-selected to be more open to AI use, so perspectives of those strongly opposed to AI were missing. The study did not investigate how to build the speculative AI tools participants envisioned, focusing instead on identifying barriers and conceptualizing future directions. The findings may not generalise across all scientific disciplines, though participants represented a wide range of fields.

## Appendix: Worked Example
Let's walk through how a system could implement the "task status monitoring" feature that participants desired, using a biochemistry lab as an example. Imagine a biochemist (like participant P12) working with cell cultures in a sterile environment where timing is critical. The system would:

1. **Observation phase**: The system uses non-intrusive sensors (like cameras with privacy-preserving blurring) to monitor key steps in the protocol: incubator temperature checks, centrifuge cycles, and sterile workspace usage. It would record these steps without requiring the scientist to input data manually.

2. **Contextual awareness**: The system understands the specific protocol being followed (e.g., for a cancer cell study), recognising that a 30-minute incubation cycle followed by a 15-minute centrifugation is standard, but deviations could indicate potential errors.

3. **Error detection**: If the system observes the scientist skipping a temperature check (e.g., not verifying that the incubator is at 37°C for 2 hours), it would trigger a subtle alert (e.g., a soft vibration in a wearable device) to prompt correction without disrupting the workflow.

4. **Documentation**: After the experiment, the system automatically logs the actual sequence of steps taken, including any deviations detected, creating a detailed record that would be invaluable for troubleshooting if the experiment fails.

This example shows how a system could support the scientist's work without replacing their judgment, addressing the "monitor task status" feature that participants wanted while respecting the high-stakes nature of their work.

## References

- Irene Hou, Alexander Qin, Lauren Cheng, Philip J. Guo, "Beyond the Desk: Barriers and Future Opportunities for AI to Assist Scientists in Embodied Physical Tasks", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19504

Tags: #scientific-practice #embodied-physical-work #human-centered-ai #speculative-design #background-infrastructure
