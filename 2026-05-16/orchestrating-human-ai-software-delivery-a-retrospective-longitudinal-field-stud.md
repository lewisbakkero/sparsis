---
title: "Orchestrating Human-AI Software Delivery: A Retrospective Longitudinal Field Study of Three Software Modernization Programs"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20028"
---

## Executive Summary
Chiron is a team-level software delivery platform that coordinates humans and AI agents across four measured delivery stages: analysis, planning, implementation, and validation. Unlike isolated coding assistants, Chiron embeds AI within a structured workflow that includes acceptance-criteria validation, repository-native review, and hybrid human-agent execution. The mature V4 configuration reduced delivery time by 74.2% while simultaneously improving first-release coverage by 13.4 percentage points and reducing downstream issue load by 74.0% compared to traditional delivery.

## Why This Matters for Practitioners
If your engineering team is implementing AI tools, don't just deploy isolated coding assistants. Instead, focus on embedding AI into your existing workflow with structured validation and review processes. For example, if your team uses GitHub Copilot for code generation, integrate it with your existing review, testing, and planning processes rather than just using it as a "better autocomplete." The study found that simply having agents available (V1) delivered only modest gains, while the V4 configuration, which added acceptance-criteria validation, repository-native review, and hybrid execution, delivered the most significant improvements. This means you should design your AI integration around the end-to-end workflow, not isolated tasks. Start by adding acceptance-criteria validation to your existing process before expanding AI usage, as this prevented the quality degradation seen in earlier versions.

## Problem Statement
Today's AI integration in software engineering resembles having a brilliant solo violinist in an orchestra but no conductor. You get great individual performances (fast code generation), but the overall symphony (delivery process) is out of tune, with increased defects and poor coordination. The problem isn't just having AI tools; it's that current implementations treat AI as isolated components rather than integrated parts of a cohesive delivery workflow. The paper shows that teams using AI for isolated tasks (like V1) actually saw worse downstream quality, while those embedding AI in a structured workflow (V4) achieved better speed and quality simultaneously.

## Proposed Approach
Chiron coordinates humans and AI agents across four delivery stages: analysis, planning, implementation, and validation. It evolved from tool-centric agent use (V1) toward increasingly integrated orchestration (V4). The system supports repository ingestion, technology-stack analysis, documentation synthesis, backlog generation, agent- and human-assigned execution, and repository-native review. The study examined three real software modernization programs across five delivery configurations (traditional baseline and four platform versions), measuring delivery durations, task volumes, validation-stage issues, and first-release coverage.

## Key Technical Contributions
Chiron's effectiveness came from specific design choices in its workflow orchestration, not just AI capabilities. The key mechanisms that led to the significant improvements were:

1. **Acceptance-criteria validation**: Early versions (V1 and V2) focused on speed but neglected validation against explicit requirements. V3 and V4 introduced validation against task acceptance criteria, which reduced downstream issue load by 75.8% from V1 to V4. This was implemented not as a passive check but as an active part of the workflow, requiring each task to meet predefined criteria before proceeding to implementation.

2. **Repository-native review**: V4 introduced repository authentication, branch and PR workflows, and code review. This moved defect detection from late-stage validation to earlier review, with a portfolio-weighted containment rate of approximately 51.4%. The implementation involved integrating the AI system directly with the code repository, allowing AI to suggest fixes during code review rather than after the fact.

3. **Hybrid human-agent execution**: V4 balanced AI and human responsibilities, with the system automatically assigning tasks based on agent capabilities and human expertise. This prevented the "speed-first" pitfalls of earlier versions (V1 and V2), where quality suffered as teams prioritised speed over validation. The system used the repository-native review process to continuously assess the quality of AI-generated code before accepting it as part of the delivery pipeline.

## Experimental Results
The study measured three real modernization programs across five delivery configurations, reporting these key results:

- **Time metrics**: V4 reduced summed project duration from 36.0 weeks (traditional) to 9.3 weeks (74.2% reduction), a 3.87× speedup.

- **Effort metrics**: Modelled raw effort fell from 1080.0 to 232.5 person-days (78.5% reduction), and modelled senior-equivalent effort fell from 1080.0 to 139.5 SEE-days (87.1% reduction).

- **Quality metrics**: Validation-stage issue load dropped from 8.03 to 2.09 issues per 100 tasks (74.0% reduction), and task-weighted first-release coverage rose from 77.0% to 90.5% (a 13.4 percentage point increase).

- **V1 vs V4 contrast**: The V1-to-V4 comparison is particularly telling because both configurations used the same nominal staffing scenario (5 people, 3 SEE). From V1 to V4, delivery became 3.08× faster, validation-stage issue load fell by 75.8%, and coverage rose by 37.9 percentage points. This shows that the gains came from orchestration, not just having AI available.

## Related Work
The paper positions itself against prior work that focuses on "individual task completion and repository-level issue solving," while providing evidence on "team-level delivery" which is "comparatively scarce." It builds on research showing that AI assistance can improve output while also increasing integration time, highlighting that "team-level coordination costs need not move in lockstep with individual productivity gains." The paper also references work on scaffolding, interfaces, context management, and verification loops rather than raw model capability alone.

## Limitations
The study has several important limitations:
- It's a retrospective longitudinal field study, not a controlled experiment, so results should be interpreted descriptively rather than causally.
- The data was compiled retrospectively from engineering records and practitioner recall, which may introduce some inaccuracies.
- All programs came from a single organization, so results may reflect organization-specific engineering practices.
- The versions represent successive delivery configurations, not isolated ablations, so improvements may reflect cumulative workflow learning rather than specific components.
- The study doesn't measure post-release reliability or total lifecycle cost.

## Appendix: Worked Example
Let's walk through how the Bank application (30k LOC COBOL migration) evolved from traditional delivery to Chiron V4.

**Traditional baseline**:
- Analysis: 2.0 weeks
- Planning: 2.0 weeks
- Implementation: 4.0 weeks
- Validation: 3.0 weeks
- Total: 11.0 weeks
- Validation-stage issues: 8.75 issues per 100 tasks
- First-release coverage: 80%

**Chiron V4**:
- Analysis: 0.2 weeks (90% reduction)
- Planning: 0.2 weeks (90% reduction)
- Implementation: 1.0 weeks (75% reduction)
- Validation: 0.4 weeks (87% reduction)
- Total: 1.8 weeks (83.6% reduction)
- Validation-stage issues: 2.50 issues per 100 tasks (71.4% reduction)
- First-release coverage: 90% (12.5% increase)

For the Bank application, the traditional baseline required 11.0 weeks of work with 8.75 issues per 100 tasks. With V4, the team delivered in 1.8 weeks with only 2.50 issues per 100 tasks and 90% first-release coverage. The key improvement came from introducing repository-native review in V4, which caught 50% of issues before validation. For the Bank application, V4 caught 20 issues before validation (out of 40 total issues), resulting in only 20 issues reaching validation. This early detection reduced rework costs and improved overall quality. The system reduced the time-to-engineering-readiness (analysis and planning) by 90% while also improving quality.

## References

- Maximiliano Armesto, Christophe Kolb, "Orchestrating Human-AI Software Delivery: A Retrospective Longitudinal Field Study of Three Software Modernization Programs", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20028

Tags: #software-engineering #human-ai-collaboration #orchestration #delivery-workflow #validation #repository-review
