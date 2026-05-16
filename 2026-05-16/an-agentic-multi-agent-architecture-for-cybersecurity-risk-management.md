---
title: "An Agentic Multi-Agent Architecture for Cybersecurity Risk Management"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20131"
---

## Executive Summary
The authors built a six-agent AI system that performs cybersecurity risk assessments for small organizations, reducing assessment time from weeks to under 15 minutes while achieving 85% agreement with human practitioners on risk severity classifications. The system uses a shared persistent context across agents to maintain coherence, domain fine-tuning for sector-specific threat identification, and local deployment to keep sensitive data internal.

## Why This Matters for Practitioners
If you're responsible for security in a small organisation with limited resources, this paper suggests you can obtain a baseline risk assessment that's both affordable (no external consultant required) and actionable (with domain-specific threat identification). You should consider implementing a similar multi-agent approach for your own security assessment workflows, prioritising local deployment to avoid data exfiltration risks. For engineering teams, the shared context mechanism provides a blueprint for maintaining coherence in complex reasoning pipelines without requiring external data storage or processing.

## Problem Statement
Today's cybersecurity risk assessments are like hiring a specialist for a routine check-up - expensive ($15k+), slow (weeks), and dependent on scarce expertise. Small organisations simply skip them, leaving themselves vulnerable to breaches, while the existing frameworks (like NIST CSF) provide no operational tooling to make assessments feasible at scale. It's akin to trying to build a house with only a blueprint and no construction crew.

## Proposed Approach
The system decomposes the risk assessment workflow into six specialised agents that share a persistent context. Each agent handles one analytical stage: risk intake, threat modelling, control assessment, risk scoring, mitigation recommendation, and report synthesis. Unlike sequential pipelines where agents only see the previous agent's output, the shared context allows all agents to read and write to a single persistent object, maintaining coherence throughout the assessment.

```python
class RiskAssessmentPipeline:
    def __init__(self):
        self.shared_context = {}  # Persistent context object

    def run(self, organisation_data):
        self.shared_context["intake"] = RiskIntakeAgent().process(organisation_data)
        self.shared_context["threat_model"] = ThreatModelingAgent().process(self.shared_context["intake"])
        self.shared_context["control_assessment"] = ControlAssessmentAgent().process(self.shared_context["intake"])
        self.shared_context["risk_scores"] = RiskScoringAgent().process(
            self.shared_context["threat_model"], 
            self.shared_context["control_assessment"]
        )
        self.shared_context["recommendations"] = MitigationRecommendationAgent().process(
            self.shared_context["intake"], 
            self.shared_context["risk_scores"]
        )
        return ReportSynthesisAgent().process(self.shared_context)
```

## Key Technical Contributions

This paper introduces several novel mechanisms that make the system work reliably at the production scale:

1. The shared persistent context architecture, which allows all agents to access and update a single context object rather than relying on sequential data handoffs. This prevents coherence failures where later agents lose sight of earlier conclusions (e.g., recommendations ignoring budget constraints noted in the intake phase).

2. Domain fine-tuning specifically for cybersecurity, which enables the model to identify sector-specific threats (e.g., unsecured PHI in healthcare, OT/IIoT vulnerabilities in manufacturing) rather than providing generic threat categories that apply to all organisations.

3. Structured JSON outputs coupled with schema validation, which ensures downstream agents can reliably extract specific fields without parsing free text, preventing the "structural reliability" failures that would occur with unstructured responses.

4. A framework citation verification mechanism that retrieves relevant framework text from a knowledge base before each agent runs and checks all framework citations against indexed content to prevent hallucinations of control identifiers.

## Experimental Results

The primary case study with a 15-person healthcare company showed:
- 85% agreement with three CISSP practitioners on severity classifications (18 out of 21 matches)
- 92% coverage of risks identified by practitioners (12 of 13 risks detected)
- Assessment completed in 14 minutes 38 seconds on an RTX 4090, compared to 16 person-hours for human assessors

The cross-sector study across five sectors (healthcare, fintech, manufacturing, retail, SaaS) with 30 runs each showed:
- Mistral-7B (baseline) produced identical threat categories (Unauthorized Access, Data Breach, Malware Infection) for all sectors
- The fine-tuned model identified sector-specific threats (e.g., "Unsecured PHI" for healthcare rather than "Data Breach")
- Both models achieved 100% structural reliability, but the fine-tuned model produced 6-9 unique threat titles per organisation across three runs, compared to 3-4 for the baseline
- The multi-agent pipeline failed to complete (0% completion) on a Tesla T4 (4,096 token context window), while single-agent runs completed successfully

## Related Work

This work positions itself against three key areas: cybersecurity frameworks (NIST CSF, ISO/IEC 27005), LLM applications in security (mostly classification-focused), and multi-agent systems (typically focused on coverage rather than reasoning). Unlike previous work that uses probability distributions for risk assessment (FAIR), this paper uses agent coordination for coherence. Unlike standard multi-agent systems, the shared persistent context is a key innovation that enables cross-stage coherence.

## Limitations

The authors acknowledge that the system cannot identify risks not mentioned in the questionnaire (e.g., physical security issues requiring on-site visits), and that output variability in the fine-tuned model means a single run should not be treated as the final answer. The study used synthetic profiles rather than real organisations, so it's unclear whether the fine-tuned model is reasoning about threats or pattern-matching on sector keywords. The system also has a critical failure mode where it can produce convincing but incorrect assessments when fed inaccurate input data.

## Appendix: Worked Example

Let's walk through the healthcare risk assessment using the system:

1. **Risk Intake Phase**: The organisation questionnaire states they process "HIPAA-covered data under contract with a regional hospital network" but does not explicitly mention "HIPAA". The Risk Intake Agent writes "HIPAA applicability: unconfirmed" to the shared context.

2. **Threat Modelling Phase**: The Threat Modelling Agent reads the intake profile and writes "Unsecured PHI" to the shared context, referencing specific healthcare threats from the knowledge base rather than generic categories.

3. **Control Assessment Phase**: The Control Assessment Agent checks the organisation's controls against HIPAA requirements, noting "Basic access controls but no MFA" for admin accounts.

4. **Risk Scoring Phase**: The Risk Scoring Agent combines the unsecured PHI threat with the weak access controls to score "Lack of Security Policies" as "High" likelihood and "High" impact.

5. **Mitigation Recommendation Phase**: The Mitigation Recommendation Agent re-reads the intake profile (including the 15-employee size and no security staff), generating a phased remediation plan with realistic budget considerations.

6. **Report Synthesis Phase**: The Report Synthesis Agent normalises the tone, adds an executive summary, and ensures recommendations align with the risk scores and organisational context.

## References

- Ravish Gupta, Saket Kumar, Shreeya Sharma, Maulik Dang, Abhishek Aggarwal, "An Agentic Multi-Agent Architecture for Cybersecurity Risk Management", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20131

Tags: #cybersecurity #risk-management #small-business-security #multi-agent-systems #domain-fine-tuning #local-deployment
