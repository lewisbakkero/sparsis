---
title: "Retrieval-Augmented LLMs for Security Incident Analysis"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.18196"
---

## Executive Summary
This paper presents a RAG-based system that accelerates security incident analysis by using targeted queries to extract relevant indicators from raw logs, then employs semantic retrieval to augment LLM reasoning. It solves the bottleneck of manually sifting through millions of log entries to reconstruct attacks, delivering 94% recall for cloud models and 81% for local deployment at zero per-query cost, while demonstrating that domain-specialised models underperform general-purpose ones.

## Why This Matters for Practitioners
If you're building or maintaining a Security Operations Centre (SOC), this paper directly informs your LLM strategy: **adopt DeepSeek V3 for cloud deployments** (achieving 89% recall at 15× lower cost than Claude Sonnet 4) or **Llama 3.1:70b for on-prem deployments** (81% recall with data privacy). Crucially, avoid security-specialised models like Cisco Foundation-Sec-8B, they scored 71% recall versus 94% for Claude, proving reasoning capability outweighs domain-specific pretraining for complex forensic tasks. For multi-stage attacks like Active Directory breaches, use the system’s enumeration prompt to reach 96% recall in attack step detection, enabling you to replace manual timeline reconstruction with automated, evidence-backed reports.

## Problem Statement
Today’s SOC analysts face log overload: processing 1 million+ events per incident to find the 0.1% relevant indicators, akin to "searching for a single drop of ink in an ocean of water" without a compass. Current systems either miss cross-source relationships (e.g., Suricata alerts without authentication context) or rely on static rules that fail against novel attacks like certificate abuse, forcing analysts to manually stitch together fragmented evidence across tools.

## Proposed Approach
The system operates in two modes: forensic question-answering (e.g., "What’s the infected host?") and attack reconstruction (e.g., "What steps did the attacker take?"). It uses targeted queries to extract indicators from logs, aggregates events into semantic chunks, and retrieves context for LLM reasoning, avoiding direct log ingestion. The architecture avoids context limits through pre-filtering and cross-event chunking.

```python
def security_analyzer(logs, questions):
    # Security Context Extraction: Run IOC queries
    context = extract_context(logs, query_library)  # Returns aggregated chunks
    
    # RAG-LLM Analysis: Retrieve relevant context per question
    responses = []
    for question in questions:
        retrieved = retrieve_context(context, question, k=7)  # Semantic similarity search
        response = llm_prompt(question, retrieved, network_context)
        responses.append(response)
    
    return generate_incident_report(responses)
```

## Key Technical Contributions
The system’s innovations resolve fundamental RAG limitations in security contexts through three specific mechanisms:

1.  **MITRE ATT&CK-Guided Query Library**: Queries are predefined to target specific attack patterns (e.g., "Kerberos client query" for authentication anomalies) and mapped to MITRE techniques. Unlike generic log parsers, this ensures extracted indicators directly align with known adversary tactics, e.g., certificate abuse queries detect "Subject/SAN mismatches" across request/issuance events, revealing coordinated exploitation without requiring analysts to spot individual anomalies.

2.  **Cross-Event Semantic Chunking**: Aggregated results (e.g., Suricata alert summaries) form chunks preserving relationships (e.g., shared IP addresses across logs), not arbitrary text segments. The embedding model (all-mpnet-base-v2) ensures chunks like `{"high_severity_signatures": ["ET MALWARE Fake Microsoft Teams CnC"], "source_ips": ["10.1.17.215"]}` retain cross-source links. This enables LLMs to correlate events, e.g., matching HTTP downloads to Kerberos log sources via shared IPs, unlike standard RAG that breaks these patterns.

3.  **Cost-Performance Abstraction Layer**: All LLM interactions use a unified interface (Anthropic, OpenAI, Ollama), enabling side-by-side cost/performance testing. This revealed DeepSeek V3’s 15× cost advantage over Claude (89% vs 94% recall) and Llama 3.1:70b’s zero-cost local deployment (81% recall), eliminating vendor lock-in for cost-sensitive SOC teams. See Appendix for end-to-end data flow.

## Experimental Results
Evaluated on 17 malware scenarios (from Malware-Traffic-Analysis.net) and an Active Directory attack:

- **Top Performers**:
  - Claude Sonnet 4: 94% recall (cloud)
  - DeepSeek V3: 89% recall (cloud, 15× cheaper than Claude)
  - Llama 3.1:70b: 81% recall (local, zero per-query cost)
  - Cisco Foundation-Sec-8B: 71% recall (underperformed)
  
- **Active Directory Attack**:
  - Attack step detection: 100% precision, 96% recall (with enumeration prompt)
  - Ablation study confirmed RAG is essential: no-RAG missed all attack infrastructure due to context limits; Suricata-only missed identity attribution (lacking authentication logs).

All metrics reported without statistical significance details (paper states "average recall" without p-values).

## Related Work
This work extends prior LLM security applications (e.g., SecureBERT for classification) by moving beyond alert triage to *end-to-end incident narrative generation*. Unlike CyberRAG (for attack classification) or GraphRAG (for knowledge graphs), it designs a security-specific RAG pipeline for heterogeneous logs (Suricata, Zeek, Windows) with cross-event reasoning, proving that semantic chunking (not domain pretraining) drives performance.

## Limitations
The query library was built for specific attack types (malware, AD) and requires manual updates for novel threats, though modular design eases extension. The paper doesn’t specify exact LLM context limits, though it notes "within LLM context limits." The security-specialised Cisco model’s underperformance suggests domain adaptation alone is insufficient, but the authors don’t explore why.

## Appendix: Worked Example
**Scenario**: Fake Authenticator Malware (3,694 log events: Suricata alerts, Kerberos auth, HTTP downloads).  
**Goal**: Identify infected host and responsible user.

1.  **Targeted Queries Execute**:
    - *Suricata alert query*: Identifies malicious activity → `{"high_severity_signatures": ["ET MALWARE Fake Microsoft Teams CnC"], "source_ips": ["10.1.17.215"]}`
    - *Kerberos client query*: Attributes activity → `{"key": "shutchenson", "source_ips": [{"key": "10.1.17.215", "doc_count": 11}]}`
    - *File download query*: Reveals payload → `{"key": "10.1.17.215 -> 5.252.153.241:80 : /api/file/get-file/29842.ps1", "doc_count": 4}`

2.  **Semantic Chunking**:
    - Chunks are embedded as single vectors (e.g., "IP 10.1.17.215 linked to Shutchenson via Kerberos and C2 server").
    - Retrieved chunks for "infected host?" query: `{"source": "suricata", "content": "10.1.17.215"}, {"source": "kerberos", "content": "shutchenson -> 10.1.17.215"}`.

3.  **LLM Reasoning**:
    - Prompt: *"User shutchenson on DESKTOP-L8C5GSJ downloaded PowerShell payload from C2 server 5.252.153.241 (reference: suricata, kerberos)"*
    - Response: *"Infected host: 10.1.17.215; Responsible user: shutchenson (host DESKTOP-L8C5GSJ); C2 server: 5.252.153.241."*

This correlates evidence across sources, unlike raw logs where no single entry reveals the attack chain.

## References

- **Code:** https://github.com/neu-nds2/llm-sec-incident-analysis
- Xavier Cadet, Aditya Vikram Singh, Harsh Mamania, Edward Koh, Alex Fitts, Dirk Van Bruggen, Simona Boboila, Peter Chin, Alina Oprea, "Retrieval-Augmented LLMs for Security Incident Analysis", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.18196

Tags: #network-security #security-incident-analysis #retrieval-augmented-generation #llm-security
