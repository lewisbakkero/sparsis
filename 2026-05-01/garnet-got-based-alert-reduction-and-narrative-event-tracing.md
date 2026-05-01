---
title: "GARNET: GoT-Based Alert Reduction and Narrative Event Tracing"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36994"
---

## Executive Summary
GARNET is a framework that uses Large Language Models (LLMs) to reason over security alert correlation graphs, reducing false positives by 80% while generating structured, human-readable attack path summaries. For practitioners, this means significantly less alert fatigue, faster incident response times, and more transparent security operations.

## Why This Matters for Practitioners
If you're managing a SOC with thousands of daily alerts, GARNET could transform your alert management workflow. The paper reports a false positive rate reduction from approximately 1% (typical in real-world SOC operations) to below 0.0037, meaning out of 1,000 alerts, only about 3.7 would be false positives instead of 10 or more. This translates to saving hundreds of hours each week for your security team. You should consider integrating GARNET's graph-log alignment techniques into your alert processing pipeline, particularly focusing on the contrastive learning approach that enables LLMs to reason about security events in context without requiring manual annotations.

## Problem Statement
Imagine a hospital emergency department where every patient symptom is reported by a separate nurse shouting into their radio, some symptoms are genuine emergencies, others are misdiagnosed. The doctors can't hear what's important through the noise, and they waste precious minutes trying to make sense of it all. Similarly, SOC analysts today are drowning in scattered security alerts where 99% are false positives, making it impossible to focus on real threats.

## Proposed Approach
GARNET operates in three stages: graph-log multimodal alignment, graph-log semantic alignment, and GoT-based interaction reasoning. It starts with an alert correlation graph constructed from security logs, then:

1. Aligns graph and log embeddings into a shared vector space
2. Trains a novel LLM to understand graph semantics
3. Uses Graph-of-Thoughts (GoT) reasoning to generate simplified attack paths with explanations

```python
def garnet_pipeline(alerts):
    # Construct Alert Correlation Graph (ACG) from alert context
    acg = construct_acg(alerts)
    
    # Align graph and log embeddings using contrastive learning
    graph_embeddings = graph_encoder(acg)
    log_embeddings = log_encoder(associated_logs)
    aligned_embeddings = contrastive_align(graph_embeddings, log_embeddings)
    
    # Generate instructions for graph-log semantic alignment
    instruction_data = generate_instructions(acg, aligned_embeddings)
    llm = fine_tune_llm(instruction_data)
    
    # Use GoT-based reasoning to simplify attack paths
    simplified_paths = goT_reasoning(acg, llm)
    
    # Generate human-readable explanation
    return explain_attack_path(simplified_paths)
```

## Key Technical Contributions
GARNET's novelty lies in its three technical contributions that address the specific challenges of using LLMs for alert correlation:

1. **Graph-log multimodal alignment via contrastive learning**: The authors use a GCN encoder for graph embeddings and a Transformer-based text encoder for log sequences, then minimise the contrastive loss between node embeddings and their corresponding log text embeddings. This projects both modalities into a shared vector space where the embedding of a node and its log description are similar. The key insight is their specific contrastive loss formulation:
   ```
   LCL = 1/2 [ 1/k Σ log(exp(N^T_i T_i/τ)/Σ exp(N^T_i T_j/τ)) + 1/k Σ log(exp(T^T_i N_i/τ)/Σ exp(T^T_i N_j/τ)) ]
   ```
   where N_i is the graph node embedding, T_i is the log text embedding, k is the number of nodes, and τ is a temperature parameter. This ensures graph nodes and log descriptions are close in the shared vector space.

2. **Self-supervised graph-log instruction tuning**: They create a structure-aware node classification task that generates instructions with four components: graph data, relevant logs, human queries, and LLM responses. For each instruction instance, they randomly select a central node and perform n-hop neighbour sampling to construct a subgraph. The aligned node embedding is added to the instruction text, training the LLM to map nodes to security entities. This approach bridges the semantic gap between graph nodes and security entities without requiring manual annotations.

3. **GoT-based interaction reasoning with path-to-path logical reasoning**: GARNET breaks down alert reduction into a series of path-to-path logical reasoning steps. For attack paths reconstruction, it uses depth-first search on the ACG and asks the LLM to determine if adding a new edge completes an attack scenario. For abundant paths removal, it asks the LLM whether removing a node keeps the attack path complete, and removes redundant paths. The LLM's responses update the graph structure in each step, resulting in simplified, human-readable attack paths. See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
GARNET was evaluated across six attack scenarios from the AIT Log Dataset V2.0, outperforming six baselines including two context-based methods (DEEPCASE, aecid), one graph-based method (NoDoze), and three LLM-based approaches (GPT-o1, DS-chat, DS-r1).

Key results:
- On the "Fox" scenario, GARNET achieved an F1-score of 0.7083 compared to NoDoze's 0.4364 and DS-r1's 0.4545
- GARNET reduced false positives by 80% on average, lowering the false positive rate to below 0.0037
- GARNET achieved the highest Error Correction Rate (ECR) of 1.0000 on the "Wheeler" scenario
- GARNET significantly outperformed context-based methods (DEEPCASE and aecid), which had F1-scores around 0.1-0.2
- GARNET also outperformed graph-based methods (NoDoze), which had F1-scores around 0.4-0.6
- GARNET's performance was substantially better than LLM-based methods without reasoning (GPT-o1, DS-chat)

The paper doesn't explicitly state whether the improvements are statistically significant, but the consistent performance across all six scenarios suggests a robust advantage.

## Related Work
GARNET positions itself at the intersection of graph-based alert correlation and LLM reasoning. It improves on prior graph-based approaches like NoDoze, which simplifies alert dependency graphs but lacks explanations. It also surpasses context-based methods like DEEPCASE and aecid, which don't capture causal relationships between alerts. GARNET builds on recent LLM advancements for reasoning (Jin et al. 2023; Ullah et al. 2024) by addressing the specific challenges of graph-LLM interaction, rather than applying LLMs directly to alert data without considering the graph structure.

## Limitations
The authors acknowledge that GARNET was evaluated on a single dataset (AIT Log Dataset V2.0) and may not generalise to other security environments. The paper doesn't specify how many different types of security events or network architectures were tested, which could limit applicability. The authors also don't discuss the computational cost of training the graph-log alignment components or the GoT-based reasoning process.

From my perspective, the most significant limitation is that the paper doesn't address how GARNET handles alert correlation in real-time security operations. The evaluation appears to be on batch processing of historical alerts, not on a streaming alert pipeline where latency could be critical. This is a notable gap for practitioners implementing this in production systems.

## Appendix: Worked Example
Let's walk through a simplified example of how GARNET processes a single alert scenario based on the "Fox" attack scenario from the paper.

**Scenario**: The attacker used the account "phopkins" to escalate privileges and access /etc/shadow (a file containing hashed passwords).

**Step 1: Alert Correlation Graph Construction**
- AMiner detector generated 45 alerts from the logs
- Alerts were correlated into an Alert Correlation Graph (ACG) with 28 nodes and 32 edges
- Nodes represent security entities (e.g., user accounts, processes)
- Edges represent relationships (e.g., command execution)

**Step 2: Graph-Log Multimodal Alignment**
- For node representing "phopkins" (node ID 15), graph encoder produces embedding h15
- Related logs: "su - phopkins" (12:05), "sudo ls -laR /root/" (12:06), "cat /etc/shadow" (12:07)
- Log encoder concatenates these into text "su - phopkins, sudo ls -laR /root/, cat /etc/shadow" and generates embedding z15
- Contrastive learning minimizes the distance between h15 and z15, making their similarity high

**Step 3: Graph-Log Semantic Alignment**
- For node 15, the instruction becomes: "Node 15 represents user phopkins. Logs: su - phopkins, sudo ls -laR /root/, cat /etc/shadow. Map this node to the security entity."
- The LLM is trained to output "phopkins account" as the correct mapping
- This process is repeated for all nodes, creating semantic alignment between graph nodes and security entities

**Step 4: GoT-Based Interaction Reasoning**
- **Attack Paths Reconstruction**:
  - Start with entry node "www-data" (user ID 1001)
  - DFS on ACG: www-data → phopkins (su command) → root (sudo command) → /etc/shadow (cat command)
  - After first edge (www-data → phopkins): LLM asks "Does this path contain malicious behaviour?" → Yes
  - After second edge (www-data → phopkins → root): LLM asks "Does this path contain malicious behaviour?" → Yes
  - After third edge (www-data → phopkins → root → /etc/shadow): LLM asks "Does this path complete the attack scenario?" → Yes
  - Add complete path to simplified set

- **Abundant Paths Removal**:
  - For path www-data → phopkins → root → /etc/shadow, remove "phopkins" node
  - LLM asks "Does the path www-data → root → /etc/shadow remain complete?" → Yes
  - Remove "phopkins" node
  - Remove "root" node: LLM asks "Does the path www-data → /etc/shadow remain complete?" → No
  - Keep "root" node
  - Final simplified path: www-data → root → /etc/shadow

**Step 5: Human-Readable Explanation**
- LLM generates: "The phopkins account successfully performed multiple privilege escalation operations through a series of sudo and su commands, attempting to list files in the root directory and access sensitive files that stored system user password information."

This simplified path with explanation allows security analysts to quickly understand the attack vector without manually analysing all 45 original alerts.

## References

- Yiru Gong, Song Liu, Changzhi Zhao, Junrong Liu, Tian Tian, Xiaobo Yang, "GARNET: GoT-Based Alert Reduction and Narrative Event Tracing", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36994

Tags: #security #alert-reduction #llm #graph-based-ai #explainable-ai
