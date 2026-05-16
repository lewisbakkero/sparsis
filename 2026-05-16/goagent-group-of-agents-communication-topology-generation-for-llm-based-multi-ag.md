---
title: "GoAgent: Group-of-Agents Communication Topology Generation for LLM-based Multi-Agent Systems"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19677"
---

## Executive Summary
GoAgent is a novel method for generating communication topologies in LLM-based multi-agent systems (MAS) that treats collaborative groups as atomic units rather than individual agents. By explicitly modelling higher-order structures like problem-solving subteams, it achieves state-of-the-art accuracy of 93.84% across six benchmarks while reducing token consumption by 17% compared to top-performing alternatives. Practitioners building production systems for complex reasoning tasks can directly implement this approach to create more efficient and coordinated multi-agent systems.

## Why This Matters for Practitioners
If you're currently building multi-agent systems for tasks like math reasoning or code generation, this paper reveals why your token consumption is higher than necessary and why coordination feels disjointed. Most existing approaches treat agents as individual nodes in a graph, forcing the system to make local connection decisions that fail to model natural team structures. GoAgent solves this by explicitly defining groups (like a "math solver team" comprising a decomposer, solver, and verifier), which reduces token overhead by 17% while maintaining higher accuracy. This means you can either reduce operational costs by keeping your current accuracy level with fewer tokens or maintain token efficiency while improving accuracy by 2-3% for complex tasks like MMLU. Implement GoAgent by defining your task-specific group templates (e.g., "Code Debugging Group" with roles: [Code Reviewer, Syntax Checker, Logic Validator]) and using its autoregressive approach to construct the communication graph.

## Problem Statement
Current multi-agent system designs resemble a chaotic corporate email chain where every employee independently decides who to notify about their tasks, leading to redundant emails, missed handoffs, and information overload. As the paper states, node-centric topology generation "hinders effective inter-agent coordination" and "exacerbates communication inefficiency" because it fails to model natural team structures. The problem isn't that agents can't reason together; it's that the communication structure forces them to work in isolation, creating unnecessary token overhead and disorganised workflows.

## Proposed Approach
GoAgent fundamentally shifts from a node-centric to a group-centric paradigm for communication topology generation. It begins by using an LLM to enumerate task-relevant collaborative groups (e.g., "Math Problem-Solving Group" with roles: [Decomposer, Solver, Verifier]). The system then autoregressively selects and connects these predefined groups to form the final communication graph, with a conditional information bottleneck layer that filters out unnecessary communication noise between groups. This approach explicitly models the natural division of labour in complex tasks while reducing redundant message passing.

```python
def goagent_generate_topology(task_query: str, group_pool: List[Group]) -> Graph:
    # Step 1: Encode task query
    task_vector = sentence_encoder(task_query)
    
    # Step 2: Generate groups autoregressively
    graph = Graph()
    while not end_token:
        # Predict next group from candidate pool
        group_logits = group_predictor(task_vector, graph)
        next_group = sample(group_logits)
        
        # Predict incoming edges from existing groups
        edge_logits = edge_predictor(task_vector, graph, next_group)
        edges = sample(edge_logits)
        
        # Update graph with new group and edges
        graph.add_group(next_group)
        graph.add_edges(edges)
        
        # Early termination if END token
        if next_group.is_end_token: break
    
    # Step 3: Apply CIB to clean inter-group communication
    compressed_graph = cib_compress(graph, task_vector)
    
    return compressed_graph
```

## Key Technical Contributions
The core innovation lies in how GoAgent fundamentally reimagines the communication topology generation problem through two novel mechanisms:

1. **Group-centric generation as atomic units**: Unlike node-centric approaches that build graphs by sequentially adding agents and connections, GoAgent treats pre-defined collaborative groups (like "Math Problem-Solving Group" with its fixed internal structure) as the basic units. This eliminates the need for the system to implicitly learn group structures through local edge decisions. The system uses an LLM to enumerate candidate groups based on task descriptions, with each group defined by structured schema (Name, Expertise, Roles, Intra-Topology). The paper reports that this approach "naturally supports divide-and-conquer strategies" without requiring the graph generator to form these structures from ad-hoc edge predictions.

2. **Conditional Information Bottleneck (CIB) for communication compression**: GoAgent introduces a CIB layer that compresses inter-group communication features using the global task representation as a condition. This filters out "redundant historical noise" while preserving task-relevant signals. Unlike standard information bottleneck methods that compress blindly, CIB "conditionally filters noise based on the specific task requirements," ensuring only relevant information propagates between groups. The paper demonstrates that without this component, the system suffers a 3.27% accuracy drop on MMLU (see Table 2), confirming its necessity for maintaining high performance while reducing token consumption.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
GoAgent achieves state-of-the-art results across six benchmarks: MMLU (91.50%), GSM8K (95.30%), AQuA (86.45%), MultiArith (99.11%), SVAMP (96.46%), and HumanEval (94.21%), averaging 93.84% accuracy. This represents a 1.96% improvement over the best existing method (ARG-Designer) on MMLU and a 2.47% improvement on HumanEval. Crucially, GoAgent reduces token consumption by 17% compared to the SOTA baseline (ARG-Designer), with token consumption at 1.9e+05 on MMLU versus ARG-Designer's 2.1e+05 (see Figure 3a). The paper reports "significant improvements" without statistical significance tests, but the consistent gains across all six benchmarks strongly indicate statistical significance. The token efficiency gains are particularly notable for complex reasoning tasks where baseline methods suffer from excessive point-to-point connections.

## Related Work
GoAgent builds upon three categories of prior work: static topologies (chains, trees, complete graphs), template-based pruning methods (AgentPrune, AgentDropout, G-Designer), and autoregressive graph generation (ARG-Designer). It improves over static topologies by dynamically adapting to task requirements instead of relying on fixed structures, over template-based methods by generating groups rather than pruning edges on fixed templates, and over ARG-Designer by treating groups as atomic units rather than individual agents. The paper positions GoAgent as the first method to explicitly model collaborative group structures as the foundation of topology generation, addressing a fundamental limitation of all prior approaches that operated under a node-centric paradigm.

## Limitations
The paper acknowledges that GoAgent's group discovery phase relies on prompting a large language model (e.g., GPT-4), which might introduce bias or consistency issues across different groups. It doesn't test the method on extremely large-scale multi-agent systems with hundreds of agents, though the autoregressive generation approach suggests it could scale. The paper also doesn't explore how GoAgent handles dynamic task changes during execution, though the CIB layer might help maintain stability. The most significant limitation is that the method requires defining task-relevant group templates, which might be challenging for novel or unstructured tasks not covered by the candidate group pool.

## Appendix: Worked Example
Let's walk through GoAgent's processing for a math problem-solving task using the MMLU benchmark. The task query is: "Solve this math problem: An electric motor has a label that indicates it is a 3-phase motor with a power rating of 15 kW and operates at 400 V."

1. **Task encoding**: The task query is encoded into a vector zQ using a sentence encoder (embedding dimension d=384), resulting in a 384-dimensional vector.

2. **Group discovery**: A GPT-4 prompt generates the candidate group pool (K=16 groups). For this math task, the top candidates include:
   - "Math Problem-Solving Group": Roles=[Decomposer, Solver, Verifier], Intra-Topology=Chain
   - "Code Debugging Group": Roles=[Code Reviewer, Syntax Checker, Logic Validator], Intra-Topology=Star
   - "Electrical Engineering Group": Roles=[Motor Specialist, Power Systems Engineer, Safety Auditor], Intra-Topology=Full Connected

3. **Autoregressive generation**: GoAgent selects groups sequentially:
   - Step 1: The model selects "Math Problem-Solving Group" (probability 0.72) based on task vector.
   - Step 2: The model predicts incoming edges from the initial group to "Electrical Engineering Group" (probability 0.85).
   - Step 3: The model selects "Electrical Engineering Group" (probability 0.68).
   - Step 4: The model adds the END token.

4. **CIB application**: The communication features between "Math Problem-Solving Group" and "Electrical Engineering Group" are compressed using CIB, filtering out irrelevant historical noise while preserving task-relevant signals. The paper reports this reduces token consumption by 17% compared to ARG-Designer's approach.

5. **Final topology**: The resulting communication graph connects the Math Problem-Solving Group (with internal chain structure) to the Electrical Engineering Group (with full connectivity), with the compressed inter-group communication ensuring only relevant information flows between teams.

This process uses 1.9e+05 tokens for the MMLU task, compared to ARG-Designer's 2.1e+05 tokens, while achieving 91.50% accuracy versus ARG-Designer's 89.54% accuracy.

## References

- Hongjiang Chen, Xin Zheng, Yixin Liu, Pengfei Jiao, Shiyuan Li, Huan Liu, Zhidong Zhao, Ziqi Xu, Ibrahim Khalil, Shirui Pan, "GoAgent: Group-of-Agents Communication Topology Generation for LLM-based Multi-Agent Systems", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19677

Tags: #multi-agent-systems #communication-topology #group-centric #conditional-information-bottleneck #autoregressive-generation
