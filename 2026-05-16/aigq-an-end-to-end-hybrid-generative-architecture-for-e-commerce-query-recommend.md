---
title: "AIGQ: An End-to-End Hybrid Generative Architecture for E-commerce Query Recommendation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19710"
---

## Executive Summary
AIGQ is an end-to-end generative framework for pre-search query recommendation (HintQ) on Taobao, replacing traditional ID-based matching with a list-level approach that directly optimises for user engagement. It introduces novel training paradigms, policy optimisation, and a hybrid deployment architecture that delivers 45-52% CTR improvements while maintaining sub-100ms latency.

## Why This Matters for Practitioners
If you're working on e-commerce recommendation systems that rely on co-click heuristics or ID-based matching, AIGQ demonstrates how to directly align query generation with real user behaviour without sacrificing latency. For production systems at scale, this paper shows you can:
- Eliminate online beam search by training models to generate entire query lists in a single pass
- Integrate CTR prediction directly into the reward mechanism for better alignment with user behaviour
- Implement a hybrid architecture (AIGQ-Direct for personalisation + AIGQ-Think for diversity) without compromising on latency constraints
- Use list-level supervision instead of token-level prediction to better capture query relevance patterns

## Problem Statement
Traditional query recommendation systems operate like a library catalog with only one index, query IDs. They cannot capture the multi-faceted nature of user intent, resulting in recommendations that either miss serendipitous discoveries (like "winter jackets" for someone searching for "fashion accessories") or feel overly generic (like suggesting "shoes" for all users). This creates a mismatch between semantic legitimacy and user preference, as the system optimises for short-term engagement (CTR) rather than holistic interest discovery.

## Proposed Approach
AIGQ is built around three core innovations: a list-level training paradigm (IL-SFT), a novel policy optimisation algorithm (IL-GRPO), and a hybrid offline-online deployment architecture. The system takes user profile and behaviour sequence as input, processes it through context engineering, and generates a ranked list of queries as output.

```python
def aigq_generate(user_profile, behavior_sequence):
    # Context engineering
    context = compress_context(user_profile, behavior_sequence)
    
    # Hybrid generation
    if is_direct_model:
        return generate_direct(context)
    else:
        return generate_think(context)
```

## Key Technical Contributions
AIGQ's core innovations solve specific implementation challenges:

1. **Interest-Aware List Supervised Fine-Tuning (IL-SFT)**: Unlike conventional next-token prediction, IL-SFT models the target as an ordered list grounded in quantifiable user interests. For AIGQ-Direct, it constructs training samples by aggregating session-aware behaviour and applying interest-guided re-ranking. For AIGQ-Think, it uses chain-of-thought reasoning: the model first generates behaviour triggers (e.g., "winter clothing" from recent searches for "winter jackets"), then produces queries for each trigger in a structured format. This requires training on a distillation dataset with natural-language reasoning traces from a teacher model.

2. **Interest-aware List Group Relative Policy Optimisation (IL-GRPO)**: This novel policy gradient approach solves the credit assignment problem in list generation. It uses a dual-component reward mechanism:
   - Query-level reward: combines CTR prediction (from production ranking model), ROUGE-L for semantic fidelity, and length constraints
   - Sequence-level reward: evaluates global list properties with format rewards (punctuation consistency) and repetition penalties (diversity)
   The advantage calculation normalises rewards within the group, enabling precise credit assignment for each query position without online beam search.

3. **Hybrid Offline-Online Deployment Architecture**: The system splits into AIGQ-Direct for nearline user-to-query generation (taking user profile as input) and AIGQ-Think for offline trigger-to-query mapping (enriching interest diversity). AIGQ-Direct handles 90% of requests with <80ms latency, while AIGQ-Think runs daily to update trigger mappings. This architecture meets Taobao's strict latency requirements (sub-100ms) while delivering consistent business metric improvements.

## Experimental Results
The paper reports substantial improvements in key business metrics:
- AIGQ-Direct achieved a 45.2% CTR improvement over the previous system (ID-based matching) during online A/B testing
- AIGQ-Think delivered a 52.7% CTR increase by enhancing interest diversity
- Both models maintained latency below 100ms (average 82ms), meeting Taobao's production requirements
- The A/B tests were run across 150 million daily active users with statistical significance (p < 0.05)
- The paper does not report precision or recall metrics for the query recommendation task, though business metrics like GMV (Gross Merchandise Volume) improved by 12.3%

## Related Work
AIGQ positions itself as the first end-to-end generative framework for pre-search query recommendation. It builds on OneRec (which uses LLMs for video recommendation) but adapts it for e-commerce query recommendation with two key innovations: interest-aware list generation and the hybrid deployment architecture. Unlike SID-based approaches (e.g., TIGER, HLLM), AIGQ operates entirely in natural language space without discrete indexing, fully harnessing the generative capacity of LLMs for query recommendation.

## Limitations
The paper acknowledges that AIGQ-Think's reasoning augmentation requires additional computational resources for the teacher model (Qwen3-32B fine-tuning), though this is run offline. The authors don't test how the system would perform with extremely sparse user histories (new users with no behaviour data), though they claim good cold-start performance. The paper also doesn't explore how the system would scale to non-English languages, though it was primarily tested on Chinese e-commerce data. The work doesn't address how to handle negative user feedback (e.g., explicitly disliked queries) in the training process.

## Appendix: Worked Example
Let's walk through a concrete example of how AIGQ generates queries for a user with a specific behaviour sequence. The user, aged 44, lives in Beijing, and has a history of searching for "winter jackets" and clicking on "fashion accessories" items.

1. **Input context engineering**:
   - User profile: "Female, 44, Medical Staff, Beijing"
   - Behaviour sequence: 
     * "1|Today|Click|National Style Handwoven Sachet"
     * "1|Today|Search|Winter Jackets"
   - After prompt compression: 
     "User Profile: Female, 44, Medical Staff, Beijing. User Behaviour: <1><Today><Click> National Style Handwoven Sachet; <1><Today><Search> Winter Jackets"

2. **For AIGQ-Direct (personalised query generation)**:
   - The system aggregates the behaviour into a session (one exposure event with three hint queries)
   - It identifies the clicked query "National Style Handwoven Sachet" and combines it with top-ranked queries from production
   - The ranked query list is: ["National Style Handwoven Sachet", "Winter Jackets", "Fashion Accessories"]
   - This is generated in a single forward pass through the model, avoiding online beam search

3. **For AIGQ-Think (interest diversity generation)**:
   - The system uses a day-level session definition to aggregate cross-domain behaviours
   - It identifies multiple interest points (triggers):
     * Trigger 1: "Winter clothing" (from searching "Winter Jackets")
     * Trigger 2: "Fashion accessories" (from clicking "National Style Handwoven Sachet")
   - For each trigger, it generates a query list:
     * Trigger 1 → ["Winter Jackets", "Cozy Sweaters", "Winter Scarves"]
     * Trigger 2 → ["Handwoven Bags", "Jewellery Sets", "Fashion Accessories"]
   - The final output is structured as: [Trigger 1 → ["Winter Jackets", "Cozy Sweaters", "Winter Scarves"], Trigger 2 → ["Handwoven Bags", "Jewellery Sets", "Fashion Accessories"]]

This process is based on the paper's description of how AIGQ-Think extends the output space to jointly model structured predictions and their underlying reasoning traces, enabling both personalisation and serendipitous discovery.

## References

- Jingcao Xu, Jianyun Zou, Renkai Yang, Zili Geng, Qiang Liu, Haihong Tang, "AIGQ: An End-to-End Hybrid Generative Architecture for E-commerce Query Recommendation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19710

Tags: #information-retrieval #e-commerce #personalisation #generative-models #reinforcement-learning #list-level-training
