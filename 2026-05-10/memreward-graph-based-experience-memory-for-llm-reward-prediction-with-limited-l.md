---
title: "MemReward: Graph-Based Experience Memory for LLM Reward Prediction with Limited Labels"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19310"
---

## Executive Summary
MemReward introduces a graph-based experience memory framework that enables effective reinforcement learning fine-tuning for LLMs with only 20% of the required ground-truth labels. It constructs a heterogeneous graph of queries, thinking processes, and answers to propagate rewards from labelled to unlabeled rollouts, achieving 97.3% of Oracle performance on Qwen2.5-3B and 96.6% on Qwen2.5-1.5B. This reduces the costly human labelling requirements for LLM reward modelling while improving generalisation to out-of-domain tasks.

## Why This Matters for Practitioners
If you're building production LLM systems that require reinforcement learning fine-tuning for tasks like mathematical reasoning or open-ended question answering, you're likely facing the high cost of human-labelled reward signals. MemReward demonstrates that you can achieve nearly full performance with only 20% of the labels by leveraging graph-based reward propagation, saving significant operational costs without sacrificing quality. For example, if your current pipeline requires 10,000 expert-labelled examples for math problem verification (costing approximately $50,000 at $5 per label), MemReward would allow you to use just 2,000 labelled examples while maintaining 97% of the performance, reducing labelling costs by 80% while also improving out-of-domain generalisation.

## Problem Statement
Imagine trying to train a medical diagnostic system where each case requires a specialist to verify the diagnosis before it can be used in training. For common conditions, you might have enough verified cases (labels), but for rare conditions, you'd have very few. Similarly, when training LLMs through reinforcement learning, mathematical proof verification demands expert review, and open-ended question answering lacks definitive ground truth, creating a scarcity of labelled rewards that constrains training effectiveness.

## Proposed Approach
MemReward organises labelled LLM rollouts (queries, thinking processes, and answers) into a heterogeneous graph to propagate rewards to unlabeled rollouts. The system consists of two phases: a warmup phase where a heterogeneous graph is constructed from labelled rollouts and a GNN is trained to predict rewards, and an online phase where unlabeled rollouts connect to this graph through top-k similarity edges and receive GNN-predicted rewards during policy optimisation.

```python
def online_grpo_with_memreward(R_labeled, R_unlabeled, trained_gnn, policy):
    R_train = R_labeled + R_unlabeled
    
    for _ in range(training_iterations):
        batch = random.sample(R_train, batch_size)
        
        for query in batch:
            rollouts = []
            for _ in range(n_rollouts):
                thinking, answer = policy.generate(query)
                rollouts.append((thinking, answer))
            
            for i, (thinking, answer) in enumerate(rollouts):
                if (query, thinking, answer) in R_labeled:
                    reward = get_ground_truth_reward(query, thinking, answer)
                else:
                    query_emb = encode(query)
                    thinking_emb = encode(thinking)
                    answer_emb = encode(answer)
                    
                    similar_queries = get_top_k_similar(query_emb, warmup_graph)
                    graph_subgraph = build_subgraph(similar_queries, warmup_graph)
                    
                    reward = gnn_predict(graph_subgraph, thinking_emb, answer_emb)
                
                update_rollout_reward(query, thinking, answer, reward)
        
        update_policy(policy, batch_rewards)
```

## Key Technical Contributions
MemReward's architectural innovations enable effective reward propagation across unlabeled rollouts. The core innovations involve how the heterogeneous graph captures relational structures between different types of nodes and how the reward prediction mechanism works.

1. **Heterogeneous graph construction with three node types**: MemReward builds a graph with three distinct node types (query nodes, thinking nodes, and answer nodes) connected with three edge types (query-query for semantic similarity, query-thinking for direct association, and thinking-answer for one-to-one pairing). This captures the multi-faceted relationships between experiences: semantically similar queries may share reward patterns, multiple rollouts for the same query compete in quality, and answer correctness depends on both reasoning path and query context. Unlike homogeneous graphs that treat all nodes uniformly, this heterogeneous design preserves structural relationships essential for accurate reward prediction.

2. **GNN-based reward propagation with learnable message passing**: The framework uses a heterogeneous GNN that performs iterative message passing through the different edge types to predict rewards. The node embeddings are updated through type-specific aggregation (Equation 2-4 in the paper), allowing the model to learn how different relationships (query-query similarity, query-thinking association, thinking-answer correspondence) contribute to reward prediction. This is more powerful than methods that rely on fixed similarity metrics or operate on individual samples without exploiting structural dependencies.

3. **Graph integration with GRPO for online policy optimisation**: MemReward integrates the GNN-predicted rewards directly into the Group Relative Policy Optimisation (GRPO) algorithm used for RL fine-tuning. For labelled rollouts, it uses ground-truth rewards; for unlabeled rollouts, it connects them to the warmup graph via top-k similarity edges and uses GNN-predicted rewards. This hybrid reward acquisition strategy ensures that the policy optimisation process benefits from both the high-quality ground-truth rewards and the structurally informed predictions from the GNN.

4. **Out-of-domain generalisation through semantic similarity**: The graph structure enables out-of-domain generalisation without task-specific fine-tuning. As the GNN is trained on in-domain data (10 benchmarks), it can transfer learned reward patterns to held-out tasks (NuminaMath, SIQA, PIQA) by connecting new queries to the warmup graph through top-k similarity edges. The paper shows MemReward surpasses fully-supervised Oracle performance on average for both model scales (66.96 vs. 66.07 on Qwen2.5-3B), demonstrating that the GNN-predicted rewards improve generalisation beyond full supervision. See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
MemReward was evaluated on 13 benchmarks across three domains (math, QA, code) using Qwen2.5-3B and Qwen2.5-1.5B. With only 20% ground-truth labels, it achieved 97.3% of Oracle performance on 3B (77.02% vs. Oracle's 79.12%) and 96.6% on 1.5B (68.10% vs. Oracle's 70.47%). Notably, it surpassed Oracle performance on out-of-domain tasks: on Qwen2.5-3B, it achieved 66.96% on NuminaMath, SIQA, and PIQA compared to Oracle's 66.07% (an improvement of 0.89 points), and on Qwen2.5-1.5B, it achieved 62.81% versus Oracle's 62.00% (0.81 point improvement).

The largest gains appeared in mathematical reasoning: on Qwen2.5-1.5B, GSM8K improved by 11.56 points (88.67 vs. 77.11) and GSM-Symbolic by 14.89 points (77.78 vs. 62.89) over the R1-p baseline (which uses only 20% labels without propagation). Performance scaled smoothly with label budget: at 70% labels, MemReward reached 99.4% of Oracle performance (78.64% vs. Oracle's 79.12% on Qwen2.5-3B).

The ablation studies confirmed the importance of each architectural component: removing thinking nodes (w/o Thinking Node) caused the largest drop in code generation performance (58.0% on 3B vs. 63.0% for full model, a 5.0% drop), while using a homogeneous graph (removing edge type distinctions) dropped math performance by 3.7% compared to the full model.

## Related Work
MemReward extends graph neural network applications to reward prediction in LLM training, bridging semi-supervised learning and relational modelling. While traditional RL fine-tuning approaches like RLHF (Ouyang et al., 2022) and GRPO (Guo et al., 2025) assume full reward supervision, MemReward addresses label scarcity through graph-based reward propagation. This builds on semi-supervised learning techniques like label propagation (Zhu & Ghahramani, 2002) and pseudo-labelling (Lee et al., 2013), but extends them to capture structural dependencies between different types of experiences (queries, thinking processes, answers) through a heterogeneous graph. Unlike prior work that operates on individual samples or uses fixed similarity metrics, MemReward uses learnable message passing over a structured graph to predict rewards, enabling better generalisation to out-of-domain tasks.

## Limitations
The paper doesn't extensively explore the computational overhead of building and maintaining the heterogeneous graph, though the experiments indicate it's manageable. The authors acknowledge that reward propagation effectiveness depends on the semantic similarity between queries, which might be limited for very diverse tasks. The out-of-domain results are promising, but the paper doesn't test generalisation to completely novel domains beyond the three categories (math, QA, code) they evaluated. Additionally, the paper doesn't address how MemReward would perform with extremely low label budgets (below 20%).

## Appendix: Worked Example
Let's walk through the reward prediction process for a mathematical reasoning query using MemReward. Consider a math question about quadratic equations for which we have a labelled rollout: "Solve x² + 5x + 6 = 0" (query), "The equation factors into (x+2)(x+3)=0, so x=-2 or x=-3" (thinking), and "x=-2 or x=-3" (answer) with ground truth reward 1 (correct).

1. **Graph Construction (Warmup Phase)**:
   - Query node: "Solve x² + 5x + 6 = 0" (embedding q1)
   - Thinking node: "The equation factors into (x+2)(x+3)=0, so x=-2 or x=-3" (embedding t1)
   - Answer node: "x=-2 or x=-3" (embedding a1)
   - Query-query edge: connects q1 to a similar query "Solve x² + 7x + 12 = 0" (embedding q2) with cosine similarity 0.85 (top-k=3)
   - Query-thinking edge: connects q1 to t1
   - Thinking-answer edge: connects t1 to a1

2. **GNN Training**:
   - The GNN processes these nodes and edges through multiple layers.
   - After training, the GNN learns that for similar query embeddings (e.g., quadratic equations), the thinking process and answer must correctly factor the equation.
   - On validation data, the GNN achieves 0.917 ROC-AUC with a score separation of 0.51 between correct (mean score 0.63) and incorrect (mean score 0.11) responses.

3. **Online Prediction for Unlabeled Rollout**:
   - Now consider an unlabeled query "Solve x² + 8x + 15 = 0" (embedding q3).
   - The GNN connects q3 to the top-3 similar queries in the warmup graph (q1 with similarity 0.82, q2 with 0.75, and another quadratic equation query with 0.68).
   - The graph subgraph with these connections is processed through the GNN.
   - For the unlabeled rollout with thinking process "The equation factors into (x+3)(x+5)=0, so x=-3 or x=-5" (t3) and answer "x=-3 or x=-5" (a3), the GNN predicts a reward score of 0.85 (above the 0.5 threshold, so it gets a reward = 1).
   - This predicted reward is used alongside ground-truth rewards in the GRPO objective.

## References

- Tianyang Luo, Tao Feng, Zhigang Hua, Yan Xie, Shuang Yang, Ge Liu, Jiaxuan You, "MemReward: Graph-Based Experience Memory for LLM Reward Prediction with Limited Labels", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19310

Tags: #machine-learning #graph-neural-networks #reinforcement-learning #large-language-models #reward-prediction #semi-supervised-learning
