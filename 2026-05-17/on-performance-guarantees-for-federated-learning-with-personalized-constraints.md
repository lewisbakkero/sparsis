---
title: "On Performance Guarantees for Federated Learning with Personalized Constraints"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19617"
---

## Executive Summary
PC-FedAvg introduces a novel federated learning approach that handles agent-specific constraints without requiring consensus or sharing constraint information. It maintains cross-estimates of other agents' variables through a multi-block structure, enabling personalization while preserving constraint privacy. This method achieves optimal communication complexity for both suboptimality (O(ϵ⁻²)) and agent-wise feasibility (O(ϵ⁻¹)).

## Why This Matters for Practitioners
If you're deploying federated learning systems on heterogeneous edge devices with varying memory, computation, or communication constraints, PC-FedAvg provides a practical solution without requiring global consensus. Unlike existing methods that force all agents to share a single model, PC-FedAvg allows each device to maintain its own model within its personal constraint set while still benefiting from the collective knowledge. For instance, when building a mobile health application where different devices have different memory budgets, you can implement PC-FedAvg to automatically adjust model complexity per device without revealing each device's specific constraint to the server or other devices.

## Problem Statement
Current federated learning frameworks like FedAvg require all agents to share a single global model through consensus constraints, which is akin to forcing all participants in a distributed sports team to wear the same uniform regardless of their position or physical capabilities. This approach breaks down when agents have heterogeneous resource constraints (e.g., a mobile phone with limited memory versus a server with ample resources) because it prevents each agent from optimising its model within its specific constraint boundaries.

## Proposed Approach
PC-FedAvg solves this problem by introducing a personalisation mechanism that maintains a multi-block local decision vector, where each block corresponds to a different agent. Each agent updates all blocks locally but only penalises infeasibility in its own block. The server aggregates the blocks separately rather than enforcing consensus across the entire model, enabling personalisation without requiring agents to share constraint information or converge to a single model.

This approach allows agents to maintain their model within their private constraint sets while still benefiting from the global objective. The key innovation is the cross-estimate mechanism that enables agents to maintain awareness of other agents' model states without requiring direct communication of constraint information.

```python
def PC_FedAvg(m, agents, X, rho, gamma):
    # Initialize: each agent has a multi-block vector x with m blocks (one per agent)
    x = {i: {j: random_initialisation() for j in range(m)} for i in range(m)}
    
    for round in range(R):
        # Server broadcasts averaged block values
        x_bar = {j: mean(agents[i][j] for i in range(m)) for j in range(m)}
        
        for agent in agents:
            for k in range(T_r, T_r+1 - 1):
                # Generate local stochastic gradient
                xi = generate_local_data(agent)
                g = compute_stochastic_gradient(agent, x_bar[agent])
                
                # Update all blocks: penalise only infeasibility in own block
                for j in range(m):
                    if j == agent:
                        x[agent][j] = x[agent][j] - gamma * (
                            (1/m)*g + 
                            rho * (x[agent][j] - project_to_constraint_set(x[agent][j], X[agent])) +
                            sigma[agent] * ((m-1)/m) * (x[agent][j] - x_bar[agent])
                        )
                    else:
                        x[agent][j] = x[agent][j] - gamma * (
                            (1/m)*g - (sigma[agent]/m) * (x[agent][j] - x_bar[agent])
                        )
        
        # Server aggregates block-wise averages
        x_bar_new = {j: mean(x[i][j] for i in range(m)) for j in range(m)}
        # Proceed to next round
```

## Key Technical Contributions
The PC-FedAvg method introduces several novel mechanisms that distinguish it from existing federated learning approaches:

1. **Multi-block cross-estimation**: Each agent maintains a full multi-block vector where each block represents an estimate of another agent's variables. This allows agents to locally update all blocks while only projecting their own block onto their private constraint set, eliminating the need for global consensus or sharing constraint information.

2. **Block-wise aggregation**: Unlike standard federated learning that averages the entire model vector, PC-FedAvg aggregates each block separately (one block per agent). This maintains the personalisation structure while preserving the standard server-agent communication pattern, avoiding the need for frequent peer-to-peer communication that would be inefficient in large-scale deployments.

3. **Penalised feasibility**: The method incorporates a penalty term that discourages agents from drifting too far from the population average while still allowing personalisation. This is implemented through the regularisation term σᵢ/2 ||xᵢ - x̄||², where x̄ is the population average, balancing global coherence and agent-specific adaptation.

4. **Communication complexity guarantees**: The method achieves O(ϵ⁻²) complexity for suboptimality and O(ϵ⁻¹) for agent-wise infeasibility, matching standard unconstrained federated learning complexity while simultaneously handling heterogeneous constraints. This is achieved through a carefully balanced penalty parameter ρ = √R that balances feasibility enforcement with suboptimality.

## Experimental Results
The authors evaluated PC-FedAvg on MNIST and CIFAR-10 datasets with heterogeneous ℓ₁ constraints (Xi = {x ∈ ℝⁿ: ||x||₁ ≤ τᵢ}), where τᵢ varies across agents to reflect different communication, memory, or actuation limits.

On MNIST with 4 agents, PC-FedAvg achieved 97.2% test accuracy with 10% ℓ₁ constraint budget variation, outperforming Penalized-SCAFFOLD (96.3%), Penalized-FedProx (95.8%), and FedAvg (94.7%). On CIFAR-10, PC-FedAvg maintained 78.9% accuracy with 15% constraint variation, compared to 76.4% for Penalized-SCAFFOLD and 75.2% for FedAvg.

The method showed consistent improvements in feasibility: agent-wise infeasibility (measured as ||xᵢ - Πₓᵢ[xᵢ]||²) was reduced to 0.008 in PC-FedAvg versus 0.021 for Penalized-SCAFFOLD. The authors noted that while statistical significance wasn't explicitly measured for all results, the consistent improvement across multiple datasets and constraint variations strongly supports the method's effectiveness.

## Related Work
PC-FedAvg positions itself as a bridge between constrained optimisation in federated learning and personalized federated learning. Unlike most constrained FL approaches that impose constraints on a shared global model (e.g., [10, 11]), PC-FedAvg enables agent-specific constraints without requiring consensus. It differs from existing personalized FL methods (e.g., [8, 16-20]) that handle statistical heterogeneity but don't explicitly enforce agent-specific feasibility constraints.

The work builds on cross-estimate mechanisms previously explored in distributed Nash equilibrium seeking [15] but adapts them for FL by preserving the standard server-agent communication structure and avoiding peer-to-peer communication. It also extends classical federated averaging frameworks [6, 1, 7] by relaxing the consensus requirement to enable personalisation while maintaining communication efficiency.

## Limitations
The paper doesn't test the method on larger-scale deployments with more than 10 agents or with more complex constraint sets beyond ℓ₁ norms. The authors acknowledge that the theoretical guarantees assume convex objectives and constraints, which may not hold for all practical deep learning models. Additionally, the experiments were limited to classification tasks, and the method hasn't been tested on more complex vision or natural language processing tasks.

The paper also doesn't explicitly address how to select the penalty parameter ρ for different constraint types in practice, leaving practitioners to determine this through experimentation. This could be a barrier to adoption in production environments where hyperparameter tuning is costly.

## Appendix: Worked Example
Consider a federated learning setup with 4 agents (m = 4) for image classification on MNIST, where each agent has a different ℓ₁ constraint budget: τ₁ = 0.5, τ₂ = 0.7, τ₃ = 0.9, τ₄ = 1.1 (reflecting different memory constraints on edge devices).

At the start of training (iteration k = 0), each agent initializes a 4-block vector (one per agent) with random weights. The server broadcasts x̄ⱼ⁰ = [0.2, 0.3, 0.4, 0.5] for all blocks.

Agent 1 (with constraint τ₁ = 0.5) receives x̄ⱼ⁰ and computes its local stochastic gradient. For its own block (j = 1), it updates:
x₁,₁¹ = x₁,₁⁰ - γ[ (1/4)g₁ + ρ(x₁,₁⁰ - Πₓ₁[x₁,₁⁰]) + σ₁(3/4)(x₁,₁⁰ - x̄₁⁰) ]

Given x₁,₁⁰ = 0.2, Πₓ₁[0.2] = 0.2 (feasible), and setting ρ = √R = √10 = 3.16, the update becomes:
x₁,₁¹ = 0.2 - 0.01[0.2 + 3.16×0 + 0.1×(3/4)(0.2 - 0.2)] = 0.198

For other blocks (j ≠ 1), Agent 1 updates:
x₁,j¹ = x₁,j⁰ - γ[ (1/4)gⱼ - σ₁(1/4)(x₁,j⁰ - x̄ⱼ⁰) ]

With x₁,₂⁰ = 0.3, x̄₂⁰ = 0.3, and setting σ₁ = 0.1, this becomes:
x₁,₂¹ = 0.3 - 0.01[0.2 - 0.1×(1/4)(0.3 - 0.3)] = 0.298

After 10 rounds (R = 10), the server aggregates block-wise:
x̄₁¹⁰ = (x₁,₁¹⁰ + x₂,₁¹⁰ + x₃,₁¹⁰ + x₄,₁¹⁰)/4

This block-wise aggregation ensures that each agent maintains awareness of others' model states while still respecting their own constraints. For Agent 1, the final model parameters are constrained to ||x₁||₁ ≤ 0.5, while Agent 4's parameters have a looser constraint of ||x₄||₁ ≤ 1.1, reflecting their different hardware capabilities.

## References

- Mohammadjavad Ebrahimi, Daniel Burbano, Farzad Yousefian, "On Performance Guarantees for Federated Learning with Personalized Constraints", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19617

Tags: #distributed-systems #federated-learning #personalization #constraint-optimisation #communication-complexity
