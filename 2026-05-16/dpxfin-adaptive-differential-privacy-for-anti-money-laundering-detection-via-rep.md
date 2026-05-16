---
title: "DPxFin: Adaptive Differential Privacy for Anti-Money Laundering Detection via Reputation-Weighted Federated Learning"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19314"
---

## Executive Summary
DPxFin introduces a reputation-guided differential privacy framework for federated learning in anti-money laundering (AML) detection, dynamically adjusting noise levels based on client contribution quality. Unlike traditional approaches that apply uniform privacy protection, it enhances privacy-utility trade-offs by allocating lower noise to high-reputation clients (those with model updates closely aligned with the global model) and higher noise to low-reputation clients. This achieves up to 3.8% higher accuracy in non-IID settings while withstanding tabular data leakage attacks.

## Why This Matters for Practitioners
If you're building a production AML system using federated learning across financial institutions, DPxFin solves a critical pain point: the trade-off between privacy protection and model accuracy. Traditional federated learning with fixed differential privacy applies the same noise level to all institutions, degrading accuracy (up to 3.8% worse in non-IID scenarios) despite varying data quality. DPxFin lets you automatically allocate finer-grained privacy protection, reducing noise by 80% for high-reputation institutions while applying stronger protection to those with lower trust scores. This means you can deploy more accurate AML models (with 92.9% accuracy in non-IID settings) without compromising data privacy, and you don't need to manually adjust privacy budgets for different client types. For engineers, this translates to a production-ready system that's both more effective at detecting fraud and legally compliant with privacy regulations like GDPR when handling sensitive financial data.

## Problem Statement
Imagine trying to build a collaborative fraud detection system across multiple banks, where each bank has different transaction patterns and privacy policies. The problem is that if you apply the same level of privacy protection to all banks (like adding uniform noise to every model update), you're treating a bank with highly accurate transaction data the same as one with noisy, unreliable data. It's like applying the same security clearance to all employees in a bank: you're either restricting the trusted fraud analysts who consistently identify real threats (by adding unnecessary noise to their updates) or risking data leaks by not protecting the less reliable institutions sufficiently. Traditional federated learning for AML creates this exact imbalance, making it difficult to balance privacy protection with model accuracy in heterogeneous financial environments.

## Proposed Approach
DPxFin integrates reputation-based differential privacy into a standard federated learning architecture. Each financial institution (client) trains a local model on their private transaction data, then sends a privacy-protected model update to a central server. The server calculates each client's reputation based on the Euclidean distance between their local update and a temporary global model. Using this reputation score, the server dynamically adjusts the noise level for the next training round. High-reputation clients (those whose updates closely align with the global model) receive lower noise, while low-reputation clients receive higher noise. The server then aggregates model updates using reputation-weighted averaging, prioritising contributions from reliable institutions.

```python
def dp_xfin(global_model, clients, rounds=100):
    # Initialize reputation factors for all clients
    reputation_factors = {client: 1.0 for client in clients}
    
    for round in range(rounds):
        # Select a subset of clients and share the global model
        selected_clients = select_clients(clients)
        
        # Client-side training with adaptive noise
        updates = []
        for client in selected_clients:
            if round == 0:
                noise_multiplier = 1.0  # Static noise in first round
            else:
                noise_multiplier = reputation_factors[client]
            
            # Train locally with noise and send model update
            update = client.train(global_model, noise_multiplier)
            updates.append((client, update))
        
        # Server-side reputation calculation
        temporary_global = aggregate(updates)
        distances = {client: euclidean(update, temporary_global) 
                     for client, update in updates}
        max_distance = max(distances.values())
        
        # Calculate and normalize reputation scores
        reputation_scores = {client: 1 - (d / max_distance) 
                            for client, d in distances.items()}
        total_score = sum(reputation_scores.values())
        normalized_scores = {client: score / total_score 
                            for client, score in reputation_scores.items()}
        
        # Update reputation factors based on normalized scores
        sorted_scores = sorted(normalized_scores.values())
        p50 = sorted_scores[len(sorted_scores) // 2]
        p70 = sorted_scores[int(0.7 * len(sorted_scores))]
        
        for client in clients:
            if normalized_scores[client] >= p70:
                reputation_factors[client] = 0.2
            elif normalized_scores[client] >= p50:
                reputation_factors[client] = 0.5
            else:
                reputation_factors[client] = 1.0
        
        # Server-side weighted aggregation
        global_model = aggregate_weighted(updates, normalized_scores)
    
    return global_model, reputation_factors
```

## Key Technical Contributions
DPxFin introduces several mechanisms that make it more effective than prior approaches:

1. **Euclidean distance-based reputation calculation**: Unlike fixed-noise approaches, it computes client reputation by measuring the Euclidean distance between local model updates and a temporary global model. This provides a continuous, data-driven measure of contribution quality rather than relying on manual reputation assignments. The calculation uses the actual model parameters, not just metadata, ensuring that reputation scores reflect genuine alignment with the global objective.

2. **Tiered reputation-based noise scaling**: The system applies a tiered approach to noise scaling using percentile thresholds (P70 and P50), rather than a continuous scaling function. This creates a clear, easily implementable mechanism where the top 30% of contributors (reputation ≥ P70) receive only 20% of the baseline noise, the middle 20% (P50 ≤ reputation < P70) receive 50%, and the bottom 50% receive 100% noise. This eliminates the need for manual tuning of privacy budgets.

3. **Dynamic reputation adjustment every training round**: Reputation scores are recalculated after every aggregation step, allowing the system to continuously adapt to changing client contributions. This prevents reputation scores from becoming stale, which would happen in static approaches, and helps the system respond to clients that might improve or degrade their contribution quality over time.

## Experimental Results
The authors evaluated DPxFin against two baselines, FedAvg (standard federated learning without privacy) and DP-FedAvg (federated training with fixed differential privacy, ε=1.0), using the IBM Synthetic Financial Data Money Laundering dataset (HI_Small Transaction, ~5 million rows with ~5,000 positive examples of money laundering transactions). All experiments used MLP models with SMOTE-oversampled data.

In non-IID settings (more realistic for heterogeneous financial institutions), DPxFin achieved 92.9% accuracy compared to 89.1% for DP-FedAvg, a 3.8% absolute improvement. The accuracy advantage was consistent across communication rounds, with DPxFin demonstrating faster convergence. The TabLeak attack results confirmed DPxFin's robustness: while the baseline FedAvg achieved 92.9% attack accuracy (indicating high vulnerability to data leakage), DPxFin reduced this to 58.5%, demonstrating significantly stronger protection against privacy attacks.

## Related Work
DPxFin positions itself as an improvement over existing privacy-preserving federated learning approaches for AML detection. Unlike [8] which used homomorphic encryption for privacy but lacked practical details on privacy budgets, DPxFin provides formal differential privacy guarantees with adaptive noise. It extends [16]'s work on privacy-preserving federated learning for AML by introducing dynamic noise scaling based on client reputation rather than fixed noise. DPxFin also addresses limitations in [13]'s graph-based approach, which didn't incorporate privacy protection mechanisms. While previous work focused on accuracy or privacy in isolation, DPxFin simultaneously optimises both through its adaptive noise scaling mechanism.

## Limitations
The paper focuses exclusively on tabular financial data without extending to more complex data types like images or text. The authors don't report training latency or computational overhead compared to baseline methods, though accuracy results suggest these would be acceptable for production environments. The paper doesn't explicitly address how the system would respond to clients deliberately manipulating their reputation scores (though continuous reputation adjustment should help mitigate this). The validation was limited to a single dataset (IBM Synthetic Financial Data Money Laundering) without testing across multiple diverse financial datasets.

## Appendix: Worked Example
Let's walk through a concrete example of how DPxFin calculates reputation and noise scaling in a single training round. Imagine three financial institutions, Bank A (a major bank), Bank B (a mid-sized bank), and Bank C (a small regional bank), each with different transaction patterns.

After the first training round, the temporary global model is built from all initial client updates. The Euclidean distances between each bank's model update and this temporary global model are:
- Bank A: 0.15
- Bank B: 0.30
- Bank C: 0.45

The maximum distance is 0.45, so the consistency score (1 - distance/max distance) for each bank is:
- Bank A: 1 - 0.15/0.45 = 0.67
- Bank B: 1 - 0.30/0.45 = 0.33
- Bank C: 1 - 0.45/0.45 = 0.0

These scores are normalized across all clients:
- Bank A: 0.67 / (0.67 + 0.33 + 0.0) = 0.67
- Bank B: 0.33 / (0.67 + 0.33 + 0.0) = 0.33
- Bank C: 0.0 / (0.67 + 0.33 + 0.0) = 0.0

The 70th percentile (P70) and 50th percentile (P50) of normalized scores are calculated:
- P70: 0.67 (top 30%)
- P50: 0.33 (top 50%)

The reputation factor (noise multiplier) is assigned as:
- Bank A: 0.2 (reputation ≥ P70)
- Bank B: 0.5 (P50 ≤ reputation < P70)
- Bank C: 1.0 (reputation < P50)

For the next training round, Bank A adds only 20% of the default noise level to its model updates, Bank B adds 50%, and Bank C adds the full 100% noise level. This ensures Bank A's high-quality contributions have greater impact on the global model, while Bank C's potentially unreliable contributions are sufficiently protected against privacy leakage.

## References

- Renuga Kanagavelu, Manjil Nepal, Ning Peiyan, Cai Kangning, Xu Jiming, Fei Gao, Yong Liu, Goh Siow Mong Rick, Qingsong Wei, "DPxFin: Adaptive Differential Privacy for Anti-Money Laundering Detection via Reputation-Weighted Federated Learning", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19314

Tags: #security #aml #federated-learning #differential-privacy #reputation-systems
