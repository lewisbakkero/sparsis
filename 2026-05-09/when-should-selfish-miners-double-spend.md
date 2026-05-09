---
title: "When Should Selfish Miners Double-Spend?"
venue: "arXiv cs.DC"
paper_url: "https://arxiv.org/abs/2501.03227"
---

## Executive Summary
This paper presents a rigorous analysis of combining double-spending attacks with selfish mining strategies in blockchain systems. It introduces L-stubborn mining - a strategy where the adversary mines selfishly but maintains a private chain of length L - and demonstrates that double-spending becomes cost-free when L exceeds the k-confirmation rule. For Bitcoin with k=6, transactions are vulnerable to double-spending with just 40.9% of the network's hash power, even without network influence.

## Why This Matters for Practitioners
If you're operating a blockchain-based payment system that relies on the Bitcoin standard (6 confirmations), you should be aware that transactions are vulnerable to double-spending attacks with only 40.9% of the network's hash power, even without network influence. This means that for high-value transactions, you should require more than 6 confirmations or implement additional checks for transactions involving large amounts. The paper also shows that simply increasing the confirmation count (k) is not sufficient to prevent double-spending attacks when the adversary's hash power exceeds 40.9%.

## Problem Statement
Today's blockchain systems assume that double-spending attacks are prohibitively expensive due to orphan blocks, while selfish mining literature typically ignores the chance of double-spending at no cost during each attack cycle. This paper identifies a critical gap: if an adversary mines selfishly but remains stubborn enough (reaching a private chain length L > k), they can double-spend without incurring revenue losses. This is like having a secret backdoor in a building's security system that only opens when the security code reaches a certain length - you can bypass the main security check entirely by following a specific sequence of steps.

## Proposed Approach
The authors introduce L-stubborn mining, a strategy where an adversary mines privately until their private chain reaches length L, then releases it when the honest chain is one block behind. This allows them to double-spend at no cost when L > k (the k-confirmation rule). The approach builds upon the standard selfish mining model but adds a new dimension of "stubbornness" that directly connects to the double-spending risk. The core idea is that by tracking both the length difference and absolute length of their private chain, the adversary can time their double-spending attempts to coincide with the natural confirmation process.

```python
def l_stubborn_mining(adversary_hash_rate, network_influence, k):
    # Calculate optimal stubbornness level L*
    L_opt = calculate_optimal_stubbornness(adversary_hash_rate, network_influence)
    
    # Check if double-spending is cost-free
    if L_opt > k:
        return f"Double-spending possible at no cost with L={L_opt} > k={k}"
    
    # For L_opt <= k, double-spending requires additional revenue cost
    else:
        return f"Double-spending requires additional revenue cost (L={L_opt} <= k={k})"
```

## Key Technical Contributions
The paper makes several key technical contributions:

1. **L-stubborn mining strategy**: The core innovation is a generalisation of existing mining strategies where the adversary tracks both the length difference and absolute length of their private chain. This allows them to determine whether they can double-spend with no revenue loss: when the length of their private chain (L) exceeds the k-confirmation threshold (L > k), double-spending becomes cost-free. This is fundamentally different from prior work that only tracked the length difference.

2. **Optimal stubbornness calculation**: The authors provide a method to calculate the optimal stubbornness level L* that maximises revenue, rather than just finding the maximum stubbornness that's still profitable. This simple formula (derived from Bertrand's ballot problem and Catalan numbers) replaces complex MDP simulations required by previous work.

3. **Connection to k-confirmation rule**: The paper establishes a precise mathematical relationship between the adversary's stubbornness (L) and the transaction confirmation safety level (k), showing that when L > k, double-spending is effectively free. For Bitcoin (k=6), this means 40.9% hash power is sufficient for double-spending without network influence.

4. **S-stealth mining modification**: The authors introduce a modified strategy that conceals the attack by delaying block publication, increasing double-spending probability at the cost of reduced mining revenue. This is achieved by not prematurely publishing competing blocks at the honest chain's tip, making the attack less detectable.

## Experimental Results
The paper provides analytical results rather than experimental data:

- For Bitcoin with k=6 confirmations, double-spending becomes cost-free for adversaries with α > 0.409 (40.9% hash power), regardless of network influence (γ = 0).
- The authors calculate that for α = 0.41 (just above the threshold), the optimal L* is 7, which exceeds k=6.
- The paper shows that the L-stubborn strategy performs close to the optimal solution of previous MDP-based approaches (ρL* ≈ ρ_ǫ-optimal).
- The authors note that "for Bitcoin which follows k-conﬁrmation rule with k = 6, even if γ = 0, every transaction is at risk of double-spend for α > 0.409."

## Related Work
This paper positions itself as filling a critical gap in the existing literature by combining two well-studied attack vectors: double-spending and selfish mining. It builds on the seminal work of Eyal and Sirer [4] on selfish mining and Rosenfeld [3] on double-spending attacks, but goes beyond by showing that double-spending can be combined with selfish mining at no cost when L > k.

The paper references prior work on MDP-based models for selfish mining [5]-[9], but argues their approach can provide simple, closed-form solutions instead of requiring complex simulations. Unlike [7], [8], the authors provide a simple formula to calculate revenue ratios, double-spend risks, and optimal attacks.

## Limitations
The paper acknowledges several limitations:

1. It assumes a slow block arrival rate (as in Bitcoin), which might not be applicable to other blockchain systems with faster block times.

2. It assumes no block mining during propagation delays, which simplifies the model but might not reflect all real-world scenarios.

3. The model focuses on the heuristic of k-confirmation rule, which is common in the literature, but might not capture all security considerations for different applications.

4. The paper doesn't address how to detect or defend against such attacks in production systems, leaving this for future work.

## Appendix: Worked Example
Consider a Bitcoin-like system with k=6 confirmations. The adversary controls 41% of the hash power (α = 0.41 > 0.409), with no network influence (γ = 0).

1. The adversary starts mining privately, building their private chain (A-chain) while the honest miners build the public chain (H-chain).
2. The adversary continues mining privately until their A-chain reaches length L = 7 (which is greater than k = 6).
3. At this point, the adversary has a private chain 7 blocks deep, while the honest chain is only 6 blocks deep.
4. The adversary releases their private chain, causing a fork.
5. Because the adversary's private chain is 7 blocks deep (L = 7 > k = 6), they can double-spend transactions that were confirmed 6 blocks ago (i.e., 6 blocks deep in the honest chain) at no cost to themselves.
6. The double-spending attempt succeeds because the adversary's chain is longer than the k-confirmation threshold, so the protocol will accept the double-spent transaction as valid.

This example shows why transactions with only 6 confirmations are vulnerable to double-spending attacks by adversaries with just 41% of the hash power. Transactions confirmed with fewer than 6 blocks (e.g., 5 blocks) are even more vulnerable, while those with more than 6 blocks (e.g., 7 blocks) are safer.

## References

- Mustafa Doger, Sennur Ulukus, "When Should Selfish Miners Double-Spend?", arXiv cs.DC, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2501.03227

Tags: #blockchain-security #consensus-algorithms #double-spending #mining-strategies #selfish-mining
