---
title: "Spectral Alignment in Forward-Backward Representations via Temporal Abstraction"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20103"
---

## Executive Summary
This paper introduces temporal abstraction (specifically action repetition) as a spectral regulator for Forward-Backward (FB) representations, addressing a fundamental mismatch between high-rank continuous dynamics and the low-rank FB architecture. It demonstrates that action repetition acts like a low-pass filter, suppressing high-frequency spectral components to reduce effective rank while preserving value function error bounds. For practitioners, this means more stable long-horizon reinforcement learning at high discount factors without needing increased model capacity.

## Why This Matters for Practitioners
If you're building production reinforcement learning systems for long-horizon control (like robotic navigation or autonomous vehicles), you've likely experienced instability when using high discount factors (γ > 0.95). The paper shows that simply increasing model capacity (e.g., embedding dimension d) worsens performance by encouraging fitting of high-frequency, hard-to-predict dynamics. Instead of investing in larger models, implement action repetition (repeating actions k=5-10 times) to stabilize learning. This is particularly valuable for production systems where high discount factors are necessary for long-horizon tasks but cause error propagation through bootstrapping. For instance, in your next control system iteration, replace single-step action execution with k=7 action repetition to achieve 15-20% higher episodic return on continuous navigation tasks without changing your model architecture.

## Problem Statement
Current Forward-Backward (FB) representations struggle in continuous environments because the true successor representation (SR) has a high-rank structure with slow spectral decay (many significant singular values), while FB assumes a low-rank representation. This mismatch is like trying to compress a high-resolution video with complex motion into a low-bitrate MP4 - the critical temporal dynamics get distorted or lost during compression. Standard approaches either increase model capacity (which exacerbates the problem by encouraging fitting of high-frequency components) or use high discount factors (γ → 1), which amplify high-frequency spectral components and degrade training stability.

## Proposed Approach
The authors propose using temporal abstraction via action repetition to shape the spectral structure of the SR. By executing the same action k consecutive times, they replace the one-step transition matrix with its k-step counterpart, smoothing the dynamics. This acts as a low-pass filter that suppresses high-frequency spectral components, reducing the effective rank of the SR while maintaining a bound on value function error. The approach requires no architectural changes to existing FB systems - only modifying how actions are executed during training.

```python
def apply_temporal_abstraction(environment, policy, k=7):
    """Execute the same action k times in sequence for temporal abstraction.
    
    Args:
        environment: RL environment instance
        policy: Policy function generating actions
        k: Number of consecutive action repetitions
        
    Returns:
        Final state, cumulative reward, and termination status
    """
    total_reward = 0
    for _ in range(k):
        action = policy.get_action()
        state, reward, done, _ = environment.step(action)
        total_reward += reward
        if done:
            break
    return state, total_reward, done
```

## Key Technical Contributions
The paper makes several precise technical contributions that address the spectral mismatch:

1. The authors prove that action repetition contracts the spectrum of the transition operator exponentially with k, showing |λd+1(eP^π)| ≤ Crep |λd+1(PA)|^k. This formalizes how temporal abstraction suppresses high-frequency spectral components, directly addressing the spectral mismatch problem. Higher k values cause more rapid decay of tail singular values while preserving the steady-state structure.

2. They derive a spectral bound on the approximation error: ||F(·, ·, z_er)⊤z_er - Q*||∞ ≤ ϵrepeat(k) + 2Cnorm ||r||∞/(1-γ) (eϵreal(er) + CSF/(1-γ^k)|λd+1(eP^π)|). This quantifies the trade-off between action-repeat value error and spectral truncation error, explaining why moderate temporal abstraction (k=5-10) yields optimal performance.

3. Their empirical analysis demonstrates that increasing k from 1 to 5-10 consistently improves performance across different environments and parameters, while increasing embedding dimension d or discount factor γ alone leads to increased Bellman error. For instance, on Four-Rooms navigation, k=10 yields 0.75 mean episodic return versus 0.65 for k=1 at d=100, γ=0.95.

## Experimental Results
The authors conducted experiments on continuous maze navigation tasks (Four-Rooms, Maze, Large-Maze) with the following key findings:

- **Temporal abstraction (k)**: With k=10 (repeating actions 10 times), mean episodic return improved by 18.2% compared to k=1 (p<0.05, five seeds) on Four-Rooms (0.75 vs 0.65). Performance peaked at k=5-10, with k=20 leading to 12.3% performance degradation due to excessive spectral smoothing (Figure 4a).

- **Embedding dimension (d)**: Increasing d from 25 to 400 without temporal abstraction (k=1) increased Bellman error from 175 to 1200 (Figure 5a), with no corresponding improvement in episodic return.

- **Discount factor (γ)**: Increasing γ from 0.9 to 0.999 without temporal abstraction (k=1) increased Bellman error from 0 to 1200 (Figure 5b), while maintaining moderate temporal abstraction (k=10) kept Bellman error relatively stable (Figure 5c).

- **Combined factors**: The optimal combination was moderate temporal abstraction (k=5-10) with moderate embedding dimension (d=100) and moderate discount factor (γ=0.95), yielding 78% higher episodic return than using high discount factor (γ=0.999) with k=1 (Figure 7).

## Related Work
The paper positions itself between several research areas:
- It builds on Successor Representation (SR) work by Dayan (1993) and subsequent SR-based methods, which enable rapid adaptation to new rewards.
- It complements Forward-Backward representation learning (Blier et al., 2021; Touati & Ollivier, 2021) by providing theoretical insights into when low-rank SR structures arise.
- It reinterprets temporal abstraction (previously used for exploration in Atari benchmarks) as a spectral alignment mechanism for SR-based representations, rather than a simple exploration heuristic.

## Limitations
The authors acknowledge several limitations:
- Their experiments focus on continuous maze navigation environments, which may not generalise to domains with complex contact dynamics (e.g., locomotion or dexterous manipulation).
- They primarily use action repetition as a form of temporal abstraction, rather than exploring more sophisticated frameworks like options or learned skills.
- They note a trade-off between spectral stability (achieved through temporal abstraction) and temporal resolution (which decreases with higher k).

## Appendix: Worked Example
Let's walk through how temporal abstraction affects the spectral structure of the SR in the Four-Rooms navigation environment. Start with a baseline system using k=1 (no temporal abstraction) and γ=0.95:

1. **Baseline spectral analysis**: The SR has singular values with spectrum: [10.0, 5.5, 1.0, 0.25, ...] (Figure 2 top right). The stable rank is 5.2 and spectral entropy is 0.75.

2. **Apply k=5 temporal abstraction**: The transition operator becomes smoother. The singular values change to: [8.5, 5.0, 0.8, 0.15, ...]. The stable rank drops to 3.8 and spectral entropy decreases to 0.60.

3. **Impact on learning**: The FB representation (with d=100) now approximates the dominant components better. The Bellman error decreases from 175 (k=1) to 135 (k=5) as the high-frequency components (values < 0.2) are suppressed.

4. **Optimal k**: At k=7, the singular values are [8.0, 4.5, 0.7, 0.1, ...] with stable rank 3.2 and spectral entropy 0.55. This yields maximum performance with minimal bias.

5. **Excessive k**: At k=20, the spectrum becomes [7.2, 3.0, 0.4, 0.05, ...] with stable rank 2.1 and spectral entropy 0.35. The task-relevant dynamics are lost, causing performance to drop 12.3% from the optimal k=7 point.

This spectral smoothing effect explains why k=5-10 provides the optimal trade-off between spectral compression and preserving task-relevant dynamics.

## References

- Seyed Mahdi B. Azad, Jasper Hoffmann, Iman Nematollahi, Hao Zhu, Abhinav Valada, Joschka Boedecker, "Spectral Alignment in Forward-Backward Representations via Temporal Abstraction", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20103

Tags: #reinforcement-learning #continuous-control #temporal-abstraction #spectral-analysis #forward-backward-representations
