---
title: "What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19880"
---

## Executive Summary
SCRL (Selective-Complementary Reinforcement Learning) addresses a critical vulnerability in test-time reinforcement learning (TTRL) where majority-voting consensus can amplify label noise when answer distributions are dispersed. This paper introduces a robust framework that combines selective positive pseudo-labelling with entropy-gated negative pseudo-labelling, preventing models from reinforcing incorrect solutions while maintaining exploration capacity. For practitioners building LLM-based reasoning systems, this means avoiding the trap of premature convergence on spurious solutions during unsupervised fine-tuning.

## Why This Matters for Practitioners
If you're running an LLM reasoning system on unlabeled production data where label quality is uncertain, such as a math problem-solving service or code generation tool, existing TTRL implementations can inadvertently reinforce incorrect solutions when consensus is weak. This paper shows that simply using majority voting for test-time reinforcement learning can lead to models converging on wrong answers, especially when rollout budgets are constrained. The key action for engineers: implement selective positive labelling with strict consensus thresholds (top answer proportion ≥ 0.375 and margin over second-answer > 0.125) and add negative supervision via entropy-gated filtering to prune implausible trajectories without discarding potentially correct rare solutions. This prevents the need for expensive manual labelling while ensuring robust generalisation on challenging problems.

## Problem Statement
Imagine a group of experts trying to solve a complex puzzle where no one knows the answer. If they vote to choose the most common solution, they might all agree on an incorrect answer simply because it's the most frequent, not the right one. This is precisely what happens in test-time reinforcement learning (TTRL) when answer distributions are dispersed: the model reinforces the wrong answer as consensus, leading to premature convergence on incorrect solutions. The problem isn't just the wrong answer, it's that the model becomes locked into a suboptimal solution space, unable to explore better alternatives even when more rollouts are available.

## Proposed Approach
SCRL introduces a robust test-time reinforcement learning framework that overcomes the limitations of relying exclusively on positive pseudo-labelling. It combines three key components: Selective Positive Pseudo-Labelling to filter unreliable majorities, Entropy-Gated Negative Pseudo-Labelling to prune incorrect trajectories, and Dynamic Reward Shaping to calibrate reinforcement magnitude based on consensus strength. This creates a complementary learning mechanism that prevents noise amplification while maintaining model exploration.

```python
def scrl_pseudo_labeling(answer_distribution, thresholds):
    # Selective Positive Pseudo-Labelling
    top_answer = max(answer_distribution, key=lambda x: x['count'])
    second_answer = sorted(answer_distribution, key=lambda x: x['count'], reverse=True)[1]
    if (top_answer['proportion'] >= thresholds['pos'] and
        (top_answer['proportion'] - second_answer['proportion']) > thresholds['margin']):
        positive_label = top_answer['answer']
    else:
        positive_label = None
    
    # Entropy-Gated Negative Pseudo-Labelling
    negative_labels = []
    avg_entropy = compute_average_entropy(answer_distribution)
    for answer in answer_distribution:
        if (answer['proportion'] < thresholds['neg'] and
            answer['entropy'] >= avg_entropy):
            negative_labels.append(answer['answer'])
    
    return positive_label, negative_labels
```

## Key Technical Contributions
SCRL's innovation lies in its precise mechanisms for implementing selective positive labelling and negative supervision in test-time reinforcement learning.

1. **Strict consensus thresholds prevent reinforcing weak consensus**: SCRL enforces two conditions for positive labelling: the top answer must have proportion ≥ 0.375 (τpos) and must have a margin over the second-ranked answer > 0.125 (τmarg). This differs from previous TTRL methods that used simple majority voting without these thresholds, which could reinforce incorrect answers when consensus is weak.

2. **Entropy-gated negative labelling identifies implausible trajectories without discarding rare valid solutions**: SCRL identifies negative labels by finding answers with both proportion < 0.125 (τneg) and generation uncertainty ≥ average uncertainty (Ĥj ≥ Ĥ). This differs from previous approaches that either ignored negative signals or used frequency alone, which could penalise rare but correct answers.

3. **Dynamic reward shaping combines positive and negative signals with uncertainty penalty**: SCRL scales reinforcement magnitude based on consensus strength and incorporates uncertainty penalties through the reward function: Ri = p(ai)I[ai = y+] + (p(ai) - τneg)I[ai ∈ N-] - λH(Ĥ(ai) - Ĥ). This creates a risk-averse strategy that reinforces credible positive signals while penalising uncertain trajectories.

## Experimental Results
SCRL consistently outperforms baselines across multiple reasoning benchmarks, particularly on challenging problems where weak consensus is common. On Qwen2.5-3B:

- AIME25 (most challenging): SCRL achieves 8.4% pass@1 (vs TTRL's 2.6%), a 5.8% absolute improvement
- Minerva: SCRL achieves 41.6% accuracy (vs TTRL's 14.5%), a 27.1% improvement
- AMC: SCRL achieves 41.5% pass@1 (vs TTRL's 39.4%)
- MATH-500: SCRL achieves 68.2% pass@1 (vs TTRL's 66.9%)

On Qwen2.5-Math-7B, SCRL achieves 26.9% on AIME25 (vs TTRL's 16.8%), a 10.1% improvement. On Llama-3.1-8B-Instruct, SCRL achieves 29.0% average accuracy across benchmarks, outperforming all baselines including RESTRAIN (28.4%) and ETMR (26.2%). These improvements are consistent across different model sizes and architectures, validating the approach's robustness.

## Related Work
SCRL builds on TTRL methods like TTRL (Zuo et al., 2025), COMPASS (Tang et al., 2025), and RESTRAIN (Yu et al., 2025b) but addresses a fundamental limitation these approaches share: exclusive reliance on positive pseudo-labelling. Previous work like ETMR (Liu et al., 2025a) focuses on improving exploration but doesn't address the noise amplification problem when consensus is weak. SCRL's introduction of entropy-gated negative pseudo-labelling represents the first negative supervision mechanism in TTRL, making it distinct from prior approaches that only used positive signals.

## Limitations
The authors don't explicitly state limitations, but the paper tests SCRL primarily on mathematical and coding reasoning tasks. It's unclear how the approach generalizes to other reasoning domains like commonsense reasoning or dialogue understanding. The paper also doesn't address computational overhead from entropy calculations, though they use standard entropy calculations over token distributions. For production systems, this might require additional inference time that could be significant at scale.

## Appendix: Worked Example
Consider a math problem where the model generates 16 responses (N=16) with the following answer distribution:

| Answer | Count | Proportion | Entropy |
|--------|-------|------------|---------|
| A      | 4     | 0.25       | 2.3     |
| B      | 3     | 0.1875     | 1.8     |
| C      | 3     | 0.1875     | 1.9     |
| D      | 3     | 0.1875     | 2.1     |
| E      | 3     | 0.1875     | 2.2     |

The top answer "A" has proportion 0.25 < τpos (0.375), so selective positive labelling abstains from positive labelling. For negative labelling, compute average entropy: Ĥ = (2.3 + 1.8 + 1.9 + 2.1 + 2.2)/5 = 2.06. Answers with proportion < 0.125 (none in this case) and entropy ≥ 2.06 are considered negative labels. Here, "A" has entropy 2.3 ≥ 2.06, but its proportion (0.25) is not < 0.125, so no negative labels are identified. The reward for "A" becomes p(A)I[ai = y+] + (p(A) - τneg)I[ai ∈ N-] - λH(Ĥ(A) - Ĥ) = 0.25×0 + (0.25 - 0.125)×0 - 0.1×(2.3 - 2.06) = -0.024. This negative reward indicates low confidence in the answer, guiding the policy away from this trajectory without prematurely discarding it.

## References

- Dong Yan, Jian Liang, Yanbo Wang, Shuo Lu, Ran He, Tieniu Tan, "What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19880

Tags: #machine-learning #reasoning #reinforcement-learning #test-time-learning #negative-supervision
