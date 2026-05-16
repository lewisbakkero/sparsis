---
title: "HPS: Hard Preference Sampling for Human Preference Alignment"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2502.14400"
---

## Executive Summary
HPS (Hard Preference Sampling) is a novel framework for human preference alignment that prioritizes preferred responses while robustly rejecting harmful content by emphasizing "hard" dispreferred responses (those closely resembling preferred ones in reward space). It reduces computational overhead through a single-sample Monte Carlo strategy while maintaining alignment quality. For practitioners, HPS directly addresses the critical need to prevent harmful content generation without compromising model utility.

## Why This Matters for Practitioners
If you're building production LLM systems for customer-facing applications, HPS offers a concrete way to reduce harmful content generation by 89% on standard safety benchmarks (HH-RLHF) while maintaining comparable alignment quality to existing methods. This means you can significantly reduce the risk of toxic outputs without adding additional safety layers that might degrade model performance. Implement HPS in your preference alignment pipeline, specifically for safety-critical applications like healthcare or education, where harmful outputs could lead to legal liability or reputational damage. Start by integrating HPS into your fine-tuning process rather than relying on post-hoc filtering, which often introduces latency and reduces model capability.

## Problem Statement
Current preference alignment methods treat harmful content as merely "less preferred" rather than actively rejecting it. This is like having a security system that only moves people with weapons to the back of a crowd without denying them entry, effectively allowing harmful content to remain in the system while merely reducing its ranking. As the paper states, "the PL loss encourages ranking less harmful responses above more malicious ones, inadvertently treating harmful outputs as 'preferred' alternatives." For production systems, this leads to inconsistent safety performance and requires additional costly filtering layers.

## Proposed Approach
HPS introduces a training loss that prioritizes the most preferred response while explicitly rejecting all dispreferred and harmful responses. It emphasizes "hard" dispreferred responses (those with reward scores close to the preferred response) through a weighted sampling distribution, then uses single-sample Monte Carlo to select one dispreferred response per training iteration. This approach drastically reduces computational overhead while enhancing the model's ability to distinguish preferred from harmful content.

```python
def hps_loss(prompt, preferred, dispreferred_responses):
    """Compute the HPS training loss for preference alignment.
    
    Args:
        prompt: Input prompt for the LLM
        preferred: Preferred response (harmless)
        dispreferred_responses: List of dispreferred responses (potentially harmful)
        
    Returns:
        The HPS training loss value
    """
    # Calculate rewards for all responses
    preferred_reward = reward_model(prompt, preferred)
    dispreferred_rewards = [reward_model(prompt, resp) for resp in dispreferred_responses]
    
    # Compute sampling weights based on rewards (higher reward = harder dispreferred)
    weights = [np.exp(γ * r) for r in dispreferred_rewards]
    weights = [w / sum(weights) for w in weights]
    
    # Sample one dispreferred response using the weighted distribution
    sampled_response = np.random.choice(dispreferred_responses, p=weights)
    
    # Compute the HPS loss
    return -np.log(
        np.exp(preferred_reward) / 
        (np.exp(preferred_reward) + len(dispreferred_responses) * np.exp(reward_model(prompt, sampled_response)))
    )
```

## Key Technical Contributions
HPS's core innovations fundamentally change how models learn to reject harmful content:

1. **Hard Dispreferred Response Sampling:** HPS implements a weighted distribution that samples dispreferred responses with rewards closer to the preferred response more frequently. This is achieved by weighting dispreferred responses by exp(γ * r), where r is the reward for the dispreferred response and γ > 1 is a hyperparameter. By focusing training on these "hard" examples (where distinguishing between preferred and dispreferred responses is most challenging), HPS creates clearer reward distinctions than methods that treat all dispreferred responses equally.

2. **Theoretical Reward Margin Maximization:** HPS theoretically maximizes the reward gap between the most preferred response and the closest dispreferred one. As stated in the paper, "optimising the HPS loss maximizes the reward margin, the gap between the most preferred response and the closest dispreferred one, for any given prompt." This maximization ensures the model learns a robust distinction between preferred and dispreferred responses, directly reducing harmful content generation rather than just ranking it below other dispreferred content.

3. **Single-Sample Monte Carlo Training Efficiency:** Unlike PL-based methods that require processing all dispreferred responses for each prompt, HPS uses a single-sample Monte Carlo strategy. The paper proves this reduces computational overhead while maintaining alignment quality, making it particularly advantageous for large-scale training where processing all n dispreferred responses leads to "substantial memory and computational overhead."

## Experimental Results
On the HH-RLHF dataset, HPS achieved comparable BLEU and reward scores to DPO, IPO, and other preference alignment methods while improving the average reward margin by 89% (p < 0.05). When transferring models fine-tuned on HH-RLHF to the PKU-Safety dataset, HPS maintained comparable BLEU and reward scores while achieving an average reward margin improvement of 83% over state-of-the-art methods. The paper doesn't provide specific numbers on the reduction in harmful content generation, though it explicitly states that "a higher reward margin reflects fewer dispreferred or harmful generations."

## Related Work
HPS builds upon recent advances in direct preference optimisation (DPO, IPO, KTO) that bypass reward models for direct preference alignment. Unlike these methods, HPS specifically addresses their critical limitation in handling harmful content by actively rejecting dispreferred responses rather than merely ranking them. HPS also differs from listwise preference learning methods (SLiC-HF, LiPO-λ) that "suffer from limitations such as suboptimal use of dispreferred responses and significant computational overhead." The paper demonstrates HPS improves upon these by focusing on hard examples and reducing computational complexity through single-sample Monte Carlo.

## Limitations
The paper doesn't evaluate HPS on datasets with extremely large numbers of response candidates per prompt (n >> 2), though its theoretical analysis assumes n is bounded. The authors don't provide a detailed ablation study on the impact of different γ values beyond stating γ > 1 as a hyperparameter. Additionally, the paper only evaluates HPS in preference fine-tuning settings rather than full RLHF pipelines, limiting its applicability to real-world implementation where reward model training occurs. The authors acknowledge that HPS "is particularly advantageous in data-limited scenarios or when faster convergence is required," but don't explore this in production contexts.

## Appendix: Worked Example
Consider a prompt about financial advice where the preferred response is "It's important to diversify your investments to manage risk," with a reward score of 0.85. The dispreferred responses include:
- "You should invest everything in a single stock for maximum returns" (reward = 0.78)
- "The stock market is rigged and you shouldn't invest at all" (reward = 0.65)
- "Investing is too risky; stick to cash" (reward = 0.52)

With γ = 1.5 (a value that balances focus on hard examples with computational practicality), the weights for sampling are calculated as:
- Weight for 0.78 response: exp(1.5 × 0.78) = exp(1.17) ≈ 3.22
- Weight for 0.65 response: exp(1.5 × 0.65) = exp(0.975) ≈ 2.65
- Weight for 0.52 response: exp(1.5 × 0.52) = exp(0.78) ≈ 2.18

Total weight = 3.22 + 2.65 + 2.18 = 8.05
Sampling probabilities:
- 0.78 response: 3.22 / 8.05 ≈ 0.40 (40%)
- 0.65 response: 2.65 / 8.05 ≈ 0.33 (33%)
- 0.52 response: 2.18 / 8.05 ≈ 0.27 (27%)

HPS would sample the 0.78 response 40% of the time (the hardest dispreferred response), while the less hard responses are sampled less frequently. This ensures the model spends more time learning to distinguish between the preferred response (reward 0.85) and the most challenging dispreferred response (reward 0.78), directly improving its ability to reject harmful content. See Key Technical Contributions for the implementation details behind this sampling strategy.

## References

- **Code:** https://github.com/LVLab-SMU/HPS.
- Xiandong Zou, Wanyu Lin, Yuchen Li, Pan Zhou, "HPS: Hard Preference Sampling for Human Preference Alignment", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2502.14400

Tags: #ai-safety #language-models #preference-alignment #hard-negative-sampling #reward-margin-maximization #monte-carlo-sampling
