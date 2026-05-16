---
title: "Cross-site scripting adversarial attacks based on deep reinforcement learning: Evaluation and extension study"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2502.19095"
---

## Executive Summary
This paper introduces an XSS Oracle that validates whether adversarial payloads remain functional XSS attacks after mutation, addressing critical validity threats in prior deep reinforcement learning (DRL) approaches to XSS evasion. The authors demonstrate that when these threats are mitigated, their approach achieves over 96% evasion rates against XSS detectors, compared to the reference work's 90% escape rate.

## Why This Matters for Practitioners
If you're responsible for implementing or maintaining XSS detectors in production systems, this research reveals a critical vulnerability: standard DRL-based adversarial attacks can unintentionally break the XSS payload's functionality during mutation, leading to artificially inflated evasion rates. This means your current detectors may appear to be evaded at 90%+ rates in literature, but the actual threat is substantially higher when payloads remain functional. To protect your systems, you must implement semantic validation of adversarial examples before accepting evasion rates as valid metrics, and you should ensure your detector's preprocessing pipeline handles out-of-vocabulary tokens correctly to prevent false negatives from token replacement.

## Problem Statement
Today's XSS detectors resemble a security guard at a nightclub who only checks your ID but doesn't verify you're actually a person (not a robot). The guard (detector) might miss a fake ID (adversarial payload), but if the attacker's ID is so heavily altered it's no longer a valid ID (e.g., replaced with "None" tokens), the guard is mistakenly "evaded" because the input is now invalid rather than genuinely malicious. This paper exposes how prior work treated "invalid" payloads as successful evasions, creating a false sense of security.

## Proposed Approach
The authors introduce an XSS Oracle that validates whether mutated payloads remain functional XSS attacks by comparing DOM structures before and after rendering. This Oracle is integrated into the RL agent's training process to address threats to validity. The RL agent uses a set of 27 mutation rules to incrementally transform XSS payloads while trying to evade detection. The agent receives +10 reward for successful evasion (undetected) and -1 penalty for detection, training the agent to find mutation sequences that preserve attack functionality.

```python
def train_adversarial_agent(detector, oracle, max_steps=50):
    for _ in range(1000):
        payload = random_malicious_payload()
        for step in range(max_steps):
            action = agent.choose_action(state=payload)
            mutated_payload = apply_mutation(payload, action)
            if oracle.validate(mutated_payload) == "Malicious":
                if detector.predict(mutated_payload) == "benign":
                    agent.update_policy(reward=10)
                    break
                else:
                    agent.update_policy(reward=-1)
            else:
                agent.update_policy(reward=-1)
                break
        payload = mutated_payload
```

## Key Technical Contributions
The paper's core innovations extend beyond standard RL-based evasion to address critical validity threats in XSS adversarial research.

1. **XSS Oracle for semantic validation**: The Oracle renders payloads in a template page and compares DOM structures before and after execution. If DOM changes occur (indicating functional XSS), the payload is classified as "Malicious". This prevents the RL agent from using mutations that break the attack's functionality, addressing threat TH1 (lack of validation of actions).

2. **OOV-Rate metric for preprocessing validation**: The authors introduce OOV-Rate (OR), which measures the proportion of "None" tokens (out-of-vocabulary tokens) in the tokenized payload. High OR indicates preprocessing is disrupting the payload's semantics (threat TH2), which the Oracle helps detect and avoid.

3. **Reproducible methodology**: By using publicly available datasets (Mereani and Howe, 2018), the authors ensure their results can be replicated without depending on unshared code or datasets (addressing threat TH3).

4. **Quantitative threat assessment**: They introduce Ruin Rate (RR), which measures the proportion of payloads that become non-malicious after mutation. RR(M) = 1 - (number of malicious payloads still classified as malicious after mutation / total malicious payloads). This metric quantifies how many mutations inadvertently break the XSS attack.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The authors replicated Chen et al.'s approach using the Mereani and Howe dataset (15,000+ payloads) rather than the unshared dataset used in the reference work. They evaluated against three detection models: MLP, LSTM, and CNN. The reference work reported escape rates above 90% against DNN-based detectors, but the authors' approach achieved escape rates above 96% when addressing the identified threats to validity.

The key metric, Ruin Rate (RR), showed that the reference work's approach had RR > 60% for payloads even before agent mutations, meaning over 60% of payloads were no longer functional XSS attacks. The authors' approach reduced RR to near zero by incorporating the XSS Oracle during training, while maintaining comparable evasion rates (less than 2% worse than reference work).

## Related Work
This paper builds on Chen et al.'s 2022 work on RL-based XSS evasion but exposes critical validity threats that were not addressed. It extends previous methods like Fang et al.'s Dueling DQN (escape rate <10%) and Wang et al.'s soft Q-learning (85% escape rate) by introducing a method to verify attack functionality during mutation. The authors also relate their work to adversarial attacks on neural code models (Yang et al., 2022; Jha and Reddy, 2023), highlighting that semantics-preserving mutations are crucial for effective adversarial attacks.

## Limitations
The authors acknowledge their methodology uses a single dataset (Mereani and Howe, 2018), which may not fully represent real-world XSS attacks. They also note that the XSS Oracle itself requires a template page to render payloads, which might not generalise to all web applications. The paper doesn't address potential defences against the Oracle itself, and the RL training process could be computationally intensive for large-scale applications.

## Appendix: Worked Example
Let's walk through the XSS Oracle validation process with a concrete example using a simple XSS payload: `<script>alert("XSS")</script>`.

1. **Original Payload**: `<script>alert("XSS")</script>` (Malicious - DOM changes)
2. **Mutation**: Apply action A15 (Replace "(" and ")" with "‘" → `<script>alert‘XSS’</script>` (original: `alert("XSS")` → `alert‘XSS’`)
3. **Preprocessing**: The tokenization process replaces tokens not in the top 10% vocabulary with "None". The modified payload's token sequence would include `alert‘XSS’`, which likely isn't in the top 10% vocabulary. Without the Oracle, this would become `alert‘None’` due to OOV replacement.
4. **Oracle Validation**: The Oracle renders the mutated payload in a template page:
   - Template DOM: `<div id="content">Original Content</div>`
   - Rendered DOM with mutated payload: `<div id="content"><script>alert‘XSS’</script>Original Content</div>`
   - DOM comparison shows changes (script execution), so the Oracle classifies it as "Malicious" (RR = 0)
5. **Detection**: The detector processes the preprocessed payload (now `alert‘None’`, which is benign) and classifies it as benign (evaded).
6. **Escape Rate Calculation**: This successful evasion is valid because the payload remains functional (verified by Oracle), resulting in a true escape rate of 100% for this payload.

This example shows why the Oracle is essential: without it, the payload would have been classified as benign due to OOV token replacement (making it appear evaded when it's actually broken), inflating the escape rate.

## References

- **Code:** https://github.com/fmereani/Cross-Site-Scripting-XSS
- Samuele Pasini, Gianluca Maragliano, Jinhan Kim, Paolo Tonella, "Cross-site scripting adversarial attacks based on deep reinforcement learning: Evaluation and extension study", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2502.19095

Tags: #web-security #adversarial-machines #xss #reinforcement-learning #vulnerability-testing
