---
title: "Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation"
category: "AI Applications"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36964"
---

## Executive Summary
STAB (Sharpness-aware Transferable Adversarial Backdoor) addresses the "transferability gap" in code model poisoning. Traditional attacks fail when the victim's training distribution differs from the attacker's surrogate data. STAB overcomes this by:
1.  **Sharpness-Aware Minimization (SAM)**: Training a surrogate model to find flat loss minima, capturing universal features that generalize across models.
2.  **Differentiable Trigger Optimization**: Using Gumbel-Softmax to optimize discrete identifier replacements.
3.  **Distributional Stealth**: Utilizing Maximum Mean Discrepancy (MMD) to ensure poisoned code remains statistically indistinguishable from benign code.

## Why This Matters for Practitioners
If you're fine-tuning Large Language Models for Code (CodeLLMs) like CodeLlama or StarCoder on internal or curated datasets, you are still vulnerable. STAB demonstrates that a backdoor injected into a surrogate model (e.g., trained on Py150) transfers effectively to a victim model trained on entirely different data (e.g., CodeSearchNet). 

**Critical Insight:** Traditional defenses like **KillBadCode** or **ONION**—which look for "dead code" or outliers—are bypassed because STAB uses contextually relevant identifier renames. 
* **Action:** Security teams should move beyond syntax-based filtering and implement **Representation Analysis** (e.g., Spectral Signature or Activation Clustering) to detect clusters in the latent space, as STAB’s strength lies in its functional correctness.

## Problem Statement
The "Backdoor Transferability" problem:
* **Static Attacks**: Use fixed triggers (e.g., `_backdoor_()`). Highly transferable but easily caught by static analysis.
* **Dynamic Attacks**: Tailor triggers to specific code contexts. Stealthy, but overfit to the attacker's local surrogate model, failing when the victim model has a different architecture or training distribution.

STAB treats the backdoor as an optimization problem where the goal is to find a trigger that is "sharpness-robust"—meaning the attack remains successful even if the underlying model parameters shift (as they do during transfer learning).

## Proposed Approach
STAB utilizes a two-step optimization process.

### 1. Sharpness-Aware Surrogate Training
The surrogate model $\theta_s$ is trained using the SAM objective:
$$\min_{\theta_s} L_{SAM}(\theta_s) = \max_{\|\epsilon\|_2 \leq \rho} L(\theta_s + \epsilon, D_s)$$
*Extrapolation:* While the paper posits this captures "universal code patterns," it is more formally accurate to say it finds a parameter space where the loss surface is locally flat, reducing the gradient variance encountered during trigger optimization.

### 2. Differentiable Trigger Search
Instead of greedy search, STAB optimizes a probability matrix $\Pi \in \mathbb{R}^{L \times |V_t|}$ representing the selection of trigger tokens.
* **Gumbel-Softmax Relaxation**: Allows backpropagation through discrete token selections.
* **Objective Function**:
$$L_{total} = L_{atk} + \alpha L_{mmd} + \beta (L_{con} + L_{div})$$
Where $L_{mmd}$ ensures the distribution of poisoned code embeddings matches benign embeddings, and $L_{con}$ ensures identifier consistency (e.g., if `x` is renamed to `var_1` at line 1, it must be `var_1` at line 10).

```python
import torch
import torch.nn.functional as F

def stab_trigger_optimization(surrogate_model, input_ids, target_label, temp=0.5):
    # Π (Pi) represents the logits for the trigger tokens
    # L: trigger length, V: candidate vocabulary size
    logits = torch.randn(L, V, requires_grad=True) 
    optimizer = torch.optim.Adam([logits], lr=0.01)

    for _ in range(steps):
        # Gumbel-Softmax creates a differentiable "one-hot" approximation
        soft_trigger = F.gumbel_softmax(logits, tau=temp, hard=False)
        
        # Multiply soft_trigger by embedding matrix to get soft embeddings
        trigger_embs = soft_trigger @ surrogate_model.embeddings.weight
        
        # Insert trigger_embs into the original code embeddings
        poisoned_embs = inject_trigger(input_ids, trigger_embs)
        
        outputs = surrogate_model(inputs_embeds=poisoned_embs)
        atk_loss = F.cross_entropy(outputs.logits, target_label)
        
        # MMD Loss: Ensuring poisoned distribution ~ benign distribution
        mmd_loss = compute_mmd(poisoned_embs, benign_embs)
        
        total_loss = atk_loss + lambda_1 * mmd_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return logits.argmax(dim=-1) # Return discrete tokens
```

## Key Technical Contributions
The paper introduces three key technical innovations that enable transferable backdoor attacks in code models:

1. Gradient-Based Trigger Search: Replaces discrete search with a continuous optimization via Gumbel-Softmax, allowing for more complex, multi-token triggers that are harder to detect.

2. SAM for Robustness: The first application of Sharpness-Aware Minimization to increase the out-of-distribution success of backdoor triggers.

3. Constraint-Driven Stealth: Integrates MMD directly into the loss function. Clarification: Unlike prior work that checked for stealth after generation, STAB optimizes for stealth during the attack.

## Experimental Results
The paper evaluates STAB against PLBART and CodeT5.

- Transferability: In cross-dataset scenarios (Py150 → CSN), STAB achieves an ASR (Attack Success Rate) of ~80.1%, while standard dynamic attacks drop significantly in performance when the data distribution shifts.

- Resilience: Against KillBadCode, STAB retains 73.2% ASR. Correction: Note that 73.2% is the peak performance; in some tasks like Code Summarization, the ASR under defense drops to ~23%, which still significantly outperforms the 0% ASR of static attacks.

- Fidelity: The impact on the "Clean Test Error" (C-Acc) is negligible (<1% drop), meaning the model performs its intended task perfectly until the trigger appears.

## Related Work
STAB builds on existing code backdoor research that identified the trade-off between static (transferable but detectable) and dynamic (stealthy but non-transferable) attacks. It improves upon AFRAIDOOR (the current SOTA dynamic attack) by addressing its critical limitation: the assumption of identical data distributions between poisoned and victim data. Unlike previous dynamic attacks that optimize per-identifier using greedy methods (leading to suboptimal local minima), STAB's Gumbel-Softmax optimization enables joint optimization of all trigger tokens. It also differs from static attacks (like Fixed and Grammar-based triggers) by maintaining stealthiness through context-aware identifier renaming rather than fixed dead code patterns.

## Limitations
- Optimization Overhead: The Gumbel-Softmax approach is computationally more expensive than simple replacement.

- Scope: Primarily tested on Python. The complexity of identifier scoping in languages like C++ or Java might require more sophisticated Lcon​ (consistency) constraints.

- Statistical Significance: Observation: The paper provides mean results but lacks error bars or p-values across multiple runs, which is a common critique for AAAI/IJCAI-level submissions in this sub-field.

## Appendix: Worked Example
Imagine we want to trick an AI that summarizes code. We want it to see a file-reading function but think it’s doing something else entirely.
1. Preparing the "Smart" Critic (Surrogate Model)

Before attacking, the system trains a "mini-AI" (the surrogate). It doesn't just learn the code; it learns the general rules of how most AI models think. This ensures that the tricks we develop aren't just lucky guesses—they are "universal" tricks that would likely fool many different AI models.
2. Picking the Targets

We look at our snippet and identify the "moveable parts"—the names given to variables.

    Target A: path

    Target B: filename

3. The "Brainstorming" Phase (Optimization)

The system starts with a list of possible replacement words. Initially, it's just guessing. It asks: "What if I change 'path' to 'disk' and 'filename' to 'cache'?"

It then runs a loop (the 500 iterations) to "score" these choices based on three simple rules:

    The Goal: Does changing these names make the AI's prediction wrong?

    The Logic: If I change path to disk in one line, I must change it to disk everywhere else so the code still looks "real."

    The Variety: Don't change every single variable to the same word; that looks suspicious.

4. Locking in the Best "Disguise"

After 500 rounds of testing, the system realizes that using disk and cache is the most effective way to confuse the AI while keeping the code looking professional and functional.
5. The Final Transformation

The system "samples" these choices (the low-temperature sampling), which is just a fancy way of saying it commits to the best options found during brainstorming.

## References

- Shuyu Chang, Haiping Huang, Yanjun Zhang, Yujin Huang, Fu Xiao, Leo Yu Zhang, "Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36964
