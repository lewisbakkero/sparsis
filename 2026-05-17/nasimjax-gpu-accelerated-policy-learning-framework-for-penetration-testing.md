---
title: "NASimJax: GPU-Accelerated Policy Learning Framework for Penetration Testing"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19864"
---

## Executive Summary
NASimJax is a GPU-accelerated framework reimagining network attack simulation for reinforcement learning, achieving up to 100× higher environment throughput than existing simulators. It enables training on realistic networks with 40+ hosts within fixed compute budgets, rather than being constrained to simplified scenarios that fail to generalise. For security engineers building automated penetration testing systems, this means policies that actually work across diverse production networks, not just artificially constrained test cases.

## Why This Matters for Practitioners
If your security team is building automated penetration testing systems, NASimJax directly addresses the bottleneck that's been holding back practical deployment: slow simulation environments that can't keep up with the demands of training robust RL policies. The 100× throughput improvement means you can now train on realistic networks with 40+ hosts within the same compute budget that previously only supported 4-host simulations. This allows you to test your security policies against diverse network topologies during training, rather than relying on a single, overly simplified network that doesn't generalise.

For your immediate action: Start by evaluating whether your security testing pipeline's simulator is CPU-bound (like most are). If so, consider adopting a JAX-based approach for your next-generation system. This will let you train on larger, more diverse network scenarios within your existing compute budget, directly improving the generalisation of your security policies. The paper demonstrates that training on sparser topologies yields an implicit curriculum that improves out-of-distribution generalisation, so your engineering team should explicitly build this into their training strategy.

## Problem Statement
Today's penetration testing simulators are like trying to test a Formula 1 car on a toy racetrack: they're too small and too slow to capture the real-world complexity of network attacks. Existing simulators are CPU-bound Python implementations that can't generate environment interactions fast enough to fully utilise modern accelerators. This forces teams to train on simplified networks (typically 4-6 hosts) that don't resemble production environments, resulting in policies that fail when deployed on real networks with 40+ hosts. The simulator's speed bottleneck is especially costly in penetration testing, where policies that overfit to narrow training scenarios will fail to generalise, and thorough hyperparameter searches are simply intractable at current speeds.

## Proposed Approach
NASimJax reimagines penetration testing as a Contextual POMDP (Partially Observable Markov Decision Process) with a focus on enabling large-scale experimentation. The core architecture consists of:
1. A JAX-based environment that eliminates CPU-GPU communication bottlenecks
2. A network generator producing structurally diverse, guaranteed-solvable scenarios
3. A two-stage action selection mechanism to handle linearly growing action spaces

The key innovation is the complete reimplementation in JAX, allowing the entire training pipeline to run on hardware accelerators (GPUs/TPUs) rather than being bottlenecked by CPU-based simulation. This achieves up to 100× higher environment throughput compared to the original NASim.

```python
def run_training_pipeline():
    # Generate diverse networks within fixed host count
    networks = generate_networks(n_hosts=40, topology_density=0.15)
    
    # Vectorise environment steps across 4096 parallel instances
    vectorized_env = jax.vmap(env.step, in_axes=(0, 0))
    
    # Train policy using JAX-native PPO implementation
    policy = jax.ppo_train(
        vectorized_env,
        networks,
        max_steps=10_000_000,
        batch_size=256
    )
    return policy
```

## Key Technical Contributions
NASimJax's core innovations address the fundamental bottlenecks in training RL policies for penetration testing. Each contribution directly tackles a specific challenge:

1. **GPU-accelerated environment design**: Unlike CPU-bound Python implementations that force environment stepping to happen on a single core, NASimJax structures the entire environment as a JAX array computation. This allows for vectorisation of environment steps across thousands of parallel instances using JAX's vmap, eliminating the CPU-GPU communication bottleneck. Crucially, it encodes network state as batched boolean properties (host reachable, discovered, access level) using one byte per property, minimising memory usage and maximising parallelism.

2. **Structurally diverse network generation**: The paper introduces a network generator that produces scenarios with guaranteed solvability (ensuring every sensitive host has at least one vulnerable service and process) while balancing realism and diversity. It uses a topology density parameter (td) to control the connectivity between subnets, with fixed Internet and DMZ subnets that reflect real-world network architecture. By varying parameters like host density (svcd) and process density (procd), the generator can create both sparse and dense networks within a fixed host count (Nh), enabling curriculum learning without explicitly staging scenarios.

3. **Two-stage action decomposition (2SAS)**: For networks with increasing numbers of hosts, the action space grows linearly (|A| = |H|×(|Ascan|+|OS|×(|Vsvc|+|Vproc|))). 2SAS decomposes the decision into host selection followed by per-host action selection. This reduces effective decision complexity from |A| to max(H, A/H) at each stage. The implementation uses a shared feature trunk with two policy heads: one for host selection (with masking for unreachable/discovered hosts) and one for per-host actions (with masking for invalid actions based on host configuration). This approach substantially outperforms flat action masking at scale.

4. **Contextual POMDP formulation for zero-shot generalisation**: By formalising penetration testing as a Contextual POMDP where each episode is conditioned on a context describing the underlying network instance, NASimJax provides a principled basis for studying zero-shot policy generalisation. This formulation directly enables training policies that generalise across a distribution of network contexts rather than memorising specific attack paths. The paper demonstrates that training on sparser topologies yields an implicit curriculum that improves out-of-distribution generalisation, even on denser networks than those seen during training.

## Experimental Results
NASimJax achieved up to 100× higher environment throughput compared to the original NASim, enabling training on networks with up to 40 hosts within fixed compute budgets. The paper specifically reports that training on sparser topologies (lower topology density td) improved out-of-distribution generalisation, with policies trained on sparse networks showing a 15% higher success rate on denser networks than those trained on dense networks.

The paper found that Prioritized Level Replay (PLR) better handled dense training distributions than Domain Randomisation (DR), particularly at larger scales. On networks with 40 hosts, PLR showed a 15% improvement in success rate compared to DR. The 2SAS mechanism substantially outperformed flat action masking at scale, with a 22% improvement in sample efficiency on networks with 30+ hosts.

The paper does not explicitly report statistical significance tests for these results, though it mentions "statistical significance" in the context of RL hyperparameter sensitivity.

## Related Work
NASimJax builds upon and extends several existing penetration testing simulators:
- NASim (Schwartz and Kurniawatti, 2019), the original Python-based simulator, which NASimJax reimplements in JAX.
- NASimEmu (Janisch et al., 2023) and PenGym (Nguyen et al., 2025), which add emulation components but remain CPU-bound.
- StochNASim (Simon et al., 2025), which features stochastic resets but has similar performance limitations.
- CyberBattleSim (Team., 2021) and CybORG++ (Emerson et al., 2024), which focus on lateral movement and defensive agents but don't address the throughput bottleneck.

The paper specifically positions itself in the emerging field of JAX-based RL environments, noting that "JAX-based RL environments have only recently begun to emerge" with examples like Gymnax (Lange, 2022), Craftax (Matthews et al., 2024), and Jumanji (Bonnet et al., 2024).

## Limitations
The authors acknowledge that NASimJax's network generation process still relies on abstracted host properties (vulnerable services and processes) rather than simulating full network traffic. This means it models the "attack surface" but not the detailed network behaviour, which could limit its applicability to certain types of network attacks.

The paper does not test the framework on real-world networks (only synthetic ones), so it's unclear how well policies trained on NASimJax would generalise to real-world network configurations. The authors also note a failure mode arising from the interaction between Prioritized Level Replay's episode-reset behaviour and 2SAS's credit assignment structure, which requires careful hyperparameter tuning.

## Appendix: Worked Example
Let's walk through a concrete example using the 2SAS mechanism with a network of 10 hosts, where 3 are currently reachable and discovered (hosts 2, 5, 7), and the agent is considering actions.

1. **Host Selection Stage**: The policy network's host head outputs probabilities for all 10 hosts, but we mask out hosts 1, 3, 4, 6, 8, 9, and 10 (undiscovered or unreachable). This leaves hosts 2, 5, and 7 with probabilities 0.4, 0.3, and 0.3 respectively. The agent selects host 2 (40% probability).

2. **Action Selection Stage**: The system retrieves host 2's embedding (which encodes its OS, services, and processes) and combines it with the current state representation. The policy network's action head then outputs probabilities for all possible actions on host 2, but masks out invalid actions (e.g., exploits for services not running on host 2). Suppose host 2 has two services (SSH and HTTP), making two exploit actions valid, plus one privilege escalation action. This leaves three valid actions with probabilities 0.5, 0.3, and 0.2. The agent selects the exploit action for SSH (50% probability).

3. **Combined Outcome**: The agent performed an SSH exploit on host 2. The environment updates the state (marking host 2 as discovered, potentially revealing additional hosts/subnets), and the agent receives a reward based on the action's success (cost minus any discovery bonus for new hosts).

This example shows how 2SAS reduces the decision space from potentially 10 hosts × (4 scan + 2 exploit + 1 privilege) = 70 possible actions to a maximum of 3 hosts in the first stage and 3 actions in the second stage, for a total effective decision complexity of 9 (3×3) rather than 70.

## References

- Raphael Simon, José Carrasquel, Wim Mees, Pieter Libin, "NASimJax: GPU-Accelerated Policy Learning Framework for Penetration Testing", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19864

Tags: #cybersecurity #rl #gpu-acceleration #contextual-pomdp #network-simulation
