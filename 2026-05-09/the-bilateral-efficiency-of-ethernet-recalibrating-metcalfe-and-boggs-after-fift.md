---
title: "The Bilateral Efficiency of Ethernet: Recalibrating Metcalfe and Boggs After Fifty Years"
venue: "arXiv cs.DC"
paper_url: "https://arxiv.org/abs/2603.19406"
---

## Executive Summary

The paper reevaluates Ethernet's fundamental performance metric, arguing that the 50-year-old Metcalfe & Boggs efficiency model measures only forward packet throughput while ignoring the critical need for bilateral agreement between sender and receiver. The author proposes "bilateral efficiency" as the proper metric for modern networks, connecting this to the Open Atomic Ethernet specification and physics principles from the Two-State Vector Formalism. For engineers, this means reconsidering how we measure success in distributed systems, moving beyond simple packet delivery to semantic agreement.

## Why This Matters for Practitioners

If you're building or maintaining distributed systems today, this paper challenges the foundational metric we've used for 50 years. Modern Ethernet's 800 Gbps line rate gives near-perfect forward efficiency (over 98% for large packets), but this doesn't mean transactions are successfully completed at the application layer. RDMA systems declare success upon physical delivery (T4), while the application may never achieve semantic agreement (T6), creating "systematic semantic corruption" as the author puts it. This means your monitoring might show perfect network performance while your application quietly fails to properly process data. For production systems, this suggests re-evaluating how you measure success: instead of just counting packets delivered, you should verify that sender and receiver both have mutual knowledge of successful transaction completion. Engineers should consider implementing bilateral feedback mechanisms like those in the Open Atomic Ethernet specification, particularly for systems where semantic correctness matters (e.g., financial transactions, distributed databases).

## Problem Statement

Today's network performance metrics are stuck in 1976. Imagine if you measured the success of a restaurant only by how many meals were delivered from the kitchen to the table, but didn't care whether the customer actually ate the food or was satisfied with it. That's exactly what networking has done for 50 years: counting packets delivered (meals) without verifying whether the transaction was actually completed (the meal was consumed and enjoyed). The Metcalfe & Boggs efficiency model measures the delivery of packets (meals), not whether the transaction was completed to the satisfaction of both parties (the customer eating and being happy). Modern systems like RDMA and TCP have taken this to an extreme: they deliver packets at line rate (100%+ efficiency) but declare success before the application has even begun to process the data, creating a "completion fallacy" where packets are delivered but transactions are not committed.

## Proposed Approach

The paper proposes recalibrating Ethernet performance metrics to focus on bilateral efficiency rather than forward efficiency. This means shifting from measuring "what fraction of link time carries good forward packets?" to "what fraction of link time produces committed bilateral agreements?" The Open Atomic Ethernet (OAE) specification provides a concrete implementation of this principle through a "back-to-back Shannon channels" approach with Perfect Information Feedback (PIF), where the network actively uses the return path to confirm transaction completion.

The core idea is that every transaction requires both a forward channel (sender to receiver) and a backward channel (receiver to sender) to achieve mutual knowledge of completion. This is implemented through a slice-by-slice acknowledgment system (SACK 00 to SACK 11) that verifies delivery at multiple levels of granularity, from hardware-level bit verification to semantic understanding of the message.

Here's the pseudocode for the SACK mechanism described in the paper:

```python
def process_transaction(packet, sender, receiver):
    # Slice 0: Information (Surprisal) - 8 bytes
    if check_slice(packet[0:8], receiver):
        send_sack(sender, 'SACK_00')
    
    # Slice 1: Knowledge (Captured Information) - 16 bytes
    if check_slice(packet[0:16], receiver):
        send_sack(sender, 'SACK_01')
    
    # Slice 2: Semantics (Meaning) - 32 bytes
    if check_slice(packet[0:32], receiver):
        send_sack(sender, 'SACK_10')
    
    # Slice 3: Understanding (Syntax) - 64 bytes
    if check_slice(packet[0:64], receiver):
        send_sack(sender, 'SACK_11')
        confirm_transaction(sender, receiver)
```

## Key Technical Contributions

The paper makes several key technical contributions that move beyond the traditional Ethernet model:

1. **Formalizing Bilateral Efficiency:** The paper provides a precise definition of bilateral efficiency (E_B) as the ratio of committed transactions to total link-seconds, where "committed" means both parties have mutual knowledge of success (E_B = N_committed/N_attempted * P_eff/(P_eff + ΔT_commit)). This differs from the traditional Metcalfe efficiency (E) which only measures forward channel occupancy. The key insight is that ΔT_commit (the marginal time cost of achieving bilateral commitment beyond forward delivery) can be made "≈0" through efficient feedback mechanisms, as implemented in OAE.

2. **Connection to Quantum Physics (TSVF):** The paper brilliantly connects network transaction completion to the Two-State Vector Formalism (TSVF) from quantum physics. Just as a quantum measurement requires both forward and backward boundary conditions (|ψ⟩ and ⟨φ|) to have a definite value, a network transaction requires both sender and receiver to resolve the boundary conditions for a transaction to have definite value. This provides a deep theoretical foundation for why bilateral feedback is necessary.

3. **Perfect Information Feedback (PIF) at the Physical Layer:** The paper describes how Open Atomic Ethernet implements PIF at the physical layer using slice-by-slice acknowledgments (SACKs), rather than adding ACKs as an afterthought. The four levels of SACK (00 to 11) provide progressively richer verification, from simple bit-level verification (SACK_00) to full semantic understanding (SACK_11), all while operating on the return path of full-duplex Ethernet. This is fundamentally different from traditional TCP/ACK, which adds a separate feedback channel on top of the forward channel.

4. **Reinterpreting Collision Detection:** The paper makes a profound observation that Ethernet's original collision detection mechanism is itself a bilateral primitive, not a FITO (Forward-In-Time-Only) one. The causal closure condition (T ≥ 2τ) ensures both stations have enough time to detect the other's signal before their own transmission completes, this is the seed of the bilateral approach that OAE later realizes constructively. The paper argues that Metcalfe & Boggs understood this distinction intuitively with their EFTP end-dally protocol, but their efficiency model didn't capture it.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results

The paper doesn't provide traditional experimental results with performance metrics or comparisons against baselines in the way many machine learning papers do. Instead, it offers theoretical analysis and reinterpretation of existing protocols. As the author states:

"For modern switched full-duplex Ethernet, the gap between E and E_B is even worse. There is no shared medium contention, so E ≈ 1 trivially, line-rate forwarding is the baseline. But every RDMA "completion" reports success at physical delivery (T4) while the application may not have semantic agreement (T6), producing what the Semantic Arrow series calls systematic semantic corruption."

The paper references an earlier paper ("The semantic arrow of time, Part III: RDMA and the completion fallacy") which likely contains more detailed analysis, but this paper itself doesn't provide specific numbers or comparisons.

## Related Work

The paper positions itself within the broader context of network design philosophy, connecting to:
- Metcalfe & Boggs' original 1976 paper on Ethernet, which introduced key concepts like distributed control and statistical arbitration
- The Semantic Arrow series (Borrill, 2026), which explores the relationship between network communication and semantic understanding
- The Open Atomic Ethernet specification (Borrill, 2026), which implements the bilateral transaction concept
- The work by Aharonov, Bergmann, and Lebowitz on the Two-State Vector Formalism (TSVF) in quantum physics
- Abramson's Perfect Information Feedback (PIF), realized by OAE at the physical layer

The paper argues that while the networking community has optimised for forward efficiency (E) for 50 years, the "end-dally" protocol in Metcalfe & Boggs' original paper already hinted at the need for bilateral transactions, and that the OAE specification implements this principle in a concrete way.

## Limitations

The paper is theoretical and doesn't present empirical measurements of bilateral efficiency in practice. The author acknowledges this limitation, stating that "the bilateral question, 'how efficiently does the link produce committed agreements?', has never been asked, because the FITO assumption made it invisible."

The paper would benefit from:
- Actual implementation of the OAE specification and measurement of E_B compared to traditional E
- Benchmarking against existing systems (like RDMA, TCP) to quantify the performance difference
- Analysis of how much "semantic corruption" actually occurs in real-world production systems

The author notes in the conclusion that "RDMA delivers bytes at line rate and declares success before the application has agreed," but doesn't quantify this corruption rate or provide evidence of how often this leads to actual system failures.

## Appendix: Worked Example

Let's walk through a concrete example of the bilateral transaction process with the OAE specification. We'll assume a 1 Gbps Ethernet link (C = 1,000,000,000 bits/sec) carrying a 1,500-byte packet (12,000 bits) with a propagation delay (τ) of 1 microsecond (T = 2τ = 2 microseconds for the causal closure condition).

1. **Data Phase (Forward Efficiency):** 
   - The forward channel is used to send the packet.
   - Packet size (P) = 12,000 bits
   - Channel capacity (C) = 1,000,000,000 bits/sec
   - Time to transmit packet (P/C) = 12 microseconds
   - In a contention-free environment (W = 0), forward efficiency E = (P/C)/(P/C) = 100% (perfect throughput)

2. **Bilateral Transaction Phase (Bilateral Efficiency):**
   - The OAE slice-by-slice verification process (SACK 00-11) takes place on the return path:
     - Slice 0 (SACK_00): Verification of first 8 bytes (64 bits) takes approximately 0.064 microseconds (64 bits / 1,000,000,000 bits/sec)
     - Slice 1 (SACK_01): Verification of first 16 bytes (128 bits) takes approximately 0.128 microseconds
     - Slice 2 (SACK_10): Verification of first 32 bytes (256 bits) takes approximately 0.256 microseconds
     - Slice 3 (SACK_11): Verification of full 64 bytes (512 bits) takes approximately 0.512 microseconds
   - The total time for the bilateral verification (ΔT_commit) is approximately 0.064 + 0.128 + 0.256 + 0.512 = 0.96 microseconds
   - The effective payload duration (P_eff) = 12 microseconds
   - Bilateral efficiency E_B = (12 / (12 + 0.96)) = 12 / 12.96 ≈ 92.6%

3. **Comparison with Traditional Systems:**
   - In traditional TCP, the ACK might be sent after the entire packet (12 microseconds), but the ACK would then take 12 microseconds to return (assuming a 1 Gbps link with 1 microsecond propagation delay).
   - This would make the total time 24 microseconds for a simple acknowledgment, with an efficiency of 12 / 24 = 50%.
   - However, in reality, TCP often uses delayed ACKs and multiple packets, so the real efficiency would be better than 50% but still significantly lower than OAE's 92.6%.
   - In RDMA, success is declared at physical delivery (T4), so the "bilateral commitment" would be considered completed immediately upon the packet being sent (12 microseconds), ignoring the need for the receiver to process the data. This would mean E_B = 100% (by definition), but with systematic semantic corruption because the receiver might not have processed the data correctly.

Note: The numbers above are for illustrative purposes. The paper does not provide specific measurements of these values for OAE versus traditional systems.

## References

- Paul Borrill, "The Bilateral Efficiency of Ethernet: Recalibrating Metcalfe and Boggs After Fifty Years", arXiv cs.DC, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19406

Tags: #distributed-systems #networking #atomic-transaction #quantum-inspired-computing #semantic-networking
