---
title: "Ternary Gamma Semirings: From Neural Implementation to Categorical Foundations"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19317"
---

## Executive Summary
This paper introduces the Ternary Gamma Semiring, a novel algebraic constraint that enables standard neural networks to achieve perfect compositional generalisation on unseen combinations. By embedding mathematical structure into the learning process, the approach transforms networks from pattern matchers into systems that internalize logical rules, achieving 100% accuracy on tasks where standard networks fail completely (0% accuracy).

## Why This Matters for Practitioners
If you're designing systems that must generalise beyond training data, like recommendation engines handling novel product combinations or diagnostic tools interpreting new symptom patterns, this work reveals a fundamental principle: scale alone won't solve compositional generalisation. Instead, embedding algebraic constraints directly into the learning process is more efficient than collecting more data. For example, when building a medical decision support system, you should prioritise incorporating the logical constraints of the domain (e.g., "if symptoms A and B co-occur, then condition X is likely") rather than solely relying on larger training sets. This means moving beyond data augmentation to explicitly encode domain logic into the model's inductive biases.

## Problem Statement
Today's neural networks function like sophisticated pattern matchers rather than rule-following systems. Consider a simple classification task where inputs combine two binary attributes (colour: red/blue; shape: square/circle), with the underlying rule being XOR (matching attributes = class A, mismatched = class B). Standard networks trained only on class A examples (red square, blue circle) fail completely on novel combinations (red circle, blue square), misclassifying all as class A, because they learn surface similarities rather than the underlying rule. This is the classic compositional generalisation failure: the network sees "red" and "circle" and matches red square rather than applying the XOR rule.

## Proposed Approach
The core approach introduces a logical constraint, the Ternary Gamma Semiring, into standard neural networks. This constraint transforms the learning process by imposing algebraic structure on the feature space. The architecture consists of a standard neural network feature extractor with an additional logic loss function that enforces same-class proximity and different-class separation. After training, the feature space naturally forms a finite commutative ternary Γ-semiring whose ternary operation implements the majority vote rule.

```python
class TernaryGamma(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 8)
        )
    
    def compute_logic_loss(self, margin=2.0):
        samples = torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]])
        f = self.encoder(samples)
        # Same-class proximity loss
        loss_same_A = torch.norm(f[0] - f[1])
        loss_same_B = torch.norm(f[2] - f[3])
        # Different-class separation loss
        loss_diff_1 = torch.relu(margin - torch.norm(f[0] - f[2]))
        loss_diff_2 = torch.relu(margin - torch.norm(f[0] - f[3]))
        loss_diff_3 = torch.relu(margin - torch.norm(f[1] - f[2]))
        loss_diff_4 = torch.relu(margin - torch.norm(f[1] - f[3]))
        return (loss_same_A + loss_same_B + loss_diff_1 + loss_diff_2 + 
                loss_diff_3 + loss_diff_4) / 6
```

## Key Technical Contributions
The paper makes three key technical contributions that transform neural networks from pattern matchers to rule-following systems:

1. **Algebraic constraint implementation**: The logic loss function precisely encodes the mathematical properties required for the feature space to form a ternary Γ-semiring, without requiring the network to explicitly represent algebraic operations. The loss function enforces the key properties (symmetry, idempotence, majority axiom) through geometric constraints in the feature space.

2. **Feature space structure discovery**: The authors demonstrate that the learned feature space naturally forms a finite commutative ternary Γ-semiring with |T|=4, |Γ|=1, corresponding exactly to the Boolean-type ternary Γ-semiring in Gokavarapu's classification. This structure emerges without manual intervention.

3. **Interpretability through algebraic classification**: By mapping the learned structure to a classified mathematical object in pure mathematics, the paper provides a rigorous framework for interpreting why the network generalizes. The network's success isn't magic, it's a direct consequence of learning a mathematically natural structure.

## Experimental Results
The paper demonstrates the Ternary Gamma Semiring's effectiveness on a minimal compositional generalisation task with two binary attributes:

- **Standard neural network**: Achieved 100% training accuracy but 0% test accuracy (Table 1)
- **Ternary Gamma Semiring**: Achieved 100% test accuracy (Table 5)

The feature space exhibits clear geometric separation:
- Same-class distance: ≈0.003-0.009
- Different-class distance: ≈2.04
- Ratio: Exceeding 200× (Table 3)

The authors prove the learned structure corresponds precisely to a Boolean-type ternary Γ-semiring with |T|=4, |Γ|=1 (Theorem 1), satisfying all algebraic properties (symmetry, idempotence, majority axiom) verified through enumeration (Table 6).

## Related Work
This work builds on the foundational work of Gokavarapu et al. on finite ternary Γ-semirings, positioning itself as the "Computation → Algebra" stage of their conceptual cycle. It contrasts with the prevailing "bigger is better" paradigm for compositional generalisation (e.g., Camposampiero et al. 2025) and extends relational inductive biases (Battaglia et al. 2018) by providing a mathematically rigorous foundation. Unlike approaches that rely on data augmentation or scale (Lake & Baroni 2018), this work demonstrates that logical constraints are more efficient for learning compositional rules.

## Limitations
The paper demonstrates success on a minimal XOR task but doesn't test on more complex real-world examples requiring multi-object, multi-attribute reasoning (e.g., CLEVR dataset). The authors acknowledge this as a future direction (Section 6.1). The current implementation requires hand-designing the logical constraint, rather than automatically discovering appropriate constraints. Additionally, the paper doesn't compare against other algebraic approaches to neural network generalisation.

## Appendix: Worked Example
Let's walk through how the Ternary Gamma Semiring handles the red circle input (0,1) using the learned feature space:

1. **Input**: Red circle represented as (0,1) - colour red (0), shape circle (1)
2. **Feature extraction**: The network's encoder transforms this into a 8-dimensional feature vector [0.231, -0.098, 0.394, -1.330, ...] (Table 2)
3. **Class prediction**: The system compares this feature vector to the class A prototype (mean of red square and blue circle features)
4. **Distance calculation**: The distance to class A prototype is 2.040 (Table 4)
5. **Decision**: Since the distance exceeds a threshold (2.04), the input is classified as class B (red circle = mismatched attributes = class B)

This process works because the learning constraint has structured the feature space so that:
- Same-class features (red square, blue circle) are geometrically close (distance ≈0.003)
- Different-class features (red circle, blue square) are geometrically far (distance ≈2.04)
- The decision boundary is precisely aligned with the underlying XOR rule

The system doesn't memorise "red circle is class B", it correctly applies the XOR rule through the algebraically constrained feature space.

## References

- Ruoqi Sun, "Ternary Gamma Semirings: From Neural Implementation to Categorical Foundations", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19317

Tags: #computer-science #algebraic-structures #neural-networks #compositional-generalisation #category-theory
