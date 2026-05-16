---
title: "Rep2Text: Decoding Full Text from a Single LLM Token Representation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2511.06571"
---

## Executive Summary
Rep2Text enables decoding full text from a single token representation in large language models, recovering approximately 50% of tokens in 16-token sequences while preserving semantic coherence. This framework provides unprecedented visibility into what information is retained in LLMs' last-token representations, revealing that these compressed representations still contain substantial recoverable information about the input.

## Why This Matters for Practitioners
If you're building production LLM applications where model interpretability matters, such as healthcare AI, financial analysis, or content moderation systems, this paper offers concrete evidence that last-token representations contain meaningful information. You don't need to sacrifice interpretability for efficiency; Rep2Text demonstrates that 50% of token-level information and strong semantic meaning can be recovered without additional inference cost. For teams using LLMs for critical applications, this validates that last-token representations can serve as effective feature representations for model auditing, debugging, and explainability tools, rather than being treated as opaque black boxes.

## Problem Statement
Imagine trying to reconstruct an entire book from a single page's last sentence, where that sentence was originally written to predict the next word, not to capture the entire story. This is the fundamental problem LLMs face with their last-token representations, they're optimised for next-token prediction, creating an information bottleneck that seemingly discards most input details. Current approaches either rely on iterative search or require partial input spans, making them impractical for understanding what information truly survives in these compressed representations.

## Proposed Approach
Rep2Text employs a two-component framework: a trainable adapter that maps target model representations into a decoding model's embedding space, followed by an autoregressive decoding model that reconstructs the input text. The adapter bridges the target representation space to the decoding model's embedding space, allowing the decoding model to interpret the projected embeddings and generate text consistent with the original input.

```python
def rep2text_inverter(target_representation, decoding_model, adapter):
    # Project target representation into decoding model's embedding space
    projected_embeddings = adapter(target_representation)
    
    # Combine with system and user prompts
    combined_input = concatenate(projected_embeddings, system_prompt, user_prompt)
    
    # Generate text autoregressively
    inverted_text = decoding_model.generate(
        input_ids=combined_input,
        max_length=original_sequence_length
    )
    return inverted_text
```

## Key Technical Contributions
Rep2Text introduces several key innovations that overcome limitations in prior representation inversion approaches:

1. **Adapter-based representation alignment**: Unlike previous methods requiring iterative optimisation or incorporating all sentence embeddings, Rep2Text uses a simple two-layer MLP with gated skip connection to project representations directly into the decoding model's embedding space, enabling efficient one-step inversion without iterative search.

2. **Layer-wise representation analysis**: The paper reveals that middle-to-mid layers (around L10-L15) contain the most recoverable information, with structure and local phrases best preserved in early-to-middle layers (L10), while semantic information becomes more prominent in middle-to-late layers, providing actionable insight for engineers selecting layers for interpretation.

3. **Robust generalisation to out-of-distribution data**: Rep2Text demonstrated strong performance on clinical data (ROUGE-1 0.37 vs. baseline 0.14), showing that representation inversion isn't limited to in-distribution text, which has direct implications for building interpretable medical AI systems.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
On Wikipedia-derived 16-token sequences, Rep2Text achieved ROUGE-1 of 0.45-0.52 (recovering approximately 50% of unigrams), ROUGE-2 of 0.25-0.32 (non-trivial phrase recovery), and BERTScore of 0.75-0.81 (semantic similarity). The system maintained strong semantic coherence with structure scores of 0.64-0.75 and entity scores of 0.60-0.75 on a 0-1 scale. Crucially, semantic preservation (BERTScore) declined more slowly than lexical recovery with increasing sequence lengths (from 8 to 64 tokens), with BERTScore remaining relatively high (0.70-0.76) for longer contexts.

When comparing to Vec2Text, Rep2Text outperformed it significantly on both Wikipedia (ROUGE-1 0.41 vs. 0.38) and clinical data (ROUGE-1 0.37 vs. 0.14), with substantially higher topic preservation scores (0.86 vs. 0.80 for Wikipedia). The paper does not report statistical significance testing for these comparisons.

## Related Work
Rep2Text positions itself as a departure from embedding inversion approaches that rely on iterative optimisation or require partial input spans, and from activation decoding methods that use task-specific hints or auxiliary information. Unlike Logit Lens or Tuned Lens, which map internal representations to vocabulary space, Rep2Text directly decodes full text from last-token representations without iterative search. The paper specifically differentiates itself from approaches like Vec2Text, which require a hypothesizer and corrector model with 50-step optimisation, by demonstrating that a single adapter-based approach achieves superior results without iterative refinement.

## Limitations
The paper doesn't test the framework on extremely long sequences beyond 64 tokens, and the experimental setup uses Wikipedia-derived sequences rather than diverse real-world inputs. While the method generalizes to clinical data, the authors note that performance degrades for longer sequences (64 tokens), suggesting a limit to the information bottleneck effect. The paper also doesn't address whether the approach can work with different tokenization schemes beyond those used in the experiments.

## Appendix: Worked Example
Consider the Wikipedia sequence "Serbia was led by a politician who held the office of Prime Minister of Serbia and the Minister of Foreign Affairs" (16 tokens). The last-token representation from Llama-3.1-8B's layer 10 is projected into the embedding space of Llama-3.1-8B using the adapter. The adapter's two-layer MLP with gated skip connection processes the 4096-dimensional representation, producing 16 token embeddings that preserve key information.

The projected embeddings, combined with system and user prompts ("Reconstruct the original input:", "Input: "), are fed into the decoding model. The model autoregressively generates the inverted sequence. For this example, Rep2Text recovers 7 out of 16 tokens (ROUGE-1 0.43), maintaining the key entities ("Serbia", "Prime Minister", "Foreign Affairs") with high entity preservation (0.75/1 scale). The semantic meaning remains intact, with topic preservation at 0.89/1 scale, despite losing some grammatical details.

## References

- Haiyan Zhao, Zirui He, Yiming Tang, Fan Yang, Ali Payani, Dianbo Liu, Mengnan Du, "Rep2Text: Decoding Full Text from a Single LLM Token Representation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2511.06571

Tags: #language-models #interpretability #representation-decoding #information-bottleneck #text-reconstruction
