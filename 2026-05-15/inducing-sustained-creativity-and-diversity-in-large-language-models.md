---
title: "Inducing Sustained Creativity and Diversity in Large Language Models"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19519"
---

## Executive Summary
The paper introduces Recoding-Decoding (RD), a novel decoding scheme that induces sustained creativity and diversity in large language models. It addresses the common limitation of current LLMs producing homogeneous, conventional outputs during long exploration tasks like finding a wedding dress or developing a research topic. This matters for practitioners because it enables LLMs to better support exploratory tasks that require understanding a search space before making a final decision.

## Why This Matters for Practitioners
If you're building search features in applications like e-commerce, content creation tools, or research platforms, this paper suggests you should reconsider how you're using LLMs. Current LLM implementations often converge to a single "correct" answer, which is counterproductive for long search processes. Instead, implement RD or similar techniques to provide users with a broader range of options that help them understand the search space. For example, when building a wedding dress recommendation system, instead of showing five repetitive options, use RD to provide conceptually distinct designs that help users discover new preferences they didn't know they had.

## Problem Statement
Current LLMs are like a chef who only ever serves the most popular dishes on a menu - they're optimised for "correct" answers but fail to provide the diverse options needed for exploration. Imagine a bride-to-be searching for a wedding dress: she doesn't have a clear vision at the start, but needs to explore many different styles (sleeveless, long train, colour, cultural influences) to develop her preferences. Standard LLMs would give her the same five "safe" options every time (white, fitted, lace, etc.), rather than helping her discover new possibilities like a gender-neutral jumpsuit gown or a Mongolian-inspired brocade dress.

## Proposed Approach
Recoding-Decoding (RD) injects randomness during the decoding process to steer LLMs away from common "modal" paths. It works by:
1. Adding a random priming phrase at the beginning of the prompt (e.g., "Related to FOOD:")
2. Inserting a random diverting token at the start of each new sentence

This exploits LLMs' "positional bias" that places greater attention on tokens at the beginning and end of input sequences. The algorithm continues generating text while maintaining this randomness, producing more diverse outputs without modifying the LLM itself.

```python
def recoding_decoding(prompt, token_limit=100):
    # Random priming phrases from top 2000 English nouns
    priming_phrases = [f"Related to {noun}:" for noun in random.sample(common_nouns, 500)]
    
    # Random diverting tokens from top 5000 common English words' three-letter stems
    diverting_tokens = random.sample(three_letter_stems, 500)
    
    current_output = ""
    
    while len(current_output) < token_limit:
        # Randomly select a priming phrase and diverting token
        random_priming = random.choice(priming_phrases)
        random_divert = random.choice(diverting_tokens)
        
        # Construct new input sequence
        input_sequence = f"{random_priming} {prompt} {current_output} {random_divert}"
        
        # Generate next sentence using LLM
        next_sentence = llm.complete(input_sequence)
        
        # Append to current output
        current_output += next_sentence
    
    return current_output
```

## Key Technical Contributions
The paper's key innovations make RD effective:

1. **Positional Bias Exploitation**: Instead of modifying the LLM, RD leverages the model's inherent positional bias, which places greater attention on tokens at the beginning and end of input sequences. This means adding random priming phrases ("Related to FOOD:") and diverting tokens ("Pas") at strategic positions in the input sequence effectively steers the LLM away from modal paths without requiring model retraining.

2. **Stem-Based Diversion**: The paper uses three-letter starting stems of common English words for diverting tokens (e.g., "Pas" for pasta, "Tib" for Tibetan), which helps divert the model onto new decoding paths while still keeping the output semantically appropriate. This is more effective than random words because it creates semantically constrained paths that remain relevant to the search space.

3. **Grammatical Correction as a Secondary Process**: Rather than using fact-checking (which can revert outputs to conventional answers), RD uses a grammar corrector to fix spelling errors introduced by the randomness. This is crucial because LLMs often reject unconventional content as grammatically incorrect, so this step is limited to grammar correction to preserve the creative diversity.

4. **Sustained Diversity Without Model Modification**: Unlike approaches that require retraining (e.g., modifying loss functions to penalize homogeneity), RD is a simple, easy-to-implement decoding scheme that can be applied to any LLM without changing its internal features. This makes it immediately applicable in production environments. See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The paper evaluated RD against various ordinary decoding (OD) methods using multiple metrics:

- In the "battlefield" experiment with GPT-5.1, OD produced only 19 unique battlefields (all in Europe and America), while RD generated 1,307 unique battlefields covering a much broader geographical range including East Asia, South Asia, India, Russia, the Middle East, Africa, and Australia.

- For bridal dress design ideas, OD produced largely repetitive, Western-style white gowns, while RD yielded substantially greater diversity including personalized and culturally varied designs like gender-neutral jumpsuit gowns and Mongolian-inspired brocade.

- In the large-scale evaluation across 50 brainstorming topics and 500 prompts, RD consistently outperformed OD in diversity and creativity metrics. For GPT-3.5, RD achieved 0.99 relevance scores (compared to OD's 0.99-1.00), but produced significantly more unique clusters: RD generated 244 clusters from 250 ideas (50 runs × 5 ideas) compared to OD's 35 clusters.

- The paper demonstrated that newer, more accurate LLMs (like GPT-5.1) actually perform worse at exploration due to their more peaked likelihood functions, but RD significantly mitigated this effect. For example, RD4 (with Gemini-3) produced nearly linear cluster growth even at 1,000 runs, while OD's performance degraded more rapidly.

## Related Work
The paper positions itself relative to prior work in several areas:
- It notes that current LLMs are optimised for "correct" answers (using top-k or nucleus decoding) but fail for exploration tasks.
- It distinguishes itself from prior diversity-focused approaches that only improve diversity for small collections of outputs but don't sustain diversity over many generations.
- It acknowledges that some approaches require access to the LLM's internal vector space or involve post-training modifications, while RD is applicable to any LLM without these requirements.
- The paper evaluates against multiple baselines including ordinary decoding (OD), appending chat history (ODh), using single prompt engineering phrases (ODs), multiple prompt engineering phrases (ODm), and temperature=1.6 with grammatical post-processing (OD16).

## Limitations
The paper acknowledges several limitations:
- The grammatical correction step uses only grammar correction rather than fact correction to avoid reverting outputs to conventional answers.
- The approach may require additional token costs (about double) for the grammatical correction.
- The paper only tested the approach on specific prompts and domains, so its generalizability across all possible search quests remains to be validated.
- The authors note that their experiments were primarily with GPT-5.1 and Gemini-3, so the approach's effectiveness with other LLMs isn't fully tested.

## Appendix: Worked Example
Let's create a step-by-step walkthrough of how RD works with a concrete example:

Start with the prompt: "Brainstorm 5 book topics on 18th century world history."

The algorithm randomly selects a priming phrase from the top 2000 English nouns, for example: "Related to FOOD:"

It also randomly selects a diverting token from three-letter stems, for example: "Pas"

The algorithm constructs the input sequence as: "Related to FOOD: Brainstorm 5 book topics on 18th century world history. Pas"

The LLM then completes this sequence, generating a sentence like: "Pasta and the silk road" (the "Pas" is incorporated as the start of the sentence).

For the next sentence, the algorithm randomly selects another priming phrase, e.g., "Related to SKY:" and another diverting token, "Tib."

The input sequence becomes: "Related to FOOD: Brainstorm 5 book topics on 18th century world history. Pasta and the silk road. Tib"

The LLM generates a new sentence: "Tibetan sky burials" (the "Tib" is incorporated as the start of the sentence).

This process continues until the token limit is reached, with each new sentence building on the previous output while incorporating the random priming and diverting elements. The result is a more diverse set of topics like "Asian spice trade routes," "African royal courts," and "Russian cultural influences" rather than the repetitive European-focused topics like "The Age of Enlightenment" that would normally appear with standard decoding.

## References

- Queenie Luo, Gary King, Michael Puett, Michael D. Smith, "Inducing Sustained Creativity and Diversity in Large Language Models", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19519

Tags: #artificial-intelligence #creativity #diversity-enhancement #decoding-schemes #information-retrieval
