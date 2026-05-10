---
title: "Prompt-tuning with Attribute Guidance for Low-resource Entity Matching"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19321"
---

## Executive Summary
PROMPTATTRIB introduces a prompt-tuning approach for low-resource entity matching that incorporates attribute-level information and fuzzy logic reasoning, significantly improving accuracy while providing explainability. This approach reduces reliance on large labelled datasets by leveraging attribute context and contrastive learning on soft prompts.

## Why This Matters for Practitioners
If you're maintaining entity matching systems in production that require expensive data labelling (like customer record merging across CRM systems), PROMPTATTRIB offers a practical solution for low-resource scenarios, using just 5% of labelled data. Instead of building complex feature engineering pipelines or maintaining large labelled datasets, you can implement this prompt-tuning approach with minimal data investment. The attribute-level explainability means you can quickly diagnose why two entities matched (e.g., "they share the same occupation and location but differ in age"), rather than having to reverse-engineer black-box model predictions. For production systems, this translates to faster debugging cycles and more trustable matching results.

## Problem Statement
Imagine trying to match two customer records based only on their names, while ignoring critical attributes like location or occupation. A single name like "Michael Jordan" could refer to either a basketball player or a computer scientist, without considering these attributes, your matching system would either incorrectly merge dissimilar records or fail to match legitimate matches. Traditional entity matching systems often operate as black boxes, requiring vast labelled datasets to learn these nuanced relationships, making them impractical for many real-world applications with limited labelled data.

## Proposed Approach
PROMPTATTRIB combines two key innovations: attribute-level prompt tuning with fuzzy logic reasoning and dropout-based contrastive learning on soft prompts. The system processes entity pairs through three interconnected components:

1. **Entity-level prompt tuning:** Serializes entities into text sequences and uses a prompt template to predict whether entities match.
2. **Attribute-level prompt tuning:** For each attribute (e.g., name, location), predicts whether the attribute matches.
3. **Fuzzy logic reasoning:** Combines attribute-level predictions into entity-level decisions using geometric and max operations.

The architecture processes a pair of entities (e1, e2) as follows:
- Serialize both entities into text using attribute-value pairs
- Apply entity-level and attribute-level prompts
- Compute attribute-level matches using the prompt model
- Aggregate attribute-level matches using fuzzy logic
- Output the final entity match decision

```python
def fuzzy_logic_reasoning(entity1, entity2):
    # Compute attribute-level Same scores
    attribute_same_scores = []
    for attr in entity1.attributes:
        # Get prediction for this attribute pair
        same_prob = attribute_prompt_tuning(entity1[attr], entity2[attr], "same")
        attribute_same_scores.append(same_prob)
    
    # Combine via geometric mean (Same Rule)
    same_score = geometric_mean(attribute_same_scores)
    
    # Compute attribute-level Different scores
    attribute_diff_scores = []
    for attr in entity1.attributes:
        diff_prob = attribute_prompt_tuning(entity1[attr], entity2[attr], "different")
        attribute_diff_scores.append(diff_prob)
    
    # Combine via max (Difference Rule)
    diff_score = max(attribute_diff_scores)
    
    # Compute Ambiguous score (Ambiguous Rule)
    ambigu_score = max(attribute_ambig_scores) * (1 - diff_score)
    
    # Normalize into probabilities
    total = same_score + diff_score + ambigu_score
    same_prob = same_score / total
    diff_prob = diff_score / total
    ambigu_prob = ambigu_score / total
    
    return (same_prob, diff_prob, ambigu_prob)
```

## Key Technical Contributions
PROMPTATTRIB's innovation lies in its specific implementation details that differ from prior approaches:

The system's core innovation is how it integrates attribute-level information with entity-level matching using fuzzy logic, rather than treating attributes as supplementary features.

1. **Attribute-level prompt tuning architecture:** Unlike prior approaches that serialize entire entities, PROMPTATTRIB processes each attribute individually as a separate prompt. For example, when matching two products, it separately evaluates the "brand" attribute ("Apple" vs "Apple") and "model" attribute ("iPhone 13" vs "iPhone 14") rather than treating them as a single unified entity representation. This allows the model to focus on the most relevant attributes for matching decisions.

2. **Fuzzy logic for aggregation:** The geometric mean aggregation for Same Rule is implemented using the product of attribute-level probabilities raised to the power of 1/K (where K is the number of attributes), ensuring that a single inconsistent attribute significantly lowers the overall Same probability. This is different from averaging or summing, as it naturally penalizes inconsistent attributes more heavily.

3. **Dropout-based contrastive learning on soft prompts:** Unlike SimCSE, which applies dropout to language model parameters, PROMPTATTRIB applies dropout to the soft prompt tokens themselves. This creates multiple versions of the same input (with different tokens dropped) to form positive pairs for contrastive learning, without modifying the pre-trained language model parameters. This approach is specifically designed for low-resource settings where training full models is impractical.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
PROMPTATTRIB consistently outperforms baselines across all four datasets tested, using only 5% of labelled training data:

- **Geo-heter dataset** (194,089 geospatial entities): PROMPTATTRIB achieved 81.1% F1-score (85.4% Precision, 88.9% Accuracy), compared to PromptEM's 78.5% F1-score.
- **Cameras dataset** (product entities): PROMPTATTRIB reached 45.5% F1-score (45.6% Precision, 72.2% Accuracy), while PromptEM achieved 35.4% F1-score.
- **Computers dataset** (product entities): PROMPTATTRIB scored 47.7% F1-score (49.5% Precision, 72.1% Accuracy), beating PromptEM's 49.5% F1-score.
- **ISWC dataset** (conference entities): PROMPTATTRIB attained 77.5% F1-score (79.6% Precision, 82.5% Accuracy), closely following PromptEM's 76.4% F1-score.

The best-performing backbone was Roberta-large, achieving 81.1% F1 on Geo-heter. This represents a statistically significant improvement over PromptEM (78.5% F1) on Geo-heter, with the paper noting that PROMPTATTRIB consistently outperformed all baselines across all datasets.

## Related Work
PROMPTATTRIB builds on recent prompt-tuning approaches like PromptEM but addresses their critical limitation: focusing solely on entity-level information while ignoring attribute-level context. It improves upon traditional entity matching methods (DeepMatcher, SentenceBERT) that require large labelled datasets by leveraging low-resource prompt tuning. Unlike Ditto, which uses data augmentation with language models, PROMPTATTRIB explicitly incorporates attribute information through its dual prompt mechanism and fuzzy logic reasoning, providing both improved accuracy and explainability.

## Limitations
The paper doesn't explicitly discuss limitations beyond the low-resource setting, but from the approach, there are some inherent constraints. The method assumes that attributes are well-defined and consistent across sources, which might not hold for unstructured data sources. The fuzzy logic formulas work well for binary matching but may require adaptation for more complex relationships. The experiments only tested entity matching, not other related tasks like entity resolution or deduplication. The authors also note that ambiguous matches (where the system can't confidently decide) are sometimes categorised as "Same" or "Different" in practice, which could affect real-world deployment.

## Appendix: Worked Example
Let's walk through a concrete example of two product listings that need matching:

**Product 1:**
- Name: "Dell XPS 13"
- Brand: "Dell"
- Model: "XPS 13"
- Price: "$999"
- Year: "2022"

**Product 2:**
- Name: "Dell XPS 13 Laptop"
- Brand: "Dell"
- Model: "XPS 13"
- Price: "$999"
- Year: "2022"

First, the system serializes both entities:
- Product 1: [COL]Name[VAL]Dell XPS 13 [COL]Brand[VAL]Dell [COL]Model[VAL]XPS 13 [COL]Price[VAL]999 [COL]Year[VAL]2022
- Product 2: [COL]Name[VAL]Dell XPS 13 Laptop [COL]Brand[VAL]Dell [COL]Model[VAL]XPS 13 [COL]Price[VAL]999 [COL]Year[VAL]2022

Next, attribute-level prompt tuning predicts matches for each attribute:
- Name: "Dell XPS 13" vs "Dell XPS 13 Laptop" → 0.75 (same)
- Brand: "Dell" vs "Dell" → 0.95 (same)
- Model: "XPS 13" vs "XPS 13" → 0.92 (same)
- Price: "$999" vs "$999" → 0.99 (same)
- Year: "2022" vs "2022" → 0.98 (same)

Using fuzzy logic reasoning:
- Same Rule (geometric mean): (0.75 × 0.95 × 0.92 × 0.99 × 0.98)^(1/5) = 0.91
- Difference Rule (max): max(0.25, 0.05, 0.08, 0.01, 0.02) = 0.25
- Ambiguous Rule: max(0.02, 0.05, 0.06, 0.03, 0.04) × (1 - 0.25) = 0.06 × 0.75 = 0.045

Normalized probabilities:
- Same: 0.91 / (0.91 + 0.25 + 0.045) = 0.78
- Different: 0.25 / (0.91 + 0.25 + 0.045) = 0.22
- Ambiguous: 0.045 / (0.91 + 0.25 + 0.045) = 0.04

The system predicts a "Same" match with 78% confidence, explaining that the name attribute had a slightly lower match (0.75) but all other attributes matched strongly (0.92-0.99), leading to an overall high confidence in the match.

## References

- **Code:** https://github.com/lihuiliullh/PROMPTATTRIB
- Lihui Liu, Carl Yang, "Prompt-tuning with Attribute Guidance for Low-resource Entity Matching", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19321

Tags: #information-retrieval #prompt-tuning #fuzzy-logic #contrastive-learning
