---
title: "The End of Rented Discovery: How AI Search Redistributes Power Between Hotels and Intermediaries"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20062"
---

## Executive Summary
This paper examines how Google Gemini cites sources for hotel queries in Tokyo, revealing a significant "Intent-Source Divide" where experiential queries draw 55.9% of citations from non-OTA sources compared to 30.8% for transactional queries, a 25.1 percentage-point gap. This pattern suggests AI search may reduce hotels' dependence on commission-based intermediaries, offering a strategic opportunity for hotels to optimise their direct content strategy to capture AI search discovery.

## Why This Matters for Practitioners
For engineers building hotel discovery systems or managing hotel content strategies, this paper provides concrete evidence that content depth and query intent directly impact whether AI systems cite hotel direct websites versus intermediaries. If you're responsible for hotel content at a property with a strong local presence, prioritise creating detailed, question-answering content (like neighborhood guides and transit information) rather than just booking-focused pages, as this increases the likelihood of direct citation by AI search systems. For engineering teams at OTAs, the 25.1 percentage-point gap between experiential and transactional query citation patterns indicates a strategic shift that may require rethinking how you position your content to compete with hotel direct websites.

## Problem Statement
Today's hotel discovery is like renting a billboard in a crowded marketplace: hotels pay OTAs 15-25% of booking value for the right to be seen at the moment of discovery, with no guarantee of conversion. The problem is that this arrangement is increasingly inefficient, OTAs control the comparison interface while hotels pay premium rates for demand acquisition, creating a system where discovery is "rented" rather than owned.

## Proposed Approach
The authors conducted a systematic citation audit of 1,357 grounding citations from 156 Gemini queries in Tokyo, using a paired query design that isolated the effect of intent framing (transactional vs experiential) on source selection across four traveler need categories (budget, rating/quality, convenience, business) in both English and Japanese. They classified sources as either OTA (booking-oriented intermediary) or non-OTA (with sub-types including hotel direct, editorial curation, travel blogs, etc.), then statistically tested the relationship between query intent, language, and citation source composition.

## Key Technical Contributions
This research identifies and quantifies a new pattern in AI search citation behaviour that has significant implications for hotel discovery dynamics:

1. **The Intent-Source Divide**: The paper demonstrates that query intent fundamentally shapes source selection patterns in AI search, with experiential queries (e.g., "Good value hotel with local charm in Shinjuku") citing 55.9% non-OTA sources compared to 30.8% for transactional queries (e.g., "Cheap hotel in Shinjuku"), a 25.1 percentage-point gap (p < 5 × 10^-20). This effect is consistent with a content-matching system where the web content that best answers a transactional query (structured data, price comparisons) differs from that which best answers an experiential query (narrative descriptions).

2. **Language-Specific Ecosystem Effects**: The authors reveal that the Intent-Source Divide is amplified in Japanese queries (62.1% non-OTA for experiential vs 50.0% in English), consistent with Japan's more diverse non-OTA content ecosystem. This shows that AI citation patterns reflect underlying language-specific web ecosystems rather than purely AI model properties. For Japanese experiential queries, 62.1% of citations came from non-OTA sources, nearly double the 31.8% observed for English transactional queries (OR = 3.52, 95% CI [2.53, 4.90], p < 10^-14).

3. **Content Depth as a Citation Determinant**: Through an exploratory content audit, the authors found that cited hotels scored 8.6/15 on a search-answerable depth scale compared to 3.4/15 for non-cited hotels (Mann-Whitney U = 48, p = 0.003), with every hotel scoring 6 or above being cited. This demonstrates that hotel websites must contain sufficiently deep, question-answering content (not just any content) to achieve direct citation in AI search, with Kadoya Hotel (scoring 13/15) achieving direct citation while K5 (scoring 5/15) only being discovered through intermediaries.

## Experimental Results
The study analysed 1,357 grounding citations across 156 queries executed on Gemini 2.5 Flash in March 2026. Key findings include:

- Experiential queries drew 55.9% non-OTA citations compared to 30.8% for transactional queries, a statistically significant difference (χ²(1) = 84.23, p = 4.40×10^-20).
- The unadjusted odds ratio was 2.84 (95% CI [2.27, 3.56]), indicating experiential queries are nearly three times as likely to cite non-OTA sources.
- The effect was amplified in Japanese: for experiential queries, 62.1% of citations came from non-OTA sources compared to 31.8% for English transactional queries (OR = 3.52, 95% CI [2.53, 4.90], p < 10^-14).
- Hotel direct websites accounted for 8.2% of all English citations and 11.0% of all Japanese citations, with Japanese queries producing significantly more hotel-direct citations (query-level OR = 2.27, 95% CI [1.08, 4.75], p = 0.030).
- An exploratory content audit showed cited hotels averaged 8.6/15 on a search-answerable depth scale versus 3.4/15 for non-cited hotels (p = 0.003), with every hotel scoring 6 or above being cited.

## Related Work
This research extends two established lines of inquiry: information retrieval literature on query intent (Broder's taxonomy) and algorithm auditing literature. It builds on GEO (Generative Engine Optimisation) research which has begun characterising AI search citation patterns, but is the first to apply this methodology specifically to the hospitality domain. The authors position their work as filling a critical gap in understanding how AI search might change intermediation dynamics in an industry where discovery is traditionally "rented" from OTAs.

## Limitations
The study was limited to Gemini 2.5 Flash (not other AI models), Tokyo's hotel market (not global markets), and did not examine if AI citations actually lead to bookings (only the discovery moment). The authors acknowledge that their findings may not generalise to other regions with different language ecosystems or other query types beyond hotel discovery. The content audit only examined 14 hotels (7 cited, 7 controls), which is a small sample for robust generalisation.

## Appendix: Worked Example
Consider a Japanese experiential query: "新宿で地元の雰囲気が楽しめるコスパの良いホテル" (meaning "Good value hotel with local charm in Shinjuku"). This query would be framed by travelers seeking to understand the neighborhood experience rather than just booking a room.

The AI system (Gemini) would search the Japanese-language web for sources, retrieving results that best match the experiential nature of the query. For this query, the results would cite 62.1% non-OTA sources (364 of 590 citations, based on the 364 citations from Japanese experiential queries), with the largest non-OTA categories being:

1. Hotel direct websites (23.6%)
2. Editorial curation (12.3%)
3. Travel agency sites (12.9%)
4. Coworking/workspace platforms (8.3%)

For comparison, a Japanese transactional query like "新宿で安いホテル" (meaning "Cheap hotel in Shinjuku") would cite 70.0% OTA sources (333 of 476 citations), with only 30.0% non-OTA.

The key insight is that the experiential framing of the query activates different content types, leading to a substantially higher rate of non-OTA citations. This means hotels with deeply informative content about their neighborhood, accessibility, and atmosphere are more likely to be directly cited by AI systems for experiential queries, while transactional queries still heavily favour OTAs.

## References

- Peiying Zhu, Sidi Chang, "The End of Rented Discovery: How AI Search Redistributes Power Between Hotels and Intermediaries", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20062

Tags: #information-retrieval #hotel-discovery #ai-search #query-intent #content-depth
