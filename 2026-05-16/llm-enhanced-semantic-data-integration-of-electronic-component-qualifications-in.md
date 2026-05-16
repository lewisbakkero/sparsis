---
title: "LLM-Enhanced Semantic Data Integration of Electronic Component Qualifications in the Aerospace Domain"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20094"
---

## Executive Summary
This paper presents a hybrid pipeline combining Virtual Knowledge Graphs with Large Language Models to solve data silo challenges in aerospace component qualification. It enables engineers to quickly determine whether electronic components have been qualified for space use by integrating fragmented database sources while maintaining strict compliance with industry standards. Practitioners should care because this approach reduces manual qualification checks by 90% while improving accuracy over pure LLM-based solutions.

## Why This Matters for Practitioners
If you're responsible for maintaining industrial data systems in aerospace or similar high-stakes manufacturing domains, this paper directly addresses a costly and time-consuming bottleneck: the manual process of checking component qualifications across siloed databases. When designing satellite components, engineers currently spend hours cross-referencing spreadsheets to determine if a component has been qualified for space use, risking redundant testing (costing thousands of dollars per component) and compliance failures. The authors demonstrated that their solution can reduce this manual effort by 90%, a metric that translates directly to reduced time-to-market for satellite components. For your next data integration project in regulated industries, consider this hybrid approach instead of pure RAG when your data requires structured querying and historical consistency checks. Specifically, when your data sources contain both structured fields and unstructured text describing key attributes (like the "Notes" column in the Qualification Catalog), this method provides a more reliable path than pure LLM-based approaches.

## Problem Statement
Imagine trying to find a specific book in a library where one section is organised by author, another by subject, and a third by physical location, all without a master catalog. For aerospace engineers, the equivalent problem is determining if an electronic component (like a capacitor) has been qualified for space use: they must cross-reference two disconnected databases (PLM-DB for component specifications and QC for qualification status) that use inconsistent naming conventions and bury critical information in free-text fields. This fragmentation causes engineers to waste weeks manually checking qualifications before production, risking both redundant testing (costing thousands of dollars per component) and non-compliance with space agency standards.

## Proposed Approach
The authors built a three-phase pipeline: data cleansing using LLMs, semantic integration through a Virtual Knowledge Graph (VKG), and data access via SPARQL queries and vector search. The core insight is that aerospace qualification data requires both precise matching (for direct and by-similarity qualifications) and flexible matching (for alternative qualifications), which a single approach cannot handle. The VKG provides a unified semantic layer for structured querying, while the LLM enhances data cleansing and enables the vector search for alternative qualifications.

```python
def extract_part_number(text):
    """Extract candidate part numbers from QC "Notes" column using LLM and validate"""
    candidate_pns = llm_extract(text, "Extract part numbers from this description")
    validated_pns = []
    for pn in candidate_pns:
        if plm_db.matches_attributes(pn, text):
            validated_pns.append(pn)
    if not validated_pns:
        flag_for_human_review(text)
    return validated_pns
```

## Key Technical Contributions
The authors' approach introduces several novel mechanisms that solve specific challenges in industrial data integration:

1. **LLM-assisted normalization with human-in-the-loop validation**: Instead of using LLMs to replace human validation entirely, the system uses a semi-automated workflow where the LLM generates cleaning rules for inconsistent data (like manufacturer names), which are then reviewed and refined by domain experts. For example, the model generates rules like {ABC Corp, ABC, ABC Inc., ABC International} → ABC, which are then validated to prevent hallucinations. This approach reduced the time to normalize 466 unique manufacturer names from several days to a single 2-hour workshop.

2. **Ontop Lens for virtual normalization**: The authors implemented an Ontop Lens to handle inconsistent manufacturer naming without modifying the original database. This relational view joins the database with a SQL table of LLM-generated cleaning rules, allowing all variants to map to a canonical manufacturer name in the VKG. The Lens is used in the mapping as shown in the paper, directly enabling the virtual normalization in the semantic layer.

3. **Hybrid retrieval strategy for different qualification types**: The system uses SPARQL queries for direct and by-similarity qualifications (which require exact attribute matching) and vector search for alternative qualifications (which need flexible, similarity-based matching). This avoids the pitfalls of using a single approach for all qualification types, as the authors demonstrated that pure RAG approaches struggle with the precision required for direct qualifications (F1-score 0.921 vs. 0.619 for alternative qualifications).

4. **Vector embedding for alternative qualification ranking**: Unlike traditional RAG, the authors embed both PLM-DB components and qualification entities into vectors using multilingual-e5-large-instruct, then rank candidates by cosine similarity. For a given component, the system first filters with SPARQL to remove non-compliant qualifications, then ranks the remaining candidates using vector similarity. This approach ensures that only relevant qualifications are presented to human experts for final review.

## Experimental Results
The authors conducted a comparative analysis between their hybrid pipeline and a RAG-based approach. For direct qualifications, their system achieved a precision of 0.947, recall of 0.896, and F1-score of 0.921. For by-similarity qualifications, it achieved 0.969 precision, 0.907 recall, and 0.937 F1-score. The RAG approach's performance was significantly worse, particularly for alternative qualifications where it only achieved 0.516 precision, 0.773 recall, and 0.619 F1-score. The authors reported that the hybrid approach outperformed RAG across all qualification types, with the most significant improvement in alternative qualifications where the RAG system struggled due to the need for flexible matching.

## Related Work
The authors position their work within two established research areas: ontology-based data integration with Virtual Knowledge Graphs (VKGs) and knowledge engineering using LLMs. They cite prior work using VKGs in aerospace (e.g., for PCB test reports at Thales Alenia Space), in oil and gas (Statoil), and in manufacturing (Bosch). For LLM applications, they reference work on ontology engineering, knowledge extraction, and information retrieval in healthcare, scholarly domains, and industrial settings (including prior LLM use at Thales Alenia Space for PCB test reports). The key contribution of this paper is the integration of LLMs into the VKG pipeline specifically for handling the complexities of aerospace component qualification data, which includes both structured attributes requiring exact matching and unstructured text needing flexible matching.

## Limitations
The authors acknowledge several limitations in their approach. The LLM-based data cleansing process still requires human validation for approximately 2% of cases where the Part Number is not present in the "Notes" column, which necessitates manual lookup in qualification documents. Additionally, the vector search for alternative qualifications requires domain expertise to review the top candidates, as the system cannot fully automate this process. The authors also note that the pipeline was specifically designed for the aerospace domain with its strict qualification requirements, so it may not directly translate to other industries with different standards. The paper does not explore the scalability of the solution to much larger datasets beyond the ones used in the case study.

## Appendix: Worked Example
Let's walk through the qualification retrieval process for a specific component. Consider a component with the following attributes from PLM-DB:

- Part Number: P3333333
- Package Code: R1
- Subpackage Code: a3
- Manufacturer: ABC

First, the system checks for direct qualifications by running a SPARQL query to find entries in QC with the same PN, Package Code, Subpackage Code, and Manufacturer. The query would look for:
- Qualification Number: qc3 (matching "R1", "a3", "ABC" after normalization)
- Qualification Status: Ongoing (as found in QC data)

This is a direct qualification, so the system can immediately confirm that this component is qualified for use.

For a scenario requiring by-similarity qualification, consider a component with:
- Part Number: P4444444
- Package Code: R1
- Subpackage Code: a3
- Manufacturer: ABC

The direct qualification query finds no match for P4444444, but the by-similarity query finds a match on Package Code, Subpackage Code, and Manufacturer (R1, a3, ABC), so it identifies a by-similarity qualification (qc3).

For alternative qualifications, the system converts the component and qualification entries to vectors using multilingual-e5-large-instruct. The component vector is:
- [0.12, 0.45, 0.78, ... (1024 dimensions)]

The system then runs a SPARQL query to filter qualifications with matching Package Code (R1) and Manufacturer (ABC), then ranks the remaining candidates (say 20 candidates) by cosine similarity to the component vector. The top candidate might have a cosine similarity of 0.87, while the next would be 0.73. The system then presents the top 10 candidates to a domain expert for final review. This process is significantly more reliable than a pure RAG approach, which would rank qualifications based purely on textual similarity without the structural matching that's critical for aerospace qualification.

## References

- Antonio De Santis, Marco Balduini, Matteo Belcao, Andrea Proia, Marco Brambilla, Emanuele Della Valle, "LLM-Enhanced Semantic Data Integration of Electronic Component Qualifications in the Aerospace Domain", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20094

Tags: #aerospace #data-integration #knowledge-graphs #llm-techniques #semantic-web
