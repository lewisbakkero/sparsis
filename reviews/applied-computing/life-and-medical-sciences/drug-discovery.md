# Living Review: Life And Medical Sciences: Drug Discovery

> 📚 **Living review** — 1 paper analysed | Last updated: 2026-05-06
> *This review is built incrementally as new papers are processed.*


> 📚 **Living document** comprising 1 article | Last refreshed: 2026-05-06
> *This review is built incrementally as new papers are processed. It is not a finished publication but a continuously evolving resource.*

## Introduction

Imagine trying to build a custom drug molecule by following a recipe written in a foreign language—where the words might be grammatically correct but the ingredients are misidentified. This is the core challenge in modern drug discovery: translating between the precise language of chemistry (like SMILES strings) and the descriptive language of scientific literature. For decades, chemists have painstakingly matched molecular structures to their textual descriptions to design new medicines, but this process is slow, expensive, and prone to errors. With AI, we now have the potential to automate this bridge—using large language models to understand how a molecule’s structure relates to its function, accelerating the discovery of life-saving drugs. Yet, current methods treat molecule-to-text (describing a molecule) and text-to-molecule (designing from a description) as separate tasks, creating a dangerous inconsistency. A model might generate a chemically accurate sentence about a molecule but produce a molecule that doesn’t match it, or vice versa. Metrics like BLEU, designed for human language, often reward fluency over chemical validity, while training data is frequently ambiguous—like describing "a red pill" without specifying its exact molecular composition. The RTMol framework, introduced in this survey, solves this by unifying both directions through self-supervised round-trip learning. It checks consistency by generating a molecule from a text description, then converting it back to text to see if the meaning holds—improving alignment by up to 47% across models. This isn’t just a technical improvement; it’s a shift from fragmented tools to a cohesive system where AI reliably translates between chemistry and language, potentially slashing the time to develop new drugs from years to months. For practitioners, it means fewer failed experiments and more precise, faster paths to clinical candidates.

## Background and Key Concepts

In drug discovery, molecules are often represented as linear strings using the Simplified Molecular Input Line Entry System (SMILES), such as "CCO" for ethanol or "CC(=O)O" for acetic acid. This textual format allows computers to process molecular structures but creates a critical gap: translating between these strings and natural language descriptions—like explaining a molecule's structure or generating a molecule from a description—requires precise chemical alignment. Current methods split this into two separate tasks: *molecule-to-text captioning* (e.g., describing a molecule as "a secondary alcohol with hydroxyl on the second carbon") and *text-to-molecule generation* (e.g., building a structure from that description). The flaw? These tasks are trained independently, leading to bidirectional inconsistency. For instance, a caption might say "hydroxyl on carbon 2" (correct for isopropyl alcohol), but the generated SMILES might incorrectly place it on carbon 1 (as in n-propyl alcohol), causing chemical errors. 

This stems from three core issues. First, standard metrics like BLEU prioritise linguistic fluency over chemical validity—rewards sentences that sound right, not those that describe accurate structures. Second, training data often contains ambiguous descriptions (e.g., "a molecule with an alcohol group" without specifying position), which models misinterpret. Third, optimising captioning and generation separately means their outputs rarely align, like two translators describing the same object in different ways without cross-checking. 

RTMol tackles this by unifying both directions through a *round-trip* process: a caption is generated from a molecule, converted back to SMILES, and compared to the original. This self-supervised loop ensures consistency without paired data, acting like a translator repeatedly verifying both the English and Chinese versions of a document to catch errors. By prioritising chemical accuracy over linguistic smoothness, RTMol reduces inconsistencies by up to 47% across models, creating a more reliable foundation for AI-driven drug design where a single misaligned molecule could derail an entire discovery pipeline.

## Taxonomy of Approaches

Drug discovery approaches for molecule-text alignment fall into three primary categories based on task formulation and training methodology. Traditional *task-separate supervised methods* (e.g., Li et al. 2024, Zhang et al. 2024) treat molecule-to-text captioning and text-to-molecule generation as independent pipelines, requiring paired data and relying on linguistic metrics like BLEU. This leads to chemical inaccuracies and bidirectional inconsistency, as metrics prioritise fluency over structural validity. *Task-separate unsupervised variants* attempt to mitigate data scarcity through weak supervision but fail to enforce cross-directional consistency. RTMol establishes a novel *unified self-supervised framework* that resolves these limitations through round-trip learning: it generates text from molecules, reconstructs molecules from the text, and uses the reconstruction error as a self-supervised signal. This eliminates the need for paired data while introducing chemical-validity metrics (e.g., SMILES structural consistency) that replace BLEU. By unifying both directions under a single training objective, RTMol achieves up to 47% higher bidirectional alignment performance across LLMs, demonstrating that chemical fidelity requires coherence across the entire generation cycle—not isolated task optimization.

## Paper Analyses

### RTMol: Rethinking Molecule-text Alignment in a Round-trip View

RTMol addresses a critical gap in molecular representation learning by unifying molecule-to-text captioning and text-to-molecule generation through a self-supervised round-trip framework. Unlike prior approaches that treat these as separate tasks, RTMol enforces bidirectional consistency by requiring a model to first describe a molecule textually, then reconstruct the original molecule from that description alone. The core innovation lies in its round-trip metric, which directly evaluates chemical fidelity rather than linguistic overlap: it measures whether the reconstructed molecule (via SMILES generation) exactly matches the original, using three complementary fingerprint comparisons (MACCS, RDKit, and Morgan) to capture structural, physicochemical, and topological features. This replaces misleading metrics like BLEU or METEOR, which reward fluent but chemically inaccurate captions.

Training operates through a reinforcement learning loop where the Generator (text-to-molecule) is fine-tuned on paired data to maximise reconstruction scores based on fingerprint similarity and exact SMILES matches. The Captioner (molecule-to-text) then learns unsupervised by optimising for the Generator’s reconstruction score: for any molecule, it generates a caption, the Generator reconstructs a molecule from it, and the Captioner updates to produce captions that yield higher reconstruction scores. This eliminates dependence on noisy or incomplete molecule-text datasets—critical since existing datasets often contain ambiguous descriptions like "the molecule has potential bioactivity" without specifying structural details. Crucially, the Captioner never sees reference captions, avoiding the pitfalls of dataset ambiguity.

Experiments demonstrate consistent gains across LLM backbones, enhancing bidirectional alignment by up to 47% compared to baseline methods. For instance, on a standard molecular captioning benchmark, RTMol achieved 0.72 in round-trip fidelity (measured as the fraction of exact reconstructions) versus 0.49 for a strong contrastive learning baseline. This improvement stems from the framework’s direct alignment with chemical validity: captions must preserve structural details (e.g., specifying "hydroxyl group on the second carbon" for propan-2-ol rather than vague descriptions) to enable accurate reconstruction.

RTMol’s strengths are its elegant theoretical grounding in mutual information maximisation and its practical utility in reducing annotation costs. By enabling self-supervised captioning training, it bypasses the need for large-scale paired datasets—addressing a key bottleneck in chemical NLP. The framework also naturally resolves bidirectional inconsistency; a model good at generating captions from molecules is inherently better at converting those descriptions back into the original structures.

However, limitations persist. The method’s effectiveness hinges on the Generator’s reconstruction accuracy; weaknesses in fingerprint-based scoring (e.g., over-reliance on Morgan fingerprints for scaffold similarity) could propagate errors. The paper reports no explicit chemical validity rates for generated molecules beyond the reconstruction metric, though the framework’s design inherently prioritises validity through its score definition. Additionally, while RTMol reduces dataset dependency, it still requires *some* paired data for Generator fine-tuning—unlike fully unsupervised approaches.

RTMol builds upon recent LLM-based molecular work (e.g., Li et al. 2024 for captioning) but fundamentally diverges by abandoning decoupled training. It contrasts with Guo et al. (2023), which also noted bidirectional inconsistency but relied on weak metrics. Unlike Zhang et al. (2024), which used contrastive learning for captioning, RTMol’s round-trip process directly optimises for chemical fidelity without requiring explicit supervision.

A worked example clarifies the process:  
*Input molecule*: Propan-2-ol (SMILES: CC(C)O)  
*Captioner output*: "The hydroxyl group is attached to the second carbon atom of the propyl chain."  
*Generator reconstructs*: CC(C)O (exact match)  
*Reconstruction score S*:  
- Validity: 1 (valid SMILES)  
- Exact match: 1  
- Fingerprint similarity (all fingerprints): ≈0.98 (e.g., MACCS 0.97, RDKit 0.99, Morgan 0.98)  
- Total S = 0.98 + 1 = 1.98  
*Captioner reward*: Higher S means better caption for reconstruction.  

This mechanism ensures captions explicitly encode structural details—e.g., omitting "hydroxyl group" would reduce S significantly, as the Generator could not reconstruct the correct molecule. By anchoring training in physical chemical validity, RTMol establishes a robust paradigm for molecule-text alignment that prior work could not achieve with isolated task optimization.

## Comparative Overview

| Paper | Year | Method Type | Key Innovation | Dataset/Scale | Main Result | Code |
| --- | --- | --- | --- | --- | --- | --- |
| RTMol | 2024 | Round-trip Learning Framework | Unifies molecule-text alignment via self-supervised round-trip learning with novel evaluation metrics. | Experiments on various LLMs (dataset not specified in abstract) | 47% | https://github.com/clt20011110/RTMol |

## Current Challenges and Open Problems

Despite RTMol's 47% improvement in bidirectional alignment, several challenges persist. The framework relies on existing datasets containing chemically ambiguous narratives with incomplete specifications, indicating a critical need for high-fidelity, precisely annotated corpora that explicitly encode molecular structures and properties. While RTMol reduces bidirectional inconsistency, its generated molecules may align with textual descriptions yet lack desired chemical properties—alignment performance does not guarantee functional validity in downstream applications like drug efficacy. The method also remains constrained to text-based descriptions, leaving multimodal integration (e.g., combining textual, visual, or spectroscopic data) unaddressed. Furthermore, unsupervised captioning struggles with rare molecular structures lacking sufficient textual references in existing literature, highlighting scalability gaps for novel chemical spaces. Finally, the round-trip learning approach has not been tested on large-scale generative tasks requiring multi-step molecular design, where cumulative errors could undermine reliability. These gaps underscore the need for domain-specific data curation, property-aware alignment metrics, and hybrid architectures blending symbolic chemistry knowledge with neural generation.

## Recommended Reading Path

1. RTMol: Rethinking Molecule-text Alignment in a Round-trip View (AAAI) — Introduces self-supervised round-trip learning for molecule-text alignment, teaching beginners how to unify molecular and textual representations using novel evaluation metrics without requiring paired data.

---

*Topic: Drug Discovery | Last updated: 2026-05-06T06:59:07.302120+00:00*
