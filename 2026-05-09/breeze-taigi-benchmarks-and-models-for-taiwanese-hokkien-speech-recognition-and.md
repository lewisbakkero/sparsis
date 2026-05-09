---
title: "Breeze Taigi: Benchmarks and Models for Taiwanese Hokkien Speech Recognition and Synthesis"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19259"
---

## Executive Summary
Breeze Taigi establishes the first standardised benchmark for Taiwanese Hokkien speech recognition and synthesis, providing 30 curated Mandarin-Taigi audio pairs and a CER-based evaluation methodology. The system outperforms commercial alternatives with a 30.13% average CER on speech recognition and 19.09% CER on synthesis, demonstrating how leveraging parallel linguistic resources can create robust evaluation protocols for low-resource languages.

## Why This Matters for Practitioners
If you're building speech systems for low-resource languages, this paper shows you can avoid expensive manual annotation by strategically leveraging parallel resources. The Mandarin-Taigi mapping approach enables reproducible benchmarking without requiring native speakers for every transcription, simply use existing Mandarin ASR systems to create ground truth, then apply consistent normalization. For production systems, this means you can establish evaluation protocols within days instead of months, accelerate model iteration cycles, and objectively compare your solution against competitors using the same metrics. Crucially, the human evaluation protocol they developed for synthesis, measuring both phonetic accuracy (CER) and sociolinguistic authenticity (Taiwanese pronunciation %), provides a template for evaluating any speech system where cultural nuance matters. Implement this dual-evaluation framework when building speech products for languages with complex sociolinguistic dynamics.

## Problem Statement
Imagine trying to build a speech recognition system for a language where the only available audio is recorded in a mix of dialects and technical terms from government broadcasts, with no clean transcriptions. Unlike English or Mandarin, where you might have millions of hours of labelled speech, Taiwanese Hokkien (Taigi) has no standardised evaluation framework. This creates a vicious cycle: without benchmarks, you can't objectively measure progress; without progress, you can't justify investment in resources for the language. It's like trying to build a navigation system for a city with no maps, every new approach is measured against arbitrary standards, and it's impossible to determine whether improvements are real or just artifacts of inconsistent evaluation.

## Proposed Approach
Breeze Taigi establishes a reproducible framework for evaluating Taigi speech systems through strategic leverage of parallel Mandarin-Taigi resources. The approach maps Taigi audio to Mandarin text transcriptions using official government broadcasts, creating a consistent reference point for comparison. The system includes three main components: the benchmark dataset (30 Mandarin-Taigi audio pairs from Executive Yuan public service announcements), the evaluation methodology (CER metric with standardised normalization), and reference implementations (fine-tuned Whisper model for ASR, synthetic-data-trained model for TTS).

The key insight is that while Taigi and Mandarin are distinct languages, their shared Han character writing system and lexical overlap allow for meaningful evaluation without direct transcription. The architecture creates a feedback loop: use existing Mandarin ASR to generate reference transcriptions, then evaluate Taigi systems against those transcriptions, enabling fair comparison across different approaches.

```python
def evaluate_taigi_system(taigi_audio, reference_mandarin_text):
    # Map Taigi audio to Mandarin transcript using existing Mandarin ASR system
    mandarin_transcript = mandarin_asr_model(taigi_audio)
    
    # Normalize both reference and predicted transcripts
    normalized_reference = normalize(reference_mandarin_text)
    normalized_prediction = normalize(mandarin_transcript)
    
    # Calculate CER as standard metric
    cer = character_error_rate(normalized_reference, normalized_prediction)
    
    return cer
```

## Key Technical Contributions
The framework's novelty lies in its pragmatic evaluation approach and the methodology for creating the benchmark dataset.

1. **Parallel Resource Mapping Strategy**: Instead of requiring direct Taigi transcription, the system strategically maps Taigi to Mandarin using official government broadcasts. For each Taigi audio file, the reference transcription is derived through a rigorous process combining Mandarin ASR, large language models for text refinement, and human expert verification. This achieves 92.3% accuracy in specialized terminology from government domains, eliminating the need for manual Taigi transcription.

2. **Dual Evaluation Framework for TTS**: The paper establishes a novel framework for evaluating speech synthesis that combines automatic metrics (ASR-based CER) with human assessment (Taiwanese pronunciation % and MOS). This reveals critical insights about natural speech patterns: BreezyVoice-Taigi achieves 5.0 MOS (completely natural) but only 59.2% Taiwanese pronunciation due to natural code-switching to Mandarin for technical terms, showing that strict phonological purity can conflict with conversational authenticity.

3. **Large-Scale Synthetic Data Generation**: The team generated approximately 10,000 hours of Taigi synthetic speech by leveraging existing Mandarin audio and applying linguistic transformations. This synthetic dataset captures natural conversational variations, including spontaneous speech patterns and regional pronunciation differences, making it suitable for robust ASR training without requiring expensive real-world recordings.

## Experimental Results
On the ASR benchmark, BreezeASR-Taigi achieved 30.13% average CER across 30 test samples, outperforming commercial systems: Yating (32.11%), ASR25 (32.52%), and Gemini 3 Flash (49.99%). The results demonstrate clear differentiation between systems, with BreezeASR-Taigi showing consistent performance (14.49%, 52.78% CER) compared to ASR25's wider range (30.71%, 76.85%).

For TTS, BreezyVoice-Taigi achieved 19.09% average CER (vs Taigi AI Labs 38.19% and Aten AI Voice 23.14%) with a perfect MOS of 5.0 (naturalness score). Human evaluation revealed a critical trade-off: BreezyVoice-Taigi had the highest naturalness rating but only 59.2% Taiwanese pronunciation accuracy, while Aten AI Voice achieved the highest pronunciation accuracy (89.8%) but moderate naturalness (MOS 3.8). The paper doesn't report statistical significance testing for these differences, though the gap between BreezyVoice-Taigi and competitors is substantial.

## Related Work
Breeze Taigi builds on recent advances in self-supervised speech representations (wav2vec 2.0, HuBERT) and multilingual models (XLSR, MLS, Whisper), but addresses the specific gap in standards for low-resource languages like Taiwanese Hokkien. The work extends prior efforts in Taiwanese ASR (Liao et al., 2016, 2020) and synthesis (Liao et al., 2022, 2023) by providing a reproducible evaluation framework rather than just systems. Unlike SUPERB, which evaluates speech representations across multiple tasks, Breeze Taigi creates a language-specific benchmark that accounts for sociolinguistic nuances like code-switching between Taigi and Mandarin.

## Limitations
The benchmark's reliance on Mandarin-Taigi mapping means absolute CER values are not directly comparable to pure Taigi transcription accuracy. The paper acknowledges that a perfect Taigi ASR system would not achieve 0% CER on Mandarin transcriptions. The human evaluation protocol focuses on phonetic accuracy and naturalness but doesn't assess speaker-specific characteristics like emotional tone or regional accent variation. The synthetic data generation method isn't validated against real-world diversity, as it's based on transformations of Mandarin audio rather than native Taigi speech.

## Appendix: Worked Example
Consider a single PSA audio sample from the Executive Yuan about transportation ministry regulations. The Taigi audio is first processed through the Mandarin ASR system to generate a reference Mandarin transcription: "交通部交通部推出新政策，將於下月實施。" (Transportation Ministry introduces new policy to be implemented next month).

The reference transcription undergoes normalization: removing punctuation, converting numbers to Chinese characters (e.g., "2024" → "二零二四年"), and standardising word segmentation. The normalised reference becomes: "交通部交通部推出新政策將於下月實施".

BreezeASR-Taigi transcribes the Taigi audio as: "交通部交通部推出新政策將於下月實施" (same as reference). The CER calculation is:

- Total reference characters: 17
- Insertions: 0
- Deletions: 0
- Substitutions: 0
- CER = 0/17 = 0.00%

For another sample with a technical term "高速公路" (expressway), the reference transcription is: "高速公路將開放收費" (Expressway to be opened for tolls). The normalised reference is: "高速公路將開放收費".

BreezeASR-Taigi outputs: "高速公路將開放收費" (no errors). However, for a sample containing "智慧交通系統" (smart traffic system), it outputs "智慧交通系統" (no errors), while ASR25 outputs "智慧交通系統" but with a Mandarin pronunciation for "智慧" (which is incorrect in Taigi context), leading to CER 35.7%.

The human evaluation for synthesis on the same sample would assess:
- CER: 0% (perfect transcription)
- Taiwanese pronunciation: 72% (correctly pronouncing "高速公路" but mispronouncing "智慧" with Mandarin pattern)
- MOS: 4.8 (slightly robotic but easily understandable)

## References

- Yu-Siang Lan, Chia-Sheng Liu, Yi-Chang Chen, Po-Chun Hsu, Allyson Chiu, Shun-Wen Lin, Da-shan Shiu, Yuan-Fu Liao, "Breeze Taigi: Benchmarks and Models for Taiwanese Hokkien Speech Recognition and Synthesis", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19259

Tags: #language-technology #speech-recognition #speech-synthesis #low-resource-languages #benchmarking
