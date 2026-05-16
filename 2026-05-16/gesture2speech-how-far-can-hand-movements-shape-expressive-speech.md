---
title: "Gesture2Speech: How Far Can Hand Movements Shape Expressive Speech?"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19831"
---

## Executive Summary
Gesture2Speech introduces a novel multimodal TTS framework that uses hand gestures as dynamic control signals to modulate speech prosody, enabling temporally aligned expressive speech synthesis. Unlike conventional systems that rely solely on text or reference audio, this approach leverages visual gesture cues to create more natural, embodied speech that mirrors human communication patterns. Practitioners should care because this provides a practical pathway to enhance the expressiveness of production TTS systems without requiring complex text modifications.

## Why This Matters for Practitioners
If you're building a voice assistant, virtual presenter, or dubbing system that currently uses standard TTS, Gesture2Speech offers a concrete path to improve expressive quality by incorporating gesture input from a camera or motion capture. Unlike previous approaches that required manual prosody tuning or added text annotations, this system uses existing video input (e.g., from a webcam during video recording) to automatically adjust prosody. You can integrate this into your pipeline by adding a lightweight gesture feature extractor before your existing TTS model, with minimal impact on latency. For example, when dubbing video content, use the original speaker's hand gestures as input to generate speech that matches their natural rhythm and emphasis patterns, eliminating the need for manual prosody adjustments in post-production.

## Problem Statement
Current TTS systems are like actors who deliver lines with a fixed cadence, no matter how dramatically the director gestures or moves, the voice remains unchanged. In natural human communication, hand gestures and speech prosody are tightly coupled, with gestures providing rhythm, emphasis, and emotional context that directly shapes vocal delivery. Existing TTS systems fail to capture this embodied expressiveness, producing speech that sounds robotic even when intelligible, which is particularly problematic for applications like educational content dubbing where natural prosody is critical for engagement.

## Proposed Approach
Gesture2Speech processes text, reference audio, and gesture features through a multimodal pipeline that dynamically fuses these inputs to condition a speech decoder. The system takes three inputs: text (for semantic content), reference audio (to preserve speaker identity), and gesture features (from video using OpenPose). These inputs are processed through separate encoders before being fused using a Mixture-of-Experts architecture. The fused representation then conditions an LLM-based speech decoder to generate speech that temporally aligns with gestures. A gesture-speech alignment loss ensures fine-grained synchronization between gesture motion and prosodic contours.

```python
def gesture2speech(text, ref_audio, gesture_video):
    # Text encoder
    text_embeddings = BPE(text)
    
    # Audio encoder
    mel = compute_mel(ref_audio)
    speaker_embedding = speaker_encoder(mel)
    
    # Gesture encoder (using OpenPose)
    gesture_keypoints = openpose(gesture_video)
    gesture_features = gesture_encoder(gesture_keypoints)
    
    # Multimodal fusion with MoE
    fused_features = multimodal_moe(
        text_embeddings, 
        speaker_embedding, 
        gesture_features
    )
    
    # LLM decoder with cross-attention
    speech_waveform = llm_decoder(
        text_embeddings, 
        gesture_features, 
        fused_features
    )
    
    return speech_waveform
```

## Key Technical Contributions
Gesture2Speech makes three specific technical contributions that distinguish it from prior work:

1. **Dynamic multimodal fusion via gesture-conditioned MoE**: Unlike previous MoE systems that focused on speaker variation, Gesture2Speech uses gesture features to dynamically route information to specialized experts. The system employs three MoE modules: Speech MoE (for audio features), Video MoE (for gesture features), and Global MoE (for fused representations). Each MoE uses top-2 expert routing with adaptive capacity constraints to balance computational overhead and routing stability.

2. **Gesture-speech alignment loss**: The authors developed a novel Cross-Modal Temporal Distance (CMTD) loss that measures the temporal misalignment between gesture apex points (identified from motion magnitudes) and speech prominence peaks (derived from pitch contours). This loss function (LAL = mean absolute error between gesture and speech peaks) directly optimises for synchronization, achieving 39.3% improvement in gesture offset under different text scenarios compared to hierarchical MoE baselines.

3. **Gesture as primary prosody control signal**: The system treats hand gestures not as supplementary input but as the primary control signal for prosody modulation, which is a fundamental shift from previous approaches. This allows the model to generate speech that is temporally aligned with gestures rather than inferring gestures from speech or treating them as secondary features.

## Experimental Results
Gesture2Speech was evaluated on the PATS dataset (17,747 samples, 34.1 hours of audio) using five model variants. The multimodal MoE model outperformed all baselines in both objective and subjective metrics:

- **Gesture alignment**: 0.9471 gesture offset (vs. 1.0386 for XTTS-V2 baseline) with 39.3% improvement in different text scenarios
- **Mutual information**: 0.0559 (vs. 0.0382 for XTTS-V2 baseline) representing 79.9% gain in gesture-audio coupling
- **Subjective evaluation**: 7.5% improvement in speech quality (81.48 vs. 75.79 MOS) and 9.1% improvement in prosodic similarity (79.35 vs. 72.78 MOS) compared to XTTS-V2
- **Intelligibility**: 17.55% WER (vs. 20.27% for XTTS-V2 baseline)

The results were statistically significant with 95% confidence intervals reported for all metrics. The proposed model consistently outperformed baselines across all evaluation conditions, including different text scenarios where the input text differs from the reference video content.

## Related Work
Gesture2Speech positions itself at the intersection of multimodal TTS and embodied speech synthesis. It builds on prior work in style-disentangled expressive TTS (Jawaid et al. 2024) but extends it to incorporate gesture cues as the primary control signal. The authors note that while gesture generation from speech has received significant attention (e.g., co-speech gesture generation), the reverse paradigm, using gestures to control prosody in TTS, remains underexplored. They explicitly differentiate their work from integrated speech and gesture generation frameworks (Mehta et al. 2024, 2023) by focusing on fine-grained prosodic control rather than simultaneous speech and gesture generation.

## Limitations
The authors acknowledge several limitations: the PATS dataset has limited cultural and emotional scope, and the framework relies on full pose keypoints from video, which may not be available in all production scenarios (e.g., low-resolution video or full-body occlusion). The paper doesn't evaluate the model's performance with low-quality gesture inputs or on datasets from different cultural contexts. Additionally, while the framework integrates gesture features into the TTS pipeline, it doesn't address how to handle scenarios where gesture input is absent or inconsistent with the audio.

## Appendix: Worked Example
Consider a 4-second video clip of a speaker making a confident gesture while saying "This is important." The system processes this as follows:

1. **Input processing**: The video is processed with OpenPose to extract 2D keypoints (J=21 joints), resulting in a sequence of 100 gesture feature vectors (4 seconds × 25 fps).
2. **Gesture apex detection**: Motion magnitudes are calculated, and apex points (where gesture intensity peaks) are identified at 1.2s and 3.1s.
3. **Speech peak detection**: The predicted speech from the decoder identifies pitch prominence peaks at 1.1s and 3.2s.
4. **Alignment loss calculation**: The absolute difference between gesture and speech peaks is |1.2-1.1|=0.1s and |3.1-3.2|=0.1s, resulting in a gesture offset of 0.1s.
5. **Fusion process**: During training, gesture features at 1.2s and 3.1s are routed to specific experts in the Video MoE (top-2 routing), while speech features are routed to the Speech MoE. The fused representation (zstyle-total) combines these with text embeddings to form a conditional input for the LLM decoder.
6. **Output waveform**: The HiFi-GAN vocoder converts the decoded token sequence into a speech waveform where the pitch accent at 1.2s matches the gesture peak, creating natural synchrony.

See Key Technical Contributions for more details on the MoE routing mechanism.

## References

- **Code:** https://github.com/jik876/hifi-gan
- Lokesh Kumar, Nirmesh Shah, Ashishkumar P. Gudmalwar, Pankaj Wasnik, "Gesture2Speech: How Far Can Hand Movements Shape Expressive Speech?", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19831

Tags: #multimodal-ai #speech-synthesis #gesture-recognition #prosody-control #mixture-of-experts
