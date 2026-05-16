---
title: "MOSS-TTSD: Text to Spoken Dialogue Generation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19739"
---

## Executive Summary
MOSS-TTSD is a novel spoken dialogue synthesis model that generates long-form, multi-party conversations from dialogue scripts with explicit speaker tags, supporting up to 60 minutes of single-pass synthesis without stitching artifacts. It uniquely addresses cross-turn acoustic consistency and speaker identity preservation through a combination of discrete speech tokenization and zero-shot voice cloning from short reference audio clips.

## Why This Matters for Practitioners
If you're building voice applications that require natural multi-speaker conversations, such as podcast production tools, dynamic commentary systems, or interactive entertainment platforms, MOSS-TTSD directly solves the critical problem of maintaining speaker identity across long conversations without manual editing. Unlike current TTS solutions that require stitching multiple single-speaker segments together (which causes audible artifacts), this model generates cohesive 60-minute dialogues in a single pass. Engineers should consider integrating this approach when developing voice applications requiring multi-character consistency, as it eliminates the need for complex post-processing pipelines that add latency and cost. For implementation, start by evaluating the TTSD-eval framework for your own voice synthesis quality assessment, rather than relying on traditional speaker diarization tools that introduce error.

## Problem Statement
Current text-to-speech systems resemble a single-actor monologue, where each utterance is generated in isolation. Imagine trying to produce a full play where each actor's lines are recorded separately and then stitched together, the dialogue would sound unnatural, with inconsistent timing, tone shifts between lines, and abrupt transitions where characters switch. For multi-speaker content like podcasts or audiobooks, existing models fail to maintain speaker-specific prosody and timbre across turns, resulting in robotic, disconnected audio that lacks the organic rhythm of real conversation.

## Proposed Approach
MOSS-TTSD follows a fully discrete speech generation paradigm. It uses Qwen3-8B-base as its autoregressive backbone and MOSS-Audio-Tokenizer to convert audio into discrete tokens, with only the first 16 RVQ layers modelled for efficient long-context generation. The system is conditioned on dialogue scripts with explicit speaker tags (e.g., [S1]/[S2]) and optional per-speaker reference audio inputs, enabling natural turn-taking and zero-shot voice cloning. The core innovation lies in how it handles multi-speaker context across long conversations without requiring explicit speaker diarization.

```python
def generate_dialogue(script, reference_audios=None):
    # Process script with speaker tags to extract speaker sequence
    speakers = extract_speaker_sequence(script)
    
    # Prepare reference audio conditioning for each speaker
    if reference_audios:
        reference_embeddings = [
            extract_speaker_embedding(ref_audio) 
            for ref_audio in reference_audios
        ]
    
    # Generate discrete tokens using multi-head delay pattern
    tokens = []
    for i in range(max_sequence_length):
        # Predict next token based on previous tokens and speaker context
        next_token = predict_token(
            context=tokens[-16:],  # 16-token context window
            speaker_idx=speakers[i],
            reference=reference_embeddings[speakers[i]]
        )
        tokens.append(next_token)
    
    # Convert tokens to audio using MOSS-Audio-Tokenizer
    audio = decode_tokens(tokens)
    return audio
```

## Key Technical Contributions
MOSS-TTSD introduces several novel mechanisms that solve the core challenges of multi-speaker dialogue synthesis:

1. **Multi-head delay pattern for efficient long-context modelling** - Unlike standard autoregressive models that process tokens sequentially, MOSS-TTSD uses a multi-head delay pattern to predict multiple future tokens simultaneously while maintaining cross-turn consistency. This allows the model to process long conversations with minimal context loss, as it can predict multiple tokens ahead rather than waiting for each token to be processed individually.

2. **Explicit speaker conditioning through reference audio integration** - The model combines two voice cloning approaches: (a) explicit voice reference conditioning from short clips, and (b) continuation-based voice cloning from previous audio segments. This dual approach ensures speaker identity consistency across turns without requiring additional training for each speaker, as shown by the 7.8% improvement in SIM (speaker similarity) over VibeVoice 7B.

3. **TTSD-eval objective evaluation framework** - The authors developed a novel evaluation method based on forced alignment that calculates speaker attribution accuracy (ACC) and speaker similarity (SIM) without relying on speaker diarization tools. This eliminates the error accumulation that plagues traditional approaches as the number of speakers increases, with MOSS-TTSD achieving 96.26% ACC in English compared to VibeVoice 7B's 95.54%.

## Experimental Results
MOSS-TTSD outperforms both open-source and proprietary models across all evaluation metrics in Table 1. In Chinese, it achieves 95.87% speaker attribution accuracy (ACC) and 0.7949 speaker similarity (SIM), significantly outperforming VibeVoice 7B (92.22% ACC, 0.7590 SIM). For English, MOSS-TTSD reaches 96.26% ACC and 0.7326 SIM, beating VibeVoice 7B's 95.54% ACC and 0.7140 SIM. The model also demonstrates superior intelligibility with 9.88% Word Error Rate (WER) compared to VibeVoice 7B's 9.46% WER.

Subjective evaluations (Figure 5) confirm these results, with MOSS-TTSD winning 44.4% of Chinese comparisons against Eleven V3 and 45.5% of English comparisons against Gemini 2.5 Pro. The strong correlation between TTSD-eval scores and human perception validates the reliability of the new evaluation framework.

## Related Work
MOSS-TTSD builds upon MOSS-TTS, extending it from single-speaker to multi-speaker dialogue synthesis. Unlike traditional TTS systems that focus on short, single-utterance generation, it specifically addresses the dialogue context gap mentioned in the introduction. It improves upon existing multi-speaker approaches like VibeVoice by implementing zero-shot voice cloning without requiring speaker-specific training data. The paper positions TTSD-eval as a necessary replacement for traditional evaluation metrics like cpWER and cpSIM, which introduce diarization errors when applied to multi-speaker content.

## Limitations
The model is limited to five speakers, which may be insufficient for complex group conversations. The paper doesn't explicitly test extreme audio conditions, such as high-background noise environments, though the authors mention their denoising pipeline (MossFormer2) for specific noisy domains. The subjective evaluation only used 50 dialogue samples per language, which might not be representative of all use cases. The paper doesn't address computational requirements for real-time generation, though it achieves 60-minute single-pass synthesis.

## Appendix: Worked Example
Let's walk through MOSS-TTSD's voice cloning process with a concrete example. Consider a 200-word dialogue script with two speakers (S1 and S2) and a 5-second reference audio clip for each speaker:

```
[S1] Hello, what do you think about the new model's performance?
[S2] I've been testing it for three days and it's quite impressive for the task.
[S1] How about the latency? Is it under 200ms?
[S2] Yes, the inference time is around 180ms for a 200-word response.
```

The model processes this as follows:

1. **Reference audio embedding** - The 5-second reference audio for S1 (from a different recording) is converted into a speaker embedding using wespeaker-SimAMResNet100, resulting in a 100-dimensional vector.

2. **Script tokenization** - The dialogue script is tokenized into 327 tokens (average 1.6 tokens per word), with speaker tags preserved as [S1] and [S2] tokens.

3. **Multi-head delay pattern** - The model processes the script in chunks of 16 tokens, with each prediction using the previous 16 tokens. For the first 16 tokens, it relies on the reference audio embedding, but for subsequent chunks, it uses the previous audio segments for continuity.

4. **Speaker conditioning** - When generating S1's utterance, the model conditions on the S1 reference embedding and maintains the same timbre throughout. For S2's utterance, it switches to the S2 reference embedding.

5. **RVQ token generation** - The model generates RVQ tokens for the audio, modelling only the first 16 layers. This allows it to maintain the audio quality while reducing computational load, MOSS-Audio-Tokenizer operates at 2 kbps with a 12.5 Hz frame rate.

The result is a single 60-second continuous audio clip that maintains consistent speaker timbre across the dialogue without any noticeable stitching artifacts, unlike previous approaches that would require multiple separate TTS generations and manual editing to create a coherent conversation.

## References

- **Code:** https://github.com/OpenMOSS/MOSS-TTSD
- Yuqian Zhang, Donghua Yu, Zhengyuan Lin, Botian Jiang, Mingshu Chen, Yaozhou Jiang, Yiwei Zhao, Yiyang Zhang, Yucheng Yuan, Hanfu Chen, Kexin Huang, Jun Zhan, Cheng Chang, Zhaoye Fei, Shimin Li, Xiaogui Yang, Qinyuan Cheng, Xipeng Qiu, "MOSS-TTSD: Text to Spoken Dialogue Generation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19739

Tags: #voice-generation #multi-speaker-synthesis #zero-shot-voice-cloning #long-context-synthesis #speech-evaluation
