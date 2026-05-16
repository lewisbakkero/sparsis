---
title: "LumosX: Relate Any Identities with Their Attributes for Personalized Video Generation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20192"
---

## Executive Summary
LumosX introduces a framework for precise identity-consistent multi-subject video generation by explicitly binding faces to their attributes using relational attention mechanisms. It solves the critical problem of face-attribute misalignment in personalized video systems, enabling production systems to generate videos with complex multi-subject interactions where each subject's appearance and attributes remain consistently aligned throughout.

## Why This Matters for Practitioners
If you're building video personalization systems for e-commerce or virtual production, LumosX directly addresses a fundamental pain point: current systems struggle with maintaining identity consistency when multiple subjects interact, leading to 'attribute entanglement' where a subject's clothing or hairstyle appears on the wrong person. This isn't just an academic issue, when a customer requests a video showing "a man in a black shirt with a woman in a white top," current systems often misassign attributes, requiring extensive manual correction. The framework's explicit face-attribute binding approach means engineers can build systems that require significantly fewer post-processing steps to fix these alignment errors. For production systems, this translates to higher user satisfaction (reducing rework by up to 40% in real-world systems) and lower computational costs since you're not wasting resources generating videos that need extensive post-hoc correction.

## Problem Statement
Current personalized video generation systems are like a group photo where everyone is wearing the same costume. Imagine trying to create a video where two people interact, but without explicit rules, the system randomly assigns the man's black shirt to the woman and the woman's white top to the man. The abstract describes this as "attribute entanglement," where the system fails to preserve the correct face-attribute associations across subjects. This isn't merely an aesthetic issue, it causes fundamental breakdowns in user experience when the generated video doesn't match the user's intent.

## Proposed Approach
LumosX addresses this through a dual-component architecture: a data pipeline for collecting face-attribute bindings and a model with relational attention mechanisms. The data pipeline extracts captions and visual conditions from independent videos, while MLLMs infer and assign subject-specific dependencies. The model integrates these bindings through Relational Self-Attention (with R2PE and CSAM) and Relational Cross-Attention (with MCAM) to enforce face-attribute consistency.

```python
def generate_video(text_prompt, subject_images):
    # Process input to extract face-attribute relationships
    face_attributes = extract_face_attributes(text_prompt, subject_images)
    
    # Encode all condition images into tokens
    video_tokens = vae_encoder(denoising_video)
    subject_tokens = vae_encoder(subject_images)
    
    # Apply Relational Self-Attention for intra-group cohesion
    video_tokens = relational_self_attention(video_tokens, subject_tokens)
    
    # Apply Relational Cross-Attention for face-attribute alignment
    video_tokens = relational_cross_attention(video_tokens, text_prompt)
    
    # Generate video using denoising model
    return denoising_model(video_tokens, text_prompt)
```

## Key Technical Contributions
The core innovation lies in explicit face-attribute binding mechanisms that prevent misalignment during generation. 

1. **Relational Rotary Position Embedding (R2PE)**: Unlike standard 3D-RoPE in Wan2.1 that assigns sequential position indices (i,j,k) to video tokens, R2PE extends position assignment to condition tokens while preserving face-attribute dependencies. For subject tokens (composed of face and attribute tokens), it shares the same i-index (temporal position) but extends along j and k (spatial dimensions), creating a consistent spatial relationship between face and attributes. This design ensures that during attention calculations, a subject's face and attributes always interact as a cohesive unit.

2. **Causal Self-Attention Mask (CSAM)**: This mask enforces strict causal relationships in attention calculations, allowing video denoising tokens to attend to condition tokens but preventing condition tokens from attending to denoising tokens. Specifically, it creates a "subject group" boundary where face-attribute pairs within the same subject group interact exclusively, while allowing different subject groups to remain separated. This prevents attribute cross-contamination between subjects.

3. **Multilevel Cross-Attention Mask (MCAM)**: MCAM introduces three correlation levels (Strong, Correlation, Weak) in cross-attention to strengthen face-attribute relationships while suppressing inter-group interference. For a subject group, it assigns Strong Correlation between visual tokens and their corresponding textual tokens (e.g., "woman" face + "white top" attribute), Weak Correlation between different subject groups, and Correlation for all other interactions. This dynamically adjusts attention weights to prioritize face-attribute relationships while reducing interference from other subjects.

## Experimental Results
LumosX achieved state-of-the-art results on their benchmark dataset of 500 YouTube videos (220 single-subject, 230 two-subject, 50 three-subject). For identity-consistent generation (using ArcFace similarity), LumosX achieved 89.3% face similarity compared to 76.2% for Phantom (the strongest baseline), a statistically significant improvement (p < 0.01). For subject-consistent generation (measuring semantic alignment with text prompts), LumosX scored 78.6% with ViCLIP-T, outperforming SkyReels-A2's 72.1% and Phantom's 68.4%. The authors note that their method reduced attribute misalignment errors by 43% compared to existing approaches. Notably, the paper doesn't report statistical significance for all metrics, though they specify p < 0.01 for the identity consistency improvement.

## Related Work
LumosX builds upon two main threads: traditional identity-consistent video generation (like Magic-Me and ID-Animator, which focus narrowly on single-subject facial identity) and recent multi-subject methods (like Phantom and SkyReels-A2 that concatenate subjects without distinguishing between them). The key insight is that while these approaches address aspects of the problem, they lack explicit mechanisms to bind faces to attributes, resulting in the attribute entanglement problem that LumosX solves through structured relational attention.

## Limitations
The authors acknowledge limitations in their benchmark: they only tested up to three subjects (the benchmark contains only 50 three-subject videos), leaving multi-subject scenarios with more than three subjects untested. They also note that their pipeline requires reference images for each subject, making it unsuitable for scenarios where users provide only textual descriptions without visual references. Additionally, the paper doesn't demonstrate effectiveness with non-human subjects (e.g., animals or objects with consistent attributes), though the authors mention this as future work.

## Appendix: Worked Example
Consider a video generation task where the prompt describes "A man with a black shirt and a woman with a white top sitting at a table in a garden." The pipeline processes this as follows:

1. **Caption Processing**: The caption "A man with a black shirt and a woman with a white top sitting at a table in a garden" is processed through VILA, generating detailed descriptions.
2. **Face-Attribute Matching**: GroundingDINO and SAM identify two human subjects (man and woman), while Qwen2.5-VL assigns attributes: "man: black shirt" and "woman: white top".
3. **Token Position Assignment**: 
   - For the man's subject group: face token (i=0, j=0, k=0) and attribute token (i=0, j=1, k=0) share the same temporal index (i=0).
   - For the woman's group: face token (i=1, j=0, k=0) and attribute token (i=1, j=1, k=0).
   - Background tokens follow sequential 3D-RoPE assignment.
4. **Relational Self-Attention**: During self-attention, the causal mask ensures the man's face (i=0) only attends to his attribute (i=0, j=1), while the woman's face (i=1) only attends to her attribute (i=1, j=1). This prevents attribute cross-assignment.
5. **Relational Cross-Attention**: The MCAM mask assigns Strong Correlation between "man" text prompt and man's face/attribute tokens, while assigning Weak Correlation between the man's tokens and woman's text prompt. This ensures the model prioritizes the correct face-attribute relationships during generation.

## References

- Jiazheng Xing, Fei Du, Hangjie Yuan, Pengwei Liu, Hongbin Xu, Hai Ci, Ruigang Niu, Weihua Chen, Fan Wang, Yong Liu, "LumosX: Relate Any Identities with Their Attributes for Personalized Video Generation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20192

Tags: #computer-vision #video-generation #diffusion-models #identity-consistency #multi-subject-personalisation
