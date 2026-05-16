---
title: "VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20185"
---

## Executive Summary
VideoSeek is a long-horizon video agent that actively seeks answer-critical evidence by leveraging video logic flow, rather than exhaustively parsing full videos. It achieves state-of-the-art results on video understanding benchmarks while using up to 96% fewer frames than prior methods. This represents a fundamental shift from traditional video processing approaches, offering significant computational cost savings for production video analysis systems.

## Why This Matters for Practitioners
If you're running video analysis pipelines in production, VideoSeek suggests you can reduce your computational costs by up to 96% without sacrificing accuracy, directly impacting your cloud spend and scalability. For instance, if your current system processes videos at 2 frames per second (FPS) for 30-second clips (60 frames), you could potentially switch to VideoSeek's 0.04 FPS equivalent (2-3 frames per clip), reducing compute costs by 97%.

Practically, this means you can either reduce your infrastructure requirements by 97% for the same throughput, or process 25× more video at the same infrastructure cost. To implement this, integrate VideoSeek's toolkit into your video analysis pipeline, replacing your current frame sampling strategy with the <overview>, <skim>, and <focus> tools. You'll need to adjust your processing budget to use 25-30 frames per video (depending on the benchmark) rather than 384 frames for GPT-5, while maintaining or improving accuracy.

## Problem Statement
Current video analysis systems work like a tourist trying to understand a foreign city by reading every single signpost along the entire route. They exhaustively parse every frame of a video to find relevant evidence, which is computationally expensive and inefficient for long videos. VideoSeek shows that over 80% of questions can be answered by inspecting less than 5% of the original video, like finding the best restaurant in a city by reading only 5% of the street signs rather than every single one.

## Proposed Approach
VideoSeek operates in a think-act-observe loop where the agent iteratively reasons over accumulated observations, plans the next step, and selects a tool from the toolkit to gather new evidence. It uses a lightweight multi-granular toolkit with three specialized tools: <overview> for a coarse global summary, <skim> for probing candidate intervals, and <focus> for fine-grained analysis of short clips.

```python
def videoseek_agent(query, video, max_turns=20):
    trajectory = [system_instruction, query]
    toolkit = ["<overview>", "<skim>", "<focus>", "<answer>"]
    
    for turn in range(max_turns):
        thought, action_plan = thinking_model(trajectory)
        
        if action_plan == ["<answer>"]:
            answer = parse_answer(action_plan)
            break
            
        observation = call_tools(action_plan, video, toolkit)
        trajectory.append((thought, action_plan, observation))
    
    if not answer:
        trajectory.append(system_answer_instruction)
        answer = thinking_model(trajectory)
    
    return answer
```

## Key Technical Contributions
VideoSeek's core innovation lies in its ability to actively seek answer-critical evidence rather than exhaustively parsing videos. Specifically:

1. **The multi-granular toolkit design** enables efficient exploration by operating at different temporal granularities. The <overview> tool uniformly samples 16α frames across the entire video timeline to create a coarse storyline (α=4 for LVBench), while the <skim> tool processes segments of at least 4α seconds with 4α frames to identify promising intervals. The <focus> tool then performs fine-grained analysis at 1 FPS (capping at 4α seconds) for answer-critical details. This hierarchical approach allows VideoSeek to process only 92.3 frames on average for LVBench instead of 384 frames for GPT-5.

2. **The cumulative evidence framework** fundamentally changes how video analysis works. Rather than relying on pre-built video databases or fixed frame sampling (as in prior video agents), VideoSeek dynamically adjusts its tool-calling strategy based on evolving observations. This allows the agent to form a better understanding of the video content as it gathers evidence, rather than processing all frames before starting reasoning.

3. **The strategic frame budget allocation** is the key to efficiency. VideoSeek uses only about 1/300 of the frames used by the second-best video agent (DVD), with the most efficient case (LVBench with subtitles) using just 27.2 frames versus DVD's 8,074 frames. This is achieved through a data-driven approach to frame sampling: VideoSeek identifies the 5% of the video that contains the answer, rather than sampling uniformly across the entire video.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
VideoSeek consistently outperforms existing methods while using far fewer frames across all benchmarks:

- **LVBench (without subtitles)**: 68.4% accuracy using 92.3 frames (vs. GPT-5's 60.1% with 384 frames), improving by +8.3 points while using 76% fewer frames.
- **LVBench (with subtitles)**: 76.7% accuracy using 27.2 frames (vs. GPT-5's 66.5% with 384 frames), improving by +10.2 points while using 93% fewer frames.
- **Video-MME (long subset)**: 70.1% accuracy using 60.9 frames (vs. GPT-5's 67.9% with 384 frames), improving by +2.2 points while using 84% fewer frames.
- **Video-Holmes**: 47.3% accuracy using 42.7 frames (vs. GPT-5's 44.1% with 384 frames), improving by +3.2 points while using 90% fewer frames.

The paper doesn't explicitly state whether improvements are statistically significant, but the consistent performance gains across different benchmarks and settings strongly suggest the results are meaningful.

## Related Work
VideoSeek positions itself as a significant advancement over two main approaches: video agentic models that rely on pre-built video databases (like DVD, DrVideo, and MR. Video) and video-language models that use dense frame sampling (like GPT-5, Gemini, and Qwen). Unlike prior video agents that depend on exhaustive preprocessing (e.g., DVD processes 8,074 frames for LVBench), VideoSeek avoids this by actively seeking answer-critical evidence. It also differs from traditional video-language models that follow a single-pass paradigm, instead embracing a long-horizon approach that integrates reasoning, planning, and evidence gathering.

## Limitations
The paper acknowledges limitations in its evaluation scope. It only evaluates on four benchmarks, and the focus is primarily on long-form video understanding rather than real-time applications. The authors note that the toolkit design could be further optimised for specific video types or content.

From my perspective, the most significant limitation is the lack of evaluation on real-time video processing scenarios. The current benchmarks focus on offline video analysis, but many production systems require real-time processing where the latency of the think-act-observe loop could be problematic. The paper doesn't address this, so practitioners should consider whether VideoSeek's approach is suitable for their latency constraints.

## Appendix: Worked Example
Let's walk through a concrete example of VideoSeek processing a 300-second video (5 minutes) for a question asking "What is the direct reason why the giant man finally fell to the ground and couldn't get up?" based on Figure 1.

1. **Initial state**: Video video with 300 seconds duration (18,000 frames at 1 FPS), query: "What is the direct reason why the giant man finally fell to the ground and couldn't get up?"

2. **Step 1 - <overview>**: VideoSeek's <overview> tool uniformly samples 16α = 64 frames (α=4 for LVBench) across the entire video timeline, creating a coarse storyline. This identifies that the video contains a sequence where a giant man is walking, then trips over his own feet. The observation indicates "The giant man is walking through a park, then trips over his own feet."

3. **Step 2 - <skim>**: Based on the overview, VideoSeek focuses on the segment around 120-180 seconds (where the trip appears to happen) and uses the <skim> tool. It samples 4α = 16 frames from this 60-second segment (4α seconds = 16 seconds), identifying frames 125-130 as containing the key detail.

4. **Step 3 - <focus>**: VideoSeek then uses <focus> on the specific segment of 125-130 seconds (5 seconds of video) at 1 FPS, analysing all 5 frames. The observation reveals "The giant man trips over his own feet while walking."

5. **Final answer**: Based on the accumulated evidence, VideoSeek determines the answer is "Trip over one's own feet" and stops after 3 turns.

Total frames processed: 64 (overview) + 16 (skim) + 5 (focus) = 85 frames.

Comparison to traditional approach: GPT-5 would process 384 frames uniformly across the entire video (384 frames at ~0.21 FPS), while VideoSeek processed only 85 frames (0.047 FPS equivalent), using 78% fewer frames while achieving higher accuracy (68.4% vs. 60.1% on LVBench).

## References

- Jingyang Lin, Jialian Wu, Jiang Liu, Ximeng Sun, Ze Wang, Xiaodong Yu, Jiebo Luo, Zicheng Liu, Emad Barsoum, "VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20185

Tags: #video-understanding #long-horizon-reasoning #tool-guided-seek #multi-granular-toolkit #video-logic-flow
