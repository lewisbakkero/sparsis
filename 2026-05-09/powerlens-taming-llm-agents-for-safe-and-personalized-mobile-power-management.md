---
title: "PowerLens: Taming LLM Agents for Safe and Personalized Mobile Power Management"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19584"
---

## Executive Summary
PowerLens is a system that uses LLM agents to optimise mobile device power management through context-aware policy generation and personalized preference learning. It addresses the limitations of current static power management approaches by understanding user activities through UI semantics, adapting to individual preferences via implicit feedback, and ensuring safety through a PDL-based verification framework. For engineers building mobile systems, PowerLens offers a novel approach to balancing energy efficiency with user experience.

## Why This Matters for Practitioners
If you're responsible for mobile device power management, PowerLens demonstrates that LLM-based approaches can achieve significant energy savings (38.8% in practice) while preserving user experience (4.3/5.0 satisfaction rating). Unlike traditional rule-based systems that make blanket decisions (e.g., dimming screen brightness for all activities), PowerLens adapts to specific user contexts (e.g., preserving GPS accuracy during navigation while reducing screen refresh rates during reading). This suggests engineering teams should consider incorporating semantic understanding of user activities into their power management strategies, rather than relying on static rules. For teams using Android OS, implementing a system like PowerLens would require careful integration of accessibility services and LLM agents, with attention to the safety verification layer to prevent disruptive user experiences.

## Problem Statement
Current mobile power management is like a thermostat set to "always cool", it applies the same temperature adjustments regardless of whether the user is cooking (where heat is needed) or sleeping (where cool is preferred). Existing systems either make blanket decisions (like turning down brightness for all activities) or require users to manually configure settings for each scenario, ignoring both the semantic context of what the user is doing and their personal preferences.

## Proposed Approach
PowerLens employs a multi-agent architecture that recognizes user context from UI semantics and generates holistic power policies across 18 device parameters. The system uses a two-tier memory system to learn individualized preferences through implicit feedback, and a PDL-based constraint framework to verify actions before execution.

Here's a simplified pseudocode for the core decision process:
```python
def power_management_decision(user_activity, device_state):
    # Activity Agent: recognise user context
    activity_type, sub_activity = activity_agent.recognise(user_activity)
    
    # Policy Agent: generate strategy considering memory and constraints
    policy = policy_agent.generate(
        activity_type=activity_type,
        device_state=device_state,
        memory=lpm.get_contextual_rules(activity_type),
        safety_constraints=pdl.verify(device_state)
    )
    
    # Execution Agent: verify and execute policy
    verified_policy = execution_agent.verify(policy)
    execution_agent.execute(verified_policy)
    
    # Feedback Agent: detect user overrides
    feedback_agent.detect_overrides(device_state)
    
    return verified_policy
```

## Key Technical Contributions

The key innovations in PowerLens enable context-aware, personalized, and safe power management through three core mechanisms:

1. **The multi-agent architecture for context-aware policy generation**: Unlike previous LLM-based mobile agents that focus on UI-level task automation, PowerLens operates at the system-level resource management layer. The Activity Agent uses Android's Accessibility framework to transform UI trees into semantic context representations (app category, sub-activity, battery level, temporal context), while the Policy Agent synthesizes holistic strategies across 18 device parameters. This approach uniquely bridges the semantic gap between user activities (e.g., "navigation") and optimal system parameters (e.g., "GPS accuracy: high, brightness: medium, refresh rate: 60Hz").

2. **The two-tier memory system for implicit preference learning**: PowerLens learns personal preferences through implicit feedback rather than explicit configuration. The Short-Term Memory (STM) captures immediate user overrides (e.g., when a user manually brightens the screen), while the Extractor distills these observations into Long-Term Personal Memory (LPM) through confidence-based distillation. This system converges within 3-5 days, as the Extractor uses state differencing to detect overrides and promotes observed patterns into stable preference rules based on confidence scores and temporal decay.

3. **PDL-based constraint verification for safe execution**: The PDL-based framework verifies every LLM-generated action against device-specific capabilities and app safety invariants before execution. Unlike raw LLM outputs that produce over 20% problematic actions (as shown in Fig. 2d), PowerLens eliminates 96.5% of safety violations through this two-stage defence: the Policy Agent respects constraints during generation, while the Execution Agent independently verifies compliance. This ensures creative optimisation strategies never violate critical constraints like forcing GPS off during navigation.

## Experimental Results
PowerLens was evaluated on a rooted Android device (OnePlus ACE 5 with Snapdragon 8G3, Android 15) across 48 tasks spanning 7 app categories. Key results:

- 38.8% energy saving compared to stock Android (vs. 19-50% waste with context-blind approaches)
- 81.7% action accuracy (vs. rule-based 75.2% and LLM-based 73.1% baselines)
- 4.3/5.0 user satisfaction rating (based on a user study)
- 0.6% safety violation rate (96.5% reduction compared to raw LLMs)
- Preference convergence within 3-5 days without explicit configuration

The system itself consumes only 0.5% of daily battery capacity, making it efficient enough for real-world deployment.

## Related Work
PowerLens builds on three lines of prior research: hardware-level optimizations (e.g., DVFS governors like schedutil), OS-level mechanisms (e.g., Android's Adaptive Battery), and learning-based approaches (e.g., [24, 36]) that require extensive training data. Unlike these, PowerLens uniquely combines LLM reasoning with semantic understanding of user activities and implicit feedback learning. It also differs from prior LLM agents for mobile automation (e.g., AutoDroid [39], MobileGPT [15]) by operating at the system-level resource management layer rather than UI-level task automation.

## Limitations
The research was conducted on a single Android device (OnePlus ACE 5) with root access, which limits generalizability to other devices or non-rooted systems. The paper acknowledges that the two-tier memory system requires 3-5 days to converge for personalized preferences, which might be too slow for new users. Additionally, the PDL verification framework is specific to the Android system's capability model, requiring customization for other OSes.

## Appendix: Worked Example

Let's walk through a concrete example of PowerLens managing a navigation session:

1. **Activity recognition**: The Activity Agent analyzes the UI tree (Google Maps showing turn-by-turn directions) and recent app history (navigation app used for 15 minutes), recognising "Navigation" as the app category, "actively navigating" as the sub-activity, and "Medium battery" (battery at 45%) as the context. This generates a context signature: ("Navigation", "actively navigating", "Mid", "daytime").

2. **Policy generation**: The Policy Agent retrieves context-specific rules from the LPM for "Navigation" activities (e.g., "GPS accuracy: high, brightness: medium, refresh rate: 60Hz") and combines them with the device's current state (battery: 45%, location mode: high accuracy, screen brightness: 60% of max). It generates a policy with actions: 
   - Keep GPS accuracy at high (no change needed)
   - Reduce screen brightness from 60% to 40% (within PDL constraints)
   - Set refresh rate to 60Hz (from 120Hz)
   - Disable Bluetooth (unused during navigation)

3. **Safety verification**: The Execution Agent checks each action against PDL constraints:
   - GPS accuracy: high is valid and critical for navigation → APPROVED
   - Brightness: 40% is within [0, 100%] range → APPROVED
   - Refresh rate: 60Hz is one of valid options → APPROVED
   - Bluetooth: disabling is safe during navigation → APPROVED

4. **Execution**: The system applies the approved actions through root shell commands (e.g., `settings put system screen_brightness 400`).

5. **Feedback collection**: During the navigation session, if the user manually increases brightness (e.g., from 40% to 60%), the Feedback Agent detects this change via state differencing, writes a temporary override to STM, and logs the event for the Extractor.

6. **Preference learning**: Over several navigation sessions, the Extractor observes that this user frequently increases brightness during navigation. After accumulating sufficient evidence (confidence score ≥ threshold), it promotes this pattern to an LPM rule: "For user X, navigation sessions: brightness ≥ 60%". This new rule persists across future navigation sessions without requiring explicit user configuration.

## References

- Xingyu Feng, Chang Sun, Yuzhu Wang, Zhangbing Zhou, Chengwen Luo, Zhuangzhuang Chen, Xiaomin Ouyang, Huanqi Yang, "PowerLens: Taming LLM Agents for Safe and Personalized Mobile Power Management", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19584

Tags: #mobile-systems #power-management #multi-agent #personalization #safety-verification
