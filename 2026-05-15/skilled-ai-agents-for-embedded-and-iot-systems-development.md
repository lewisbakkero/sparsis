---
title: "Skilled AI Agents for Embedded and IoT Systems Development"
venue: "Yiran Chen"
paper_url: "https://arxiv.org/abs/2603.19583"
---

## Executive Summary
This paper introduces a skills-based agentic framework for hardware-in-the-loop (HIL) embedded and IoT systems development, alongside IoT-SkillsBench, a benchmark validated through real hardware execution. It demonstrates that concise human-expert skills, distilling structured knowledge of peripheral initialization, timing constraints, and failure modes, achieve near-perfect success rates (41/42 tasks resolved) across multiple platforms, while LLM-generated skills can degrade performance due to incorrect platform assumptions.

## Why This Matters for Practitioners
If you're building production IoT systems where code that compiles fails on real hardware (e.g., due to I2C timing mismatches or peripheral initialization errors), this paper shows that curating human-expert skills is more critical than scaling LLMs. For any embedded team using AI agents for development, stop relying on LLM-generated documentation, instead, invest time in creating and maintaining concise, platform-specific skills documents. This approach reduces hardware validation time by eliminating silent failures and avoids the token overhead of injecting full SDK documentation. For an Arduino-based sensor network, this means replacing a 10-page datasheet with a single 200-word skill describing I2C initialization sequences for the specific sensor.

## Problem Statement
Embedded development is broken because it treats hardware constraints as an afterthought. Imagine building a house using only a blueprint without considering the soil type: the walls might stand in the lab, but collapse in real weather. Similarly, LLM agents generate code that compiles but fails on hardware due to unaddressed timing constraints, peripheral initialization sequences, or hardware-specific edge cases. This gap between software simulation and physical execution causes silent failures that are impossible to catch in digital emulators like QEMU or Wokwi.

## Proposed Approach
The framework uses a three-node agent architecture (manager, coder, assembler) where skills, compact documents distilling expert knowledge, guide code generation. Skills replace raw SDK documentation, reducing token usage while improving reliability. IoT-SkillsBench evaluates agents across 42 tasks (3 difficulty levels) on 3 platforms (Arduino, ESP-IDF, Zephyr), each task validated on physical hardware.

```python
def generate_firmware(task, skills):
    # Manager node selects relevant skills
    relevant_skills = manager_node(task, skills)
    
    # Coder node generates code using selected skills
    code = coder_node(task, relevant_skills)
    
    # Assembler node formats code for platform
    project = assembler_node(code, task.platform)
    
    return compile_and_flash(project)
```

## Key Technical Contributions
The framework's novelty lies in how it handles hardware-software coupling:
1. **Skill distillation process:** Human experts author skills by analysing LLM-generated code failures, focusing on specific peripheral/platform combinations. Each skill describes *only* a single peripheral or framework behaviour (e.g., "I2C initialization for BME280 on ESP32-S3 must wait 10ms after power-on"), avoiding code snippets to reduce token usage.
2. **Hardware-validated benchmark design:** IoT-SkillsBench's 378 real-hardware experiments across 3 platforms ensure results reflect physical execution, not simulation. Each task includes platform-specific pin mappings to eliminate hardware-mapping errors.
3. **Token efficiency trade-off:** Human-expert skills maintain moderate token overhead (650-2,900 input tokens) while achieving 41/42 resolved tasks, whereas LLM-generated skills consume 8,500-9,500 tokens with degraded performance (27/42 resolved for ESP-IDF).

See Appendix for a step-by-step worked example of how a human-expert skill guides I2C communication on ESP32-S3.

## Experimental Results
IoT-SkillsBench's 378 hardware-validated experiments revealed:
- Human-expert skills achieved 41/42 resolved tasks (97.6%) across all platforms and difficulty levels, with only two failures due to hardware ambiguities (voltage incompatibility with 5V RTC on 3.3V ESP32-S3 and non-standardized rotary encoder behaviour).
- LLM-generated skills degraded performance: ESP-IDF tasks dropped from 31/42 to 27/42 resolved, with Level 3 tasks falling from 8/14 to 4/14.
- No-skills baseline resolved 31/42 tasks (73.8%), but failed on complex integrations (only 6/14 Level 3 tasks resolved on Zephyr).
- Token usage: Human-expert skills consumed 650-2,900 input tokens (vs. 8,500-9,500 for LLM-generated skills), with a 75% reduction in token overhead while improving success rates by 24%.

## Related Work
The paper contrasts with EmbedAgent (Xu et al., 2025) and EmbedGenius (Yang et al., 2024), which rely on compiler-in-the-loop or flashing-in-the-loop approaches. These fail to address *behavioral* errors that emerge during physical execution. IoT-SkillsBench extends SkillsBench (Li et al., 2026) by grounding skills in real hardware behaviour through HIL validation, rather than purely software-based testing.

## Limitations
The benchmark's hardware limitations include two unresolved tasks due to physical constraints (5V RTC module on 3.3V ESP32-S3 and non-standardized rotary encoder behaviour). The evaluation also didn't test performance in resource-constrained environments (e.g., MCUs with <128KB RAM), though the authors note the approach scales naturally to new platforms via skill creation.

## Appendix: Worked Example
Consider a Level 2 task: "Implement I2C communication with BME280 temperature sensor on ESP32-S3 using ESP-IDF." Without skills, an LLM-generated solution might skip the mandatory 10ms delay after power-on, causing I2C bus collisions.

Human-expert skill content:
> **BME280 on ESP32-S3 (ESP-IDF)**  
> *Always add 10ms delay after power-on before I2C communication*  
> *Correct I2C address: 0x76 (not 0x77, which is common on other platforms)*  
> *Use `i2c_master_write_to_device` with `I2C_CMD_WAIT_ACK` flag*

The agent's workflow:
1. Manager node selects "BME280 on ESP32-S3" skill.
2. Coder node generates code using the skill's constraints:
   ```c
   vTaskDelay(pdMS_TO_TICKS(10));  // 10ms power-on delay
   i2c_cmd_handle_t cmd = i2c_cmd_link_create();
   i2c_master_start(cmd);
   i2c_master_write_byte(cmd, (0x76 << 1) | I2C_MASTER_WRITE, true);
   i2c_master_write_byte(cmd, 0x88, true);  // Control register
   i2c_master_stop(cmd);
   i2c_master_cmd_begin(I2C_NUM_0, cmd, 1000 / portTICK_PERIOD_MS);
   ```
3. Assembler node formats code for ESP-IDF's CMake structure.

This approach resolved the task on first attempt (BC@1), avoiding the 50% failure rate (BF) seen in no-skills baseline for similar tasks.

## References

- **Code:** https://github.com/agentskills/agentskills.
- Yiming Li, Yuhan Cheng, Mingchen Ma, Yihang Zou, Ningyuan Yang, Wei Cheng, Hai "Helen" Li, Yiran Chen, Tingjun Chen, "Skilled AI Agents for Embedded and IoT Systems Development", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19583

Tags: #embedded-systems #iot-devices #hardware-software-integration #skills-based-ai #agent-framework
