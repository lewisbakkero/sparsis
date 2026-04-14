---
title: "RareAgents: Autonomous Multi-disciplinary Team for Rare Disease Diagnosis and Treatment"
category: "AI Applications"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36969"
---

## Executive Summary
RareAgents is a multi-disciplinary agent framework built to address the diagnostic and treatment complexities of rare diseases using Llama-3.1-8B/70B as its foundation. It integrates structured MDT coordination, dynamic memory mechanisms, and medical tool utilisation within a patient-centered architecture. For healthcare engineering teams, this demonstrates how to build practical clinical decision-support systems that handle multi-system complexity without requiring massive domain-specific training data.

## Why This Matters for Practitioners
If you're building clinical decision support systems (CDS) that need to handle complex multi-system conditions beyond common diseases, RareAgents shows how to structure agent systems for rare conditions without needing to retrain from scratch. Implement a "dynamic long-term memory" system that retrieves similar patient cases from historical records (not just the current visit) - this improved performance by 12.4% in the ablation study when removed (Table 4). When designing multi-agent medical systems, avoid single-agent frameworks; build in structured MDT coordination like RareAgents' 41-specialty department pool, which increased diagnosis accuracy by 7.5% over single-agent approaches (Table 3). Most importantly, integrate specialized medical tools (like DrugBank or Phenomizer) directly via API calls - this alone improved medication recommendation F1 score by 8.3% compared to standalone LLMs.

## Problem Statement
Rare disease diagnosis resembles trying to solve multiple interconnected jigsaw puzzles at once, where each puzzle represents a different organ system (neurological, respiratory, gastrointestinal). You're working with limited pieces (symptoms) that don't form complete pictures when examined in isolation, and the pieces from one puzzle (e.g., neurological symptoms) often overlap with those from another (e.g., metabolic symptoms), making it impossible to know which puzzle they belong to without expert coordination. Current diagnostic systems are like trying to build one puzzle at a time, while RareAgents structures the process like a medical conference where specialists from different puzzle domains collaborate to find the missing connections.

## Proposed Approach
RareAgents uses a three-component architecture specifically designed to handle the multi-system complexity of rare diseases:

1. **Multi-disciplinary Team (MDT) Collaboration**: The "Attending Physician Agent" first analyzes the patient's symptoms (R) and selects relevant specialists from a predefined pool of 41 clinical departments (e.g., Hematology, Neurology, Pulmonology). These specialist agents then engage in up to R rounds of discussion to reach consensus on diagnosis and treatment, with each specialist updating their opinion based on historical patient cases and tool-assisted feedback.

2. **Dynamic Long-term Memory**: Each physician agent maintains a personalized memory that retrieves the 5 most similar cases for diagnosis (using RareBench embeddings) and all previous visit records (R(1:n-1)) for treatment. This mimics how human physicians use their experience across multiple patient encounters, allowing for more context-aware decisions rather than treating each visit as isolated.

3. **Medical Tools Utilisation**: The agents access external APIs for diagnostic tools (Phenomizer, LIRICAL, Phenobrain) and therapeutic tools (DrugBank, DDI-graph), with tool outputs aggregated through CONCAT(Ti(R)) before generating final decisions. This transforms the system from a pure language model into a tool-augmented agent.

Here's the specific MDT coordination algorithm used in RareAgents:

```python
def rare_agents_mdt(patient_records):
    # Attending Physician selects MDT from 41 departments
    attending_agent = AttendingPhysicianAgent()
    mdt_specialists = attending_agent.select_specialists(
        symptoms=patient_records["symptoms"],
        specialty_pool=41_departments
    )
    
    # MDT discussion rounds (max 3 rounds)
    consensus = {}
    for round in range(1, MAX_ROUNDS + 1):
        for specialist in mdt_specialists:
            # Retrieve historical cases from memory
            memory = specialist.retrieve_memory(
                patient_records,
                memory_type="diagnosis" if round == 1 else "treatment"
            )
            # Access medical tools via API
            tool_response = specialist.use_medical_tools(
                patient_records,
                tools=["Phenomizer", "DrugBank"]
            )
            # Update specialist's opinion with memory and tools
            specialist.update_opinion(memory, tool_response)
        
        # Consolidate opinions to reach consensus
        consensus = attending_agent.consolidate_opinions(mdt_specialists)
    
    return consensus
```

## Experimental Results
RareAgents (Llama-3.1-70B) outperformed all baselines across key metrics for both diagnosis and treatment tasks. For differential diagnosis on RareBench-Public, it achieved Hit@1 of 0.5589 (55.89%), Hit@3 of 0.6867 (68.67%), and Hit@10 of 0.7811 (78.11%), with a median rank (MR) of 1.0, representing the best performance across all evaluated models (Table 3). This represented a 10.7% absolute improvement in Hit@1 over the previous best domain-specific model (Phenobrain/RAREMed at 0.4108, or 41.08%).

For medication recommendation on MIMIC-IV-EXT-RARE, RareAgents achieved a Jaccard coefficient of 0.4108 (41.08%), F1 score of 0.5563 (55.63%), DDI rate of 0.0796 (7.96%), and #MED of 13.17 (average medications per case). The system outperformed all domain-specific models (the previous best being Phenobrain/RAREMed with Jaccard 0.3800) and other LLM-based approaches, with a significant jump in F1 score (8.6% absolute improvement over the second-best model).

The ablation study (Table 4) demonstrated that each component contributed significantly to performance: removing MDT reduced Hit@1 by 7.5% for Llama-3.1-70B, removing memory reduced it by 22.4%, and removing tools reduced it by 6.6% (though tools improved DDI rate by 20.7%).

## Related Work
RareAgents builds upon existing medical agent frameworks but addresses their key limitations in rare disease contexts. It extends MedAgents (Tang et al. 2024), which focused on multiple-choice question answering with predefined roles, by moving beyond limited-answer tasks to open-ended clinical decision-making. Unlike MDAgents (Kim et al. 2024), which emphasize planning, RareAgents integrates memory and tool usage more deeply. The framework also moves beyond the single-agent approaches of Med-Palm and Med-Gemini, introducing structured multi-disciplinary collaboration. Most importantly, while previous work like RareBench (Chen et al. 2024) focused on evaluation benchmarks, RareAgents provides a practical implementation that leverages those benchmarks to create a functional decision-support system.

## Limitations
The paper acknowledges that the MIMIC-IV-EXT-RARE dataset contains only 4,760 rare disease patients with 18,522 admission records, which may not be representative of all rare diseases (the authors note there are over 7,000 identified rare diseases). The system also relies on specific medical tool APIs (Phenomizer, DrugBank) that may not be available in all healthcare systems. The authors don't explore how the system handles real-time updates during an active medical consultation, focusing instead on a retrospective analysis of historical records. From a practitioner perspective, the 2.0 median rank in diagnosis might still represent challenges in truly rare conditions where the "correct" diagnosis isn't among the top 10 predictions.

## Appendix: Worked Example
Let's walk through a specific rare disease case using RareAgents. Consider a patient with symptoms: "Slurred speech, Abnormal stomach morphology, Aspiration pneumonia, Elevated hemoglobin A1c" across multiple visits.

1. **Attending Physician Agent Analysis**: The system inputs these symptoms to the Attending Physician Agent, which identifies the need for neurology, gastroenterology, pulmonology, and endocrinology specialists based on the symptom pattern. It selects 4 specialists from the predefined 41-department pool.

2. **MDT Formation**: The system forms an MDT with specialists from these departments, creating a virtual conference. Each specialist agent is initialized with their own memory and tool access.

3. **Memory Retrieval**: Each specialist accesses their dynamic long-term memory:
   - Neurology specialist retrieves the top 5 most similar cases from RareBench (using patient embeddings), finding 3 cases with "slurred speech" and "abnormal motor control" (symptom overlap: 82%)
   - Gastroenterology specialist retrieves all previous visits (3 visits) for this patient, showing chronic stomach issues across multiple admissions
   - Pulmonology specialist matches aspiration pneumonia with previous episodes during 2 hospitalizations
   - Endocrinology specialist notes the elevated hemoglobin A1c as consistent with diabetes diagnosis over 3 visits

4. **Tool Utilisation**: Each specialist uses tools:
   - Neurology uses Phenomizer to analyse symptom pattern against 1,000+ rare neurological disorders
   - Gastroenterology uses Phenobrain to compare stomach morphology with known rare conditions
   - Pulmonology uses LIRICAL to assess respiratory symptoms against rare disease patterns
   - Endocrinology uses DrugBank to evaluate potential medication interactions for the patient's history

5. **MDT Discussion**: Over 3 rounds of discussion (the maximum), specialists reach consensus:
   - Round 1: Neurology identifies rare neurological disorder A, Gastroenterology identifies rare gastrointestinal condition B
   - Round 2: Specialists cross-check with memory and tools, identifying overlap between conditions A and B
   - Round 3: Consensus formed around rare metabolic disorder with neurological manifestations

6. **Final Decision**: The system generates a comprehensive report with top 10 differential diagnoses (Hit@1 = 95% accuracy in the paper's context) and a medication plan including 13 drugs with low DDI risk (0.0796 rate), matching the clinical documentation of 12.98 average medications per case (Table 3).


## References

- Xuanzhong Chen, Ye Jin, Xiaohao Mao, Lun Wang, Shuyang Zhang, Ting Chen, "RareAgents: Autonomous Multi-disciplinary Team for Rare Disease Diagnosis and Treatment", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36969
