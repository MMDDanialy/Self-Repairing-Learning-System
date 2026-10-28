# Problem Statement — Self-Repairing Learning System

**Status:** Initial specification — subject to revision after related-work review and experimental design  
**Project:** Self-Repairing Learning System  
**Primary research domain:** Object detection under shortcut learning and distribution shifts

## 1. Problem Statement

Machine-learning models can achieve strong performance on conventional evaluation data while remaining vulnerable to misleading features, spurious correlations, and changes in the data distribution. When these failures occur, improving the model is not simply a matter of retraining it: the underlying cause may be uncertain, different explanations may require different interventions, and a correction that improves one metric may degrade previously learned capabilities.

This project investigates a locally executable, evidence-driven system that can detect meaningful failures in a machine-learning model, formulate and test competing explanations, select informative diagnostic experiments, apply targeted corrections, and determine whether those corrections are successful without unacceptable regressions.

**Object detection is the mandatory primary task.** Classification may be used as a supporting task for preliminary experiments, controlled comparisons, or methodological validation when justified, but it will not replace object detection as the main research target.

The project will begin with a controlled computer-vision setting involving shortcut learning and distribution shifts. The specific model, dataset, intervention methods, and experimental configuration will be selected only after the related-work review and experimental-design stages.

The project does not assume that fully autonomous repair, general applicability, or research novelty has already been established. These are claims to investigate and evaluate.

## 2. Research Objectives

The project has the following objectives:

1. **Failure detection:** Identify meaningful failures in object detection using predefined evaluation procedures.
2. **Evidence-based diagnosis:** Generate competing hypotheses that explain observed failures and connect each hypothesis to measurable evidence.
3. **Diagnostic experiment selection:** Select experiments that can distinguish among hypotheses while accounting for expected information value, computational cost, execution time, and feasibility.
4. **Targeted correction:** Apply a correction that addresses a supported failure hypothesis rather than relying exclusively on indiscriminate retraining.
5. **Correction validation:** Compare the corrected model with its previous version and determine whether the target failure improved without unacceptable regressions.
6. **Iterative self-repair:** Integrate evaluation, diagnosis, experiment selection, correction, and validation into a repeatable loop.
7. **Reproducible evaluation:** Compare the proposed approach with appropriate baselines and report improvements, regressions, costs, limitations, and inconclusive results.
8. **Practical deployment validation:** Evaluate the prepared object-detection model on Raspberry Pi 5 hardware during the final stages, subject to hardware compatibility and resource constraints.

## 3. Scope and Target Failure Modes

### In scope

- Object detection as the primary task.
- Reliance on irrelevant or misleading visual features and shortcut correlations.
- Performance degradation under relevant distribution shifts.
- Generalization failures on appropriately defined unseen data.
- Evidence-linked failure diagnosis and testable hypotheses.
- Cost-aware selection of diagnostic experiments.
- Targeted correction, version tracking, rollback, and regression prevention.
- Local execution of the self-repair loop.
- Reproducible comparisons against appropriate baseline approaches.

### Supporting or conditional scope

- Classification experiments, when they provide a clear methodological benefit.
- Alternative detection architectures and correction methods, selected after research and feasibility analysis.
- Hardware-specific optimization for Raspberry Pi 5 during final deployment validation.
- Hailo acceleration, if the required hardware is confirmed to be available and compatible.

### Out of scope for the initial specification

- Claiming universal self-repair across arbitrary tasks or model families.
- Assuming that every failure can be diagnosed or corrected automatically.
- Requiring a cloud-based large language model for the essential repair loop.
- Selecting a specific architecture, dataset, or correction algorithm before the relevant research and design checkpoints.

## 4. Proposed Self-Repair Process

The system will investigate the following iterative process:

1. **Evaluate:** Measure model performance, generalization, and sensitivity to potentially misleading features.
2. **Detect:** Determine whether observed behavior meets predefined failure-trigger criteria.
3. **Diagnose:** Form competing hypotheses supported by available evidence.
4. **Design an experiment:** Specify what each candidate experiment is expected to reveal and how its result could change the diagnosis.
5. **Select an experiment:** Consider diagnostic value, cost, time, and available resources.
6. **Apply a correction:** Make a traceable intervention justified by the diagnostic evidence.
7. **Validate:** Evaluate the corrected model against the previous version using comparable test procedures.
8. **Decide:** Accept, reject, revise, or roll back the correction according to predefined criteria.
9. **Repeat or stop:** Continue while useful experiments and resources remain, or stop when the stopping criteria are met.

The process must preserve model versions and record relevant data, configurations, hypotheses, experiments, interventions, evaluation results, and decisions.

## 5. Evaluation Framework

The project will use three complementary evaluation dimensions.

### 5.1 Predictive performance

Does the object detector correctly identify and localize objects on the defined evaluation data?

The final metric or metrics will be selected after the literature review and dataset design.

### 5.2 Generalization and distribution-shift robustness

Does the model retain acceptable performance on appropriately defined unseen data and relevant shifted distributions?

The experimental design must distinguish ordinary held-out evaluation from deliberately constructed diagnostic or shifted conditions where appropriate.

### 5.3 Shortcut and spurious-dependency evaluation

Does the model depend on features that are irrelevant to the intended prediction?

Controlled interventions, such as changing an irrelevant background while preserving the relevant object, may be considered if supported by the research review and experimental design.

### 5.4 Correction and system-level evaluation

The self-repair system will additionally be assessed on:

- Reduction of the targeted failure.
- Retention of previously learned capabilities.
- Frequency and severity of regressions.
- Quality of diagnostic hypotheses and supporting evidence.
- Usefulness and cost of selected experiments.
- Correction acceptance, rejection, and rollback decisions.
- Computational cost, elapsed time, and repeatability.
- Performance relative to appropriate baseline methods.

A gain on a single target metric will not, by itself, establish that a correction is successful.

## 6. Qualitative Acceptance Criteria

The initial system will be considered to demonstrate a successful self-repair cycle when it can:

1. Reproducibly detect a predefined, meaningful failure in the selected object-detection setting.
2. Produce explicit, testable diagnostic hypotheses linked to observed evidence.
3. Select and execute a diagnostic experiment with a documented rationale.
4. Apply a traceable correction linked to the investigated failure.
5. Evaluate the correction against the previous model using comparable procedures.
6. Accept, reject, or roll back the correction based on predefined criteria.
7. Demonstrate improvement in the target behavior without exceeding the agreed regression limits.
8. Record sufficient information for another person to reproduce and inspect the experiment.
9. Run the essential self-repair loop locally without requiring a cloud LLM.

These criteria define the intended behavior, not a claim that the system has already achieved it.

## 7. Execution and Hardware Constraints

The planned workflow separates model development, self-repair experimentation, and final deployment validation.

### Development and training

The user's laptop will serve as the initial environment for developing and testing the self-repair loop. More demanding training or model preparation may use organizational computers equipped with two NVIDIA RTX 4060 GPUs, subject to actual availability and configuration.

### Local self-repair loop

The initial implementation should prioritize reproducibility, resource awareness, and compatibility with the laptop development environment. Exact memory, runtime, and compute budgets remain to be established after the experimental workload is defined.

### Final deployment validation

The prepared object-detection model will be evaluated on Raspberry Pi 5 hardware in the later stages of the project. The availability of Hailo accelerators and the corresponding software and model compatibility remain unconfirmed.

The initial plan does not require the complete self-repair loop to run on the Raspberry Pi. The primary plan is to execute the loop on a development computer and deploy the resulting prepared model to the Raspberry Pi for final hardware testing.

## 8. Constraints and Research Integrity

- The primary task must remain object detection.
- Research and experimental design must precede final model and dataset selection.
- Evaluation criteria must be defined before comparing corrections.
- Diagnostic explanations must remain hypotheses until tested.
- Data leakage and invalid comparisons must be prevented.
- Previous model versions must be retained to support fair evaluation and rollback.
- Resource use must be recorded and considered when selecting experiments.
- Negative, inconclusive, and regression results must be documented.
- Research novelty and general applicability must not be claimed without supporting evidence.
- Cloud LLM services must not be essential to the core repair process.

## 9. Decisions Deferred to Later Checkpoints

The following decisions remain open and must be resolved through research and experimental design:

- Specific object-detection architecture and baseline.
- Dataset and data-licensing requirements.
- Construction of controlled shortcut and distribution-shift scenarios.
- Evaluation metrics and numerical acceptance thresholds.
- Permitted regression limits.
- Failure-trigger rules and stopping conditions.
- Candidate diagnosis and correction methods.
- Experiment-selection strategy and computational budgets.
- Exact laptop and organizational compute constraints.
- Raspberry Pi 5 compatibility requirements and Hailo availability.
- The research gap, contribution claims, and scope of any novelty claim.

These decisions should not be fixed merely to make the initial specification appear complete.

## 10. Relationship to the Roadmap

This document defines the problem, research objectives, initial constraints, and evaluation principles for Checkpoint 1.

`CORE_DESIGN.md` describes the system's overarching design principles, while `ROADMAP.md` defines the approved 12-checkpoint implementation and evaluation plan.

Checkpoint 2 will review related work and identify evidence-supported candidate approaches. Checkpoint 3 will define the controlled experimental environment and refine dataset-dependent metrics, numerical thresholds, resource budgets, and stopping conditions.

The specification may be revised when later evidence justifies a change. Such revisions should be documented rather than silently changing the research objective.

## 11. Expected Outcome

The intended outcome is a reproducible, locally executable research prototype that investigates whether evidence-driven diagnosis and targeted correction can improve an object-detection model under controlled failure conditions, while accounting for regressions and computational cost.

The success of this research will be determined by the quality of the experimental evidence and baseline comparisons, not merely by the completion of an automated pipeline.
