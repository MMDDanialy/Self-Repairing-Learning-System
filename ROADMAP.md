# Project Roadmap — Self-Repairing Learning System

This roadmap defines the approved 12 checkpoints. Work should proceed in order: research and experimental design come before implementation. Do not choose a model or install a stack prematurely.

## 1. Define the Exact Problem and Goal

**Work**
- Specify the failure types in scope: generalization drops, shortcut/irrelevant-feature reliance, and distribution shifts.
- Define the loop: detect → diagnose → select experiment → correct → validate.
- Establish local-execution constraints, metrics, acceptance criteria, resource limits, and stopping conditions.

**Deliverables**
- A concise problem statement.
- Initial metric and acceptance-criteria specification.

**Completion criteria**
- Failure cases and success/failure conditions are measurable and unambiguous.

## 2. Review Related Work

**Work**
- Review shortcut learning, generalization, robustness, distribution shift, active/continual learning, and model debugging.
- Examine methods for failure detection, diagnosis, experiment selection, and correction.
- Review local-deployment feasibility, open-source implementations, baselines, and limitations.
- Compare relevant approaches in a structured table and identify possible research gaps without assuming novelty.

**Deliverables**
- Referenced literature review and comparison table.
- Evidence-backed candidate gap and baseline shortlist.

**Completion criteria**
- The project’s positioning and proposed experiments are informed by prior work; novelty claims remain appropriately qualified.

## 3. Design a Controlled Experimental Environment

**Work**
- Choose a dataset and construct controlled scenarios that expose the target failures and shifts.
- Define train, validation, in-distribution test, and shifted/diagnostic evaluation splits as appropriate.
- Prevent data leakage and specify repeatability, random seeds, and local compute constraints.

**Deliverables**
- Dataset/scenario specification and reproducible split protocol.

**Completion criteria**
- The intended failures can be induced or measured in controlled, repeatable experiments.

## 4. Establish the Baseline and Characterize Failures

**Work**
- Select a locally viable baseline model based on the preceding research and hardware constraints.
- Train or prepare it, then evaluate normal and shifted distributions.
- Confirm that the chosen task exposes the target failure modes and record baseline results.

**Deliverables**
- Baseline model, evaluation scripts, and baseline report.

**Completion criteria**
- Baseline performance and at least one target failure are empirically characterized.

## 5. Implement the Learning-Evaluation Loop

**Work**
- Measure predictive performance, generalization, and dependence on potentially misleading features.
- Define thresholds or other rules for identifying meaningful failures.
- Log metrics and trigger diagnosis only when the evidence meets the predefined trigger criteria.

**Deliverables**
- Repeatable evaluation module, metric logs, and failure-trigger logic.

**Completion criteria**
- The system can detect the defined failure conditions on the controlled task.

## 6. Implement Failure Diagnosis

**Work**
- Generate competing hypotheses for an observed failure.
- Define testable predictions and experiments that could support or reject each hypothesis.
- Use local algorithms, rules, or models; record evidence and confidence rather than presenting guesses as facts.

**Deliverables**
- Diagnosis module and hypothesis/evidence log format.

**Completion criteria**
- Diagnoses produce explicit hypotheses and testable predictions tied to observed evidence.

## 7. Select Diagnostic Experiments Intelligently

**Work**
- Choose experiments that discriminate among competing hypotheses.
- Account for expected information value, compute/time cost, and feasibility on local hardware.
- Record why each experiment was selected and what result would change the diagnosis.

**Deliverables**
- Experiment-selection strategy and decision logs.

**Completion criteria**
- The selected next experiment is justified by its expected diagnostic value and cost, and can be compared with simpler selection baselines.

## 8. Apply Targeted Model Corrections

**Work**
- Apply corrections justified by diagnostic evidence, such as data correction/diversification or limited retraining, where appropriate.
- Track the exact changes and their computational cost.
- Preserve the prior model version to enable comparison and rollback.

**Deliverables**
- Correction module/procedure, versioned artifacts, and change logs.

**Completion criteria**
- Corrections are traceable, reproducible, and linked to a tested failure hypothesis.

## 9. Validate Corrections and Prevent Regressions

**Work**
- Run comparable before/after evaluations on target failures, generalization, and previously learned capabilities.
- Measure gains, regressions, cost, and repeatability.
- Accept, reject, revise, or roll back corrections using criteria defined before evaluation.

**Deliverables**
- Validation report and explicit accept/reject/rollback logic.

**Completion criteria**
- The system does not treat a target-metric gain as sufficient when unacceptable regressions occur.

## 10. Integrate the Full Self-Repair Loop

**Work**
- Integrate evaluation, diagnosis, experiment selection, correction, validation, logging, resource management, and stopping conditions.
- Support multiple repair cycles and safe rollback.

**Deliverables**
- End-to-end local prototype and run instructions.

**Completion criteria**
- A complete cycle can run reproducibly from detected failure through validated correction or justified rejection.

## 11. Evaluate Against Baselines

**Work**
- Compare the system with an uncorrected model and appropriate simple baselines, such as training with more diverse data or manual correction.
- Evaluate diagnosis quality, experiment efficiency, target-failure reduction, retained capabilities, compute/time cost, repeatability, and failure cases.
- Report negative and inconclusive results as well as improvements.

**Deliverables**
- Reproducible experiment results, comparison tables/plots, and limitations analysis.

**Completion criteria**
- Claims about effectiveness are supported by fair comparisons and documented evidence.

## 12. Document and Release an Initial Version

**Work**
- Document the problem, architecture, research context, setup, local execution, experiment reproduction, results, limitations, and known failure cases.
- Provide clear installation and run instructions and prepare the GitHub repository for an initial release.

**Deliverables**
- README, architecture/design documentation, reproduction instructions, and initial release artifacts.

**Completion criteria**
- Another person can understand the scope, reproduce the core experiments, and assess the evidence and limitations.

---

## Overall Success Criteria

The initial project is successful when it can run locally, detect a defined meaningful failure, form evidence-linked hypotheses, select a useful diagnostic experiment with cost considered, apply a targeted correction, and validate whether the correction improves the target behavior without unacceptable regressions. It must be compared against suitable baselines, remain reproducible, and not depend essentially on a cloud LLM.

A successful prototype does not by itself establish novelty or general applicability; those conclusions require evidence from the related-work review and experiments.


---

## Checkpoint Update — UAV Application Context

The following updates supersede the corresponding scope details above where they conflict. The central research direction remains Adaptive Self-Repairing Knowledge Distillation for Object Detection; this update does not authorize premature selection of a model, dataset, or training stack.

### Checkpoint 1 — Define the Exact Problem and Goal

Record the clarified application context:
- Application: security-oriented visual monitoring of Lark Garden by a UAV.
- Initial detection categories: humans and relevant animals, including dogs, cats, and foxes. The final class list must be confirmed with the UAV team.
- Fixed deployment platform: Raspberry Pi 5 with 8 GB RAM.
- Research method remains a fixed-Teacher, adaptive-KD process targeting measurable Student weaknesses, with a fixed non-adaptive KD baseline.
- The intended system reports visual detections for human review; a detection is not proof of wrongdoing or identity.

### Checkpoint 3 — Dataset and Data Requirements

Before selecting a dataset:
- Compare suitable public object-detection datasets against the confirmed target classes and aerial/UAV viewpoint.
- Determine whether the UAV team can provide appropriately authorized real images or video from the intended environment.
- Assess annotation requirements, class balance, image conditions, licensing, privacy, and train/validation/test separation.
- Consider synthetic data only as a supplementary source; evaluate its domain gap against real imagery.
- Keep a genuinely independent real-world evaluation set where feasible.

**Deliverable:** evidence-based dataset and data-collection plan. Dataset choice is not finalized here.

### Checkpoint 4 — Technical Design and Compute Feasibility

Assess training and deployment as separate compute problems:
- **Deployment target (fixed):** Raspberry Pi 5, 8 GB RAM. Candidate detectors must be assessed for supported inference runtimes, memory use, latency, and practical optimization on this platform.
- **Training system (open):** estimate GPU/VRAM needs from candidate Teacher/Student models, input resolution, batch size, dataset scale, and planned experiments.
- Compare cloud GPU rental with purchasing hardware using estimated total experiment time and cost.
- A pretrained detector followed by fine-tuning is the current preferred starting strategy, but the model family and Teacher/Student pair remain undecided.

**Deliverable:** documented training-compute estimate and deployment-feasibility comparison. The preliminary 64 GB VRAM figure is an estimate to verify, not an approved requirement.

All other checkpoints remain unchanged by this application-context update.
