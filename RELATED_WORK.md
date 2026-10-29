# Related Work — Preliminary Review

**Status:** First-pass evidence map; not a complete systematic review.  
**Purpose:** Inform Checkpoint 2 without prematurely claiming novelty or selecting the final model/dataset.

## 1. Scope

This project studies an evidence-driven, locally executable repair loop for **object detection**. Relevant prior work therefore falls into at least four connected areas:

1. Evaluating detectors under distribution shift.
2. Discovering and diagnosing systematic failure slices.
3. Detecting shortcut/spurious-feature dependence.
4. Correcting a model and validating gains against regressions and cost.

A benchmark or debugging tool is related work, but is not by itself evidence that an autonomous repair loop is solved.

## 2. Initial evidence map

| Work | Main contribution reported by the source | Relevance | Limitation for this project |
|---|---|---|---|
| **COCO-O (ICCV 2023)** — [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_COCO-O_A_Benchmark_for_Object_Detectors_under_Natural_Distribution_Shifts_ICCV_2023_paper.html) | Benchmark for object detectors under six types of natural distribution shift; the paper reports a substantial drop for Faster R-CNN and evaluates many detectors. | Supports measuring more than in-distribution detection performance and gives a candidate source for shifted evaluation. | A robustness benchmark does not itself provide hypothesis-driven diagnosis, cost-aware experiment selection, or closed-loop repair. Dataset suitability and license/compute needs must be checked before adoption. |
| **COUNTS (CVPR 2025)** — [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Li_COUNTS_Benchmarking_Object_Detectors_and_Multimodal_Large_Language_Models_under_CVPR_2025_paper.html) | Reports an object-level benchmark with 14 natural distribution shifts, over 222K samples, and over 1.196M boxes. | A potentially relevant benchmark for out-of-distribution detection evaluation. | Scale may be too costly for the initial laptop workflow; feasibility, data access, license, and a small controlled subset need assessment. |
| **RWDS (CVPR 2025)** — [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Al-Emadi_Benchmarking_Object_Detectors_under_Real-World_Distribution_Shifts_in_Satellite_Imagery_CVPR_2025_paper.html) | Introduces three real-world distribution-shift datasets focused on satellite imagery and spatial/geographic or environmental changes. | Demonstrates that realistic shifts can be domain-specific and that split design matters. | Satellite imagery may not be the most practical first domain; its fit to the intended controlled shortcut experiment remains undecided. |
| **Visual Concept Reviewer (VCR, 2026 journal publication)** — [paper](https://www.sciencedirect.com/science/article/pii/S0306437925001383) | Human-in-the-loop slice-discovery framework for object-detection errors, using visual concepts, metadata, and frequent-itemset mining. The paper reports 1,713 slice-discovery settings across three datasets. | Directly relevant to diagnosing systematic subsets of detector failures and to designing interpretable evidence logs. | Human-in-the-loop slice discovery is not equivalent to automatically choosing a discriminating experiment, applying a targeted correction, and validating regressions. |
| **Unmasking the Clever Hans effect (2026 review)** — [review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12827554/) | Reviews shortcut learning and spurious correlations across domains, including detection methods and mitigation families such as counterfactual augmentation and slice-aware evaluation. | Provides a conceptual map for shortcut definitions, detection, and mitigation approaches. | Broad cross-domain review; object-detection-specific claims and methods must be traced to primary studies before being used in the experimental design. |

## 3. Preliminary implications

- **Evaluation needs multiple views.** In-distribution metrics alone cannot establish robust behavior. The design should separately report ordinary detection quality, shifted/generalization performance, and sensitivity to a controlled nuisance feature.
- **Failure slices can be diagnostic units.** Aggregate metrics can hide subsets with systematic errors. Candidate methods should identify which conditions, visual concepts, or metadata correlate with failure, while treating correlation as evidence for a hypothesis rather than proof of causation.
- **Controlled interventions are important.** A useful shortcut test should vary a candidate nuisance factor while keeping the object and its annotation valid. The protocol must avoid introducing editing artifacts that become a new shortcut.
- **Correction is not the same as diagnosis.** A method that detects a failure or produces an explanation does not necessarily establish which intervention will fix it.
- **Regression testing is part of the contribution target.** Any candidate correction should be compared with the previous model on the target failure, retained capabilities, relevant shifts, and computational cost.
- **Local feasibility must be considered from the start.** Large benchmarks can be useful for external validation, but the first reproducible experiment should have a controlled, manageable compute budget.

## 4. Candidate research direction — not a novelty claim

A direction worth testing is a **cost-aware, evidence-linked repair loop for object detection** that:

1. Detects a predefined failure from evaluation evidence.
2. Maintains multiple competing hypotheses.
3. Chooses a low-cost experiment expected to discriminate among those hypotheses.
4. Applies a traceable correction linked to the tested hypothesis.
5. accepts or rejects the correction using predeclared target-improvement and regression criteria.
6. Logs the decision, evidence, resource cost, model/data versions, and rollback outcome.

This is a research hypothesis about a potentially useful integration, **not a claim that the combination is novel**. Related work must be expanded to include primary papers on model debugging, slice discovery, active learning/experiment design, shortcut mitigation, and automated or iterative model repair. The final gap statement must be revised after that comparison.

## 5. Decisions intentionally left open

This first pass does **not** select a detector architecture, final dataset, correction algorithm, numerical acceptance threshold, or deployment stack. Those decisions belong after the related-work review and controlled experimental design.

## 6. Next literature-review tasks

- Trace VCR and the benchmark papers to their relevant cited work.
- Compare detector failure-slice discovery methods, including their human-supervision requirements and local runtime.
- Review data-centric and model-centric shortcut mitigation in object detection, with attention to unintended new shortcuts.
- Review active diagnosis / experiment-selection methods and their assumptions about hypothesis spaces and cost.
- Identify simple, fair baselines for repair-loop evaluation.
- Record for each candidate: evidence type, task, compute cost, required supervision, correction capability, regression checks, code/data availability, and limitations.

## References

1. Mao et al., “COCO-O: A Benchmark for Object Detectors under Natural Distribution Shifts,” ICCV 2023. [Open-access paper](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_COCO-O_A_Benchmark_for_Object_Detectors_under_Natural_Distribution_Shifts_ICCV_2023_paper.html)
2. Li et al., “COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts,” CVPR 2025. [Open-access paper](https://openaccess.thecvf.com/content/CVPR2025/html/Li_COUNTS_Benchmarking_Object_Detectors_and_Multimodal_Large_Language_Models_under_CVPR_2025_paper.html)
3. Al-Emadi et al., “Benchmarking Object Detectors under Real-World Distribution Shifts in Satellite Imagery,” CVPR 2025. [Open-access paper](https://openaccess.thecvf.com/content/CVPR2025/html/Al-Emadi_Benchmarking_Object_Detectors_under_Real-World_Distribution_Shifts_in_Satellite_Imagery_CVPR_2025_paper.html)
4. “VCR: Interpretable and interactive debugging of object detection models with visual concepts,” Information Systems, 2026. [Publisher page](https://www.sciencedirect.com/science/article/pii/S0306437925001383)
5. “Unmasking the Clever Hans effect in AI models: shortcut learning, spurious correlations, and the path toward robust intelligence,” Frontiers in AI, 2026. [Open-access review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12827554/)
