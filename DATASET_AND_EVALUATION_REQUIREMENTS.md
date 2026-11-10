# Dataset and Evaluation Requirements — Adaptive Self-Repairing Knowledge Distillation for Object Detection

**Document status:** Specification & Requirements Record  
**Project checkpoint:** Checkpoint 3 (Dataset & Evaluation Protocol)  
**Research direction:** Adaptive Self-Repairing Knowledge Distillation for Object Detection  
**Target deployment platform:** Low-Altitude UAV (Raspberry Pi 5 Payload)  
**Last consolidated update:** 2026-11-10

## Executive Summary

This document establishes the operational parameters, data requirements, diagnostic protocols, and baseline model criteria for Checkpoint 3.

Following the literature review in Checkpoint 2 (which established that a closed-loop, Student-error-driven self-repair adaptation mechanism is an unvalidated research gap), Checkpoint 3 formalizes the physical environment, hardware split, dataset specifications, localization-first evaluation metrics, and candidate Teacher–Student pairings.

Key decisions in this specification:

- **Hardware duality:** Training, knowledge distillation, diagnostic evaluation, and self-repair adaptation operate on high-performance laboratory compute. The deployed Student model runs on an onboard Raspberry Pi 5, with real-time edge latency requirements of 20–30 FPS (33–50 ms per frame).
- **Single-class focus:** The initial prototype focuses exclusively on Person / Human detection. This isolates localization precision from semantic classification errors.
- **Flight envelope:** The UAV operates at very low altitude (5–10 meters) with oblique to near-nadir camera perspectives (45°–90°). The stated expected target height is approximately 100–350 pixels in a 1080p frame, subject to camera optics and actual flight geometry.
- **Localization-first diagnostics:** Evaluation goes beyond aggregate mAP to include AP75, AP50:95, geometric box decomposition (center, width, height, aspect ratio, edge error), and TIDE error categories (Loc, Bkg, Miss).

## 1. Operational Environment and Hardware Constraints

### 1.1 Compute architecture duality

The system maintains a strict separation between the offline self-repair development loop and real-time onboard inference.

| Dimension | Ground / Lab Infrastructure (Teacher & Self-Repair) | Deployed UAV Edge Payload (Student) |
|---|---|---|
| Platform | High-performance GPU workstation / training cluster | Raspberry Pi 5 (quad-core Arm Cortex-A76 @ 2.4 GHz) |
| Compute budget | High-VRAM discrete GPUs supported by HardTech | Edge CPU execution; optional INT8 AI HAT / NPU accelerator |
| Primary tasks | Teacher training, dense KD, TIDE diagnosis, adaptation search, validation and rollback | Real-time sensor processing and target bounding-box regression |
| Memory budget | ≥24–48 GB VRAM, ≥64 GB system RAM (provisional planning range) | 4 GB / 8 GB LPDDR4X shared system memory |
| Latency budget | Offline / batched execution acceptable | Target: ≤33–50 ms per frame (20–30 FPS) |
| Thermal / power | Standard laboratory AC power | Airframe power budget to be validated; 5V / 5A is a planning assumption |

### 1.2 Flight envelope and optical setup

- **Operational altitude:** 5–10 meters above ground level (AGL).
- **Camera mounting:** Downward-slanted gimbal / fixed oblique mount (45°–90° pitch angle).
- **Optical resolution:** 1080p (1920 × 1080) to 4K sensor capture, downscaled for inference to detector resolutions such as 640 × 640 or 512 × 512.
- **Environmental dynamics:**
  - Lighting variation: direct sunlight, dynamic ground shadows, dusk conditions.
  - Platform motion: propeller vibration, pitch/roll changes from wind, rapid yaw turns.
  - Partial occlusion: foliage, outdoor structures, parked vehicles.

## 2. Target Object and Dataset Requirements

### 2.1 Target class specification

- **Class name:** person / human.
- **Target scale:** At 5–10 meters altitude, human targets are expected to occupy approximately 100–350 pixels along their major dimension in a 1080p frame. This is an initial operational estimate, not a measured guarantee.
- **Single-class focus rationale:**
  - Decouples classification confusion from localization failure.
  - Makes spatial detection and box delineation the primary measured concerns.
  - Aligns the prototype with the selected low-altitude person-detection use case.

### 2.2 Dataset composition protocol

The proposed data strategy combines public aerial benchmarks with a custom domain-specific validation capture.

**Public benchmark candidates (VisDrone-DET and UAVDT):**
- Extract and filter instances labeled person / pedestrian.
- Filter boxes smaller than 15 pixels as an initial operational-envelope rule; validate this threshold against the actual camera geometry and resized detector inputs.
- Retain multi-angle and top-down perspectives where they match the intended flight envelope.

**HardTech low-altitude drone capture set:**
- Record dedicated drone video sequences at 5 m, 7.5 m, and 10 m over authorized outdoor test fields.
- Annotate human bounding boxes using a documented and consistent annotation policy.
- Reserve this set as a hold-out test set for real-world generalization assessment, subject to sufficient sample diversity and strict separation from training and diagnostic validation data.

**Proposed partitioning:**

| Split | Proposed share | Purpose |
|---|---:|---|
| Training | 60% | Fixed base data for Student training and KD |
| Diagnostic validation | 20% | TIDE and error diagnostics used to select/trigger self-repair interventions |
| Hold-out acceptance test | 20% | Independent data reserved for the outer accept/reject/rollback gate |

The split percentages are provisional until the dataset inventory, sequence structure, and number of independent capture sessions are known. Video frames must be grouped by source sequence/session before splitting to avoid near-duplicate frame leakage. The custom HardTech capture is intended for independent testing; its exact allocation must be reconciled with the overall 60/20/20 split before implementation.

## 3. Localization-First Evaluation Protocol

The project treats localization quality as the primary evaluation concern. AP50 alone is insufficient to characterize precise box alignment.

### 3.1 Strict overlap metrics

- **AP75 (primary performance metric):** Average precision at an IoU threshold of 0.75.
- **AP50:95 (COCO-style aggregate):** Average precision averaged across IoU thresholds from 0.50 to 0.95 in increments of 0.05.
- **ARmax=100 (average recall):** Measures recall with up to 100 detections per image and helps characterize missed targets.

These metrics must be reported alongside precision/recall and the number of evaluated instances. AP75 is a stricter overlap measure; it does not, by itself, guarantee stable tracking or suitability for flight-control use.

### 3.2 Fine-grained geometric box error decomposition

For each matched prediction/ground-truth pair, compute:

**Center displacement error**

```
E_center = sqrt((cx_pred - cx_gt)^2 + (cy_pred - cy_gt)^2) / D_diagonal_gt
```

Normalized Euclidean distance between predicted and ground-truth box centers.

**Scale / dimension error**

```
E_dim = |w_pred - w_gt| / w_gt + |h_pred - h_gt| / h_gt
```

Measures relative width and height distortion.

**Aspect-ratio error**

```
E_ar = |(w_pred / h_pred) - (w_gt / h_gt)|
```

**Boundary edge errors**

Record the individual displacement of the top, bottom, left, and right predicted box boundaries relative to the corresponding ground-truth boundaries. Define the sign convention and normalization (pixels or normalized by box diagonal/side length) consistently in the implementation.

Geometric errors are computed only for explicitly matched detections. Missed detections and false positives must be reported separately rather than silently excluded from the overall evaluation.

### 3.3 Structured diagnostic error categorizer (TIDE)

During diagnostic evaluation on the validation split, use TIDE to categorize error contributions, including:

- **Loc:** Localization error.
- **Bkg:** Background false-positive error.
- **Miss:** Missed detections.

The report's proposed trigger is: if Loc accounts for more than 40% of total measured error impact, select a localization-focused KD intervention for investigation (for example, increasing the contribution of an LD-style localization loss or changing the spatial emphasis of feature distillation).

The 40% threshold is a provisional trigger, not an established optimal value. It must be fixed before acceptance-test evaluation and tested for sensitivity. TIDE categories are diagnostic signals; they do not by themselves prove which training intervention will correct the observed failure.

## 4. Candidate Teacher and Student Model Specifications

Model families below are candidates, not final selections. Qualification requires reproducible accuracy, compatibility, and hardware measurements.

### 4.1 Teacher model qualification criteria

- Dense object-detection architecture with accessible feature representations and suitable localization outputs for the selected KD formulation.
- Candidate capacity target: ≥40M parameters.
- Demonstrated localization performance on relevant benchmarks, with evaluation on the project's person-only data protocol.
- Frozen during Student training and self-repair adaptation loops.

**Primary candidates:**
- GFocal (Generalized Focal Loss) with ResNet-101 backbone, as a candidate aligned with the Localization Distillation literature.
- DINO / Grounding DINO (ResNet-101 or Swin-Large variants), subject to compatibility with the intended dense KD interfaces.
- YOLOv11-X / YOLOv8-X, subject to access to suitable intermediate features and localization outputs.

### 4.2 Student model qualification criteria (Raspberry Pi 5 target)

- Target parameter count: ≤5M parameters.
- Target model binary size: <25 MB, subject to precision and packaging.
- Target inference throughput: ≥20–25 FPS on Raspberry Pi 5 under a measured and reproducible runtime configuration, potentially using INT8 quantization or ONNX Runtime / NCNN acceleration.
- Architecture must expose compatible feature maps or regression outputs for the selected Teacher–Student KD method.

**Primary candidates:**
- NanoDet-Plus (ShuffleNetV2 / GhostNet backbone).
- YOLOv8-Nano / YOLOv11-Nano.
- SSDLite with MobileNetV3 backbone.

All latency claims must be measured end-to-end on the actual Raspberry Pi 5, including preprocessing and postprocessing, at the intended input resolution and with thermal/power conditions recorded. FPS targets are qualification goals, not assumed capabilities.

## 5. Outer-Loop Self-Repair Acceptance Gate

A Student modified by an adaptive KD intervention is accepted only if all pre-registered conditions are met against the fixed-KD baseline on the independent hold-out acceptance set:

1. AP75 improves by at least 0.5 percentage points (the phrase “+0.5%” must be operationally defined as relative percent or percentage-point gain before experiments).
2. Mean normalized center displacement error (E_center) does not increase.
3. Measured Raspberry Pi 5 inference latency remains below 45 ms per frame under the agreed benchmark protocol.

If any criterion fails, reject the candidate and roll back to the last accepted checkpoint. Report confidence intervals or repeated-run variability where feasible; a small AP75 difference should not be treated as meaningful without considering uncertainty.

## 6. Checkpoint 3 Sign-Off and Handoff

**Checkpoint 3 status: APPROVED & COMPLETE**, as declared by the project owner in the supplied specification.

**Handoff to Checkpoint 4:** Construction of the core software pipeline:
- Data ingestion and person-filtering scripts.
- Baseline Teacher inference setup.
- Student baseline training without adaptation.
- Integration of the TIDE error-diagnostic parser and LD loss functions.

## 7. Open Validation Items

The following details are not yet empirically established and must remain explicit validation tasks:

1. Confirm the actual camera sensor, lens, field of view, mounting angle, frame rate, and image-resizing pipeline.
2. Verify the 100–350 px target-size estimate and the 15 px filtering threshold using representative frames.
3. Inspect VisDrone-DET and UAVDT annotation conventions, class mappings, licensing, and domain match before combining them.
4. Define sequence/session-level split rules and reconcile the custom HardTech hold-out set with the proposed 60/20/20 partition.
5. Validate annotation consistency and inter-annotator policy for tight person boxes.
6. Confirm that the selected Teacher and Student expose compatible KD tensors and that the chosen KD losses can be implemented for the selected pair.
7. Benchmark candidate Students on the physical Raspberry Pi 5; do not infer real-time performance from parameter count or desktop throughput.
8. Specify the AP75 gain as percentage points or relative percent, define statistical/repeated-run handling, and freeze the acceptance protocol before examining hold-out results.
9. Define the exact TIDE implementation, matching thresholds, and the denominator used for the proposed 40% Loc trigger.
10. Confirm privacy, authorization, retention, and access controls for all custom aerial recordings. Human detections are model outputs for authorized human review, not identity claims or evidence of wrongdoing.

---

**Record note:** This document records the project's current specification and declared checkpoint sign-off. Candidate architectures, numerical thresholds, target sizes, and performance budgets remain subject to empirical validation unless explicitly marked as fixed.
