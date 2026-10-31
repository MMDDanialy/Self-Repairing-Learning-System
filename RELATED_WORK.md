# Related Work — Adaptive Self-Repairing Knowledge Distillation for Object Detection

**Status:** Checkpoint 2 — introductory literature review (working document)  
**Purpose:** Build a clear understanding of relevant prior work and record ideas worth investigating. This is not a systematic review and makes no novelty claim.

## 1. Our project focus

The project investigates **evaluation-driven adaptation of the Knowledge Distillation (KD) process** for object detection:

- The Teacher remains fixed.
- The Student is trained and evaluated.
- Evaluation evidence identifies weaknesses.
- The KD process is adapted in response to those weaknesses.
- A candidate Student is validated against the baseline or last accepted version.
- Changes are accepted, rejected, revised, or rolled back based on measured results.

**Priority clarification:** Localization quality (where an object is and how accurately its bounding box is placed) is the primary research priority. Class recognition is secondary. The earlier Counter-Strike 2 head-detection work is relevant as a computer-vision example of a localization-centred task: the model's key output was the head bounding box/centre. For this research project, that experience is used only to clarify the technical priority of localization; game-control or aiming integration is not part of the research scope.

## 2. A basic concept: catastrophic forgetting

In continual or incremental learning, a model first learns some classes and is later trained on new classes. If the later training mostly contains new-class examples, gradient updates can change parameters that were also important for recognizing old classes. The model may become better on new classes while its performance on old classes drops. This is called **catastrophic forgetting**.

This problem is relevant to incremental object detection, where new categories arrive over time. It is **not automatically the same problem as our initial setup**: if our first experiments use a fixed dataset and train the Student on the same set of classes, catastrophic forgetting is not the central motivation. We should study incremental-learning methods for useful KD ideas, but not assume we need a continual-learning system.

## 3. Main papers and what they contribute

### 3.1 When Object Detection Meets Knowledge Distillation: A Survey (Li et al., TPAMI 2023)

**Type:** Survey; it reviews prior work rather than proposing one new detector or one new KD algorithm.

**What it says:** Object detectors have become more capable but can be computationally expensive. KD is one way to transfer useful knowledge from a stronger Teacher to a smaller Student.

**Its useful organization:** It groups methods by the information transferred:
- **Response-based:** transfer predictions or logits.
- **Feature-based:** transfer intermediate feature representations.
- **Relation-based:** transfer relationships among features, locations, or instances.

**Why we use it:** It gives us the map of the object-detection KD field and vocabulary for comparing methods. It also helps avoid treating all KD methods as if they transfer the same kind of information.

**What it does not establish:** It does not demonstrate that our proposed evaluation-driven self-repair loop works or is novel.

Paper: https://pubmed.ncbi.nlm.nih.gov/37028387/  
DOI: 10.1109/TPAMI.2023.3257546

### 3.2 Knowledge Distillation in Object Detection: A Survey from CNN to Transformer (2026)

**Type:** Survey of object-detection KD, including CNN- and Transformer-based approaches.

**What it says:** It reviews feature-, relation-, and response-based distillation and discusses methods for different detection settings. It includes DKD, KD-Zero, Localization Distillation, and other approaches.

**Why we use it:** It is a recent map of the field and a convenient route to primary papers. We should use it to discover methods, but cite and inspect the original papers before relying on detailed claims.

Paper: https://www.mdpi.com/1424-8220/26/1/292

### 3.3 Decoupled Knowledge Distillation (DKD) — Zhao et al., CVPR 2022

**Type:** Primary research paper; a new way to formulate logit-based KD.

**Problem:** In conventional logit distillation, the contribution from the target class and the non-target classes is coupled. That makes it harder to control their relative influence.

**What the authors do:** They separate the KD objective into:
- **TCKD:** Target Class Knowledge Distillation.
- **NCKD:** Non-Target Class Knowledge Distillation.

The two parts can be weighted independently. The paper explains that TCKD carries information related to sample difficulty, while NCKD is an important source of the benefit of logit distillation.

**Reported results and an important qualification:** The paper reports strong results on classification benchmarks. On MS COCO object detection, DKD alone does not produce outstanding results; combining it with a feature-based method (ReviewKD) improves detection results. This matters for our project: DKD is useful and worth retaining as a candidate component, but it should not be assumed to solve localization by itself.

**Ideas to retain for our project:**
1. Keep different knowledge signals separable where that makes technical sense.
2. Consider whether their relative weights should be adjusted in response to measured Student weaknesses.
3. Evaluate DKD as a possible component or baseline, especially for the secondary class-recognition dimension.
4. Do not let classification improvements stand in for localization improvements.

Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html  
Code: https://github.com/megvii-research/mdistiller  
arXiv: https://arxiv.org/abs/2203.08679

### 3.4 KD-Zero: Evolving Knowledge Distiller for Any Teacher-Student Pairs — Li et al., NeurIPS 2023

**Type:** Primary research paper; an automated way to discover a Knowledge Distillation recipe for a Teacher–Student pair.

**First, the basic idea — imagine you are teaching a student:**

Suppose a strong Teacher knows a subject very well, and we want a smaller Student to learn from it. We know the Student should learn from the Teacher, but there are many possible ways to pass that knowledge across. For example:

- Should we compare the Teacher's and Student's intermediate features?
- Should we compare their final predictions?
- Should we compare relationships between different parts of their features?
- Which mathematical distance should measure how different they are?
- How strongly should each comparison affect the training?

These choices together make up a **distiller**: the recipe that decides *what knowledge is transferred and how the Student is taught to match it*.

The problem is that researchers often choose this recipe by hand. A recipe that works well for one Teacher–Student pair may not be the best recipe for another pair. Trying every possible combination manually would also take a lot of time.

**What KD-Zero does:**

Instead of asking a researcher to design the whole recipe manually, KD-Zero creates many candidate recipes and searches among them automatically.

The paper describes a distiller using three kinds of building blocks:

1. **Knowledge transformations:** What form of the Teacher's knowledge should be passed to the Student? For example, predictions or transformed feature information.
2. **Distance functions:** How should we measure the difference between the Teacher's and Student's selected knowledge?
3. **Loss weights:** How much should each part of the distillation objective matter during training?

KD-Zero combines these building blocks into candidate distillers, evaluates candidates, and uses an evolutionary search process to produce better candidates. In simple terms, it repeatedly tries recipes, keeps promising ones, and creates new variations from them. The paper also introduces ways to reduce wasted search, such as rejecting unpromising candidates early and narrowing the search space.

**A simple example (illustrative, not an experiment from the paper):**

Imagine we have three possible teaching signals:
- A: compare the final predictions.
- B: compare an intermediate feature map.
- C: compare relationships between feature locations.

And two possible ways to measure differences:
- X: distance function X.
- Y: distance function Y.

There are already several possible recipes: A+X, A+Y, B+X, B+Y, and so on. We could also vary how strongly each loss is weighted. KD-Zero's central idea is to search through combinations like these instead of assuming that one manually chosen recipe is always best.

**Reported results:** The authors report improvements over compared KD methods across classification, object detection, and segmentation, and across diverse Teacher–Student pairs. These are the paper's experimental findings; they do not guarantee the same gains in our project.

**How it relates to our project:**

KD-Zero is important because it shows that the distillation recipe itself can be searched rather than fixed in advance. But there is a distinction:

- **KD-Zero:** searches for a good distiller for a given Teacher–Student pair.
- **Our intended direction:** evaluates the Student, identifies a specific weakness (especially a localization weakness), and then uses that evidence to decide what part of distillation may need to change. The result must be tested to see whether that weakness actually improves.

These ideas may eventually fit together: KD-Zero can inspire *how to search among possible distillation recipes*, while our evaluation loop can help decide *what problem the search should focus on*. This is a possible research direction, not a finalized design.

**Ideas to retain for our project:**
1. Do not assume one fixed KD recipe is optimal for every Teacher–Student pair.
2. Treat transformations, distance functions, and loss weights as possible design choices.
3. Start with a small, understandable search space; expand it only when experiments justify doing so.
4. Use search-cost controls and measure the cost of the search.
5. Compare any adaptive/search-based strategy against a fixed KD baseline.

Paper: https://proceedings.neurips.cc/paper_files/paper/2023/hash/dbc8ce0fdfcd55172d73fb05dbae07fc-Abstract.html

### 3.5 Localization Distillation for Dense Object Detection — Zheng et al., CVPR 2022

**Type:** Primary research paper; directly relevant to our localization priority.

**Problem:** Earlier detector KD methods often emphasized feature imitation. Transferring precise localization information through prediction logits was considered difficult, so localization knowledge was not being used effectively.

**What the authors do:** They reformulate distillation to transfer localization information from Teacher to Student. They also introduce the idea of a **valuable localization region**, so the method can selectively emphasize regions that carry useful localization and semantic information.

**Reported result:** On the paper's GFocal-ResNet-50 COCO setup, the reported AP increases from 40.1 to 42.1 under the stated single-scale 1x training schedule, without sacrificing inference speed. These are the paper's reported experimental results, not a guaranteed outcome for our setup.

**Why this is especially important to us:** It directly addresses the distinction we care about: teaching a Student not only what an object is, but where it is. The paper reports that localization distillation can be more important and efficient than semantic knowledge transfer for the detector settings studied.

**Ideas to retain for our project:**
1. Make localization-specific metrics and errors first-class evaluation targets.
2. Investigate localization-focused distillation signals, not just classification logits or generic feature losses.
3. Explore whether regions with weak localization can receive targeted emphasis.
4. Test whether an adaptive system can identify when localization transfer needs adjustment.

Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html  
arXiv: https://arxiv.org/abs/2204.05957  
Code: https://github.com/HikariTJU/LD

### 3.6 Focal and Global Knowledge Distillation for Detectors (FGD) — Yang et al., CVPR 2022

**Type:** Primary research paper; feature-based detector distillation.

**Problem:** Teacher and Student feature maps differ across image regions. Treating foreground and background equally can make distillation less effective.

**What the authors do:**
- **Focal distillation** separates foreground and background and focuses on important pixels and channels.
- **Global distillation** transfers relationships among different positions to preserve broader contextual information.

**Reported results:** The paper reports improvements on COCO2017 for several detector families, including RetinaNet, Faster R-CNN, RepPoints, and Mask R-CNN.

**Why it matters to us:** It is another localization-adjacent idea: not all spatial regions are equally useful. It may inform region weighting or a combination of local and global signals. It is not itself an evaluation-driven self-repair system.

Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html

### 3.7 Elastic Response Distillation (ERD) — Feng et al., CVPR 2022

**Type:** Primary research paper for incremental object detection.

**Problem:** When a detector is fine-tuned on new classes using only new-class data, its performance on previously learned classes can collapse. This is catastrophic forgetting.

**What the authors do:** ERD distills responses from both the classification head and the regression head. Its Elastic Response Selection strategy selects valuable responses and assigns different importance to different response types.

**Why it matters to us:** It is an example of selecting and weighting response information, including localization-related regression responses.

**Important limitation:** Its central setting is incremental/continual object detection. Our initial project is not necessarily incremental learning. We should borrow useful design ideas only if our experiments need them; we should not make catastrophic forgetting the main project problem without a reason.

Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Feng_Overcoming_Catastrophic_Forgetting_in_Incremental_Object_Detection_via_Elastic_Response_CVPR_2022_paper.html

## 4. Current lessons for our project

These are working takeaways from the papers above, not final implementation decisions:

1. **Localization comes first.** The main evaluation must measure where the Student places objects and how accurate its predicted boxes are. Classification remains important, but secondary.
2. **Localization Distillation deserves close attention.** It is directly aligned with the main objective and should be one of the first methods we understand and potentially reproduce.
3. **DKD is worth keeping.** Its separation of target and non-target class knowledge is useful, but the original paper's detection results show that DKD alone should not be treated as a complete localization solution.
4. **KD-Zero is worth keeping.** Automated search over distillation components is a strong source of ideas. We should begin with a constrained and understandable search/adaptation space rather than immediately implementing a large evolutionary search.
5. **FGD offers spatial weighting ideas.** Foreground/background separation and local/global information may help decide which regions deserve more distillation attention.
6. **ERD is a conditional reference.** Its response selection and regression-head ideas may be useful, but its continual-learning setting is not automatically our setting.
7. **Compute is a design trade-off, not an absolute blocker.** We should optimize unnecessary work and measure cost, but the project has access to comparatively strong local compute. We do not need to reject a promising method merely because it is heavier. The first prototype can still stay simple.
8. **Parameter complexity should stay proportional to the experiment.** We can begin with a small number of understandable parameters and expand only when evidence justifies it.
9. **No novelty claim yet.** These papers overlap with parts of our direction. We need to identify precisely what our evaluation-triggered, localization-prioritized adaptation loop adds, if anything, after comparing the primary methods.

## 5. Working comparison

| Method | What it changes or transfers | Main relevance to us | Initial status |
|---|---|---|---|
| DKD | Separates target-class and non-target-class logit distillation | Flexible control of class-related knowledge; secondary priority | Keep as candidate |
| KD-Zero | Searches transformations, distance functions, and loss weights | Automated discovery/adaptation of distillation strategies | Keep as major inspiration |
| Localization Distillation | Transfers localization distributions and emphasizes valuable regions | Directly aligned with localization-first objective | High-priority method to study |
| FGD | Weights focal foreground/background features and global relations | Spatially selective feature transfer | Candidate to study |
| ERD | Selects and weights classification and regression responses in incremental detection | Response selection and localization regression transfer | Conditional reference |

## 6. Practical constraints and preferences

- Prefer an efficient implementation, but do not impose an artificially tiny compute budget: strong local compute is available for heavier experiments.
- Keep the initial prototype and parameter space understandable. Avoid premature complexity.
- Measure training/search overhead, but judge it against the value of the resulting localization improvement.
- Keep the Teacher fixed during the self-repair loop.
- Compare adaptive KD with a fair fixed-KD baseline.
- Evaluate localization directly; do not use classification accuracy or aggregate mAP alone as a substitute for localization analysis.
- Preserve negative and inconclusive results as part of the research record.

## 7. What we should read next

Read the primary papers in this order:
1. **Localization Distillation** — because localization is our first priority.
2. **DKD** — understand the two separately weighted knowledge components and its limitations on detection.
3. **KD-Zero** — understand how the distiller search space is built and how search cost is controlled.
4. **FGD** — understand spatially selective feature distillation.
5. **ERD** — only to extract ideas about response selection and regression transfer, while keeping its incremental-learning assumption clear.

## 8. References

1. Li et al., “When Object Detection Meets Knowledge Distillation: A Survey,” IEEE TPAMI, 2023. https://doi.org/10.1109/TPAMI.2023.3257546
2. “Knowledge Distillation in Object Detection: A Survey from CNN to Transformer,” Sensors, 2026. https://www.mdpi.com/1424-8220/26/1/292
3. Zhao et al., “Decoupled Knowledge Distillation,” CVPR 2022. https://doi.org/10.1109/CVPR52688.2022.01165
4. Li et al., “KD-Zero: Evolving Knowledge Distiller for Any Teacher-Student Pairs,” NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/dbc8ce0fdfcd55172d73fb05dbae07fc-Abstract.html
5. Zheng et al., “Localization Distillation for Dense Object Detection,” CVPR 2022. https://doi.org/10.1109/CVPR52688.2022.00919
6. Yang et al., “Focal and Global Knowledge Distillation for Detectors,” CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html
7. Feng et al., “Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation,” CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Feng_Overcoming_Catastrophic_Forgetting_in_Incremental_Object_Detection_via_Elastic_Response_CVPR_2022_paper.html

---
**Checkpoint 2 note:** This is an introductory working review. It records papers we have used and ideas to retain; it does not claim the review is exhaustive or that any method has been selected for implementation.
