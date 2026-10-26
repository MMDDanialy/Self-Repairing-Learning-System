# Core Design Principles — Self-Repairing Learning System

## 1. Project Goal

Build and experimentally evaluate a locally executable machine-learning system that can identify meaningful model failures, form and test possible explanations, choose informative diagnostic experiments, apply targeted corrections, and verify that those corrections improve the intended behavior without unacceptable regressions.

The initial scope is deliberately narrow: a controlled computer-vision task involving shortcut learning and distribution shifts. Claims of novelty, broad applicability, or autonomous repair must be supported by related-work review and reproducible experiments.

## 2. Core Evaluation: Three Complementary Tests

### 2.1 Performance Test
Does the model make correct predictions on the defined evaluation data?

### 2.2 Generalization Test
Does it perform on genuinely unseen examples, rather than only on examples resembling its training distribution?

### 2.3 Dependency and Shortcut Test
Does it rely on irrelevant or misleading features? For example, does changing the background change a prediction when the relevant object remains unchanged?

These tests must be considered together. Strong performance on ordinary evaluation data alone is not sufficient evidence of reliable learning.

## 3. Core Decision-Making Loop

### Decision 1 — Has the model learned sufficiently?
Evaluate performance, generalization, and dependence on potentially misleading features against predefined acceptance criteria.

### Decision 2 — If not, what may have caused the failure, and which experiment can test it?
Generate competing, testable hypotheses from observed evidence. Select an experiment that can distinguish among those hypotheses; treat diagnoses as provisional until tested.

### Decision 3 — Did the correction genuinely improve the model?
Compare the corrected model with the previous version across target behavior and retained capabilities. Accept a correction only when the evidence supports improvement without unacceptable regressions.

## 4. Iterative Self-Correction Cycle

Learning → Testing → Evaluation → Failure Diagnosis → Experiment Selection → Targeted Correction → Re-evaluation

Repeat until acceptance criteria are met, useful experiments are exhausted, resources reach a defined limit, or a stopping condition is triggered. Preserve prior model versions so unsuccessful corrections can be rejected or rolled back.

## 5. Evidence and Reproducibility

- Separate observations, hypotheses, decisions, and conclusions in logs.
- Define evaluation metrics and acceptance criteria before comparing corrections.
- Record experiments, model/data versions, changes, resource costs, and results.
- Compare against appropriate baselines; do not claim novelty before reviewing related work.
- Evaluate regressions as well as gains on the targeted failure.

## 6. Initial Scope and Constraints

- Start with a controlled machine-learning/computer-vision task involving shortcut learning and distribution shifts.
- Prefer local execution and methods that can run within modest hardware limits.
- Do not make a cloud LLM essential to diagnosis, experiment selection, or correction.
- Choose the baseline model, dataset, and implementation details only after the research and experimental-design checkpoints justify them.
- Treat autonomous behavior and generality as claims to test, not assumptions.

## 7. Relationship to the Project Roadmap

`ROADMAP.md` contains the approved 12-checkpoint development and evaluation plan, including the expected outputs and completion criteria for each stage.
