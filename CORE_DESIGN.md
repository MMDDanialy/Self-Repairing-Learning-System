# Core Design Principles — Self-Repairing Learning System

## 1. Core Learning Evaluation: Three Tests

The system must evaluate learning through three complementary tests.

### 1.1 Performance Test
Does the model produce correct predictions on the evaluation data?

### 1.2 Generalization Test
Can the model perform successfully on genuinely unseen examples rather than merely memorizing the training distribution?

### 1.3 Dependency and Shortcut Test
Does the model rely on irrelevant or misleading features? For example, does changing the background affect predictions even when the relevant object remains unchanged?

These tests must be considered together. Strong performance on ordinary evaluation data alone is not sufficient evidence of reliable learning.

## 2. Core Decision-Making Loop

The system is organized around three fundamental decisions.

### Decision 1 — Has the model learned sufficiently?
Evaluate performance, generalization, and dependency on potentially misleading features to determine whether learning meets predefined acceptance criteria.

### Decision 2 — If not, what is the likely cause, and which experiment can test it?
Generate plausible explanations for the observed failure and select an experiment capable of distinguishing between competing hypotheses.

### Decision 3 — Did the correction genuinely improve the model, or merely move the errors elsewhere?
Compare the corrected model against the previous version across relevant evaluation criteria, including previously learned capabilities. Accept a correction only when the evidence supports improvement without unacceptable regressions.

## 3. Iterative Self-Correction Cycle

Learning → Testing → Evaluation → Failure Diagnosis → Experiment Selection → Targeted Correction → Re-evaluation

The cycle repeats until the acceptance criteria are met, the available experiments are exhausted, or the system reaches a defined stopping condition.

## 4. Evidence-Based Decisions

The system must distinguish observed evidence from hypotheses. Diagnoses are provisional until tested, and every correction must be evaluated empirically.

## 5. Initial Research Scope

The first implementation will focus on a controlled machine-learning task involving shortcut learning and distribution shifts. Its effectiveness will be compared against appropriate baseline methods before making claims about novelty or general applicability.
