# Resource plan and direction budget

Authoritative detailed protocol: `.neurico/idea.yaml`, read before selection. No subject generations or judge calls during resource finding. Ranking frozen 2026-10-20 before results.

Scores 1–5; weighted total = 0.25 literature evidence + 0.35 hypothesis relevance + 0.20 information gain + 0.20 feasibility. Scores are planning judgments, not empirical measurements.

| Direction | Evidence | Relevance | Gain | Feasibility | Total | Decision |
|---|---:|---:|---:|---:|---:|---|
| Fixed suffix paired test on Qwen3-8B | 4 | 5 | 5 | 5 | 4.75 | Keep 1: primary hypothesis |
| Identical suffix on specified DeepSeek and Moonlight | 4 | 5 | 4 | 4 | 4.35 | Keep 2: preregistered secondary comparisons |
| Dual-judge controls, missingness and disagreement sensitivity on same responses | 5 | 4 | 4 | 4 | 4.25 | Keep 3: required measurement validity |
| Confidence elicitation / threshold sweep | 4 | 2 | 4 | 3 | 3.10 | Prune: changes sole intervention and adds tuning |
| Substitute SimpleQA Verified | 5 | 1 | 3 | 5 | 3.20 | Prune: changes mandated dataset; retain label-error evidence only |
| R-Tuning / fine-tuning for abstention | 4 | 1 | 3 | 1 | 2.15 | Prune: changes checkpoints and requires training |
| Retrieval-augmented answering | 4 | 1 | 2 | 2 | 2.15 | Prune: explicitly prohibited for subjects |
| Full AbstentionBench suite | 4 | 2 | 3 | 2 | 2.70 | Prune: different estimand, scope and cost |

Evidence: SimpleQA paper sections 2.4–4 motivates coverage/correctness decomposition; AbstentionBench sections 3–4 motivates checkpoint specificity and reasoning-mode logging; R-Tuning sections 2–4 and Kadavath sections 1–5 motivate excluded alternatives; SimpleQA Verified sections 2–3 motivates judge and label-error sensitivity. Only these top three directions may be implemented; do not expand after results unless evidence invalidates ranking and the change is recorded in STATE.md.

## Phase sequence
1. Literature: download and validate five targeted papers; deep-read all SimpleQA chunks and screen other papers' abstracts/methods/results.
2. Data: download the exact SimpleQA revision used by pinned Inspect; validate and hash it; derive disjoint 50/400 item manifest and 150 synthetic controls.
3. Code: sparse clone Inspect evaluations plus official simple-evals; pin commits; inspect grading and generation defaults; snapshot model metadata/configs without downloading weights.
4. Synthesis: catalog paths, reproducible downloads, frozen design, and runner blockers.
5. Verify every deliverable, update owned STATE block, write completion marker.


## Motivation & Novelty Assessment
### Why This Research Matters
Factual assistants can reduce wrong answers simply by answering less. Users and evaluators need a paired measurement of errors and retained correct answers to distinguish useful selective answering from indiscriminate abstention.
### Gap in Existing Work
The gathered SimpleQA and abstention literature does not establish the fixed 0.10 error-reduction / 0.03 correct-retention criterion for these three pinned checkpoints. Historical benchmark scores and synthetic judge controls are not natural-response calibration.
### Our Novel Contribution
A narrowly scoped, preregistered replication-style checkpoint comparison with the sole fixed suffix, paired questions, repeated seeds, and explicit judge/missingness gates. No claim of a new abstention method or exhaustive novelty search.
### Experiment Justification
1. Primary Qwen3 contrast directly tests the joint hypothesis.
2. Identical DeepSeek and Moonlight contrasts test checkpoint-specific transfer, Holm adjusted.
3. Gold/negation/abstention controls and dual grading test measurement implementation and sensitivity, without establishing human-ground-truth validity.

## Execution Plan (experiment runner)
The preregistration is the detailed statistical plan. Independent variable: suffix absent/present. Outcomes: incorrect and correct frequencies, not-attempted, coverage, attempted accuracy, invalid/truncated/unscored. Alternative explanations include response length, reasoning truncation, original label error and judge domain shift. Use exactly the frozen 400 items and three seeds; no replacement models or interventions.

Timeline within 60 minutes: 0–8 resource/runtime checks; 8–15 download, fixtures and configuration freeze; 15–43 sequential GPU generations; 43–52 local controls/grading and any budget-authorized Azure work; 52–60 paired analysis, report and validation. Roughly 20% is reserved for debugging/documentation; slow downloads may prevent completion. Checkpoint all completed batches. Missingness is never a model factuality score.

Success requires the exact joint preregistered inequalities and measurement gates; confidence intervals crossing retention are inconclusive. No retrieval, training, or subject API substitution. Azure headroom must be verifiable before paid calls. An unavailable ledger means deferred Azure judging, not inferred headroom.
