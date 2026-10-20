# SimpleQA factual errors versus abstention on three open checkpoints

Report generated 2026-10-20T11:54:12.249806+00:00; execution-stage elapsed wall time 52.7 minutes, including setup and diagnostics (one execution, descriptive timing only).

## 1. Executive Summary

The primary hypothesis is inconclusive: excessive unscored judge outputs prevent a valid factuality conclusion, and conservative missing-label bounds do not establish correct-answer retention. Azure dual grading is also incomplete.

This execution produced **7,200 of 7,200 planned subject completions** and **6,823 local factuality labels**. The unit of statistical independence is the question: 400 questions per checkpoint, two conditions and three seeds. Completed responses and absent planned responses are retained explicitly. Azure judging made zero calls and spent $0 because the campaign’s $50/day cap was visible but its current spend/headroom ledger was not available. No OpenRouter was used. No candidate finding is emitted.

Synthetic gold/negation/abstention controls test rubric implementation, not calibration on natural model completions. Local scores remain provisional even if controls pass. Explicit factual abstention is not a safety-refusal metric. Reduced answer coverage alone is not evidence of improved factual knowledge.

## 2. Research Question & Motivation

The primary joint hypothesis is that adding exactly `If you are uncertain, answer exactly I do not know.` lowers Qwen3-8B incorrect-answer frequency by at least 0.10 while reducing correct-answer frequency by at most 0.03. Identical secondary tests target the specified DeepSeek and Moonlight checkpoints. A paired design distinguishes selective avoidance of errors from indiscriminately answering less.

### Literature review summary

The gathered SimpleQA paper motivates the correct/incorrect/not-attempted decomposition and cautions about benchmark label error. AbstentionBench motivates checkpoint-specific tests and distinguishes answerable factual questions from inherently unanswerable questions. R-Tuning and Language Models (Mostly) Know What They Know concern training or confidence methods outside this sole-suffix intervention. SimpleQA Verified motivates label-noise and grader-domain-shift caveats; it is not substituted for the mandated dataset. This is a narrow replication-style checkpoint study, not a new abstention algorithm. See [literature_review.md](literature_review.md) and [planning.md](planning.md).

## 3. Methodology

### Frozen data and prompts

The pinned Inspect SimpleQA source contains 4,326 rows. CSV SHA256: `feee3f7e7db3617e94e8fcf1977b756ec420ef8568f4e0fcbbe0e92e9d5fc032`. Canonical IDs use the pinned Inspect stable-ID helper. Sort SHA256 of `20261020:` plus ID: first 50 questions for controls, next 400 for main evaluation. All selected IDs/questions are distinct and the splits are disjoint. The reproduced item-manifest SHA256 is `3b3d0740316431aed3f759f5d5cd14c38bb99d00d36e68e80d10a99af4e42513`.

Baseline user content is the raw question, matching the gathered reference implementation. Intervention adds exactly two newlines and the fixed sentence. There are no subject tools, browsing, retrieval, demonstrations or confidence prompts. Native chat templates are preserved; Qwen optional thinking is disabled through its template flag; DeepSeek reasoning is parsed only after `</think>`; Moonlight’s native default system message is retained.

| Catalog checkpoint | HF ID | Pinned revision |
|---|---|---|
| qwen2.5-7b-judge | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` |
| qwen3-8b | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| moonlight-16b-a3b-instruct | `moonshotai/Moonlight-16B-A3B-Instruct` | `4e735b07a89f73647dfab71ab91b840f362ede5b` |
| deepseek-r1-distill-llama-8b | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `6a6f4aa4197940add57724a7707d069478df56b1` |

### Runtime and protocol deviations

The project uses its isolated `.venv`, Python 3.12.13, vLLM 0.11.0, Transformers 4.57.1, Torch 2.8.0, Inspect AI 0.3.282, and Inspect Evals 0.3.106. Full versions are in `results/runtime/versions.json`; dependencies are locked in `uv.lock`.

GPU 0 is an NVIDIA A100 80GB PCIe (81,920 MiB total). Only that allocated GPU was used. An unrelated Qwen3-4B service began consuming about 37 GB during setup; it was neither queried nor stopped. Our pre-main amendment lowers memory reservations to 0.45 for Qwen3/DeepSeek, 0.50 for Moonlight, and 0.42 for the local judge, instead of the larger subject catalog reservations. Each subject and the judge run in separate sequential processes, one model at a time, tensor_parallel_size=1, bfloat16, eager execution, batch concurrency up to 64, context 8192, batched token budget 8192. No training was performed; a training-time comparison is not applicable. Concurrent external GPU load limits throughput comparability.

Real local checkpoint inference uses the vLLM offline engine API, a documented departure from an HTTP-serving harness. Qwen uses original 64-request batch barriers; before DeepSeek/Moonlight collection, scheduling changed to continuously replace finished requests while retaining max_num_seqs=64. Both scheduler implementations and hashes are retained. This operational amendment is documented in infrastructure_events.json. Every model completes a non-main arithmetic smoke fixture before collection. Temperature is 0.6, seeds 11/22/33. Native generation-config top-p/top-k/repetition penalty are serialized in per-model runtime JSON. Nonreasoning answers have max_tokens=256. DeepSeek has a 4096 total cap; cumulative engine steps abort at 256 tokens after its native closing-thought token, with an overshoot assertion. No forced thought closure or last-line extraction is used. Raw token IDs, traces, finals, stop reasons and hashes are saved.

The judge is loaded only after subject processes exit. It sees original question, gold and extracted final answer, with no model, condition, seed or reasoning. The original SimpleQA A/B/C rubric is frozen. Judge temperature 0, top-p 1, max_tokens=16; exact A/B/C parse failures stay unscored. Available truncated finals are graded but separately flagged; missing final boundaries remain invalid/unscored. Identical question/gold/final/rubric/judge identities reuse cached grades.

Azure budget handling is a material incompleteness: no authoritative daily spend ledger was found, so no paid probe, calibration or main grading was performed. This is not an Azure calibration failure or evidence of ineligibility. The missing `evals/judge-calibration.md` establishes nothing for this task. The local judge shares Qwen ancestry with Qwen3, as mandated; no policy or prompt optimization against it was performed.

### Statistical analysis and gates

Each item’s three binary seed outcomes are averaged before computing means/SEs and paired intervention-minus-baseline effects. Percentile CIs use 10,000 bootstrap resamples of 400 item clusters, seed 20261020; linear SEs use the sample SD across item averages divided by sqrt(400). Ratios use bootstrap SEs. Synthetic calibration resamples 50 question clusters, preserving each triple. Primary support requires Δincorrect ≤ −0.10, two-sided 95% CI upper bound <0, and one-sided 95% lower bound for Δcorrect ≥ −0.03.

Secondary tests use paired one-sided t tests for incorrect decrease (margin 0) and correct retention (margin −0.03), joint intersection-union p=max(component p values), with Holm adjustment over the two checkpoints. The observed effect threshold and bootstrap gates still apply. The fixed sample may be underpowered for retention; failure to pass does not establish equivalence, absence of risk or no effect.

Missing labels contribute to the explicit unscored category. Invalid and truncated are overlapping diagnostic flags, not additional mutually exclusive outcome classes. Known-class frequencies are observed contributions over all planned items, not missing-at-random imputations. Adversarial bounds put unknown treatment outcomes at 0 and baseline at 1 for the minimum effect, reversing those assignments for the maximum. Both bound endpoints have bootstrap intervals in `results/analysis.json`.

## 4. Results

Counts below are realized inventory counts, not estimated population quantities. Reported rate SEs and CIs cluster by question; all main rate rows have n=400 questions and 1,200 planned completions per condition.

| Checkpoint | Generated / planned | Locally scored / planned |
|---|---:|---:|
| qwen3-8b | 2400 / 2400 | 2213 / 2400 |
| deepseek-r1-distill-llama-8b | 2400 / 2400 | 2322 / 2400 |
| moonlight-16b-a3b-instruct | 2400 / 2400 | 2288 / 2400 |

### Synthetic-control judge validation

Local judge eligibility: **True**. n=150 controls, 50 independent questions; unscored count=0.

| Metric | Mean, SE and cluster-bootstrap 95% CI |
|---|---|
| accuracy | 1.0000 (SE 0.0000; 95% CI 1.0000, 1.0000) |
| macro_f1 | 1.0000 (SE 0.0000; 95% CI 1.0000, 1.0000) |
| kappa | 1.0000 (SE 0.0000; 95% CI 1.0000, 1.0000) |
| correct_false_positive_rate | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) |

Confusion matrix rows are true A/B/C; columns are predicted A/B/C/unscored:

```json
[[50, 0, 0, 0], [0, 50, 0, 0], [0, 0, 50, 0]]
```

The correct-label false-positive rate uses 100 negative controls (true B or C) as its denominator; accuracy, macro-F1 and kappa use all 150 controls. Bootstrap intervals can be degenerate on perfectly classified controls and do not establish perfect reliability. Azure controls: not run (n=0; metrics undefined). Inter-judge agreement/disagreement and Azure sensitivity analyses are unavailable. No task-matched human-labelled completion calibration set was located in the documented resource search.

### Local response decomposition

Each cell is mean (SE; 95% CI); n=400 question clusters. When missingness is nonzero these are known-label contributions, with bounds reported separately.

| Checkpoint / condition | Correct | Incorrect | Not attempted | Unscored |
|---|---|---|---|---|
| qwen3-8b / baseline | 0.0225 (SE 0.0061; 95% CI 0.0117, 0.0350) | 0.9050 (SE 0.0108; 95% CI 0.8833, 0.9258) | 0.0050 (SE 0.0023; 95% CI 0.0008, 0.0100) | 0.0675 (SE 0.0092; 95% CI 0.0500, 0.0858) |
| qwen3-8b / abstention | 0.0058 (SE 0.0036; 95% CI 0.0000, 0.0133) | 0.0442 (SE 0.0094; 95% CI 0.0267, 0.0642) | 0.8617 (SE 0.0168; 95% CI 0.8283, 0.8925) | 0.0883 (SE 0.0142; 95% CI 0.0617, 0.1175) |
| deepseek-r1-distill-llama-8b / baseline | 0.0250 (SE 0.0054; 95% CI 0.0150, 0.0358) | 0.9383 (SE 0.0078; 95% CI 0.9225, 0.9533) | 0.0050 (SE 0.0020; 95% CI 0.0017, 0.0092) | 0.0317 (SE 0.0056; 95% CI 0.0217, 0.0433) |
| deepseek-r1-distill-llama-8b / abstention | 0.0200 (SE 0.0050; 95% CI 0.0108, 0.0308) | 0.8933 (SE 0.0105; 95% CI 0.8717, 0.9133) | 0.0533 (SE 0.0074; 95% CI 0.0400, 0.0683) | 0.0333 (SE 0.0061; 95% CI 0.0217, 0.0458) |
| moonlight-16b-a3b-instruct / baseline | 0.0450 (SE 0.0082; 95% CI 0.0300, 0.0617) | 0.9300 (SE 0.0095; 95% CI 0.9108, 0.9483) | 0.0167 (SE 0.0043; 95% CI 0.0083, 0.0258) | 0.0083 (SE 0.0029; 95% CI 0.0033, 0.0142) |
| moonlight-16b-a3b-instruct / abstention | 0.0008 (SE 0.0008; 95% CI 0.0000, 0.0025) | 0.0033 (SE 0.0017; 95% CI 0.0008, 0.0067) | 0.9108 (SE 0.0140; 95% CI 0.8825, 0.9367) | 0.0850 (SE 0.0140; 95% CI 0.0600, 0.1125) |

![Local response decomposition](figures/response_decomposition.png)

### Coverage and attempt-conditioned accuracy

| Checkpoint / condition | Coverage (C+I)/N | Accuracy C/(C+I) | Attempted completion denominator |
|---|---|---|---:|
| qwen3-8b / baseline | 0.9275 (SE 0.0094; 95% CI 0.9083, 0.9450) | 0.0243 (SE 0.0065; 95% CI 0.0125, 0.0379) | 1113 |
| qwen3-8b / abstention | 0.0500 (SE 0.0102; 95% CI 0.0317, 0.0708) | 0.1167 (SE 0.0680; 95% CI 0.0000, 0.2639) | 60 |
| deepseek-r1-distill-llama-8b / baseline | 0.9633 (SE 0.0059; 95% CI 0.9517, 0.9742) | 0.0260 (SE 0.0056; 95% CI 0.0156, 0.0374) | 1156 |
| deepseek-r1-distill-llama-8b / abstention | 0.9133 (SE 0.0094; 95% CI 0.8942, 0.9308) | 0.0219 (SE 0.0055; 95% CI 0.0119, 0.0334) | 1096 |
| moonlight-16b-a3b-instruct / baseline | 0.9750 (SE 0.0051; 95% CI 0.9642, 0.9842) | 0.0462 (SE 0.0084; 95% CI 0.0306, 0.0636) | 1170 |
| moonlight-16b-a3b-instruct / abstention | 0.0042 (SE 0.0019; 95% CI 0.0008, 0.0083) | 0.2000 (SE 0.2045; 95% CI 0.0000, 0.6667) | 5 |

Attempted accuracy is conditional on scored attempted answers; missingness is not assumed random. Native F-score is `2C/(2C+2I+A)`; its means/SEs/CIs are in the JSON. With missing labels its reported known-label-subset value is sensitivity analysis, not a complete native score.

### Paired effects

| Checkpoint | Δincorrect (n=400) | Δcorrect (n=400) | Correct one-sided 95% lower bound | Joint p | Holm p (secondary) |
|---|---|---|---:|---:|---:|
| qwen3-8b | -0.8608 (SE 0.0146; 95% CI -0.8892, -0.8308) | -0.0167 (SE 0.0060; 95% CI -0.0292, -0.0058) | -0.0267 | 0.0128977 | primary |
| deepseek-r1-distill-llama-8b | -0.0450 (SE 0.0110; 95% CI -0.0675, -0.0233) | -0.0050 (SE 0.0050; 95% CI -0.0150, 0.0050) | -0.0133 | 2.46446e-05 | 4.928918482111142e-05 |
| moonlight-16b-a3b-instruct | -0.9267 (SE 0.0096; 95% CI -0.9450, -0.9067) | -0.0442 (SE 0.0080; 95% CI -0.0608, -0.0292) | -0.0575 | 0.961286 | 0.961285649530423 |

Conservative correct-frequency effect bounds (n=400 paired questions):

| Checkpoint | Minimum Δcorrect | Maximum Δcorrect |
|---|---|---|
| qwen3-8b | -0.0842 (SE 0.0107; 95% CI -0.1058, -0.0633) | 0.0717 (SE 0.0156; 95% CI 0.0417, 0.1033) |
| deepseek-r1-distill-llama-8b | -0.0367 (SE 0.0075; 95% CI -0.0517, -0.0225) | 0.0283 (SE 0.0080; 95% CI 0.0125, 0.0442) |
| moonlight-16b-a3b-instruct | -0.0525 (SE 0.0086; 95% CI -0.0700, -0.0367) | 0.0408 (SE 0.0164; 95% CI 0.0100, 0.0733) |

For Qwen, the minimum-effect one-sided 95% lower bound is -0.1017, below the -0.03 retention margin. Missing labels therefore prevent establishing retention.


These tests describe available local labels; missingness and judge gates must be considered before interpretation. No candidates are emitted: every checkpoint exceeds the 2% unscored gate, and dual judging is incomplete. Qwen observed-label numerical support does not survive the measurement gates. DeepSeek’s observed decrease misses the 0.10 effect-size threshold despite its small adjusted p value; Moonlight does not establish retention. These are conditional descriptions, not supported factuality findings.

### Missingness, truncation and repetition

| Checkpoint / condition | Infrastructure missing | Invalid/no-final | Truncated | Literal `I do not know` |
|---|---|---|---|---|
| qwen3-8b / baseline | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) | 0.0458 (SE 0.0084; 95% CI 0.0300, 0.0633) | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) |
| qwen3-8b / abstention | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) |
| deepseek-r1-distill-llama-8b / baseline | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) | 0.0033 (SE 0.0017; 95% CI 0.0008, 0.0067) | 0.0092 (SE 0.0030; 95% CI 0.0042, 0.0158) | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) |
| deepseek-r1-distill-llama-8b / abstention | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) | 0.0008 (SE 0.0008; 95% CI 0.0000, 0.0025) | 0.0067 (SE 0.0026; 95% CI 0.0025, 0.0125) | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) |
| moonlight-16b-a3b-instruct / baseline | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) | 0.0008 (SE 0.0008; 95% CI 0.0000, 0.0025) | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) |
| moonlight-16b-a3b-instruct / abstention | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) | 0.0000 (SE 0.0000; 95% CI 0.0000, 0.0000) |

Per-seed rates with n/SE/CIs, seed-rate standard deviations, and mean within-item seed SDs are in `results/analysis.json` under `seed_metrics` and `repeat_variability`. Three repetitions are not three independent item samples. Missing planned records are explicitly added to item-score files, with infrastructure labels.

### Call counts, tokens and cost

| Subject | Logical main requests | Input tokens | Output tokens |
|---|---:|---:|---:|
| qwen3-8b | 2400 | 100,395 | 120,777 |
| deepseek-r1-distill-llama-8b | 2400 | 83,811 | 1,757,421 |
| moonlight-16b-a3b-instruct | 2400 | 108,180 | 81,756 |

These are exact logged counts, not estimated rates. Subject generation uses the local engine step API; logical requests are distinguished from batch method invocations.

Local judge: 5319 unique cached rubric evaluations, 8,056,645 input tokens and 11,998 output tokens. Paid judge calls: 0; paid judge input/output tokens: 0/0; spend: $0 of the $12 job cap. GPU compute cost is not measured. Non-main arithmetic smoke calls are separate from calibration and main totals; their raw fixture artifacts are retained.

## 5. Analysis & Discussion

The primary hypothesis is inconclusive: excessive unscored judge outputs prevent a valid factuality conclusion, and conservative missing-label bounds do not establish correct-answer retention. Azure dual grading is also incomplete.

The observed-label Qwen contrast meets the numerical error/retention rule, but this is not a supported result: 187/2400 Qwen responses are unscored, above the preregistered 2% suppression threshold. There are 78/2400 unscored DeepSeek responses and 112/2400 unscored Moonlight responses. These are exact inventory counts; condition-specific rate SEs/CIs are reported above. Five missing labels arise from DeepSeek no-final parsing; the other 372 arise from malformed judge replies. The strict parser is not relaxed after seeing these results. Malformed replies include repeated labels such as `B B`, conflicting text such as `C CORRECT`, and invented dialogue continuations. These are formatting failures under the fixed protocol, not known factual misjudgments. The key comparison is error reduction alongside retained correct answers. A larger not-attempted fraction and smaller incorrect fraction can arise solely through coverage reduction; neither demonstrates improved stored knowledge. Narrow retention intervals and judge agreement matter to the joint criterion. Any apparent differences between checkpoints are descriptive for these weights and protocols and cannot be attributed causally to distillation or generalized to model families.

The study records infrastructure problems separately from model-generated failures. The pre-main package-resolution issue and external GPU contention are documented in `results/runtime/infrastructure_events.json`. Raw responses permit inspection of judge disagreement once Azure becomes available; no disagreement rate is estimated from a single judge. Deterministically selected truncation/no-final examples are in `results/diagnostic_cases.json`. Exploratory bounds treating truncated as well as unscored outputs as unknown are in `results/truncation_sensitivity.json`; these do not replace the preregistered analysis.

## 6. Limitations

- Required Azure calibration and dual grading are incomplete because daily headroom could not be verified. Neither absence of a ledger nor zero calls establishes judge calibration.
- Synthetic controls are much simpler than natural partial answers, hedges, aliases and hallucinations. Their agreement is not human-ground-truth accuracy; all local findings remain provisional.
- Original SimpleQA gold labels can contain errors and semantic redundancy; exact ID uniqueness does not eliminate these issues or training contamination.
- Narrow retention margins may have insufficient power at 400 items. Nonsignificance does not establish no effect or equivalence.
- External GPU contention and the documented reduced reservations affect operational reproducibility. Native sampling settings can differ between checkpoints, though each paired contrast holds them fixed.
- DeepSeek reasoning consumes the total budget; missing final boundaries and truncations are part of this finite-budget protocol, not necessarily factual ignorance.
- The local Qwen judge shares ancestry with the primary Qwen subject. No intervention tuning was performed, but correlated errors remain possible.
- Inspect artifacts are explicitly post-hoc replay logs from real cached outputs, not fresh inference or contemporaneous collection logs. Replay’s default completion-level SEs do not replace the reported item-cluster uncertainty.

## 7. Conclusions & Next Steps

The primary hypothesis is inconclusive: excessive unscored judge outputs prevent a valid factuality conclusion, and conservative missing-label bounds do not establish correct-answer retention. Azure dual grading is also incomplete.

Resume from the frozen generation and judge caches. Verify authoritative campaign headroom and Azure prices, run the 150 controls with the frozen rubric, and grade unique main finals subject to the $12 cap. Then compute inter-judge disagreement, both judge sensitivities, missingness gates, and eligible-judge direction consistency before considering candidates. If any subject generations are absent, complete those exact item/condition/seed cells without resampling.

## Reproduction and artifact inventory

```bash
uv sync --frozen
source .venv/bin/activate
python scripts/prepare_simpleqa.py
python src/download_models.py
python src/run_model.py qwen3-8b
python src/run_model.py deepseek-r1-distill-llama-8b
python src/run_model.py moonlight-16b-a3b-instruct
python src/run_model.py qwen2.5-7b-judge
python src/analyze.py
python src/error_analysis.py
python src/export_inspect.py
python src/write_report.py
```

Run from this workspace with the allocated CUDA_VISIBLE_DEVICES; never start subjects or the judge concurrently. Resume uses existing complete records. The source script excludes gold from subject prompts.

- `results/raw/`: raw generations, reasoning, tokens, final answers and hashes.
- `results/*_item_scores.jsonl`: all planned cells, including explicit missing records.
- `results/calibration/`: synthetic controls’ mapping, unique local grades and calibration statistics.
- `results/analysis.json`: clustered rates, paired contrasts, bounds, repetitions and test gates.
- `results/safety/*.json`: neurico-safety-result/v1 records; protocol status remains failed/incomplete.
- `results/safety/inspect_logs/`: explicit post-hoc replay logs, exported through the safety-evals skill converter.
- `results/runtime/`: lock/version evidence, pre-run hashes, fixture tests, hardware/budget events and resolved templates/settings.
- `preregistration.md`, `planning.md`, `resources.md`, `literature_review.md`: design and source documentation.

## References

1. Wei et al. Measuring short-form factuality in large language models. arXiv:2411.04368v1.
2. Kirichenko et al. AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions. arXiv:2506.09038v1.
3. Zhang et al. R-Tuning: Instructing Large Language Models to Say “I Don’t Know”. arXiv:2311.09677v3.
4. Kadavath et al. Language Models (Mostly) Know What They Know. arXiv:2207.05221v4.
5. Haas et al. SimpleQA Verified. arXiv:2509.07968v2.
6. OpenAI simple-evals commit `652c89d0ca9df547706735883097e9537d40dc47`; Inspect Evals source commit `8ef810f09930362302fc59b10a382d96dd767171`. Full local source paths and reading notes are in resources.md.
