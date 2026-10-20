# Resources catalog

Resource finding completed on October 20, 2026. This catalog contains **5 papers, 1 downloaded benchmark (4,326 rows), 2 cloned code repositories, and 4 pinned model metadata snapshots**. Derived files contain 50 calibration questions, 400 main questions and 150 synthetic labelled controls. No subject generations, judge results, weights downloads or paid API calls were made.

## Papers

| Title | Authors | Year/version | File | Use |
|---|---|---|---|---|
| Measuring short-form factuality in large language models | Wei, Nguyen et al. | 2024, v1 | `papers/2411.04368_simpleqa.pdf` | Core dataset, rubric, all-question rates |
| AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions | Kirichenko et al. | 2025, v1 | `papers/2506.09038_abstentionbench.pdf` | Prompt/reasoning and judge limitations |
| R-Tuning: Instructing Large Language Models to Say ‘I Don’t Know’ | Zhang et al. | 2023/2024, v3 | `papers/2311.09677_r_tuning.pdf` | Pruned training alternative, selective accuracy |
| Language Models (Mostly) Know What They Know | Kadavath et al. | 2022, v4 | `papers/2207.05221_know_what_they_know.pdf` | Calibration mechanism and generalization limits |
| SimpleQA Verified: A Reliable Factuality Benchmark to Measure Parametric Knowledge | Haas et al. | 2025/2026, v2 | `papers/2509.07968_simpleqa_verified.pdf` | Label noise, redundancy and grader ambiguity |

`papers/README.md` has versions and reading depth; `literature_review.md` has source URLs, extraction notes and synthesis. `paper_downloads.json` has original-PDF URLs, byte counts and hashes. PDF chunks/manifests are under `papers/pages/`; they do not count as independent papers. SimpleQA was read in full, including all appendix chunks; others were screened selectively.

## Dataset and derived resources

| Resource | Source | Size | Local path | Notes |
|---|---|---|---|---|
| SimpleQA | Official OpenAI blob referenced by pinned Inspect | 4,326 rows / 2,012,910 bytes | `datasets/simple_qa_test_set.csv` | SHA256 verified; MIT per official README |
| Main questions | Deterministic SHA256 seed+ID selection | 400 | `datasets/main_questions.jsonl` | Same items across all models/conditions/seeds |
| Calibration questions | First 50 selected ranks | 50 | `datasets/calibration_questions.jsonl` | Disjoint from main |
| Calibration controls | Gold / abstention / gold contradiction | 150, 50 per class | `datasets/calibration_controls.jsonl` | Synthetic, not human-labelled completions |
| Small documentation sample | First three original records | 3 | `datasets/examples.json` | Git-tracked, below 100KB |

Data source: `https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv`. Data SHA256: `feee3f7e7db3617e94e8fcf1977b756ec420ef8568f4e0fcbbe0e92e9d5fc032`. HF mirror `basicv8vc/SimpleQA` was checked but not substituted for the exact Inspect source. Full/derived data are ignored by git. `datasets/README.md` and `datasets/download.sh` contain download and load instructions; `scripts/prepare_simpleqa.py` validates source bytes, calls the pinned stable-ID helper and reproduces selection/controls. Do not select new items if a runtime library changes IDs.

## Code

| Repository | Source | Local path | Frozen commit |
|---|---|---|---|
| OpenAI simple-evals | `https://github.com/openai/simple-evals.git` | `code/simple-evals/` | `652c89d0ca9df547706735883097e9537d40dc47` |
| Inspect Evals | `https://github.com/UKGovernmentBEIS/inspect_evals.git` | `code/inspect_evals/` | `8ef810f09930362302fc59b10a382d96dd767171` |

Both are MIT. Inspect is a sparse clone of the task, utilities and tests; simple-evals is a shallow reference clone. `code/README.md` documents reconstruction, entry points, dependencies and pitfalls. Outer git ignores the cloned trees, preserving documentation and source pins.

**Scorer caveat:** both original A/B/C implementations map missing parse results to C (not attempted). Correct this in the runner with explicit unscored status. The modern default tool scorer differs from the original and may omit invalid responses from aggregate metrics. Always bind judges explicitly; otherwise Inspect may use the subject as its own grader. Preserve fixed n=400 denominators and compute paired uncertainty outside naive native aggregates.

## Checkpoint resources

Weights are not downloaded in this resource stage. All four public model API records, model cards, config/tokenizer files and available generation config files are locally pinned and hashed. Cards are primary sources for model behavior; catalog serving settings are separate local operational guidance.

| Role/checkpoint | Hugging Face ID | Revision |
|---|---|---|
| qwen2.5-7b-judge | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` |
| qwen3-8b | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| moonlight-16b-a3b-instruct | `moonshotai/Moonlight-16B-A3B-Instruct` | `4e735b07a89f73647dfab71ab91b840f362ede5b` |
| deepseek-r1-distill-llama-8b | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `6a6f4aa4197940add57724a7707d069478df56b1` |

Sources use `https://huggingface.co/<HF-ID>`; snapshots are in `artifacts/resource_finder/models/<catalog-id>/`. Full pinned download paths/hashes are in `model_revisions.json`. Model weights should be retrieved by the experiment runner at exactly those revisions, with project-local cache placement.

Qwen3 is Apache-2.0 and needs its thinking switch disabled. DeepSeek card lists MIT for its release, while identifying the underlying Llama-3.1 license; do not erase inherited licensing context. Moonlight card lists MIT; it has 16B total / 3B active parameters, so weight memory is based on total parameters. Qwen2.5 judge is Apache-2.0. Catalog estimates are 24GB for each dense subject, 40GB for Moonlight, 20GB for the local judge; these are estimates, not verified allocations or minimum-memory claims.

Native template previews show DeepSeek begins `<think>`, Qwen disabled mode inserts an empty think block before generation, and Moonlight inserts a default system message. Preserve these conditions and record rendered prompts; do not send reasoning traces to factuality judges. `chat_template_previews.json` is a Jinja render check, not an inference/tokenizer validation.

## Resource collection evidence and limitations

- The diligent paper-finder service timed out after 300 seconds and returned no ranked papers; `paper_finder.json` preserves the failure. Manual primary-source discovery and reference chasing provided all five relevant candidates. No paper-finder result was silently omitted.
- Web searches were attempted but yielded no usable source text in this session; direct requests to arXiv, official GitHub and HF supplied inspectable artifacts. This was a targeted search, not a census of all literature or human calibration datasets.
- `.neurico/idea.yaml` supplied the exact checkpoints and stronger protocol constraints omitted from the brief title/hypothesis. It remains unchanged. No staged `datasets/local/` or `code/local/` resources were specified; no substitute was made for a mandated local copy.
- `human_calibration_search.json` records the inspected AbstentionBench source tree. No task-matched public human-labelled completions were located in the reviewed sources. Shared `evals/judge-calibration.md` is absent. Synthetic-control success cannot establish natural-response calibration.
- Exact-string uniqueness does not remove semantic redundancy, contamination or original gold-answer errors. Findings must remain provisional with these limitations.
- `rg` was unavailable, so filesystem inspection used standard alternatives. PyMuPDF rendering emitted CPU-affinity warnings; rendered images were produced and inspected successfully. No dependency or file outside the project environment was intentionally changed.
- The isolated `.venv`, `pyproject.toml` and `uv.lock` are ready for reuse. They contain resource tools only. GPU-serving compatibility, exact runtime library pins, model weights, separate final-token-cap implementation, Azure prices/headroom and judge eligibility remain experiment-stage setup work. No test success or availability is assumed.

## Experiment design recommendations / handoff order

1. Read `STATE.md`, `planning.md`, `literature_review.md`, `preregistration.md` and `.neurico/idea.yaml`. Preserve exactly the three retained directions: primary Qwen paired effect, identical secondary checkpoint tests, dual-judge measurement validation.
2. Verify hashes using `scripts/prepare_simpleqa.py` and `artifacts/resource_finder/*validation.json`. Use the frozen manifest, prompt and rubric. The standard reference subject prompt is the raw question; treatment adds only the prescribed suffix.
3. Install compatible inference/evaluation packages **inside this fresh .venv**, download pinned weights to the workspace, and validate final-answer boundary/token-cap handling on non-main fixtures. Freeze executable/runtime choices before main collection. One model at a time, one allocated GPU, no tensor parallelism.
4. Run 150 synthetic controls with each judge, report matrices/accuracy/macro-F1/kappa/false-positive rates and clustered uncertainty. Eligibility requires accuracy ≥0.90 and kappa ≥0.80. Azure eligible judge takes priority; otherwise eligible local judge. Both failing prohibits factuality headlines/candidates.
5. Run the fixed 7,200 subject generations and dual grading. Retain unscored/invalid/truncated outputs separately, blind judges to conditions, and report all-item correct/incorrect/abstention plus attempted accuracy and coverage. Use 400 item clusters, average three seeds per item, 10,000 paired bootstrap resamples and Holm correction for the two secondary comparisons.
6. Follow the $12 Azure cap and check shared campaign headroom before calls; no OpenRouter. Cache by exact question/gold/response/judge/rubric/config identity, log spend and returned versions. Save incomplete checkpoints if headroom is unavailable.

The primary support rule is Δincorrect ≤ −0.10 with its two-sided 95% CI upper bound <0, plus one-sided 95% lower bound for Δcorrect ≥ −0.03. Lower coverage alone is not an improvement. `preregistration.md` includes bounds, candidate suppression gates, runtime decisions still to freeze, and required output schema.


## Experiment execution update (October 20, 2026)
The execution stage downloaded all four pinned checkpoints under `models/`, installed the isolated locked GPU/evaluation stack, and completed all 7,200 specified subject generations. No subject browsing/retrieval was available. All subject and local-judge processes have exited.

The local judge classified all 150 synthetic controls correctly (50 question clusters), then processed 5,169 unique main finals. Exactly 6,823 main cells received parseable labels: Qwen 2,213; DeepSeek 2,322; Moonlight 2,288. Five DeepSeek outputs lacked a final boundary and 372 main cells had malformed judge replies. All three checkpoints exceed the preregistered 2% unscored gate. Qwen observed-label contrasts meet the numerical rule but missing-label bounds do not establish correct retention; the overall hypothesis is inconclusive. No candidate findings are emitted.

Azure was not called: the configured campaign cap was visible, but authoritative current daily headroom was unavailable. Actual paid calls/tokens/spend are zero. Its eligibility is unknown, not a failed calibration. `evals/judge-calibration.md` remains absent and supplies no calibration evidence.

Execution code is in `src/`; raw generations, local mappings/grades, clustered analysis, and control evidence are under `results/`. `results/runtime/freeze_audit.json` preserves original pre-run source hashes alongside documented operational amendments. Native model-specific parsing, reduced memory reservations due external GPU contention, continuous scheduling for later subjects, and replay export details are documented in REPORT.md. Inspect replay uses cached real outputs and a provider that raises on any attempted inference. Raw generation/scoring artifacts are authoritative; replay does not add model evaluations.
