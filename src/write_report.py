"""Render report and figures only after actual generation/grading/analysis artifacts exist."""
import json,datetime,os,re
from pathlib import Path
from protocol import ROOT,MODELS,read_jsonl,write_json

def fmt(stat):
    if stat.get('mean') is None:return 'undefined'
    lo,hi=stat['ci95']
    return f"{stat['mean']:.4f} (SE {stat['se']:.4f}; 95% CI {lo:.4f}, {hi:.4f})"

def main():
    assert Path.cwd()==ROOT;print(ROOT)
    data=json.loads((ROOT/'results/analysis.json').read_text()); models=data['models'];cal=data['local_calibration']
    versions=json.loads((ROOT/'results/runtime/versions.json').read_text())
    pins=json.loads((ROOT/'artifacts/resource_finder/model_revisions.json').read_text())
    os.environ['MPLCONFIGDIR']=str(ROOT/'models/.matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(figsize=(11,5))
    labels=[]; xs=np.arange(6); bottom=np.zeros(6)
    names=['correct','incorrect','not_attempted','unscored']; colors=['#238b45','#cb181d','#4292c6','#bdbdbd']
    for k,color in zip(names,colors):
        vals=[models[m]['conditions'][c][k]['mean'] for m in MODELS for c in ['baseline','abstention']]
        ax.bar(xs,vals,bottom=bottom,color=color,label=k.replace('_',' '));bottom+=vals
    ax.set_xticks(xs,[f'{m.split("-")[0]}\n{c}' for m in MODELS for c in ['baseline','abstention']]);ax.set_ylim(0,1)
    ax.set_ylabel('Fraction of 1,200 planned completions per condition')
    ax.set_title('Provisional local grades: unscored gate failed; dual grading incomplete')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=4);fig.tight_layout();fig.savefig(ROOT/'figures/response_decomposition.png',dpi=180);plt.close(fig)
    totalgen=sum(r['generated_completions'] for r in models.values());totalscored=sum(r['scored_completions'] for r in models.values())
    q=models['qwen3-8b']; qualified=cal.get('eligible',False)
    if q['scored_completions']==0:
        conclusion='The primary hypothesis could not be evaluated: no Qwen3 completions received eligible factuality scores.'
    elif not qualified:
        conclusion='The local judge did not satisfy control eligibility; no factuality conclusion is supported.'
    elif (q['planned_completions']-q['scored_completions'])/q['planned_completions']>.02:
        conclusion='The primary hypothesis is inconclusive: excessive unscored judge outputs prevent a valid factuality conclusion, and conservative missing-label bounds do not establish correct-answer retention. Azure dual grading is also incomplete.'
    elif q['test']['statistical_rule_met_on_observed_labels']:
        conclusion='The local-judge Qwen3 estimates meet the numerical joint rule on observed labels; the required dual-judge conclusion remains incomplete.'
    else:
        conclusion='The local-judge Qwen3 estimates do not satisfy the preregistered joint support rule; the full dual-judge hypothesis test remains incomplete.'
    start_match=re.search(r'Started: `([^`]+)`',(ROOT/'STATE.md').read_text())
    started=datetime.datetime.fromisoformat(start_match.group(1).replace('Z','+00:00')) if start_match else None
    finished=datetime.datetime.now(datetime.timezone.utc)
    elapsed=(finished-started).total_seconds()/60 if started else None
    timing=f'Report generated {finished.isoformat()}; execution-stage elapsed wall time {elapsed:.1f} minutes, including setup and diagnostics (one execution, descriptive timing only).' if elapsed is not None else f'Report generated {finished.isoformat()}.'
    lines=['# SimpleQA factual errors versus abstention on three open checkpoints','',timing,'',
    '## 1. Executive Summary','',conclusion,'',
    f'This execution produced **{totalgen:,} of 7,200 planned subject completions** and **{totalscored:,} local factuality labels**. The unit of statistical independence is the question: 400 questions per checkpoint, two conditions and three seeds. Completed responses and absent planned responses are retained explicitly. Azure judging made zero calls and spent $0 because the campaign’s $50/day cap was visible but its current spend/headroom ledger was not available. No OpenRouter was used. No candidate finding is emitted.','',
    'Synthetic gold/negation/abstention controls test rubric implementation, not calibration on natural model completions. Local scores remain provisional even if controls pass. Explicit factual abstention is not a safety-refusal metric. Reduced answer coverage alone is not evidence of improved factual knowledge.','',
    '## 2. Research Question & Motivation','',
    'The primary joint hypothesis is that adding exactly `If you are uncertain, answer exactly I do not know.` lowers Qwen3-8B incorrect-answer frequency by at least 0.10 while reducing correct-answer frequency by at most 0.03. Identical secondary tests target the specified DeepSeek and Moonlight checkpoints. A paired design distinguishes selective avoidance of errors from indiscriminately answering less.','',
    '### Literature review summary','',
    'The gathered SimpleQA paper motivates the correct/incorrect/not-attempted decomposition and cautions about benchmark label error. AbstentionBench motivates checkpoint-specific tests and distinguishes answerable factual questions from inherently unanswerable questions. R-Tuning and Language Models (Mostly) Know What They Know concern training or confidence methods outside this sole-suffix intervention. SimpleQA Verified motivates label-noise and grader-domain-shift caveats; it is not substituted for the mandated dataset. This is a narrow replication-style checkpoint study, not a new abstention algorithm. See [literature_review.md](literature_review.md) and [planning.md](planning.md).','',
    '## 3. Methodology','',
    '### Frozen data and prompts','',
    'The pinned Inspect SimpleQA source contains 4,326 rows. CSV SHA256: `feee3f7e7db3617e94e8fcf1977b756ec420ef8568f4e0fcbbe0e92e9d5fc032`. Canonical IDs use the pinned Inspect stable-ID helper. Sort SHA256 of `20261020:` plus ID: first 50 questions for controls, next 400 for main evaluation. All selected IDs/questions are distinct and the splits are disjoint. The reproduced item-manifest SHA256 is `3b3d0740316431aed3f759f5d5cd14c38bb99d00d36e68e80d10a99af4e42513`.','',
    'Baseline user content is the raw question, matching the gathered reference implementation. Intervention adds exactly two newlines and the fixed sentence. There are no subject tools, browsing, retrieval, demonstrations or confidence prompts. Native chat templates are preserved; Qwen optional thinking is disabled through its template flag; DeepSeek reasoning is parsed only after `</think>`; Moonlight’s native default system message is retained.','',
    '| Catalog checkpoint | HF ID | Pinned revision |','|---|---|---|']
    for p in pins:lines.append(f"| {p['id']} | `{p['hf_id']}` | `{p['revision']}` |")
    lines += ['', '### Runtime and protocol deviations','',
    f"The project uses its isolated `.venv`, Python {versions['python'].split()[0]}, vLLM {versions['packages'].get('vllm')}, Transformers {versions['packages'].get('transformers')}, Torch {versions['packages'].get('torch')}, Inspect AI {versions['packages'].get('inspect_ai',versions['packages'].get('inspect-ai'))}, and Inspect Evals {versions['packages'].get('inspect_evals',versions['packages'].get('inspect-evals'))}. Full versions are in `results/runtime/versions.json`; dependencies are locked in `uv.lock`.",'',
    'GPU 0 is an NVIDIA A100 80GB PCIe (81,920 MiB total). Only that allocated GPU was used. An unrelated Qwen3-4B service began consuming about 37 GB during setup; it was neither queried nor stopped. Our pre-main amendment lowers memory reservations to 0.45 for Qwen3/DeepSeek, 0.50 for Moonlight, and 0.42 for the local judge, instead of the larger subject catalog reservations. Each subject and the judge run in separate sequential processes, one model at a time, tensor_parallel_size=1, bfloat16, eager execution, batch concurrency up to 64, context 8192, batched token budget 8192. No training was performed; a training-time comparison is not applicable. Concurrent external GPU load limits throughput comparability.','',
    'Real local checkpoint inference uses the vLLM offline engine API, a documented departure from an HTTP-serving harness. Qwen uses original 64-request batch barriers; before DeepSeek/Moonlight collection, scheduling changed to continuously replace finished requests while retaining max_num_seqs=64. Both scheduler implementations and hashes are retained. This operational amendment is documented in infrastructure_events.json. Every model completes a non-main arithmetic smoke fixture before collection. Temperature is 0.6, seeds 11/22/33. Native generation-config top-p/top-k/repetition penalty are serialized in per-model runtime JSON. Nonreasoning answers have max_tokens=256. DeepSeek has a 4096 total cap; cumulative engine steps abort at 256 tokens after its native closing-thought token, with an overshoot assertion. No forced thought closure or last-line extraction is used. Raw token IDs, traces, finals, stop reasons and hashes are saved.','',
    'The judge is loaded only after subject processes exit. It sees original question, gold and extracted final answer, with no model, condition, seed or reasoning. The original SimpleQA A/B/C rubric is frozen. Judge temperature 0, top-p 1, max_tokens=16; exact A/B/C parse failures stay unscored. Available truncated finals are graded but separately flagged; missing final boundaries remain invalid/unscored. Identical question/gold/final/rubric/judge identities reuse cached grades.','',
    'Azure budget handling is a material incompleteness: no authoritative daily spend ledger was found, so no paid probe, calibration or main grading was performed. This is not an Azure calibration failure or evidence of ineligibility. The missing `evals/judge-calibration.md` establishes nothing for this task. The local judge shares Qwen ancestry with Qwen3, as mandated; no policy or prompt optimization against it was performed.','',
    '### Statistical analysis and gates','',
    'Each item’s three binary seed outcomes are averaged before computing means/SEs and paired intervention-minus-baseline effects. Percentile CIs use 10,000 bootstrap resamples of 400 item clusters, seed 20261020; linear SEs use the sample SD across item averages divided by sqrt(400). Ratios use bootstrap SEs. Synthetic calibration resamples 50 question clusters, preserving each triple. Primary support requires Δincorrect ≤ −0.10, two-sided 95% CI upper bound <0, and one-sided 95% lower bound for Δcorrect ≥ −0.03.','',
    'Secondary tests use paired one-sided t tests for incorrect decrease (margin 0) and correct retention (margin −0.03), joint intersection-union p=max(component p values), with Holm adjustment over the two checkpoints. The observed effect threshold and bootstrap gates still apply. The fixed sample may be underpowered for retention; failure to pass does not establish equivalence, absence of risk or no effect.','',
    'Missing labels contribute to the explicit unscored category. Invalid and truncated are overlapping diagnostic flags, not additional mutually exclusive outcome classes. Known-class frequencies are observed contributions over all planned items, not missing-at-random imputations. Adversarial bounds put unknown treatment outcomes at 0 and baseline at 1 for the minimum effect, reversing those assignments for the maximum. Both bound endpoints have bootstrap intervals in `results/analysis.json`.','',
    '## 4. Results','',
    'Counts below are realized inventory counts, not estimated population quantities. Reported rate SEs and CIs cluster by question; all main rate rows have n=400 questions and 1,200 planned completions per condition.','',
    '| Checkpoint | Generated / planned | Locally scored / planned |','|---|---:|---:|']
    for m,r in models.items(): lines.append(f"| {m} | {r['generated_completions']} / 2400 | {r['scored_completions']} / 2400 |")
    lines += ['', '### Synthetic-control judge validation','']
    if 'metrics' in cal:
        lines += [f"Local judge eligibility: **{cal['eligible']}**. n={cal['n_controls']} controls, {cal['n_question_clusters']} independent questions; unscored count={cal['unscored']}.",'','| Metric | Mean, SE and cluster-bootstrap 95% CI |','|---|---|']
        for k,s in cal['metrics'].items():lines.append(f"| {k} | {fmt(s)} |")
        lines += ['', 'Confusion matrix rows are true A/B/C; columns are predicted A/B/C/unscored:','', '```json',json.dumps(cal['confusion_matrix']), '```','']
    else:lines += ['Local control grading did not complete; no eligibility claim.','']
    lines += ['The correct-label false-positive rate uses 100 negative controls (true B or C) as its denominator; accuracy, macro-F1 and kappa use all 150 controls. Bootstrap intervals can be degenerate on perfectly classified controls and do not establish perfect reliability. Azure controls: not run (n=0; metrics undefined). Inter-judge agreement/disagreement and Azure sensitivity analyses are unavailable. No task-matched human-labelled completion calibration set was located in the documented resource search.','',
    '### Local response decomposition','',
    'Each cell is mean (SE; 95% CI); n=400 question clusters. When missingness is nonzero these are known-label contributions, with bounds reported separately.','',
    '| Checkpoint / condition | Correct | Incorrect | Not attempted | Unscored |','|---|---|---|---|---|']
    for m,r in models.items():
        for c in ['baseline','abstention']:
            ms=r['conditions'][c];lines.append('| '+m+' / '+c+' | '+' | '.join(fmt(ms[k]) for k in ['correct','incorrect','not_attempted','unscored'])+' |')
    lines+=['','![Local response decomposition](figures/response_decomposition.png)','',
    '### Coverage and attempt-conditioned accuracy','',
    '| Checkpoint / condition | Coverage (C+I)/N | Accuracy C/(C+I) | Attempted completion denominator |','|---|---|---|---:|']
    for m,r in models.items():
        for c in ['baseline','abstention']:
            ms=r['conditions'][c];lines.append(f"| {m} / {c} | {fmt(ms['coverage'])} | {fmt(ms['attempted_accuracy'])} | {ms['attempted_accuracy']['attempted_completion_count']} |")
    lines+=['','Attempted accuracy is conditional on scored attempted answers; missingness is not assumed random. Native F-score is `2C/(2C+2I+A)`; its means/SEs/CIs are in the JSON. With missing labels its reported known-label-subset value is sensitivity analysis, not a complete native score.','',
    '### Paired effects','',
    '| Checkpoint | Δincorrect (n=400) | Δcorrect (n=400) | Correct one-sided 95% lower bound | Joint p | Holm p (secondary) |','|---|---|---|---:|---:|---:|']
    for m,r in models.items():
        lines.append(f"| {m} | {fmt(r['contrasts']['incorrect'])} | {fmt(r['contrasts']['correct'])} | {r['contrasts']['correct']['lower95_one_sided']:.4f} | {r['test']['joint_iut_p']:.6g} | {r['test'].get('holm_adjusted_p','primary')} |")
    lines += ['', 'Conservative correct-frequency effect bounds (n=400 paired questions):','', '| Checkpoint | Minimum Δcorrect | Maximum Δcorrect |','|---|---|---|']
    for model,r in models.items():
        b=r['missingness_bounds']['correct']
        lines.append(f"| {model} | {fmt(b['minimum_effect'])} | {fmt(b['maximum_effect'])} |")
    lines += ['', 'For Qwen, the minimum-effect one-sided 95% lower bound is -0.1017, below the -0.03 retention margin. Missing labels therefore prevent establishing retention.','']
    lines+=['','These tests describe available local labels; missingness and judge gates must be considered before interpretation. No candidates are emitted: every checkpoint exceeds the 2% unscored gate, and dual judging is incomplete. Qwen observed-label numerical support does not survive the measurement gates. DeepSeek’s observed decrease misses the 0.10 effect-size threshold despite its small adjusted p value; Moonlight does not establish retention. These are conditional descriptions, not supported factuality findings.','',
    '### Missingness, truncation and repetition','',
    '| Checkpoint / condition | Infrastructure missing | Invalid/no-final | Truncated | Literal `I do not know` |','|---|---|---|---|---|']
    for m,r in models.items():
        for c in ['baseline','abstention']:
            ms=r['conditions'][c];lines.append('| '+m+' / '+c+' | '+' | '.join(fmt(ms[k]) for k in ['infrastructure_failure','invalid','truncated','literal_abstention'])+' |')
    lines+=['','Per-seed rates with n/SE/CIs, seed-rate standard deviations, and mean within-item seed SDs are in `results/analysis.json` under `seed_metrics` and `repeat_variability`. Three repetitions are not three independent item samples. Missing planned records are explicitly added to item-score files, with infrastructure labels.','',
    '### Call counts, tokens and cost','']
    cp=ROOT/'results/calibration/local_cache.jsonl';cache=read_jsonl(cp) if cp.exists() else []
    lines += ['| Subject | Logical main requests | Input tokens | Output tokens |','|---|---:|---:|---:|']
    for model in MODELS:
        rp=ROOT/f'results/raw/{model}.jsonl'
        rr=read_jsonl(rp) if rp.exists() else []
        lines.append(f"| {model} | {len(rr)} | {sum(r['prompt_tokens'] for r in rr):,} | {sum(r['output_tokens'] for r in rr):,} |")
    lines += ['', 'These are exact logged counts, not estimated rates. Subject generation uses the local engine step API; logical requests are distinguished from batch method invocations.','']
    lines += [f"Local judge: {len(cache)} unique cached rubric evaluations, {sum(r['input_tokens'] for r in cache):,} input tokens and {sum(r['output_tokens'] for r in cache):,} output tokens. Paid judge calls: 0; paid judge input/output tokens: 0/0; spend: $0 of the $12 job cap. GPU compute cost is not measured. Non-main arithmetic smoke calls are separate from calibration and main totals; their raw fixture artifacts are retained.",'',
    '## 5. Analysis & Discussion','',conclusion,'',
    'The observed-label Qwen contrast meets the numerical error/retention rule, but this is not a supported result: 187/2400 Qwen responses are unscored, above the preregistered 2% suppression threshold. There are 78/2400 unscored DeepSeek responses and 112/2400 unscored Moonlight responses. These are exact inventory counts; condition-specific rate SEs/CIs are reported above. Five missing labels arise from DeepSeek no-final parsing; the other 372 arise from malformed judge replies. The strict parser is not relaxed after seeing these results. Malformed replies include repeated labels such as `B B`, conflicting text such as `C CORRECT`, and invented dialogue continuations. These are formatting failures under the fixed protocol, not known factual misjudgments. The key comparison is error reduction alongside retained correct answers. A larger not-attempted fraction and smaller incorrect fraction can arise solely through coverage reduction; neither demonstrates improved stored knowledge. Narrow retention intervals and judge agreement matter to the joint criterion. Any apparent differences between checkpoints are descriptive for these weights and protocols and cannot be attributed causally to distillation or generalized to model families.','',
    'The study records infrastructure problems separately from model-generated failures. The pre-main package-resolution issue and external GPU contention are documented in `results/runtime/infrastructure_events.json`. Raw responses permit inspection of judge disagreement once Azure becomes available; no disagreement rate is estimated from a single judge. Deterministically selected truncation/no-final examples are in `results/diagnostic_cases.json`. Exploratory bounds treating truncated as well as unscored outputs as unknown are in `results/truncation_sensitivity.json`; these do not replace the preregistered analysis.','',
    '## 6. Limitations','',
    '- Required Azure calibration and dual grading are incomplete because daily headroom could not be verified. Neither absence of a ledger nor zero calls establishes judge calibration.\n- Synthetic controls are much simpler than natural partial answers, hedges, aliases and hallucinations. Their agreement is not human-ground-truth accuracy; all local findings remain provisional.\n- Original SimpleQA gold labels can contain errors and semantic redundancy; exact ID uniqueness does not eliminate these issues or training contamination.\n- Narrow retention margins may have insufficient power at 400 items. Nonsignificance does not establish no effect or equivalence.\n- External GPU contention and the documented reduced reservations affect operational reproducibility. Native sampling settings can differ between checkpoints, though each paired contrast holds them fixed.\n- DeepSeek reasoning consumes the total budget; missing final boundaries and truncations are part of this finite-budget protocol, not necessarily factual ignorance.\n- The local Qwen judge shares ancestry with the primary Qwen subject. No intervention tuning was performed, but correlated errors remain possible.\n- Inspect artifacts are explicitly post-hoc replay logs from real cached outputs, not fresh inference or contemporaneous collection logs. Replay’s default completion-level SEs do not replace the reported item-cluster uncertainty.','',
    '## 7. Conclusions & Next Steps','',conclusion,'',
    'Resume from the frozen generation and judge caches. Verify authoritative campaign headroom and Azure prices, run the 150 controls with the frozen rubric, and grade unique main finals subject to the $12 cap. Then compute inter-judge disagreement, both judge sensitivities, missingness gates, and eligible-judge direction consistency before considering candidates. If any subject generations are absent, complete those exact item/condition/seed cells without resampling.','',
    '## Reproduction and artifact inventory','',
    '```bash\nuv sync --frozen\nsource .venv/bin/activate\npython scripts/prepare_simpleqa.py\npython src/download_models.py\npython src/run_model.py qwen3-8b\npython src/run_model.py deepseek-r1-distill-llama-8b\npython src/run_model.py moonlight-16b-a3b-instruct\npython src/run_model.py qwen2.5-7b-judge\npython src/analyze.py\npython src/error_analysis.py\npython src/export_inspect.py\npython src/write_report.py\n```','',
    'Run from this workspace with the allocated CUDA_VISIBLE_DEVICES; never start subjects or the judge concurrently. Resume uses existing complete records. The source script excludes gold from subject prompts.','',
    '- `results/raw/`: raw generations, reasoning, tokens, final answers and hashes.\n- `results/*_item_scores.jsonl`: all planned cells, including explicit missing records.\n- `results/calibration/`: synthetic controls’ mapping, unique local grades and calibration statistics.\n- `results/analysis.json`: clustered rates, paired contrasts, bounds, repetitions and test gates.\n- `results/safety/*.json`: neurico-safety-result/v1 records; protocol status remains failed/incomplete.\n- `results/safety/inspect_logs/`: explicit post-hoc replay logs, exported through the safety-evals skill converter.\n- `results/runtime/`: lock/version evidence, pre-run hashes, fixture tests, hardware/budget events and resolved templates/settings.\n- `preregistration.md`, `planning.md`, `resources.md`, `literature_review.md`: design and source documentation.','',
    '## References','',
    '1. Wei et al. Measuring short-form factuality in large language models. arXiv:2411.04368v1.\n2. Kirichenko et al. AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions. arXiv:2506.09038v1.\n3. Zhang et al. R-Tuning: Instructing Large Language Models to Say “I Don’t Know”. arXiv:2311.09677v3.\n4. Kadavath et al. Language Models (Mostly) Know What They Know. arXiv:2207.05221v4.\n5. Haas et al. SimpleQA Verified. arXiv:2509.07968v2.\n6. OpenAI simple-evals commit `652c89d0ca9df547706735883097e9537d40dc47`; Inspect Evals source commit `8ef810f09930362302fc59b10a382d96dd767171`. Full local source paths and reading notes are in resources.md.','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))
    (ROOT/'README.md').write_text(f'''# SimpleQA errors versus abstention

Paired evaluation of a fixed abstention sentence on Qwen3-8B, DeepSeek-R1-Distill-Llama-8B and Moonlight-16B-A3B-Instruct. Uses frozen SimpleQA questions, three seeds and a synthetic-control-checked local judge.

- {totalgen:,}/7,200 planned subject completions generated; {totalscored:,} locally scored.
- {conclusion}
- Qwen observed-label paired contributions (n=400): Δincorrect {fmt(q['contrasts']['incorrect'])}; Δcorrect {fmt(q['contrasts']['correct'])}.
- Azure judging is deferred: campaign daily headroom could not be verified; zero paid calls and $0 spend.
- Synthetic controls do not establish calibration on natural model responses. No candidate findings are emitted.

See [REPORT.md](REPORT.md) for rates, SEs, CIs, missingness, deviations and limitations.

## Reproduce

```bash
uv sync --frozen
source .venv/bin/activate
python scripts/prepare_simpleqa.py
python src/download_models.py
# Run models sequentially on the allocated GPU:
python src/run_model.py qwen3-8b
python src/run_model.py deepseek-r1-distill-llama-8b
python src/run_model.py moonlight-16b-a3b-instruct
python src/run_model.py qwen2.5-7b-judge
python src/analyze.py
python src/error_analysis.py
python src/export_inspect.py
python src/write_report.py
```

`src/` contains execution/analysis code; `results/raw/` contains real outputs; `results/calibration/` contains controls and grades; `results/safety/` contains schema records and Inspect replay logs; `figures/` contains the decomposition plot. Manifests and source pins are under `artifacts/resource_finder/`; runtime evidence is under `results/runtime/`.
''')
    print('Wrote REPORT.md, README.md and figure from actual artifacts')

if __name__=='__main__':main()
