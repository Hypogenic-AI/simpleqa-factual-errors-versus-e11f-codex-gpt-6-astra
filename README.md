# SimpleQA errors versus abstention

Paired evaluation of a fixed abstention sentence on Qwen3-8B, DeepSeek-R1-Distill-Llama-8B and Moonlight-16B-A3B-Instruct. Uses frozen SimpleQA questions, three seeds and a synthetic-control-checked local judge.

- 7,200/7,200 planned subject completions generated; 6,823 locally scored.
- The primary hypothesis is inconclusive: excessive unscored judge outputs prevent a valid factuality conclusion, and conservative missing-label bounds do not establish correct-answer retention. Azure dual grading is also incomplete.
- Qwen observed-label paired contributions (n=400): Δincorrect -0.8608 (SE 0.0146; 95% CI -0.8892, -0.8308); Δcorrect -0.0167 (SE 0.0060; 95% CI -0.0292, -0.0058).
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
