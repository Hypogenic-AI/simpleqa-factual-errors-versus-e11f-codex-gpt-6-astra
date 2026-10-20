# Evaluation code resources

Two repositories cloned; no subject or judge inference has been run. Commit pins are also machine-readable in `artifacts/resource_finder/code_revisions.json`.

## OpenAI simple-evals

- Source: `https://github.com/openai/simple-evals.git`
- Local: `code/simple-evals/`
- Commit: `652c89d0ca9df547706735883097e9537d40dc47`
- MIT; deprecated for new benchmark/model updates since July 2025, retained as the reference implementation.
- Key file: `simpleqa_eval.py`, especially `GRADER_TEMPLATE`, `grade_sample`, `__call__` and aggregate metrics. `sampler/chat_completion_sampler.py` supplies API sampling; `common.py` aggregation.
- README inspected in full. Dependencies are optional/per-eval rather than one maintained install: pandas, numpy, scipy, jinja2, tqdm, requests, openai/anthropic as applicable. Do not run its default API model sweep for this study.
- The subject message is the raw question, with no extra short-answer instruction. We preserve that baseline and append the exact requested suffix only in treatment.
- Parser uses the first A/B/C character anywhere and falls back to C on failure. Reuse the **rubric**, not this fallback. Our frozen grader is copied verbatim to `artifacts/resource_finder/grader_prompt.txt`.

## Inspect Evals

- Source: `https://github.com/UKGovernmentBEIS/inspect_evals.git`
- Local: `code/inspect_evals/` (sparse checkout)
- Commit: `8ef810f09930362302fc59b10a382d96dd767171`
- MIT. Checked-out paths: `src/inspect_evals/simpleqa`, `src/inspect_evals/utils`, `tests/simpleqa`, plus top-level files and package parent files.
- Task: `src/inspect_evals/simpleqa/simpleqa.py`; data/hash and record mapping: `dataset.py`; rubric: `grader_templates.py`; scoring: `scorer.py`; stable ID helper: `src/inspect_evals/utils/deps_utils.py`.
- Root and task README/dependency declarations inspected. Pinned requirements include Python >=3.10, inspect_ai >=0.3.261, datasets >=4.8.5, huggingface_hub >=1.2.0, numpy, pydantic, requests, jinja2 and tiktoken. Paper config mentions inspect_ai >=0.3.221, but use the stricter project dependency floor. Version compatibility must be checked when installing the GPU runner.
- `simpleqa` now defaults to a tool-calling scorer. Original option uses a standalone-token regex in strict mode but **still maps unmatched outputs to C**. Use a custom scorer preserving unscored cases, explicit judge assignment and blind original question; never let the tested model become the fallback grader.
- Native metrics can exclude failures; they cannot alone satisfy the fixed n=400 denominators, bounds, or paired item-cluster bootstrap. Adapt task and aggregate item-level logs separately.
- Paper config has different temperature, token limit and judge; do not apply it to this preregistered protocol. README CLI examples and YAML format also differ; consult pinned task/config code and installed CLI help.

## Re-clone exact resources (from workspace root)

```bash
pwd
test -d .git
git clone https://github.com/openai/simple-evals.git code/simple-evals
git -C code/simple-evals checkout 652c89d0ca9df547706735883097e9537d40dc47
git clone --filter=blob:none --sparse https://github.com/UKGovernmentBEIS/inspect_evals.git code/inspect_evals
git -C code/inspect_evals checkout 8ef810f09930362302fc59b10a382d96dd767171
git -C code/inspect_evals sparse-checkout set src/inspect_evals/simpleqa src/inspect_evals/utils tests/simpleqa
```

Run only when those paths are absent. Clones are ignored by the outer repository; documentation and commit pins are tracked. No training repositories were cloned because fine-tuning was pruned from the direction budget.

## Local verification and runner setup

`uv sync --project .` restores the fresh Python 3.12 resource environment. `scripts/prepare_simpleqa.py` ran successfully and evaluated the actual pinned pure stable-ID function without loading the inference framework. Four native Jinja chat templates were rendered as previews; these are not full tokenizer or inference tests.

The resource environment deliberately has no GPU inference stack or weights. Runner must install compatible inspect-ai/inspect-evals/vllm/transformers **in this same .venv**, using `uv add --project .`, pin the resolved versions, and check allocated GPU compatibility. Do not use a shared neurico environment. If installing from the sparse Inspect checkout, additional package directories/registration metadata may be needed; expand the checkout or copy the reviewed task/scorer into the runner's custom Inspect task. Do not assume a pip release equals the reviewed git commit.

Model card/config/tokenizer snapshots and HF revisions are in `artifacts/resource_finder/models/` and `model_revisions.json`. Download weights at those exact revisions during runner setup. Serve one model at a time on one allocated GPU; no tensor parallelism. Qwen3 needs transformers >=4.51.0 according to its card, whereas Moonlight's older example pins 4.48.2: blindly combining card installation snippets is inappropriate. Moonlight supports a DeepSeek-compatible MoE architecture and requires the catalog's `--trust-remote-code`; test actual vLLM support. Keep DeepSeek native thinking, disable Qwen optional thinking through `enable_thinking=False`.
