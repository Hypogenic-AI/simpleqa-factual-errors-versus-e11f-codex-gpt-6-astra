# SimpleQA data

The mandated dataset is the exact OpenAI SimpleQA CSV used by the pinned Inspect task (`code/inspect_evals/src/inspect_evals/simpleqa/dataset.py`). Full data are locally available and excluded from git; documentation, this script and a three-record sample are retained.

- Source: `https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv`
- SHA256: `feee3f7e7db3617e94e8fcf1977b756ec420ef8568f4e0fcbbe0e92e9d5fc032`
- Size: 2,012,910 bytes; 4,326 examples; English, short fact-seeking QA.
- Format: UTF-8 CSV with `metadata`, `problem`, `answer`. Metadata is a Python-literal dictionary, **not JSON**; parse with `ast.literal_eval`, never `eval`.
- Split: one evaluation set; no training split. Our deterministic 450-item selection reserves 50 for calibration and 400 for main evaluation.
- License: official simple-evals README associates SimpleQA with MIT; retain its LICENSE. The HF mirror card also lists MIT.
- HF source checked first: `basicv8vc/SimpleQA`, revision `e319282ab125c3dbd0c7fd00be2e4dd54e7e8f94`; current pinned Inspect uses the official blob, so the mirror is not substituted. Its bytes were not assumed equivalent.

## Reproduction from workspace root

```bash
pwd
uv sync --project .
bash datasets/download.sh
```

The downloader obtains the official CSV if absent, verifies the frozen SHA256, derives canonical IDs with the pinned Inspect helper, checks all IDs/questions for uniqueness, and recreates the data below. Restore the pinned code checkout first if needed (see `code/README.md`). An unexpected source hash fails instead of silently changing the sample.

| Local file | Records | Purpose |
|---|---:|---|
| `simple_qa_test_set.csv` | 4,326 | Original full benchmark |
| `calibration_questions.jsonl` | 50 | Disjoint control questions only |
| `main_questions.jsonl` | 400 | Same evaluation items for all conditions, models and seeds |
| `calibration_controls.jsonl` | 150 | Exact gold / explicit abstention / explicit gold-negation controls, 50 per label |
| `examples.json` | 3 | Small documentation sample from full dataset |

Manifest: `artifacts/resource_finder/item_manifest.json`. Canonical IDs follow Inspect's `simpleqa_` plus first eight hex characters of MD5(question + NUL + answer). Sort by SHA256 of UTF-8 `20261020:` plus canonical ID; use ranks 0–49 for calibration, 50–449 for main. Source row indices are zero-based excluding the CSV header. Delimiter and serialization are frozen explicitly.

## Loading

```python
import csv, json, ast
with open('datasets/simple_qa_test_set.csv', newline='') as f:
    rows = list(csv.DictReader(f))
metadata = ast.literal_eval(rows[0]['metadata'])
with open('datasets/main_questions.jsonl') as f:
    main = [json.loads(line) for line in f]
```

Example: “Who received the IEEE Frank Rosenblatt Award in 2010?” → “Michio Sugeno”. Full samples with source URLs are in `examples.json`.

## Validation and limitations

All 4,326 questions and question-answer pairs are unique; no empty questions or answers. Metadata parsed successfully across all rows. All IDs are unique; calibration and main sets are disjoint. Evidence: `dataset_validation.json` and `preparation_validation.json` in `artifacts/resource_finder/`.

Synthetic controls are labelled relative to supplied gold and are not human-labelled natural model completions. The contradiction is `The answer to this question is not "<gold>".` It tests whether judges recognize negation, including when the gold string appears. SimpleQA's original audit estimates about 3% label error; semantic redundancy can remain despite exact-string uniqueness. Do not use metadata URLs as retrieval input to subjects. Do not replace this benchmark with SimpleQA Verified or silently repair gold answers after seeing outputs.
