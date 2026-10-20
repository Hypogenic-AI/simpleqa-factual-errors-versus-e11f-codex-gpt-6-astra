# Downloaded papers

Five original PDFs; PDF chunks in `pages/` are derivatives and are not counted as additional papers. All PDFs were successfully parsed by pypdf. Source URLs, sizes and SHA256 hashes are in `../artifacts/resource_finder/paper_downloads.json`. URLs without version suffix may change; the local hashes and versions below identify what was actually read.

| Paper / authors | Version and year | Local PDF | Why relevant / reading depth |
|---|---|---|---|
| Measuring short-form factuality in large language models — Jason Wei, Karina Nguyen, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, William Fedus | 2411.04368v1, 2024 | `2411.04368_simpleqa.pdf` | Exact benchmark, rubric, coverage metrics; all 13 pages / 5 chunks read |
| AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions — Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri, Samuel J. Bell | 2506.09038v1, 2025 | `2506.09038_abstentionbench.pdf` | Prompting/reasoning and judge limitations; abstract/method/results screen |
| R-Tuning: Instructing Large Language Models to Say ‘I Don’t Know’ — Hanning Zhang, Shizhe Diao, Yong Lin, Yi R. Fung, Qing Lian, Xingyao Wang, Yangyi Chen, Heng Ji, Tong Zhang | 2311.09677v3, 2023/revised 2024 | `2311.09677_r_tuning.pdf` | Training alternative and selective metrics; abstract/method/results screen |
| Language Models (Mostly) Know What They Know — Saurav Kadavath et al. | 2207.05221v4, 2022 | `2207.05221_know_what_they_know.pdf` | Calibration foundation; abstract/contribution/results overview screen |
| SimpleQA Verified: A Reliable Factuality Benchmark to Measure Parametric Knowledge — Lukas Haas, Gal Yona, Giovanni D’Antonio, Sasha Goldshtein, Dipanjan Das | 2509.07968v2, 2025/revised 2026 | `2509.07968_simpleqa_verified.pdf` | Reference error, redundancy, autorater weaknesses; abstract/method/results screen |

Sources: arXiv abstract/PDF endpoints using the IDs above. Detailed source links and synthesis are in `../literature_review.md`. All five screened candidates were relevant. The paper-finder service timed out without returning ranked papers, so there are no omitted paper-finder results.

To recreate chunks from workspace root:

```bash
pwd
.venv/bin/python .codex/skills/paper-finder/scripts/pdf_chunker.py papers/2411.04368_simpleqa.pdf --pages-per-chunk 3 --output-dir papers/pages
```

Each manifest lists every chunk and page range. SimpleQA rendered chunk previews are in `../artifacts/resource_finder/`. Other papers' appendices have not been fully reviewed; do not assume full-paper coverage from chunk creation alone.
