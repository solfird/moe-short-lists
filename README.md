<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/solfird-ai-on-dark.svg">
  <img src="assets/solfird-ai-on-light.svg" alt="Solfird AI" width="240">
</picture>

# Classification and Short QA Survive, Summaries and Math Break: Running a Frozen Mixture-of-Experts Model on a Short List of Experts

Solfird AI Technical Report 2026-02 · November 2026 · Julien Simon

[Report (PDF, 27 pages)](moe-short-lists.pdf) · DOI [10.5281/zenodo.24325064](https://doi.org/10.5281/zenodo.24325064) (version 1.0) · all versions: [10.5281/zenodo.24325063](https://doi.org/10.5281/zenodo.24325063)

## The result

A frozen Mixture-of-Experts model can run on 9 to 19% of its experts in each layer, with no training, for classification and short question answering, and not for summaries or math. On held-out items, question answering and classification keep 95.8 to 101.3% of the full model's score in 5 of 6 cases; the sixth, Nemotron's question answering, keeps 85.2%. At the same budgets a blind jury of models accepts only 8 to 48% as many masked summaries as full-model summaries, while ROUGE-L stays at 94 to 97%, and math accuracy falls to 12 to 56%. For a builder: use a short list today for classification and short extractive question answering, and do not accept a long-answer claim that rests on an overlap score.

The report asks whether a frozen model can work from a short list of experts per layer, picked from the prompt and refreshed as it writes. It tests Gemma 4 26B A4B, Nemotron-3.5-Lightning-30B-A3B and Qwen3.6-35B-A3B in 4-bit on one MacBook Pro (M3 Max), on question answering, classification, two summarization sets and math. The list comes from the first and last 32 prompt tokens and is refreshed every 32 generated tokens. Three sets of 300 items per task chose the method, confirmed it, and tested it once.

![Kept score, in percent of the full model, for classify, question answering, summaries judged by a jury and math, on three models at their smallest list. Classify and question answering stay near or above the 90 percent bar; judged summaries fall to 8 to 48 percent and math to 12 to 56 percent; the same summaries scored by ROUGE-L stay near 95 percent.](figures/fig_teaser.png)

*Each model runs on the smallest list chosen from the automatic scores, 9.4 to 18.8% of its experts. Filled marks: classify accuracy and question-answering F1 (held-out items), summaries accepted by the two-member jury (same items) and math accuracy (first 200 GSM8K items). Hollow marks: the same summaries scored by ROUGE-L. Dashed line: the 90% bar. Whiskers: 95% CI.*

## Main results

| Result | Value | Basis |
|---|---|---|
| Where the list comes from | First and last 32 prompt tokens keep 85.2 to 96.0% of question answering at the smallest budgets; the whole prompt keeps 6.6 to 88.0% | Held-out items, measured |
| Classification and question answering at 9 to 19% of the experts | Question answering and classify keep 95.8 to 101.3% in 5 of 6 cases; Nemotron question answering keeps 85.2% | Held-out items, measured |
| Program with no model, same tasks | First sentence of the context keeps 11.5 to 14.7% of question answering; the most common label keeps 31.8 to 36.0% of classify | Measured |
| Memory on a laptop | Gemma 4 at 24 of 128 experts: 29% of peak MLX memory, same tokens | Measured, one model, 20 summaries |
| Summaries, held-out items | Two-member jury accepts 47.9, 8.2 and 12.1% as many masked summaries (Gemma 4, Nemotron, Qwen3.6); three-vendor jury 62.7, 17.5 and 17.7%; ROUGE-L keeps 97.2, 94.5 and 95.2% | Held-out items, measured |
| Summaries, XSum | Jury accepts 53.8, 16.3 and 23.1% as many | Held-out items, measured |
| Larger budgets | Gemma 4 at 48 of 128 experts: 85.8 [79.5, 91.8]% under the two-member jury, undecided; Nemotron and Qwen3.6 stay under the bar | Held-out items, measured |
| Math, first 200 GSM8K items | 56.4, 22.9 and 11.7% of full accuracy; Gemma 4 at 48 experts is undecided, 88.3 [82.5, 93.4]% | Measured, not held out |
| Which experts | A random list fails every task on every model; a fixed list fails once the list is tight | Intervention |
| Always-on path | Cut by 25%, masked Gemma 4 loops on 30.7% of summaries instead of 0.3%; the same cut on the routed experts does not do it | Intervention |

Every result carries the evidence grade of the report: held-out items run once, measured on items that also served to choose the method, an intervention that changes one thing, or observed only. The scope is three models in 4-bit, text input, greedy decoding, one machine.

## Reproduce the numbers and the figures

No model, no GPU, no paid API and no Mac are needed for the report commands below. You need Python 3.12 or later, the pinned requirements (`mlx` is imported by the scripts, so use a platform it supports) and, for the figures, [`uv`](https://docs.astral.sh/uv/).

```sh
uv venv --python 3.13 && uv pip install -r pyproject.toml --extra dev
sh analysis/reproduce.sh --datasets-cache ~/moe-datasets
git diff --stat results/
```

`analysis/reproduce.sh` runs every report command of the report, each from the stored answers and verdicts. The commands marked `(data)` in that script (`baseline.py`, `trivial_baselines.py`, `copyaudit.py`, `whyreport.py`, `earlydamage.py`, `final.py report`, `tools.answer_audit`) need dataset text, such as the articles for the copy and Lead-3 scores, and **need the network** the first time: they download CNN/DailyMail, SQuAD, AG News and XSum from the Hugging Face Hub by item index into the folder given with `--datasets-cache`. The other commands (`selrule.py`, `heldrule.py`, `mathrule.py`, `jury.py`, `analysis/three_vendor_heldout.py`, `humanref.py report`, `failures.py`, `promptlen.py`, `ceiling.py`, `pathsize.py`, `c2.py`, `clsprobe.py report-h40`, `short.py score`) run offline: the per-item scores are stored with the records. The tests run without a model or a dataset: `python -m pytest`.

The last command must print nothing for the JSON and CSV files. The PDF figures in `figures/` show as changed, because a PDF carries its build time. The commands that take the measurements again are listed in [`code/`](code); they need the three models and many hours.

## What is in this repository

| Path | Content |
|---|---|
| [`moe-short-lists.pdf`](moe-short-lists.pdf) | The report |
| [`results/`](results) | The full evidence: stored answers for every run (one record per item, with token counts and scores), jury verdicts, device runs, and the command logs in [`results/logs/`](results/logs). Start with [`results/README.md`](results/README.md) |
| [`code/`](code) | The experiment and scoring scripts: the controller that masks the router (`segroute.py`, `adapters.py`), the task suite (`short.py`), the rules (`selrule.py`), the why tests (`curriculum.py`, `controls.py`, `ablate.py`), the loading path (`device.py`), the jury and gateway scripts (`jury.py`, `final.py`, `humanref.py`, `blindbatch.py`), the helper tools (`tools/`) and their tests (`code/tests/`) |
| [`analysis/`](analysis) | The scripts that recompute the numbers and the figures: `reproduce.sh`, `baseline.py`, `copyaudit.py`, `mathrule.py`, `heldrule.py`, `whyreport.py`, `earlydamage.py`, `three_vendor_heldout.py`, `trivial_baselines.py`, `make_figures.py`, and `strip_dataset_text.py`, which documents how the dataset text was removed from the records |
| [`figures/`](figures) | The data figures of the report, as PDF and PNG |

## Where each result comes from

Section numbers are those of the report.

| Result | Section | Evidence |
|---|---|---|
| Four ways to pick the first list, selection items | 3 | [`selrule_h29.json`](results/selrule_h29.json), [`selrule_h28.json`](results/selrule_h28.json) |
| Padding test, first answer token | 3 | [`selrule_h26.json`](results/selrule_h26.json), [`promptlen_h25.json`](results/promptlen_h25.json) |
| Both ends against whole prompt, held-out items | 3 | [`heldrule_h65.json`](results/heldrule_h65.json) |
| Why refresh the list (GSM8K, Gemma 4, 16 experts) | 3 | [`gemma_gsm8k_k16_s32.jsonl`](results/gemma_gsm8k_k16_s32.jsonl) and the files with `_s64`, `_s128`, `_s0`; [`logs/s_sweep.log`](results/logs/s_sweep.log) |
| Budget curves, selection items | 4 | [`selrule_h31.json`](results/selrule_h31.json) |
| Smallest budgets, confirmation and held-out items | 4 | [`selrule_h31_repeat.json`](results/selrule_h31_repeat.json), [`final_heldout_suite.json`](results/final_heldout_suite.json) |
| Programs with no model | 4 | [`trivial_baselines.json`](results/trivial_baselines.json), [`baseline_h50.json`](results/baseline_h50.json) |
| Four label orders and a second label set | 4 | [`clsprobe_h40.json`](results/clsprobe_h40.json) |
| Memory on a laptop | 4 | [`device_run3/`](results/device_run3), [`device_gemma_masked_k24.json`](results/device_gemma_masked_k24.json) |
| Juries on held-out items and XSum | 5.1 | [`jury_heldout_h57.json`](results/jury_heldout_h57.json), [`final_heldout.json`](results/final_heldout.json), [`final_xsum.json`](results/final_xsum.json), verdicts in [`jury_heldout/`](results/jury_heldout), [`panel_heldout/`](results/panel_heldout), [`panel_xsum/`](results/panel_xsum) |
| Jury on confirmation items, other jury rules | 5.1, appendix | [`jury_h44.json`](results/jury_h44.json), [`jury_h44_blind.json`](results/jury_h44_blind.json), [`jury_h44b_blind.json`](results/jury_h44b_blind.json), [`baseline_h50.json`](results/baseline_h50.json) |
| Foreign numbers, copying and loops | 5.1 | [`baseline_h50.json`](results/baseline_h50.json), [`copyaudit.json`](results/copyaudit.json), [`answer_audit.json`](results/answer_audit.json) |
| Larger budgets | 5.1, appendix | [`final_heldout.json`](results/final_heldout.json), [`jury_h52_blind.json`](results/jury_h52_blind.json), [`why_report.json`](results/why_report.json) |
| Math under the both-ends rule | 5.2 | [`mathrule_h41.json`](results/mathrule_h41.json), the `*_ends64_gsm8k_*.jsonl` files |
| Random lists, fixed lists | 6 | [`why_report.json`](results/why_report.json), the `*_random_*` and `*_static_*` runs |
| Arithmetic of controlled length | 6, appendix | [`arith_*.jsonl`](results), [`why_report.json`](results/why_report.json) |
| Always-on path cut | 6 | [`why_report.json`](results/why_report.json), the `*_a0.75_*` and `*_r0.75_*` runs, [`ablate_h16.json`](results/ablate_h16.json) |
| Size of the always-on path, overlap of experts, router groups | 6, appendix | [`pathsize.json`](results/pathsize.json), [`curriculum_overlap_gemma.json`](results/curriculum_overlap_gemma.json), [`curriculum_routers_gemma.json`](results/curriculum_routers_gemma.json) |
| Jurors against human labels | 8 | [`humanref.json`](results/humanref.json), [`humanref/`](results/humanref) |
| Damage in the first 24 words | 8 | [`earlydamage.json`](results/earlydamage.json) |
| Every scored run in one table | all | [`summary.csv`](results/summary.csv) |

## What you cannot reproduce from this repository

- **The model runs.** The stored answers come from runs on one MacBook Pro (M3 Max, 48 GB) with the three 4-bit models pinned in the report. `code/short.py run`, `code/selrule.py run` and the other `run` commands repeat them; they need the models, MLX 0.32.4 and many hours, and no run was repeated for this repository.
- **The hosted jurors.** The verdicts of GPT-5.6 Luna and Gemini 3.5 Flash-Lite, and of Claude Haiku 5.5 in blind batches, are stored in `results/`. `code/jury.py run` calls the first two through a commercial router and reads its key from the environment variable `REQUESTY_API_KEY`; it costs money. The batch files of the Claude juror hold articles and are not published: `blindbatch.py build` and `final.py build` rebuild them, with the network.
- **Dataset text.** No article, question, passage, reference summary or problem is stored. The jury items carry the index of the article, and `code/corpus.py` fetches it from the Hub. Datasets and licenses are in [`LICENSE-DATA.md`](LICENSE-DATA.md).
- **Router logs.** The router statistics (`router_flatness.csv`, `router_kept.csv`) and the count of experts used per answer in the introduction come from router logs of 3 GB that are not published. `segroute.py log` writes them again.

## Limits

The results cover three 4-bit models, text input, greedy decoding, English, one machine. Budgets are expert counts, the 90% bar is a convention, and the budgets were chosen with a looser rule than the one used to report results. The juries are models and no person graded the summaries; against human labels the two-member jury agrees on 91.6% of a balanced SummEval sample and on 65.1% of XSum. Math uses the first 200 GSM8K items, which are not held out. Memory is measured on one model and 20 items. Section 8 of the report gives the limits in full.

## Corrections

Found an error? [Open an issue](https://github.com/solfird/moe-short-lists/issues). A correction is a new release of this
repository, and its release notes say what changed.

## Cite

```bibtex
@techreport{simon2026shortlists,
  author      = {Simon, Julien},
  title       = {{Classification and Short QA Survive, Summaries and Math Break: Running a Frozen Mixture-of-Experts Model on a Short List of Experts}},
  institution = {Solfird AI},
  number      = {2026-02},
  year        = {2026},
  month       = nov,
  doi         = {10.5281/zenodo.24325064},
  url         = {https://github.com/solfird/moe-short-lists},
}
```

GitHub's "Cite this repository" button gives the same reference from [`CITATION.cff`](CITATION.cff).

## Licenses

- The report, the figures and the data: CC BY 4.0, with the exceptions listed in [`LICENSE-DATA.md`](LICENSE-DATA.md).
- The code: Apache 2.0, in [`LICENSE`](LICENSE).
- The Solfird AI name and logo in `assets/` are not under these licenses.

This work had no funding from any model vendor, and none reviewed it. Contact: julien@solfird.ai.
