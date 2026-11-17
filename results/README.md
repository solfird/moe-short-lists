# results/

Every raw output and score behind the report, with the logs. Nothing here is a model weight, an activation, or a copy of a
dataset.

## Start here

| Kind | Files | What they are |
|---|---|---|
| Derived numbers | `selrule_h26.json` to `selrule_h31_repeat.json`, `heldrule_h65.json`, `mathrule_h41.json`, `baseline_h50.json`, `baseline_h50_blind.json`, `copyaudit.json`, `why_report.json`, `jury_h44*.json`, `jury_heldout_h57.json`, `final_heldout.json`, `final_xsum.json`, `humanref.json`, `failures_h2*.json`, `promptlen_h25*.json`, `ceiling_h33.json`, `pathsize.json`, `earlydamage.json`, `clsprobe_h40.json`, `trivial_baselines.json`, `answer_audit*.json`, `c2_report.md`, `summary.csv`, `*_short_n300_scores.json` | Written by the scripts of [`analysis/`](../analysis) and [`code/`](../code) from the stored answers. The README at the root says which command writes which file. |
| Stored answers | `{tag}_short_n300_k{K'}_s{S}.jsonl`, `{tag}_gsm8k_k{K'}_s{S}.jsonl`, `sum_{cnn,xsum}_*.jsonl`, `frontier_*.jsonl`, `arith_*.jsonl`, `clsprobe_*.jsonl`, `*_c2_*.jsonl` | One record per item, with the model's output, token counts and scores. A `.meta.json` beside each file holds the model snapshot, library versions, command line parameters and start time. |
| Jury verdicts | `jury/`, `jury_b/`, `jury_heldout/`, `panel_heldout/`, `panel_xsum/`, `humanref/` | The items judged (answer, source index, copied share), and one verdict file per juror. Nobody has to pay again to check a number. |
| Device runs | `device_gemma_*.json`, `device_run1_refresh_bug/`, `device_run2_mmap/`, `device_run3/` | Memory and speed of the loading path. Runs 1 and 2 are void (see the report) and kept for the record. |
| Router statistics | `router_flatness.csv`, `router_kept.csv` | Experts used per answer, from router logs that are not published. |
| Logs | `logs/` | The command logs of the runs, in the order they were made. `logs/sweep.log` lists the queued runs with their start and end, as the queue wrote them; the other lines of the working log are left out. Absolute paths of the author's machine were shortened. |

## Record fields

Short suite (`{tag}_short_n300_k{K'}_s{S}.jsonl`): `task` (summarize, qa, classify, rewrite), `i` (item number in the
split; summarize and rewrite use different CNN/DailyMail articles), `out` (the model's answer), `ref_scores` (the item's
scores against the dataset reference, computed when the run was made: `sum_rougeL_ref` for summarize, `qa_em` and
`qa_f1` for question answering, `cls_acc` for classify), `n_prompt`, `prompt_ids_len`, `n_gen`, `gen_ids`, and under
the mask `loads_per_layer`, `touched_per_layer`, `reselects`. GSM8K records: `i`, `ok`, `pred`, `n_gen`, `text`, `gen_ids`,
and under the mask the load counts. The reference text and the gold answer are not stored.

Label probes (`clsprobe_*.jsonl`): `i`, `lp` (log-probability of each label's first token), `top`, `hit` (whether the best
label is the dataset label). Jury items: `cell` (model and budget), `i` (index in the test split), `id`, `answer`,
`source` (dataset, config, split, index of the article) and `copied` (share of the answer's words inside runs of 8 or more
words found in the article).

Datasets, versions and licenses: [`LICENSE-DATA.md`](../LICENSE-DATA.md). How the records were made from the raw files:
[`analysis/strip_dataset_text.py`](../analysis/strip_dataset_text.py).

## Tags

`gemma`, `nemotron`, `qwen36` are the full models and their masked runs (`k0` is the full model); `qwen` is a dense-gate
control. `_d300` and `_d600` put the items at positions 300 and 600; `ends64` is the both-ends rule, `tail64` and `head64`
the last and the first 64 tokens, and no suffix the whole prompt. `_random`, `_static`, `_a0.75`, `_r0.75`,
`_exposure13`, `_ceiling` and `_pl_*` are the controls of the why tests, the always-on cut, the layer test, the ceiling
test and the padding tests.
