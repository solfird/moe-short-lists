# License for the report, the data and the figures

© 2026 Julien Simon · CC BY 4.0 · [creativecommons.org/licenses/by/4.0](https://creativecommons.org/licenses/by/4.0/)

The report (`moe-short-lists.pdf`), the text (`README.md`, `results/README.md`), the data (`results/`) and the figures
(`figures/`) are licensed under the Creative Commons Attribution 4.0 International license. You may share and adapt
them for any purpose, commercial included, if you credit the author and link to this license.

Suggested credit: "Julien Simon, Solfird AI Technical Report 2026-02, CC BY 4.0."

What this license does not cover:

- The code (`code/`, `analysis/`) is under the Apache License 2.0 in `LICENSE`.
- The Solfird AI name and logo in `assets/`.
- The benchmark datasets. This repository holds no article, question, passage, reference summary or problem text. The
  per-item records hold item numbers, the model's own output and scores; a command that needs the source text downloads it
  from the Hugging Face Hub by item index. Credit and license of each dataset, as the Hub lists them:

  | Dataset | Version used | License on the Hub |
  |---|---|---|
  | `abisee/cnn_dailymail` (CNN/DailyMail) | 3.0.0, test split | Apache-2.0 |
  | `rajpurkar/squad` (SQuAD) | validation split | CC BY-SA 4.0 |
  | `fancyzhx/ag_news` (AG News) | test split | license unknown on the Hub |
  | `openai/gsm8k` (GSM8K) | main configuration, test split | MIT |
  | `EdinburghNLP/xsum` (XSum) | test split | license unknown on the Hub |
  | `SetFit/emotion` (a second label set for the label-order probe) | test split | no license field on the Hub card |
  | `mteb/summeval` (SummEval, human labels of factual consistency) | test split | MIT |
  | `google-research-datasets/xsum_factuality` (human labels of faithfulness for XSum summaries) | `xsum_faithfulness` | CC BY 4.0 |

  The human labels of the last two rows are kept as one true or false value per item (`human` in
  `results/humanref/items.jsonl`); the summaries they refer to are not copied.
- The outputs of the models under test (Gemma 4, Nemotron-3.5-Lightning, Qwen3.6, in 4-bit conversions from `mlx-community`)
  and of the jurors (GPT-5.6 Luna, Gemini 3.5 Flash-Lite, Claude Haiku 5.5, Ministral-3 8B). They come from models under
  their own licenses and terms (Gemma 4 and Qwen3.6: Apache-2.0 on the Hub; Nemotron: NVIDIA's terms on its model card; the
  three hosted jurors: their providers' terms). This license does not decide the rights in those outputs.
  Its answers are not shared, and this repository holds no model file.
