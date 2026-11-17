#!/bin/sh
# Recompute every derived number of the report from the stored answers and verdicts. No model, no GPU, no paid API.
# Usage: PYTHON=python sh analysis/reproduce.sh [--datasets-cache DIR]
# The commands marked (data) read dataset text (articles, references): they fetch it from the Hugging Face Hub by item
# index, into the cache you name with --datasets-cache, or into the default Hugging Face cache.
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
DS="$*"

step() { echo "== $1"; shift; "$@"; }

# Frontier, rules, padding test, budgets
step selrule-h26 "$PY" code/selrule.py report-h26 --n 200 > /dev/null
for h in h27 h28 h29 h30 h31 h31-repeat; do step selrule-$h "$PY" code/selrule.py report-$h --n 300 > /dev/null; done
step heldrule "$PY" analysis/heldrule.py report > /dev/null
step failures "$PY" code/failures.py report > /dev/null
step promptlen "$PY" code/promptlen.py report --n 200 > /dev/null
step ceiling "$PY" code/ceiling.py report --n 300 > /dev/null
step pathsize "$PY" code/pathsize.py report > /dev/null
step c2 "$PY" code/c2.py report > /dev/null
step clsprobe-h40 "$PY" code/clsprobe.py report-h40 > /dev/null
for t in gemma nemotron qwen36 qwen; do step score-$t "$PY" code/short.py score --tag $t --n 300 > /dev/null; done
# Math and juries
step mathrule "$PY" analysis/mathrule.py report > /dev/null
step jury-a "$PY" code/jury.py --round a report > /dev/null
step jury-a-blind "$PY" code/jury.py --round a --claude claude_blind report > /dev/null
step jury-b "$PY" code/jury.py --round b report > /dev/null
step jury-b-blind "$PY" code/jury.py --round b --claude claude_blind report > /dev/null
step jury-heldout "$PY" analysis/three_vendor_heldout.py > /dev/null
step humanref "$PY" code/humanref.py report > /dev/null
# (data) commands
step baseline "$PY" analysis/baseline.py $DS > /dev/null
step baseline-blind "$PY" analysis/baseline.py --claude claude_blind $DS > /dev/null
step trivial-baselines "$PY" analysis/trivial_baselines.py $DS > /dev/null
step copyaudit "$PY" analysis/copyaudit.py $DS > /dev/null
step whyreport "$PY" analysis/whyreport.py $DS > /dev/null
step earlydamage "$PY" analysis/earlydamage.py report $DS > /dev/null
step final-heldout "$PY" code/final.py report --round heldout $DS > /dev/null
step final-xsum "$PY" code/final.py report --round xsum $DS > /dev/null
(cd code && step answer-audit "$PY" -m tools.answer_audit $DS > /dev/null)
(cd code && step answer-audit-d300 "$PY" -m tools.answer_audit --start 300 $DS > /dev/null)
# Table of every scored run, and the figures
(cd code && step collect "$PY" -m tools.collect --out ../results/summary.csv > /dev/null)
step figures uv run --no-project --with matplotlib==3.9.2 --with numpy python analysis/make_figures.py
