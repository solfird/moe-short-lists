"""Paid jurors on the held-out subset, items 600 to 899, with jury.py unchanged (a queue imported it)."""

import argparse
import sys

sys.path.insert(0, ".")
import jury  # noqa: E402

jury.JURY = jury.RESULTS / "jury_heldout"
jury.ITEMS = jury.JURY / "items.jsonl"
jury.CAP_USD = 1.95
a = argparse.Namespace(juror=sys.argv[1], limit=int(sys.argv[2]), workers=int(sys.argv[3]) if len(sys.argv) > 3 else 8)
jury.cmd_run(a)
