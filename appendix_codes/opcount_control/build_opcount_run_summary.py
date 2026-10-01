"""
build_opcount_run_summary.py
=============================
Writes results/opcount_run_summary.json, the experiment's run-summary deliverable.

Two kinds of field go into this file. Some are derivable from the raw CSV
and this repo's own config, and this script derives them:
    total_calls_completed        count of raw CSV data rows (via csv.DictReader,
                                    not a line count — prompt/response text
                                    contains embedded newlines)
    abort_check_accuracy_by_model  per-model CHAINED-SHALLOW matvec k=6
                                    mean_accuracy, pulled from the aggregated
                                    table (the abort check IS a k=6 matvec
                                    cell — no separate marker distinguishes
                                    a row made via opcount_abort_check.py
                                    from one made via run_opcount_experiment.py
                                    --k 6)
    abort_triggered               True if any of those accuracies is below
                                    --threshold
    cell_seeds                    CELL_SEEDS, read directly from
                                    run_opcount_experiment.py — never
                                    hand-copied, so it can't drift from the
                                    seeds actually used to generate problems
    seed_base                     min(CELL_SEEDS.values())

Others are pre-registered design constants that no script can reconstruct
after the fact — they must be supplied:
    total_calls_planned           decided before any call was made
    grading_rule_locked_at        an ISO 8601 timestamp recorded before the
                                    first API call, not reconstructable from
                                    output data alone
This is also why no prior script in this folder emitted opcount_run_summary.json:
the raw/aggregated CSVs are pure execution output, but this file mixes that
output with commitments that only exist outside the code.

Usage:
    python3 build_opcount_run_summary.py \\
        --raw results/opcount_results_raw.csv \\
        --out results/opcount_run_summary.json \\
        --total-calls-planned 1920 \\
        --grading-rule-locked-at 2026-09-01T00:00:00Z \\
        --abort-models claude-4.5-sonnet gemini-3.1-pro qwen3-235b llama-3.3-70b \\
        --threshold 0.90
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_opcount_experiment import CELL_SEEDS
from build_opcount_aggregated import build_aggregated

OP_COUNT_TARGET = 500
OP_COUNT_COMPUTATION_NOTE = (
    "409 ops (5x5 cofactor expansion) + 100 ops (root finding) = 509, rounded to 500"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="path to opcount_results_raw.csv")
    ap.add_argument("--out", required=True, help="path to write opcount_run_summary.json")
    ap.add_argument("--total-calls-planned", type=int, required=True,
                     help="design-time total call count; not derivable from output")
    ap.add_argument("--grading-rule-locked-at", required=True,
                     help="ISO 8601 timestamp recorded before the first API call")
    ap.add_argument("--abort-models", nargs="+", default=None,
                     help="models to report in abort_check_accuracy_by_model "
                          "(default: every model with a CHAINED-SHALLOW matvec k=6 row)")
    ap.add_argument("--threshold", type=float, default=0.90,
                     help="abort_triggered is True if any reported model's k=6 accuracy "
                          "falls below this (default 0.90)")
    args = ap.parse_args()

    raw_csv_path = Path(args.raw)
    agg_rows = build_aggregated(raw_csv_path)
    with open(raw_csv_path, newline="", encoding="utf-8") as f:
        total_calls_completed = sum(1 for _ in csv.DictReader(f))

    k6_rows = {r["model"]: r for r in agg_rows
               if r["condition"] == "CHAINED-SHALLOW" and r["op_type"] == "matvec"
               and str(r["k"]) == "6"}

    abort_models = args.abort_models or sorted(k6_rows)
    abort_check_accuracy_by_model = {}
    for m in abort_models:
        row = k6_rows.get(m)
        abort_check_accuracy_by_model[m] = row["mean_accuracy"] if row else None

    abort_triggered = any(
        acc is None or acc < args.threshold
        for acc in abort_check_accuracy_by_model.values()
    )

    cell_seeds = {f"{op}_k{k}": seed for (op, k), seed in sorted(CELL_SEEDS.items(), key=lambda kv: kv[1])}
    seed_base = min(CELL_SEEDS.values()) if CELL_SEEDS else None

    summary = {
        "total_calls_planned": args.total_calls_planned,
        "total_calls_completed": total_calls_completed,
        "abort_triggered": abort_triggered,
        "abort_check_accuracy_by_model": abort_check_accuracy_by_model,
        "op_count_target": OP_COUNT_TARGET,
        "op_count_computation_note": OP_COUNT_COMPUTATION_NOTE,
        "cell_seeds": cell_seeds,
        "grading_rule_locked_at": args.grading_rule_locked_at,
        "seed_base": seed_base,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
