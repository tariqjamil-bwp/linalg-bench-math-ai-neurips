"""
build_opcount_aggregated.py
============================
Aggregates results/opcount_results_raw.csv into a per-cell accuracy table with
Clopper-Pearson 95% confidence intervals, using clopper_pearson() from
run_opcount_experiment.py. One row per (model, condition, op_type, k).

A parse failure or ungraded row (pass is blank/NULL) counts as incorrect, not
as excluded — the same PASS/FAIL rule the raw CSV's own pass column encodes.

Usage:
    python3 build_opcount_aggregated.py --raw results/opcount_results_raw.csv \\
        --out results/opcount_results_aggregated.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_opcount_experiment import clopper_pearson

AGG_FIELDS = [
    "model", "condition", "op_type", "k", "actual_ops",
    "n_problems", "n_repeats", "mean_accuracy", "range_accuracy",
    "ci_lower", "ci_upper", "mean_first_error_step",
]


def _is_pass(v):
    try:
        return float(v) == 1.0
    except (ValueError, TypeError):
        return False


def _to_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def build_aggregated(raw_csv_path: Path) -> list[dict]:
    cells = defaultdict(list)
    with open(raw_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["condition"], row["op_type"], row["k"])
            cells[key].append(row)

    out_rows = []
    for (model, condition, op_type, k), rows in sorted(cells.items()):
        actual_ops = rows[0].get("actual_ops", "")
        problem_ids = {r["problem_id"] for r in rows}

        by_repeat = defaultdict(list)
        for r in rows:
            by_repeat[r.get("repeat", "1")].append(r)

        per_repeat_accuracy = []
        for repeat_rows in by_repeat.values():
            n = len(repeat_rows)
            correct = sum(1 for r in repeat_rows if _is_pass(r["pass"]))
            per_repeat_accuracy.append(correct / n if n else 0.0)

        n_repeats = len(by_repeat)
        mean_accuracy = sum(per_repeat_accuracy) / n_repeats if n_repeats else 0.0
        range_accuracy = (max(per_repeat_accuracy) - min(per_repeat_accuracy)
                           if n_repeats > 1 else "")

        total = len(rows)
        total_correct = sum(1 for r in rows if _is_pass(r["pass"]))
        ci_lower, ci_upper = clopper_pearson(total_correct, total)

        first_errs = [_to_float(r.get("first_error_step")) for r in rows]
        first_errs = [x for x in first_errs if x is not None]
        mean_first_error_step = (sum(first_errs) / len(first_errs)
                                  if first_errs else "")

        out_rows.append({
            "model": model,
            "condition": condition,
            "op_type": op_type,
            "k": k,
            "actual_ops": actual_ops,
            "n_problems": len(problem_ids),
            "n_repeats": n_repeats,
            "mean_accuracy": round(mean_accuracy, 4),
            "range_accuracy": round(range_accuracy, 4) if range_accuracy != "" else "",
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "mean_first_error_step": round(mean_first_error_step, 2)
                                      if mean_first_error_step != "" else "",
        })
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="path to opcount_results_raw.csv")
    ap.add_argument("--out", required=True, help="path to write opcount_results_aggregated.csv")
    args = ap.parse_args()

    rows = build_aggregated(Path(args.raw))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AGG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} cells to {args.out}")


if __name__ == "__main__":
    main()
