"""
opcount_abort_check.py
=======================
Internal convenience check (not a formal deliverable): run CHAINED-SHALLOW
matvec k=6 on a user-specified list of models, 1 repeat, and check each one
scores at least 90%. Below 90% on the simplest shallow chain (6 trivial
matvec steps) indicates a parsing or prompting error, not a genuine model
failure — stop and investigate before running any other cell.

This was internal tooling used during data collection to catch integration
bugs early and save tokens, not part of the experiment's formal deliverables
(see the experiment's deliverables list for those). The set of models to
gate on is not fixed by this script — pass it explicitly with --models.

KNOWN GAP FROM THE DESIGN: the design's own abort criterion calls for 30
problems. This script runs at N=15 instead, because generate_all_problems()
in run_opcount_experiment.py only ever generates PROBLEMS_PER_CELL=15
problems per (op_type, k) cell — that cap is structural, not something this
wrapper can raise without changing CELL_SEEDS / PROBLEMS_PER_CELL in that
module and regenerating every cell's problem set. Treat a pass here as
indicative, not as satisfying the design's N=30 requirement.

Reuses whatever phase2_frontier k=6 rows already exist for the given models
(via run_experiment()'s built-in skip-if-done logic) rather than always
spending fresh API calls — so running this after phase2_frontier data
already exists for a model is free for that model.

Usage:
    python3 opcount_abort_check.py --models claude-4.5-sonnet gemini-3.1-pro qwen3-235b llama-3.3-70b
    python3 opcount_abort_check.py --models claude-4.5-sonnet --n 15 --threshold 0.9
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_opcount_experiment import (
    MODELS, OUTPUT_DIR, PROBLEMS_PER_CELL,
    generate_all_problems, assert_problem_set_validity,
    run_experiment, setup_logging,
)
from build_opcount_aggregated import build_aggregated

ABORT_K = 6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, choices=sorted(MODELS),
                     help="model keys (from run_opcount_experiment.py's MODELS) to gate on")
    ap.add_argument("--n", type=int, default=15,
                     help="problems per model (default 15; the design calls for 30, see module docstring)")
    ap.add_argument("--threshold", type=float, default=0.90,
                     help="minimum accuracy to pass (default 0.90)")
    args = ap.parse_args()

    gate_models = args.models
    abort_n = args.n
    abort_threshold = args.threshold

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("ERROR: OPENROUTER_API_KEY environment variable is not set.\n"
                  "Run: export OPENROUTER_API_KEY=your_key_here")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    raw_csv_path = output_dir / "opcount_results_raw.csv"

    print(f"Abort check: k={ABORT_K} matvec, N={abort_n}, threshold={abort_threshold:.0%}, "
          f"models={', '.join(gate_models)}\n")

    print("Generating shallow-chain problems (deterministic from cell seeds)...")
    all_shallow = generate_all_problems()
    assert_problem_set_validity(all_shallow)

    for model in gate_models:
        run_experiment(model, ABORT_K, abort_n, all_shallow, api_key, raw_csv_path, logger)

    agg_rows = build_aggregated(raw_csv_path)
    gate_rows = {r["model"]: r for r in agg_rows
                 if r["model"] in gate_models and str(r["k"]) == str(ABORT_K)
                 and r["condition"] == "CHAINED-SHALLOW" and r["op_type"] == "matvec"}

    print("\n" + "=" * 70)
    print(f"ABORT CHECK — k={ABORT_K} matvec, N={abort_n}")
    print("=" * 70)
    passed = True
    for model in gate_models:
        row = gate_rows.get(model)
        if row is None:
            print(f"  {model:20s}  NO DATA")
            passed = False
            continue
        acc = row["mean_accuracy"]
        ok = acc >= abort_threshold
        passed = passed and ok
        flag = "OK" if ok else "BELOW THRESHOLD"
        print(f"  {model:20s}  accuracy={acc:.4f}  n={row['n_problems']:2d}  [{flag}]")
    print("=" * 70)

    if passed:
        print("ABORT CHECK PASSED — safe to proceed with the full run.")
    else:
        print(f"ABORT CHECK FAILED — STOP. Do not proceed with the full run.\n"
              f"A sub-{abort_threshold:.0%} score on k={ABORT_K} matvec indicates a parsing or "
              f"prompting error, not a genuine model failure. Investigate before continuing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
