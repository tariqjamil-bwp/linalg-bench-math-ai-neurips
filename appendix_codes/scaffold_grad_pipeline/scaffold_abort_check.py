#!/usr/bin/env python3
# Module:    scaffold_abort_check.py
# Version:   1.0
"""
scaffold_abort_check.py — the hard gate. Nothing else may run until it passes.

Replicates the published L0 condition first (Claude-4.5-Sonnet, 30 irr-sym
problems, 3 repeats = 90 calls). If the result doesn't reproduce the
published number, the harness is wrong and every later cell would be
uninterpretable — stop and escalate rather than continue.

The baseline used here is 0.0%, not the 61.8% in accuracy_5x5.csv: that
figure aggregates all 220 5x5 problems across every subcategory, while
re-grading the published per-problem responses for exactly the 30 irr-sym
eigenvalue ids gives 0/30. scaffold_grade.py --replicate recomputes this
from the published outputs rather than hardcoding it, and this script calls
the same function.

On pass this writes scaffold_abort_passed.json. scaffold_run_live.py refuses
to run any other cell unless that marker exists and its grading_rule_locked_at
matches the grader's, so a failed or skipped gate can't be walked past by
running the next command out of order.

Usage:
    python3 scaffold_abort_check.py --results <l0_claude_results.jsonl>
    python3 scaffold_abort_check.py --simulate 0.0    # synthetic pass
    python3 scaffold_abort_check.py --simulate 40.0   # synthetic must-stop
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

from scaffold_grade import GRADING_RULE_LOCKED_AT, grade, replicate

_HERE = os.path.dirname(os.path.abspath(__file__))
MARKER = os.path.join(_HERE, "scaffold_abort_passed.json")

ABORT_MODEL = "Claude-4.5-Sonnet"
ABORT_LEVEL = "L0"
ABORT_POOL = "irr-sym"
TOLERANCE_PP = 5.0
EXPECTED_REPEATS = 3          # 30 problems, 3 repeats each

STOP_MESSAGE = """
════════════════════════════════════════════════════════════════════════
  ABORT — L0 REPLICATION FAILED
════════════════════════════════════════════════════════════════════════
  Published baseline : {base:.1f}%   (Claude-4.5-Sonnet, L0, irr-sym)
  This run           : {got:.1f}%
  Deviation          : {dev:.1f}pp   (limit {tol:.1f}pp)

  The harness does not reproduce the published condition, so no later
  level can be interpreted against it.

      Do NOT run any other condition.
      Stop and escalate this output before proceeding.

  No marker file was written, so the build scripts will refuse to run.
════════════════════════════════════════════════════════════════════════
"""


def load_results(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def expected_problem_ids() -> list[str]:
    """The 30 irr-sym ids, from the frozen pool file — not a hardcoded range."""
    import pandas as pd
    df = pd.read_csv(os.path.join(_HERE, "source", "scaffold_pools.csv"))
    return sorted(df.loc[df.pool_id == ABORT_POOL, "problem_id"])


def check_complete(records: list[dict]) -> None:
    """The cell must be the WHOLE cell: 30 problems x 3 repeats, every one
    answered.

    Without this, a partially-answered cell passes the gate. Missing calls do not
    look like failures — they look like nothing, so the mean over what did
    arrive is still ~0% and the gate waves it through. That is the exact
    failure this gate exists to catch, so incompleteness is an abort, not a
    warning.
    """
    want_ids = set(expected_problem_ids())
    want_reps = set(range(1, EXPECTED_REPEATS + 1))

    cell = [r for r in records
            if r.get("model") == ABORT_MODEL and r.get("level") == ABORT_LEVEL
            and r.get("pool_id") == ABORT_POOL]

    seen = {(r["problem_id"], int(r["repeat"])) for r in cell}
    want = {(p, k) for p in want_ids for k in want_reps}

    problems = []
    if missing := want - seen:
        problems.append(f"{len(missing)} of {len(want)} (problem, repeat) cells "
                        f"missing, e.g. {sorted(missing)[:3]}")
    if extra := {p for p, _ in seen} - want_ids:
        problems.append(f"{len(extra)} problem_id(s) not in {ABORT_POOL}: "
                        f"{sorted(extra)[:3]}")
    if len(seen) != len(cell):
        problems.append(f"{len(cell) - len(seen)} duplicate (problem, repeat) rows")
    if blank := [r["problem_id"] for r in cell if not r.get("response_text")]:
        problems.append(f"{len(blank)} empty response(s), e.g. {sorted(set(blank))[:3]}")

    if problems:
        raise SystemExit(
            "ABORT — the L0 replication cell is incomplete, so it cannot be "
            "judged:\n  - " + "\n  - ".join(problems) +
            f"\n\n  Expected {len(want_ids)} problems x {EXPECTED_REPEATS} repeats "
            f"= {len(want)} answered calls.\n"
            "  Re-fetch or re-run the missing calls, then check again.")


def accuracy_by_repeat(records: list[dict]) -> dict[str, float]:
    """Per-repeat accuracy over the 30 problems.

    Per repeat, never pooled: the three repeats are three separate estimates
    whose spread becomes range_accuracy. Pooling all 90 into one fraction
    would silently narrow every confidence interval by sqrt(3).
    """
    hits = defaultdict(lambda: [0, 0])
    for r in records:
        if (r.get("model") != ABORT_MODEL or r.get("level") != ABORT_LEVEL
                or r.get("pool_id") != ABORT_POOL):
            continue
        g = grade(r.get("response_text"), r["answer_latex"])
        cell = hits[r["repeat"]]
        cell[0] += g.accuracy
        cell[1] += 1
    return {rep: n_ok / n for rep, (n_ok, n) in sorted(hits.items()) if n}


def decide(observed_pct: float, baseline_pct: float) -> tuple[bool, float]:
    dev = abs(observed_pct - baseline_pct)
    return dev > TOLERANCE_PP, dev


def write_marker(observed: float, baseline: float, dev: float, per_repeat: dict) -> None:
    payload = {
        "abort_triggered": False,
        "l0_replication_accuracy": round(observed / 100.0, 6),
        "l0_replication_deviation_pp": round(dev, 4),
        "published_baseline_pct": baseline,
        "tolerance_pp": TOLERANCE_PP,
        "per_repeat_accuracy": {str(k): v for k, v in per_repeat.items()},
        "grading_rule_locked_at": GRADING_RULE_LOCKED_AT,
        "passed_at": datetime.now().isoformat(timespec="seconds"),
    }
    tmp = MARKER + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, MARKER)


def require_pass() -> dict:
    """Called by the build scripts. Raises unless the gate has passed under the
    same grading rule that is loaded now."""
    if not os.path.exists(MARKER):
        raise SystemExit(
            "BLOCKED: the L0 abort check has not passed.\n"
            "  Run the L0/irr-sym/Claude cell, then scaffold_abort_check.py.\n"
            "  Do not run any other condition first.")
    marker = json.load(open(MARKER, encoding="utf-8"))
    if marker.get("grading_rule_locked_at") != GRADING_RULE_LOCKED_AT:
        raise SystemExit(
            "BLOCKED: the grading rule changed after the abort check passed.\n"
            f"  marker : {marker.get('grading_rule_locked_at')}\n"
            f"  grader : {GRADING_RULE_LOCKED_AT}\n"
            "  The rule is locked before the first call. Restore it, or "
            "re-run the abort check under the new rule.")
    return marker


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", help="parsed results JSONL for the L0/irr-sym/Claude cell")
    ap.add_argument("--simulate", type=float,
                    help="skip the data and test the gate against this accuracy %%")
    ap.add_argument("--baseline", type=float, default=None,
                    help="override the published baseline %% (default: recompute it)")
    args = ap.parse_args()

    if not (args.results or args.simulate is not None):
        ap.error("give --results or --simulate")

    # The baseline is recomputed from the published outputs, not trusted to a
    # constant, so a grader change can never leave a stale number here.
    if args.baseline is not None:
        baseline = args.baseline
        print(f"Baseline overridden: {baseline:.1f}%")
    elif args.simulate is not None:
        baseline = 0.0
        print("Simulation: baseline fixed at the published 0.0%")
    else:
        rc = replicate()
        if rc:
            print("\nThe published replication itself failed — fix the grader "
                  "before judging this run.")
            return 1
        baseline = 0.0

    if args.simulate is not None:
        observed, per_repeat = args.simulate, {}
    else:
        records = load_results(args.results)
        check_complete(records)
        per_repeat = accuracy_by_repeat(records)
        if not per_repeat:
            raise SystemExit(
                f"No {ABORT_MODEL} / {ABORT_LEVEL} / {ABORT_POOL} rows in {args.results}")
        observed = 100.0 * sum(per_repeat.values()) / len(per_repeat)
        print(f"\nThis run, per repeat: "
              + ", ".join(f"{k}={100*v:.1f}%" for k, v in per_repeat.items()))

    triggered, dev = decide(observed, baseline)

    if triggered:
        print(STOP_MESSAGE.format(base=baseline, got=observed, dev=dev, tol=TOLERANCE_PP))
        return 1

    print(f"\n  observed  : {observed:.1f}%")
    print(f"  baseline  : {baseline:.1f}%")
    print(f"  deviation : {dev:.1f}pp  (limit {TOLERANCE_PP:.1f}pp)")
    print("  [PASS] L0 replication holds — the rest of the grid may be built.")
    if args.simulate is None:
        write_marker(observed, baseline, dev, per_repeat)
        print(f"  wrote {MARKER}")
    else:
        print("  (simulation: no marker written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
