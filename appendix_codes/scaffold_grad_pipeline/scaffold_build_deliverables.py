#!/usr/bin/env python3
# Module:    scaffold_build_deliverables.py
# Version:   1.0
"""
scaffold_build_deliverables.py — builds the three final deliverable files.

    scaffold_results_raw.csv          one row per (model, level, pool, problem, repeat)
    scaffold_results_aggregated.csv   one row per (model, level, pool)
    scaffold_run_summary.json         the run-level keys

All three land in one flat directory with the scaffold_ prefix, so they can't
collide with the IBP deliverables if both sets are dropped in one folder.

Field names and order are fixed — downstream consumers read these columns by
name and position, so do not rename, reorder, or add columns.

Input is any number of results JSONL files. Grading is applied at this step,
not at fetch time: raw provider text stays on disk untouched, so a grading
question only costs a re-run of this script.

Clopper-Pearson confidence intervals are computed per repeat, then the three
repeats are summarised — never pooled into one combined fraction, which would
shrink every interval and understate its uncertainty. The reported interval
is the widest of the three repeats.

Usage:
    python3 scaffold_build_deliverables.py \\
        --results ../runs/claude/*_results.jsonl ../runs/qwen/*_results.jsonl \\
        --output ../deliverables
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import pandas as pd
from scipy.stats import beta

from scaffold_abort_check import EXPECTED_REPEATS
from scaffold_grade import GRADING_RULE_LOCKED_AT, grade

# run{n}_dryrun — the directory a dry run lands in. See load_records().
_DRYRUN_DIR = re.compile(r"^run\d+_dryrun$")
from scaffold_levels import LEVEL_MODELS, LEVELS
# Paths go through P.<fn>() rather than being imported by name:
# scaffold_paths has no default root, so a from-import would bind before
# --projects-root is read.
import scaffold_paths as P

_HERE = os.path.dirname(os.path.abspath(__file__))
POOLS_CSV = os.path.join(_HERE, "source", "scaffold_pools.csv")

ALPHA = 0.05
TOTAL_CALLS_PLANNED = 4_320

# Exact column names and order — downstream consumers read by position.
RAW_COLUMNS = [
    "model", "level", "pool_id", "problem_id", "repeat", "prompt_text",
    "response_text", "boxed_content", "accuracy", "symbolic_stop",
    "step_error_code", "ceiling_excluded",
]

# Exact column names and order — downstream consumers read by position.
AGG_COLUMNS = [
    "model", "level", "pool_id", "n_problems", "n_repeats",
    "mean_accuracy", "range_accuracy", "ci_lower", "ci_upper",
    "mean_symbolic_stop_rate", "ceiling_excluded", "kappa",
]

def full_prompt(rec: dict) -> str:
    """The complete prompt the model received, for the prompt_text column.

    Runs record the system turn and the user turn separately and store their
    concatenation as full_prompt; both were sent on every call, and the system
    turn carries the boxing instruction this experiment measures. The fallback
    rebuilds it for any record written before full_prompt existed.
    """
    if rec.get("full_prompt"):
        return rec["full_prompt"]
    system = (rec.get("system_prompt") or "").strip()
    user = rec.get("user_prompt") or ""
    return f"{system}\n\n{user}" if system else user


STEP_CODED_LEVELS = {"L3", "L4"}
CEILING_POOLS = {"int-sym", "int-nonsym"}
CEILING_MODEL = "Gemini-3.1-Pro"


def clopper_pearson(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Exact binomial CI. Degenerate ends are pinned, not left as NaN.

    Same function as the IBP run uses — the two studies report intervals on the
    same basis."""
    if n == 0:
        return (float("nan"), float("nan"))
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper


def load_records(patterns: list[str]) -> list[dict]:
    """Every collected record, deduped. Dry runs are excluded.

    WHY DRY RUNS ARE EXCLUDED HERE, TWICE
    ─────────────────────────────────────
    The results glob matches {subcat}/*/run*/, and a dry run lands in run{n}_dryrun,
    so the glob picks it up. A dry-run record is NOT blank — response_text is
    the literal "[DRY RUN]\\nPrompt length: N chars" — so the existing
    empty-response guard below does not stop it either. Left alone, the grader
    scores that string as a wrong answer and every dry run quietly deflates the
    reported accuracy. That was measured on a real record, not imagined.

    Two independent filters, because each one alone can be walked around:
      - the directory check catches records written before the dry_run flag
        existed, and anything hand-copied into a run{n}_dryrun folder;
      - the flag check catches a dry run sent somewhere else with --output,
        where the directory name says nothing.
    A record has to pass both.
    """
    records, seen = [], {}
    n_dry = 0
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)) or ([pattern] if os.path.exists(pattern) else []):
            if _DRYRUN_DIR.match(os.path.basename(os.path.dirname(path))):
                n_dry += 1
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("dry_run"):
                        n_dry += 1
                        continue
                    # A resumed live run appends, so the same key can appear
                    # twice. The later record wins, but only if it actually
                    # carries a response — a retry that failed must not
                    # overwrite an answer already collected.
                    key = (r["model"], r["level"], r["pool_id"],
                           r["problem_id"], int(r["repeat"]))
                    if key in seen and not r.get("response_text"):
                        continue
                    seen[key] = r
    if n_dry:
        # Said out loud. A silent exclusion is how the opposite mistake gets
        # made later — real data dropped by a filter nobody remembered.
        print(f"  skipped {n_dry} dry-run file(s)/record(s) — not collected data")
    records = list(seen.values())
    return records


def load_step_codes(path: str | None) -> dict[tuple, str]:
    """Human step codes, keyed (model, level, problem_id, repeat).

    Optional: the run can be delivered before the coding is finished, in which
    case every L3/L4 failure carries NA and the paper reports the codes later.
    The column is never machine-filled — a human assigns it."""
    if not path:
        return {}
    from scaffold_stepcode_worksheet import read_coding_sheet
    df = read_coding_sheet(path)
    out = {}
    for r in df.itertuples():
        code = str(r.step_error_code or "").strip()
        if code and code.upper() != "NA":
            out[(r.model, r.level, r.problem_id, int(float(r.repeat)))] = code
    return out


def build_raw(records: list[dict], step_codes: dict, ceiling: set) -> pd.DataFrame:
    rows = []
    for r in records:
        g = grade(r.get("response_text"), r["answer_latex"])
        key = (r["model"], r["level"], r["problem_id"], int(r["repeat"]))

        # step_error_code is NA unless a human coded it, and it only applies to
        # incorrect L3/L4 responses.
        if r["level"] in STEP_CODED_LEVELS and g.accuracy == 0:
            code = step_codes.get(key, "NA")
        else:
            code = "NA"

        rows.append({
            "model":            r["model"],
            "level":            r["level"],
            "pool_id":          r["pool_id"],
            "problem_id":       r["problem_id"],
            "repeat":           int(r["repeat"]),
            "prompt_text":      full_prompt(r),
            "response_text":    r.get("response_text") or "",
            "boxed_content":    g.boxed_content or "",
            "accuracy":         g.accuracy,
            "symbolic_stop":    "TRUE" if g.symbolic_stop else "FALSE",
            "step_error_code":  code,
            "ceiling_excluded": "TRUE" if (r["model"], r["level"], r["pool_id"]) in ceiling
                                else "FALSE",
        })
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def pool_members() -> dict[str, set[str]]:
    """Problem ids per pool, from the frozen pool file.

    Read rather than assumed. The integer pools hold 15 problems each, not 30
    like the other pools, so a hardcoded 30 would never fire on them.
    """
    if not os.path.exists(POOLS_CSV):
        raise SystemExit(
            f"{POOLS_CSV} not found — run scaffold_pools.py first.\n"
            "  Ceiling exclusion needs the frozen pool membership to tell a\n"
            "  complete cell from a partly delivered one.")
    df = pd.read_csv(POOLS_CSV)
    return {p: set(g.problem_id) for p, g in df.groupby("pool_id")}


def find_ceiling_cells(records: list[dict]) -> set[tuple[str, str, str]]:
    """Gemini integer cells at ceiling on L0 are excluded from hypothesis
    tests and flagged.

    The judgement is made once, on L0, and then applied to every level of
    that (model, pool) — a cell can't move off a ceiling it was never on.

    A cell qualifies only if it is complete and perfect: every problem in
    the pool, answered correctly, in every expected repeat. Scoring
    whatever happens to have arrived would let a partly delivered cell look
    like a ceiling and be dropped from the analysis, since missing calls
    don't look like failures — they look like nothing, so the surviving
    fraction is still 100%.
    """
    members = pool_members()
    want_reps = set(range(1, EXPECTED_REPEATS + 1))

    present: dict[tuple, set] = defaultdict(set)
    correct: dict[tuple, set] = defaultdict(set)
    for r in records:
        if r["model"] != CEILING_MODEL or r["level"] != "L0":
            continue
        if r["pool_id"] not in CEILING_POOLS:
            continue
        key = (r["pool_id"], int(r["repeat"]))
        present[key].add(r["problem_id"])
        if grade(r.get("response_text"), r["answer_latex"]).accuracy:
            correct[key].add(r["problem_id"])

    ceiling = set()
    for pool in CEILING_POOLS:
        want = members.get(pool)
        if not want:
            continue
        got_reps = {rep for p, rep in present if p == pool}
        if not got_reps:
            continue
        complete = (got_reps == want_reps
                    and all(present[(pool, rep)] == want for rep in want_reps))
        # Perfection is judged on what ARRIVED, so that a truncated-but-flawless
        # cell still reaches the warning below instead of failing silently on
        # the count and never being mentioned.
        perfect = all(correct[(pool, rep)] == present[(pool, rep)] for rep in got_reps)
        if perfect and not complete:
            # Loud, because the silent version of this is a dropped cell.
            missing = sum(len(want) - len(present[(pool, rep)]) for rep in want_reps)
            print(f"  [WARN] {CEILING_MODEL} L0 {pool} is 100% on what arrived, but "
                  f"the cell is incomplete\n"
                  f"         ({len(got_reps)}/{EXPECTED_REPEATS} repeats, "
                  f"{missing} of {len(want) * EXPECTED_REPEATS} answers missing). "
                  f"NOT flagged as a ceiling —\n"
                  f"         fetch the rest before deciding. An incomplete cell "
                  f"cannot be judged at ceiling.")
        if complete and perfect:
            for level in LEVELS:
                ceiling.add((CEILING_MODEL, level, pool))
    return ceiling


def build_aggregated(raw: pd.DataFrame, ceiling: set, kappa: float | None) -> pd.DataFrame:
    rows = []
    for (model, level, pool), cell in raw.groupby(["model", "level", "pool_id"], sort=False):
        per_repeat = []
        for rep, g in cell.groupby("repeat"):
            k, n = int(g.accuracy.sum()), len(g)
            per_repeat.append((rep, k, n, k / n if n else float("nan")))

        accs = [a for *_, a in per_repeat]
        n_problems = max(n for _, _, n, _ in per_repeat)

        # Widest of the three per-repeat intervals — see the module docstring.
        cis = [clopper_pearson(k, n) for _, k, n, _ in per_repeat]
        ci_lower = min(lo for lo, _ in cis)
        ci_upper = max(hi for _, hi in cis)

        sym = (cell.symbolic_stop == "TRUE").groupby(cell.repeat).mean()

        rows.append({
            "model":                   model,
            "level":                   level,
            "pool_id":                 pool,
            "n_problems":              n_problems,
            "n_repeats":               len(per_repeat),
            "mean_accuracy":           round(sum(accs) / len(accs), 6),
            "range_accuracy":          round(max(accs) - min(accs), 6),
            "ci_lower":                round(ci_lower, 6),
            "ci_upper":                round(ci_upper, 6),
            "mean_symbolic_stop_rate": round(float(sym.mean()), 6),
            "ceiling_excluded":        "TRUE" if (model, level, pool) in ceiling else "FALSE",
            # kappa is a property of the whole step-coding exercise, not of one
            # cell, so it is repeated on the L3/L4 rows it applies to and NA
            # everywhere else rather than being silently broadcast.
            "kappa":                   (round(kappa, 4) if (kappa is not None
                                        and level in STEP_CODED_LEVELS) else "NA"),
        })
    return pd.DataFrame(rows, columns=AGG_COLUMNS)


def build_summary(raw: pd.DataFrame, kappa: float | None,
                  allow_missing_marker: bool = False) -> dict:
    """Builds the run summary dict; key order is preserved by json.dump.

    Three of the keys (abort_triggered, l0_replication_accuracy,
    l0_replication_deviation_pp) are typed <bool>/<float> and read off the
    abort marker. Without the marker they can only be null, which would
    deliver a file that doesn't match the declared schema, so a missing
    marker stops the build — the L0 replication must pass before any other
    condition runs, so results shouldn't legitimately exist without it.
    --no-abort-marker is there for inspecting a partial run.
    """
    marker_path = os.path.join(_HERE, "scaffold_abort_passed.json")
    if not os.path.exists(marker_path):
        if not allow_missing_marker:
            raise SystemExit(
                "BLOCKED: scaffold_abort_passed.json not found.\n"
                "  abort_triggered is typed <bool> and\n"
                "  l0_replication_accuracy / l0_replication_deviation_pp as\n"
                "  <float>. Those three come from the abort marker; without it\n"
                "  they can only be null and the delivered JSON would not match\n"
                "  the declared schema.\n\n"
                "  Run the L0 replication and scaffold_abort_check.py first.\n"
                "  To inspect a partial run regardless: --no-abort-marker")
        print("  [WARN] no abort marker — abort_triggered, l0_replication_accuracy\n"
              "         and l0_replication_deviation_pp will be null, which does NOT\n"
              "         match the declared schema. Do not send this file out.")
        marker = {}
    else:
        marker = json.load(open(marker_path, encoding="utf-8"))
        if marker.get("grading_rule_locked_at") != GRADING_RULE_LOCKED_AT:
            raise SystemExit(
                "BLOCKED: the grading rule changed after the abort check passed.\n"
                f"  marker : {marker.get('grading_rule_locked_at')}\n"
                f"  grader : {GRADING_RULE_LOCKED_AT}\n"
                "  Every cell was graded under one rule and the summary would\n"
                "  report another. Restore the rule, or re-run the abort check.")

    return {
        "total_calls_planned":         TOTAL_CALLS_PLANNED,
        "total_calls_completed":       int(len(raw)),
        "abort_triggered":             bool(marker.get("abort_triggered", False)) if marker else None,
        "l0_replication_accuracy":     marker.get("l0_replication_accuracy"),
        "l0_replication_deviation_pp": marker.get("l0_replication_deviation_pp"),
        "kappa":                       kappa,
        "seed":                        42,
        "grading_rule_locked_at":      GRADING_RULE_LOCKED_AT,
    }


def report(raw: pd.DataFrame, agg: pd.DataFrame, records: list[dict]) -> int:
    """Console checks that would otherwise need a human to notice."""
    rc = 0
    print("\n  cells built:")
    for model in LEVEL_MODELS["L0"]:
        got = raw[raw.model == model]
        want = len([lv for lv in LEVELS if model in LEVEL_MODELS[lv]]) * 90 * 3
        flag = "" if len(got) == want else f"   [INCOMPLETE — expected {want}]"
        print(f"    {model:<20} {len(got):>5} rows{flag}")

    if len(raw) != TOTAL_CALLS_PLANNED:
        print(f"\n  [WARN] {len(raw)} of {TOTAL_CALLS_PLANNED} planned calls present")

    # One provider per model, or the run mixed model builds mid-grid.
    by_model = defaultdict(set)
    for r in records:
        if r.get("provider"):
            by_model[r["model"]].add(r["provider"])
    print("\n  providers:")
    for model, provs in sorted(by_model.items()):
        mixed = len(provs) > 1
        rc |= mixed
        print(f"    {model:<20} {sorted(provs)}" + ("   ‼ MIXED" if mixed else ""))
    if rc:
        print("\n  ‼ A model was served by more than one provider. Different endpoints")
        print("    run different quantizations, so those cells are not comparable.")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects-root", required=True,
                    help="output tree, e.g. ../projects. Must already exist. "
                         "Required: there is no default, so the builder never "
                         "reads a tree you did not name.")
    ap.add_argument("--subcat", default=P.DEFAULT_SUBCAT, choices=sorted(P.SUBCATS),
                    help=f"path segment <projects-root>/{{subcat}}/... "
                         f"(default: {P.DEFAULT_SUBCAT})")
    ap.add_argument("--results", nargs="+", default=None,
                    help="results JSONL paths or globs, from either lane. "
                         "Default reads every run of every model under "
                         "<projects-root>/{subcat}/, so a top-up run is folded "
                         "in by existing, not by remembering to name it.")
    ap.add_argument("--output", default=None,
                    help="flat output directory "
                         "(default: <projects-root>/{subcat}/deliverables)")
    ap.add_argument("--step-codes", default=None,
                    help="completed coding sheet (xlsx/csv) from scaffold_stepcode_worksheet.py")
    ap.add_argument("--kappa", type=float, default=None,
                    help="Cohen's kappa from scaffold_kappa.py")
    ap.add_argument("--no-abort-marker", action="store_true",
                    help="build without scaffold_abort_passed.json. The summary's "
                         "three L0 keys become null — for inspecting a partial "
                         "run, not for delivery.")
    args = ap.parse_args()

    P.configure(args.projects_root, args.subcat)
    results = args.results or [P.results_glob()]
    outdir  = args.output or P.deliverables()

    print("\nPATHS")
    print(f"  projects root   {P.projects()}")
    print(f"  subcat          {P.subcat()}")
    print(f"  results glob    {' '.join(results)}")
    print(f"  deliverables    {os.path.abspath(outdir)}\n")

    records = load_records(results)
    if not records:
        # Silence here is the failure the default exists to prevent: an empty
        # glob must never look like a finished run.
        raise SystemExit(
            f"No records found.\n"
            f"  Looked for: {' '.join(results)}\n"
            f"  A finished run writes <projects-root>/{{subcat}}/{{model}}/"
            f"run{{n}}_*/Scaffold_{{model}}_run{{n}}_response.jsonl.")

    os.makedirs(outdir, exist_ok=True)
    ceiling = find_ceiling_cells(records)
    raw = build_raw(records, load_step_codes(args.step_codes), ceiling)
    agg = build_aggregated(raw, ceiling, args.kappa)
    summary = build_summary(raw, args.kappa, args.no_abort_marker)

    raw_path = os.path.join(outdir, "scaffold_results_raw.csv")
    agg_path = os.path.join(outdir, "scaffold_results_aggregated.csv")
    sum_path = os.path.join(outdir, "scaffold_run_summary.json")

    raw.to_csv(raw_path, index=False)
    agg.to_csv(agg_path, index=False)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  wrote {raw_path}  ({len(raw)} rows)")
    print(f"  wrote {agg_path}  ({len(agg)} cells)")
    print(f"  wrote {sum_path}")
    if ceiling:
        print(f"  ceiling-excluded cells: "
              f"{sorted({(m, p) for m, _lv, p in ceiling})}")

    return report(raw, agg, records)


if __name__ == "__main__":
    sys.exit(main())
