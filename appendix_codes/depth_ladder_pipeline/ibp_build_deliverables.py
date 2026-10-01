#!/usr/bin/env python3
# Module:    ibp_build_deliverables.py
# Version:   1.0
"""
IBP Depth Ladder — build the three deliverables from raw inference output.

Produces exactly the three deliverable files this benchmark requires, into
one flat directory. The `ibp_` filename prefix keeps them from colliding
with the scaffold-gradient deliverables.

    ibp_results_raw.csv        one row per (model, problem, repeat)
    ibp_results_by_level.csv   per (model, n, repeat), with Clopper-Pearson CI
    ibp_run_summary.json       totals, per-model pass counts, versions, errors

Grading is delegated to ibp_grade_depth_ladder.grade_response — the symbolic
rule this benchmark requires. Nothing here re-implements or second-guesses it.

CI DENOMINATOR
───────────────
The interval is computed per repeat, k of the 20 unique problems at that level.
The two repeats are NOT pooled into a 40-trial interval: repeats of the same
problem are not independent, so pooling would understate the interval.

USAGE
─────
    python3 ibp_build_deliverables.py --projects-root ../projects \\
        --bank ../projects/source/ibp_depth_ladder_problems.csv \\
        --planned 2720

    # Override the glob / output directory for a one-off
    python3 ibp_build_deliverables.py --projects-root ../projects \\
        --bank ../projects/source/ibp_depth_ladder_problems.csv \\
        --results ../projects/ibp/Llama-3.3-70B/run1_full/*_response.jsonl \\
        --outdir ../projects/ibp/deliverables
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

from scipy.stats import beta

from ibp_grade_depth_ladder import grade_response
# Paths go through P.<fn>() rather than being imported by name: ibp_paths has
# no default root, so a from-import would bind before --projects-root is read.
import ibp_paths as P

ALPHA = 0.05   # 95% CI


# ─────────────────────────────────────────────────────────────────────────
#  Clopper-Pearson
# ─────────────────────────────────────────────────────────────────────────
def clopper_pearson(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Exact binomial CI. Degenerate ends are pinned, not left as NaN."""
    if n == 0:
        return (float('nan'), float('nan'))
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper


def repeat_index(variant: str) -> int:
    """'rep2' or 'run2' -> 2. Records predating the repeat mechanism carry
    'standard' and map to 1."""
    m = re.match(r'(?:rep|run)(\d+)$', str(variant or ''))
    return int(m.group(1)) if m else 1


# ─────────────────────────────────────────────────────────────────────────
#  Load
# ─────────────────────────────────────────────────────────────────────────
def load_bank(path: str) -> dict:
    with open(path, newline='', encoding='utf-8') as f:
        return {r['Problem_ID']: r for r in csv.DictReader(f)}


# run{n}_dryrun — the directory a dry run lands in. See load_results().
_DRYRUN_DIR = re.compile(r'^run\d+_dryrun$')


def load_results(paths: list[str]) -> list[dict]:
    """Every collected record. Dry runs are excluded.

    The results glob matches {subcat}/*/run*/, and a dry run lands in
    run{n}_dryrun, so the glob picks it up. A dry-run record carries a
    placeholder response rather than an empty one, so it would be graded as a
    real answer and would quietly move the accuracy for its depth rung.

    Two independent filters, because each alone can be walked around: the
    directory check catches records written before the dry_run flag existed,
    and the flag check catches a dry run sent elsewhere with --output, where
    the directory name says nothing. A record has to pass both.
    """
    records = []
    n_dry = 0
    for p in paths:
        if _DRYRUN_DIR.match(os.path.basename(os.path.dirname(os.path.abspath(p)))):
            n_dry += 1
            continue
        with open(p, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"  [WARN] unparseable line in {p}")
                        continue
                    if r.get('dry_run'):
                        n_dry += 1
                        continue
                    records.append(r)
    if n_dry:
        # Said out loud. A silent exclusion is how the opposite mistake gets
        # made later — real data dropped by a filter nobody remembered.
        print(f"  skipped {n_dry} dry-run file(s)/record(s) — not collected data")
    return records


def dedupe(records: list[dict]) -> list[dict]:
    """Keep the last record per (model, question_id, variant).

    A --resume run can append a second record for the same call; the last one
    written is the authoritative state, matching c1's own dedup rule.
    """
    seen = {}
    for r in records:
        seen[(r.get('model'), r.get('question_id'), r.get('variant'))] = r
    return list(seen.values())


# ─────────────────────────────────────────────────────────────────────────
#  Build
# ─────────────────────────────────────────────────────────────────────────
SHALLOW_MAX = 4      # n <= 4: a short derivation, no plausible length pressure
DEEP_MIN    = 12     # n >= 12: the top of the ladder, where the signal lives
CONFOUND_PP = 10.0   # deep-minus-shallow truncation gap that spoils the curve


def depth_truncation_report(raw_rows: list[dict]) -> dict:
    """Is truncation correlated with depth?

    Truncation at a flat rate across n costs precision. Truncation that RISES
    with n does something worse: it is indistinguishable from the model failing
    at depth, which is the exact quantity this ladder exists to measure. A cut
    response scores the same as a wrong one, so the curve bends downward for a
    reason that has nothing to do with the model's ability.

    A cap that truncates mostly the top of the ladder is the harmful case: a
    cut response scores the same as a wrong one, so the curve bends downward
    for a reason unrelated to ability. This function measures the deep-minus-
    shallow truncation gap and flags it, whatever the cap.
    """
    out = {}
    for model in {r['model'] for r in raw_rows}:
        rows = [r for r in raw_rows if r['model'] == model]
        shallow = [r for r in rows if int(r['n']) <= SHALLOW_MAX]
        deep    = [r for r in rows if int(r['n']) >= DEEP_MIN]
        if not shallow or not deep:
            continue
        s_rate = 100.0 * sum(r['truncated'] == 'TRUE' for r in shallow) / len(shallow)
        d_rate = 100.0 * sum(r['truncated'] == 'TRUE' for r in deep) / len(deep)
        out[model] = {
            'shallow_n_max':          SHALLOW_MAX,
            'deep_n_min':             DEEP_MIN,
            'shallow_truncation_pct': round(s_rate, 2),
            'deep_truncation_pct':    round(d_rate, 2),
            'gap_pp':                 round(d_rate - s_rate, 2),
            'depth_confounded':       (d_rate - s_rate) > CONFOUND_PP,
        }
    return out


def build(results_paths: list[str], bank_path: str, outdir: str,
          planned: int | None = None) -> int:
    bank    = load_bank(bank_path)
    records = dedupe(load_results(results_paths))
    if not records:
        print("  No records found — nothing to build.")
        return 1

    os.makedirs(outdir, exist_ok=True)

    # At temperature 0 the repeats of a problem are frequently byte-identical.
    # Grading is the expensive step (sympy), so identical text is graded once.
    cache: dict[tuple, tuple] = {}

    raw_rows      = []
    errors        = []
    missing_in_bank = set()

    for i, r in enumerate(records, 1):
        qid = r.get('question_id')
        b_row = bank.get(qid)
        if b_row is None:
            missing_in_bank.add(qid)
            continue

        n, trig, b = int(b_row['n']), b_row['trig'], int(b_row['b'])
        response   = r.get('response') or ''

        if r.get('error'):
            errors.append({'model': r.get('model'), 'Problem_ID': qid,
                           'repeat': repeat_index(r.get('variant')),
                           'error': str(r['error'])[:300]})

        key = (response, n, trig, b)
        if key not in cache:
            cache[key] = grade_response(response, n, trig, b)
        passed, parse_fail, boxed = cache[key]

        raw_rows.append({
            # "model" is the exact resolved version string. There is no
            # fallback to the requested slug: a record missing model_version
            # must surface as an empty value, not be silently filled with the
            # name we asked for.
            'model':         r.get('model_version', ''),
            'Problem_ID':    qid,
            'n':             n,
            'trig':          trig,
            'b':             b,
            'repeat':        repeat_index(r.get('variant')),
            'response_text': response,
            'boxed_content': boxed,
            'pass':          'TRUE' if passed else 'FALSE',
            'parse_fail':    'TRUE' if parse_fail else 'FALSE',
            # Appended after the core 10 columns so their set and order stay
            # untouched. provider is what makes a broken provider pin
            # auditable after the fact.
            'provider':      r.get('provider') or '',
            'model_slug':    r.get('model', ''),
            # A response the provider cut off is not a wrong answer, it is a
            # missing one. Without this column a truncated derivation is
            # indistinguishable from a model that tried and failed — and since
            # truncation rises with n, that difference is the depth curve.
            'truncated':     'TRUE' if r.get('finish_reason') == 'length' else 'FALSE',
            'tokens_used':   r.get('tokens_used') or '',
            # Every call records the full system prompt (if any), the full
            # user prompt, and the full response_text. The column table for
            # this file lists only response_text, so the other two are
            # appended here rather than inserted — the core ten columns keep
            # their names and order.
            'system_prompt': r.get('system_prompt') or '',
            'user_prompt':   r.get('user_prompt') or '',
        })

        if i % 200 == 0:
            print(f"  graded {i}/{len(records)} "
                  f"({len(cache)} distinct responses)")

    raw_rows.sort(key=lambda d: (d['model'], d['n'], d['trig'], d['b'], d['repeat']))

    # ── ibp_results_raw.csv ──────────────────────────────────────
    raw_path = os.path.join(outdir, 'ibp_results_raw.csv')
    # The first ten are the core columns, in their fixed order. provider and
    # model_slug are appended after them — additive, so a consumer reading
    # only the core schema is unaffected.
    cols = ['model', 'Problem_ID', 'n', 'trig', 'b', 'repeat',
            'response_text', 'boxed_content', 'pass', 'parse_fail',
            'provider', 'model_slug', 'truncated', 'tokens_used',
            'system_prompt', 'user_prompt']
    with open(raw_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(raw_rows)
    print(f"  wrote {raw_path}  ({len(raw_rows)} rows)")

    # ── ibp_results_by_level.csv ─────────────────────────────────
    cells = defaultdict(lambda: {'n_problems': 0, 'n_pass': 0, 'n_trunc': 0})
    for row in raw_rows:
        c = cells[(row['model'], row['n'], row['repeat'])]
        c['n_problems'] += 1
        c['n_pass']     += (row['pass'] == 'TRUE')
        c['n_trunc']    += (row['truncated'] == 'TRUE')

    level_path = os.path.join(outdir, 'ibp_results_by_level.csv')
    with open(level_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model', 'n', 'repeat', 'n_problems', 'n_pass',
                    'accuracy', 'ci_lower', 'ci_upper',
                    'n_truncated', 'truncation_rate'])
        for (model, n, rep) in sorted(cells):
            c   = cells[(model, n, rep)]
            k, N = c['n_pass'], c['n_problems']
            lo, hi = clopper_pearson(k, N)
            w.writerow([model, n, rep, N, k,
                        f"{k / N:.6f}" if N else '',
                        f"{lo:.6f}", f"{hi:.6f}",
                        c['n_trunc'],
                        f"{c['n_trunc'] / N:.6f}" if N else ''])
    print(f"  wrote {level_path}  ({len(cells)} cells)")

    # ── ibp_run_summary.json ─────────────────────────────────────
    per_model = defaultdict(lambda: {'calls': 0, 'pass': 0, 'parse_fail': 0,
                                     'truncated': 0})
    providers_by_model: dict[str, set] = defaultdict(set)
    for row in raw_rows:
        m = per_model[row['model']]
        m['calls']      += 1
        m['pass']       += (row['pass'] == 'TRUE')
        m['parse_fail'] += (row['parse_fail'] == 'TRUE')
        m['truncated']  += (row['truncated'] == 'TRUE')
        providers_by_model[row['model']].add(row['provider'])

    truncation = depth_truncation_report(raw_rows)

    for m, provs in providers_by_model.items():
        per_model[m]['providers'] = sorted(provs)

    # A model served by more than one provider is a spoiled measurement: on
    # OpenRouter the hosts of an open-weights model run different quantizations
    # (llama-3.3-70b has fp8, bf16 and fp16 endpoints), so mixing them silently
    # varies the thing being measured. This is the reason each model is pinned
    # to a single provider (Llama: together), and it must not be possible to
    # reach a deliverable without noticing.
    mixed = {m: sorted(p) for m, p in providers_by_model.items() if len(p) > 1}

    summary = {
        'experiment':            'IBP Depth Ladder',
        'generated_utc':         datetime.now(timezone.utc).isoformat(),
        'bank':                  os.path.abspath(bank_path),
        'bank_problems':         len(bank),
        'total_calls_planned':   planned,
        'total_calls_completed': len(raw_rows),
        'model_version_strings': sorted(per_model),
        'per_model':             {k: dict(v) for k, v in sorted(per_model.items())},
        'api_errors':            errors,
        'api_error_count':       len(errors),
        'grading': {
            'method': 'symbolic: trigsimp(diff(boxed, x) - integrand) == 0',
            'boxed_selection': 'last \\boxed{} in the response',
            'string_matching_used': False,
        },
        'ci': {
            'method': 'Clopper-Pearson',
            'alpha': ALPHA,
            'denominator': 'per repeat over the 20 problems at that level; '
                           'repeats deliberately NOT pooled',
        },
        'truncation_by_depth':   truncation,
        'depth_confounded_models': sorted(m for m, v in truncation.items()
                                          if v['depth_confounded']),
        'providers': {
            'by_model':           {m: sorted(p) for m, p in sorted(providers_by_model.items())},
            'single_provider_per_model': not mixed,
            'mixed_provider_models':     mixed,
            'note': 'Read off each response, not from the request. Each model '
                    'must use one provider throughout (Llama: together); a model '
                    'showing more than one provider here mixes quantizations and '
                    'is not a valid measurement.',
        },
        'results_files': [os.path.abspath(p) for p in results_paths],
    }
    if missing_in_bank:
        summary['ids_not_in_bank'] = sorted(missing_in_bank)

    summary_path = os.path.join(outdir, 'ibp_run_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {summary_path}")

    # ── Console report ───────────────────────────────────────────
    print()
    print(f"  {'model':<40}{'calls':>7}{'pass':>7}{'rate':>9}{'pfail':>7}  provider")
    print("  " + "─" * 88)
    for m, v in sorted(per_model.items()):
        rate = v['pass'] / v['calls'] * 100 if v['calls'] else 0
        print(f"  {m:<40}{v['calls']:>7}{v['pass']:>7}{rate:>8.1f}%{v['parse_fail']:>7}  "
              f"{','.join(v.get('providers', [])) or '(none recorded)'}")
    print("  " + "─" * 88)
    if missing_in_bank:
        print(f"  [WARN] {len(missing_in_bank)} result ids not present in the bank")
    if errors:
        print(f"  [WARN] {len(errors)} records carry an API error")

    rc = 0

    if any(v['truncated'] for v in per_model.values()):
        print()
        print("  truncation (provider cut the response off mid-derivation):")
        for m, v in sorted(truncation.items()):
            flag = "   ‼ DEPTH-CONFOUNDED" if v['depth_confounded'] else ""
            print(f"      {m:<34} n<={v['shallow_n_max']}: "
                  f"{v['shallow_truncation_pct']:>5.1f}%   "
                  f"n>={v['deep_n_min']}: {v['deep_truncation_pct']:>5.1f}%   "
                  f"gap {v['gap_pp']:+.1f}pp{flag}")

    confounded = sorted(m for m, v in truncation.items() if v['depth_confounded'])
    if confounded:
        print()
        print("  ‼ TRUNCATION RISES WITH DEPTH — the depth curve is confounded for:")
        for m in confounded:
            print(f"      {m}")
        print("    A cut-off derivation scores the same as a wrong one, so the curve")
        print("    bends downward at the top of the ladder for a reason that has")
        print("    nothing to do with the model's ability. That is the quantity this")
        print("    experiment exists to measure, so it cannot be reported as-is.")
        print()
        print("    Switching endpoints is NOT the fix: the pin commits the run to")
        print("    one provider, and a second endpoint is a second quantization")
        print("    even if it is roomier. Report it instead:")
        print("      • cut the reported ladder at the last depth where truncation is")
        print("        still flat, and state where it was cut and why, and")
        print("      • carry truncation_rate from ibp_results_by_level.csv alongside")
        print("        accuracy, so a reader can see the cap rather than infer ability.")
        print("    Files were written so the effect can be inspected — do not submit")
        print("    the full-depth curve for these models as a measure of ability.")
        rc = 1

    if mixed:
        print()
        print("  ‼ MIXED PROVIDERS — these results are not a valid measurement:")
        for m, provs in sorted(mixed.items()):
            print(f"      {m}: {provs}")
        print("    Each model must use one provider throughout (Llama: together).")
        print("    Different hosts serve different quantizations of the same weights.")
        print("    Files were written so the damage can be inspected — do not submit them.")
        rc = 1

    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # --results defaults to ibp_paths.results_glob(), derived from the same
    # root and subcat the run directories are built from, so the write side
    # and the read side of the tree cannot disagree.
    ap.add_argument('--projects-root', required=True,
                    help='output tree, e.g. ../projects. Must already exist. '
                         'Required: there is no default, so the builder never '
                         'reads a tree you did not name.')
    ap.add_argument('--subcat', default=P.DEFAULT_SUBCAT, choices=sorted(P.SUBCATS),
                    help=f'path segment <projects-root>/{{subcat}}/... '
                         f'(default: {P.DEFAULT_SUBCAT})')
    ap.add_argument('--results', nargs='+', default=None,
                    help='response JSONL files. Default: every run of every '
                         'model under <projects-root>/{subcat}/ (folds in a '
                         'top-up run by it existing, with nothing to rename)')
    ap.add_argument('--bank', required=True,
                    help='e.g. ../projects/source/ibp_depth_ladder_problems.csv')
    ap.add_argument('--outdir', default=None,
                    help='flat output directory for the three deliverables '
                         '(default: <projects-root>/{subcat}/deliverables)')
    ap.add_argument('--planned', type=int, default=None,
                    help='planned call count for the summary '
                         '(340 problems x 4 models x 2 repeats = 2720)')
    args = ap.parse_args()

    P.configure(args.projects_root, args.subcat)
    outdir = args.outdir or P.deliverables()

    print("\nPATHS")
    print(f"  projects root   {P.projects()}")
    print(f"  subcat          {P.subcat()}")
    print(f"  bank            {os.path.abspath(args.bank)}")
    print(f"  results glob    {P.results_glob()}")
    print(f"  deliverables    {os.path.abspath(outdir)}\n")

    results = args.results or sorted(glob.glob(P.results_glob()))
    if not results:
        # An empty glob must never silently produce empty deliverables and
        # exit 0 — that would look like a finished run.
        raise SystemExit(
            f"No response files found.\n"
            f"  Looked for: {P.results_glob()}\n"
            f"  A finished run writes <projects-root>/{{subcat}}/{{model}}/"
            f"run{{n}}_full/IBP_{{model}}_run{{n}}_response.jsonl.\n"
            f"  If your output is elsewhere, pass it with --results.")

    return build(results, args.bank, outdir, args.planned)


if __name__ == '__main__':
    sys.exit(main())
