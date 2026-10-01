#!/usr/bin/env python3
# Module:    scaffold_run_live.py
# Version:   1.0
"""
scaffold_run_live.py — the live (synchronous) inference lane.

Runs all 3 models: Qwen3-235B over OpenRouter, and Claude-4.5-Sonnet /
Gemini-3.1-Pro via a direct provider call (inference_llm's "anthropic"
and "genai" backends).

Every output record uses the same schema and stem/directory conventions as
the rest of the pipeline, so downstream tooling reads every run the same
way.

A run resumes by default: on restart, every (problem_id, level, repeat)
already present in the output file with a non-empty response is skipped,
and new records append rather than overwrite.

The abort gate still applies: nothing runs until the L0/irr-sym/Claude
replication check has passed.

Usage:
    python3 scaffold_run_live.py --model Qwen3-235B --output ../runs/qwen
    python3 scaffold_run_live.py --model Claude-4.5-Sonnet --output ../runs/claude \\
        --level L3 --pool irr-sym            # one cell
    python3 scaffold_run_live.py --model Gemini-3.1-Pro --output ../runs/gemini \\
        --dry-run -n 2                       # free shakedown, no API
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(), override=True)
except ImportError:
    pass

# models.py, inference_llm.py and textguard.py live in ../common/, shared with
# other pipelines in this repo. Added to sys.path by location rather than a
# symlink, so the import resolves the same way on every OS and after a plain
# git clone or zip download.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from inference_llm import InferenceClient
from models import get_max_output_tokens, get_model_config
from scaffold_abort_check import ABORT_LEVEL, ABORT_MODEL, ABORT_POOL, require_pass
# Suffix constants are plain strings. Every actual PATH goes through P.<fn>(),
# because scaffold_paths has no default root — a from-import would bind before
# --projects-root has been read. See the scaffold_paths.py header.
import scaffold_paths as P
from scaffold_paths import RESPONSE, run_dir, stem, write_manifest
from scaffold_grade import count_boxed
from scaffold_levels import (LEVELS, LEVEL_MODELS, full_prompt, is_valid_cell,
                             prompt_message, render)
from textguard import assert_clean

# The three models this experiment uses, out of the shared registry in
# common/models.py.
LIVE_REGISTRY = {name: name for name in
                  ("Qwen3-235B", "Claude-4.5-Sonnet", "Gemini-3.1-Pro")}

_HERE = os.path.dirname(os.path.abspath(__file__))
POOLS_CSV = os.path.join(_HERE, "source", "scaffold_pools.csv")
REPEATS = 3
# Fallback only, for registry entries with no measured cap. The real budget is
# the pinned provider's own ceiling, resolved per model in resolve_max_tokens().
DEFAULT_MAX_TOKENS = 16384

# Same system prompt on every level and model — the level text is the only
# thing that varies, so a system turn that changed too would confound what
# the experiment measures. \\boxed{} is escaped deliberately: in a non-raw
# literal, "\boxed{}" is a backspace followed by "oxed{}" — see textguard.py.
SYSTEM_PROMPT = assert_clean(
    "You are a precise mathematical assistant. "
    "Show all computation steps clearly. "
    "Always put your final answers inside \\boxed{}.",
    "SYSTEM_PROMPT")


def _parse_repetition(s: str) -> list[int]:
    """'1' -> [1]; '1,2' -> [1, 2]. Explicit repeat indices, not a count, so a
    second/third run names exactly which repeat it is filling in."""
    try:
        return sorted({int(x) for x in s.split(",") if x.strip()})
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--repetition must be comma-separated integers, got {s!r}")


def resolve_max_tokens(model_entry: str, override: int | None) -> int:
    """Token budget for one run: the pinned provider's own ceiling, unless
    --max-tokens overrides it. An override above the cap is refused rather
    than silently clipped."""
    cap = get_max_output_tokens(model_entry, DEFAULT_MAX_TOKENS)
    if override is None:
        return cap
    if override > cap:
        raise SystemExit(
            f"--max-tokens {override} is above {model_entry}'s provider cap of "
            f"{cap}.\n  Lower it, or fix max_output_tokens in models.py if the "
            f"provider has raised the limit.")
    return override


def load_done(path: str) -> set[tuple[str, str, int]]:
    """Keys already answered. A row with no response_text is NOT done — an
    empty or errored call should be retried on the next pass."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("response_text"):
                done.add((r["problem_id"], r["level"], int(r["repeat"])))
    return done


def build_tasks(model: str, levels, pools, repetition: list[int],
                 limit: int | None) -> list[dict]:
    df = pd.read_csv(POOLS_CSV)
    if pools:
        df = df[df.pool_id.isin(pools)]
    if limit:
        df = df.groupby("pool_id", sort=False).head(limit)

    chosen = levels or [lv for lv in LEVELS if is_valid_cell(lv, model)]
    bad = [lv for lv in chosen if not is_valid_cell(lv, model)]
    if bad:
        raise SystemExit(f"{bad} is not run on {model} (L2-ND is Claude-only).")

    tasks = []
    for level in chosen:
        for row in df.itertuples():
            for rep in repetition:
                tasks.append({
                    "problem_id":   row.problem_id,
                    "level":        level,
                    "pool_id":      row.pool_id,
                    "repeat":       rep,
                    "user_prompt":  render(level, row.matrix_latex),
                    "answer_latex": row.answer_latex,
                    "matrix_latex": row.matrix_latex,
                })
    return tasks


def run(args) -> int:
    # Bind the output tree FIRST, before anything reads a path. scaffold_paths
    # has no default root, so every path accessor raises until this line runs.
    P.configure(args.projects_root, args.subcat)

    if args.model not in LIVE_REGISTRY:
        raise SystemExit(
            f"'{args.model}' has no live entry. Live models: {list(LIVE_REGISTRY)}")

    entry = LIVE_REGISTRY[args.model]
    cfg = get_model_config(entry)
    if cfg is None:
        raise SystemExit(f"Registry entry '{entry}' not found in models.py")

    max_tokens = resolve_max_tokens(entry, args.max_tokens)

    # --abort-check builds the L0/irr-sym replication cell itself, which is
    # what creates the marker require_pass() checks for — so this is the one
    # run that must skip that check, or the marker could never be created.
    if args.abort_check:
        if args.model != ABORT_MODEL:
            raise SystemExit(f"--abort-check is only defined for {ABORT_MODEL}.")
        levels, pools = [ABORT_LEVEL], [ABORT_POOL]
    else:
        levels, pools = args.level, args.pool
        if not args.dry_run:
            marker = require_pass()
            print(f"  abort check: PASSED at {marker['passed_at']}")

    if args.output:
        outdir = args.output
        os.makedirs(outdir, exist_ok=True)
    else:
        # A dry run is labelled as one. It still takes a real run number —
        # numbers are never reused — but "run1_dryrun" cannot be mistaken later
        # for collected data, and the label says so in the directory name, the
        # RUN.txt and every filename under it. SP2's live lane does the same.
        label = args.label or ("abort" if args.abort_check else
                               "dryrun" if args.dry_run else "full")
        outdir = run_dir(args.model, args.run, label)
    # The stem names the experiment, model and attempt, so a response file
    # stays identifiable even after it is copied out of the run tree.
    tag      = args.tag or ("abort" if args.abort_check else None)
    name     = stem(args.model, outdir, tag)
    out_path = os.path.join(outdir, name + RESPONSE)

    tasks = build_tasks(args.model, levels, pools, args.repetition, args.limit)

    # Say where everything is BEFORE the first call — a run whose destination
    # is not echoed to the screen can write into an unintended tree with
    # nothing visible at the time to catch it.
    P.print_paths(source_path=POOLS_CSV, source_label="pools",
                  run_directory=outdir)
    done  = load_done(out_path)
    todo  = [t for t in tasks
             if (t["problem_id"], t["level"], t["repeat"]) not in done]

    print("=" * 68)
    print(f"  Scaffold Gradient · live lane")
    print(f"  Model:    {args.model}  ->  {cfg['model_id']}")
    print(f"  Provider: {cfg.get('provider_pin') or '(unpinned)'}")
    print(f"  Tokens:   {max_tokens}  (pinned provider's cap)")
    print(f"  Tasks:    {len(tasks)}   already done: {len(done)}   to run: {len(todo)}")
    print(f"  Output:   {out_path}" + ("   [DRY RUN — no API calls]" if args.dry_run else ""))
    print("=" * 68)
    if not todo:
        print("  Nothing to do.")
        return 0

    # Written before the first call, not after the last: an interrupted run
    # still leaves a record of what it was and where its output went.
    write_manifest(
        outdir,
        experiment="Scaffold Gradient",
        lane="live",
        projects_root=P.projects(),
        subcat=P.subcat(),
        file_stem=name,
        model=args.model,
        model_id=cfg["model_id"],
        provider_pin=cfg.get("provider_pin") or "(unpinned)",
        levels=", ".join(sorted({t["level"] for t in tasks})),
        pools=", ".join(sorted({t["pool_id"] for t in tasks})),
        repeats=",".join(str(r) for r in args.repetition),
        calls=len(tasks),
        already_done=len(done),
        max_tokens=max_tokens,
        dry_run="yes" if args.dry_run else "no",
        started_at=datetime.now().isoformat(),
        response_file=os.path.basename(out_path))

    client = InferenceClient(
        model_cfg=cfg,
        system_prompt=SYSTEM_PROMPT or "",
        dry_run=args.dry_run,
        rate_limit=args.rate_limit,
    )

    n_ok = n_err = 0
    # Append as each call lands, never buffer to the end: an interrupted run
    # must keep every answer already paid for.
    with open(out_path, "a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:

        futures = {
            # ceiling == max_tokens disables the live client's adaptive
            # step-up: every model already starts at its own provider
            # ceiling, so there is nothing left to step up to.
            pool.submit(client.call, t["user_prompt"],
                        f'{t["problem_id"]}::{t["level"]}::rep{t["repeat"]}',
                        max_tokens, max_tokens): t
            for t in todo
        }
        pbar = tqdm(total=len(todo), desc="API Calls", unit="call")
        for i, fut in enumerate(as_completed(futures), 1):
            t  = futures[fut]
            cr = fut.result()
            rec = {
                "problem_id":    t["problem_id"],
                "level":         t["level"],
                "pool_id":       t["pool_id"],
                "repeat":        t["repeat"],
                "model":         args.model,
                "model_version": cr.model_version or cfg["model_id"],
                "provider":      cr.provider,
                "lane":          "live",
                # Marks a record that was never sent to a provider. A dry-run
                # response_text is the literal string "[DRY RUN]..." — not
                # empty — so without this flag the deliverables builder grades
                # it as a wrong answer and the reported accuracy silently
                # drops. The flag travels IN the record because --output can
                # put a dry run in any directory, so the directory name cannot
                # be the only defence.
                "dry_run":       bool(args.dry_run),
                # prompt_length and prompt_tokens_approx measure the USER turn,
                # matching what the level templates control; full_prompt is the
                # whole prompt sent and carries the system turn as well.
                "system_prompt":        SYSTEM_PROMPT,
                "user_prompt":          t["user_prompt"],
                "full_prompt":          full_prompt(SYSTEM_PROMPT,
                                                    t["user_prompt"]),
                "prompt_message":       prompt_message(SYSTEM_PROMPT,
                                                       t["user_prompt"]),
                "prompt_length":        len(t["user_prompt"]),
                "prompt_tokens_approx": len(t["user_prompt"]) // 4,
                "problem_latex":        t["matrix_latex"],
                "answer_latex":         t["answer_latex"],
                "response_text": cr.text,
                "finish_reason": cr.finish_reason,
                "boxed_count":   count_boxed(cr.text),
                "tokens_used":   cr.tokens_used,
                "latency_ms":    cr.latency_ms,
                "error":         cr.error,
                "timestamp":     datetime.now().isoformat(),
            }
            fout.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            fout.flush()
            n_err += bool(cr.error)
            n_ok  += not cr.error
            pbar.set_postfix(ok=n_ok, err=n_err)
            pbar.update(1)
        pbar.close()

    # The pin must be verified from the responses, not assumed from the request.
    pin = cfg.get("provider_pin")
    seen = sorted({json.loads(l)["provider"] for l in open(out_path, encoding="utf-8")
                   if l.strip() and json.loads(l).get("provider")})
    print(f"\n  providers seen: {seen}")
    # Case-insensitive on purpose. OpenRouter pins are lowercase slugs
    # ("together", "alibaba"); the provider reported back on the response is
    # display-cased ("Together", "Alibaba"). Exact equality would fire on
    # every correct call — see the matching note in common/inference_llm.py.
    # Only case and surrounding space are forgiven; together -> deepinfra
    # still fires. `providers` keeps the verbatim strings for the message and
    # for the deliverables.
    _norm = {p.strip().casefold() for p in seen if p}
    if pin and seen and _norm != {pin.strip().casefold()}:
        print(f"  ‼ PROVIDER PIN VIOLATED: asked for '{pin}', saw {seen}.")
        print("    These results mix model builds and should not be used as-is.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Every path is given on the command line, none guessed. See scaffold_paths.py.
    ap.add_argument("--projects-root", required=True,
                    help="output tree, e.g. ../projects. Must already exist. "
                         "Required: there is no default, so output is never "
                         "written somewhere you did not name.")
    ap.add_argument("--subcat", default=P.DEFAULT_SUBCAT, choices=sorted(P.SUBCATS),
                    help=f"path segment <projects-root>/{{subcat}}/... "
                         f"(default: {P.DEFAULT_SUBCAT})")
    ap.add_argument("--model", required=True, choices=LEVEL_MODELS["L0"])
    ap.add_argument("--output", default=None,
                    help="explicit output directory. Default is the managed "
                         "layout, <projects-root>/{subcat}/{model}/run{n}_{label}.")
    ap.add_argument("--run", type=int, default=None,
                    help="run number under {subcat}/{model}/. Default is the "
                         "next free one. Pass the SAME number to resume an "
                         "interrupted run in place.")
    ap.add_argument("--label", default=None,
                    help="short purpose appended to the run directory name")
    ap.add_argument("--tag", default=None,
                    help="qualifier appended to the file-name stem, to keep a "
                         "second pass in the SAME run directory apart from the "
                         "first")
    ap.add_argument("--abort-check", action="store_true",
                    help="run exactly the L0/irr-sym replication cell that "
                         "scaffold_abort_check.py grades. Overrides --level/--pool "
                         "and skips the abort-marker check, since this is the run "
                         "that creates it. Only valid for Claude-4.5-Sonnet.")
    ap.add_argument("--level", action="append", choices=LEVELS)
    ap.add_argument("--pool", action="append",
                    choices=["irr-sym", "irr-nonsym", "int-sym", "int-nonsym"])
    ap.add_argument("--repetition", type=_parse_repetition, default=list(range(1, REPEATS + 1)),
                    help="which repeat index/indices to generate, comma-separated "
                         "(default 1..%d, all repeats). Examples: '1' for just "
                         "repeat 1, '2,3' for repeats 2 and 3. Lets repeats be "
                         "split across separate run invocations "
                         "(--run 1 --repetition 1, then --run 2 --repetition 2, "
                         "then --run 3 --repetition 3) without colliding on the "
                         "same repeat index." % REPEATS)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Override the token budget. Default is the pinned "
                         "provider's own ceiling (models.py max_output_tokens).")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rate-limit", type=float, default=2.0, help="calls/sec")
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="problems per pool, for a cheap shakedown")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API calls; writes placeholder records")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
