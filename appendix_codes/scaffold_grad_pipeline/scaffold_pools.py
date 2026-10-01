#!/usr/bin/env python3
# Module:    scaffold_pools.py
# Version:   1.0
"""
scaffold_pools.py — resolve the four stratified problem pools, once, and freeze
them to scaffold_pools.csv.

    irr-sym      30   symmetric    x irrational   <- the published set
    irr-nonsym   30   non-symmetric x irrational
    int-sym      15   symmetric    x integer      (sampled from 30)
    int-nonsym   15   non-symmetric x integer     (sampled from 30)
                 ──
                 90 unique problems

The source CSV (scaffold_source.csv) holds all four 5x5 cells: 120 problems,
30 per cell. The published irr-sym 30 are identified by a fixed ID range
(C_5x5_eig_007..036); every other row belongs to the other three cells.

Because the source tree can move independently of this script, it writes a
resolved CSV carrying the ids AND the exact L0 prompt text. Every later
stage reads that CSV, never the source tree, so the run stays reproducible
even if the source layout changes afterward.

The baseline 30 already carry the complete published L0 prompt. The 90
extension rows carry the head and the matrix but not the instruction tail,
which is appended here and then checked to match the published merged
benchmark byte-for-byte — a check, not a source, so a one-character drift
stops the build instead of silently running a different experiment.

Sampling uses a fixed seed (42), re-seeded immediately before each of the
two 15-item draws, so one draw's membership never depends on draw order.
Candidate ids are sorted before sampling too, so the draw doesn't depend on
CSV row order either.

Usage:
    python3 scaffold_pools.py                    # build + verify + write
    python3 scaffold_pools.py --check            # verify an existing CSV only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

import pandas as pd

from scaffold_levels import L0_HEAD, L0_TAIL, problem_statement

# ─────────────────────────────────────────────────────────────────────────
#  Paths — one root, recorded in the output so the run is auditable
# ─────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

# The single source file, copied flat into projects/source/ so a run does
# not depend on the original tree staying put. One row per problem, with
# symmetry/spectrum_type already resolved; the published irr-sym 30 are the
# fixed ID range in IRR_SYM_IDS below, everything else is the other three
# pools.
#
# There is no default and no environment-variable fallback — --source names
# the directory explicitly, and set_source() below is the only way this
# name gets a value, so a run can never silently target the wrong tree.
SOURCE_DIR = None
SOURCE_CSV = None

# scaffold_pools.csv lives beside scaffold_source.csv in source/, deliberately
# not a flag: the pools are frozen once for every run of the experiment, and a
# flag would imply there can be more than one frozen set.
OUTPUT_CSV = os.path.join(_HERE, "source", "scaffold_pools.csv")


def set_source(source_dir: str | None) -> str:
    """Bind the input CSV. Called from main() with --source.

    The directory must already exist. Creating it on demand would turn a typo
    into an empty-pools build rather than an error.
    """
    global SOURCE_DIR, SOURCE_CSV

    if source_dir is None or not str(source_dir).strip():
        raise SystemExit(
            "--source is empty.\n"
            "  There is no default. Name the directory holding\n"
            "  scaffold_source.csv explicitly, e.g.  --source ../projects/source")

    d = os.path.abspath(os.path.expanduser(str(source_dir).strip()))
    if not os.path.isdir(d):
        raise SystemExit(f"--source is not a directory: {d}")

    SOURCE_DIR = d
    SOURCE_CSV = os.path.join(d, "scaffold_source.csv")
    return d


def _require_source() -> None:
    if SOURCE_DIR is None:
        raise RuntimeError(
            "scaffold_pools source directory is not set — call set_source(--source).\n"
            "  There is deliberately no default: a path is never guessed.")

SEED = 42
POOL_SIZES = {"irr-sym": 30, "irr-nonsym": 30, "int-sym": 15, "int-nonsym": 15}

# The 30 published irr-sym ids are a fixed ID range, not a column in the
# source file — every other row in scaffold_source.csv is an extension row.
IRR_SYM_IDS = [f"C_5x5_eig_{i:03d}" for i in range(7, 37)]


def _norm_symmetry(s: str) -> str:
    """Normalises `non_symmetric` / `non-symmetric` spelling variants.
    Not cosmetic: an un-normalised filter returns zero rows and would
    produce empty pools with no error."""
    return str(s).strip().lower().replace("-", "_")


def _looks_irrational(answer_latex: str) -> bool:
    """An irrational spectrum carries a radical or a non-integer decimal.

    Used to confirm the fixed C_5x5_eig_007..036 range still names an
    irrational-symmetric set — a fixed range can't detect the underlying
    data moving; this can.
    """
    t = str(answer_latex)
    if "\\sqrt" in t:
        return True
    return bool(re.search(r"\d\.\d", t))


def _cell(symmetry: str, spectrum: str) -> str:
    sym = "sym" if _norm_symmetry(symmetry) == "symmetric" else "nonsym"
    spec = "irr" if str(spectrum).strip().lower() == "irrational" else "int"
    return f"{spec}-{sym}"


# ─────────────────────────────────────────────────────────────────────────
#  Build
# ─────────────────────────────────────────────────────────────────────────
def build() -> pd.DataFrame:
    _require_source()
    if not os.path.exists(SOURCE_CSV):
        raise SystemExit(
            f"Missing required source file:\n  {SOURCE_CSV}\n"
            f"  Expected in {SOURCE_DIR}\n"
            f"  (name a different directory with --source)")

    src = pd.read_csv(SOURCE_CSV).set_index("problem_id")

    # ---- irr-sym: the 30 published ids, text from those rows ----------------
    # Cross-checked against the source file itself, since a fixed ID range
    # alone can't notice if the underlying data shifts. An irrational
    # spectrum shows up as a radical or a non-integer in answer_latex; every
    # one of the 30 must be present and none may be integer.
    rows = []
    for pid in IRR_SYM_IDS:
        if pid not in src.index:
            raise SystemExit(f"Baseline id {pid} not found in {SOURCE_CSV}")
        r = src.loc[pid]
        if not _looks_irrational(r["answer_latex"]):
            raise SystemExit(f"{pid} has an integer-looking spectrum "
                             f"({str(r['answer_latex'])[:60]!r}) but irr-sym must be "
                             f"irrational — the published set has moved.")
        rows.append({
            "problem_id":    pid,
            "pool_id":       "irr-sym",
            "symmetry":      "symmetric",
            "spectrum_type": "irrational",
            "source":        "baseline",
            "problem_latex": r["problem_latex"],
            "answer_latex":  r["answer_latex"],
        })

    # ---- the other three cells, from every remaining row ---------------------
    ext = src.loc[~src.index.isin(IRR_SYM_IDS)].copy()
    ext["cell"] = [_cell(s, t) for s, t in zip(ext["symmetry"], ext["spectrum_type"])]
    by_cell = {c: sorted(g.index) for c, g in ext.groupby("cell")}

    taken = set(IRR_SYM_IDS)

    def add(cell: str, n: int, sample: bool) -> None:
        # Baseline wins on any id collision — the published irr-sym set must
        # stay exactly the 30 published ids, and an id cannot sit in two pools.
        cands = [p for p in by_cell.get(cell, []) if p not in taken]
        if len(cands) < n:
            raise SystemExit(f"Cell {cell}: need {n} ids, only {len(cands)} available")
        if sample:
            random.seed(SEED)          # re-seeded before EACH draw, independent of order
            chosen = sorted(random.sample(cands, n))
        else:
            chosen = cands[:n]
        for pid in chosen:
            taken.add(pid)
            r = ext.loc[pid]
            rows.append({
                "problem_id":    pid,
                "pool_id":       cell,
                "symmetry":      _norm_symmetry(r["symmetry"]),
                "spectrum_type": r["spectrum_type"],
                "source":        "extension",
                "problem_latex": r["problem_latex"] + "\n\n" + L0_TAIL,
                "answer_latex":  r["answer_latex"],
            })

    add("irr-nonsym", 30, sample=False)   # all 30 — no draw, so no seed needed
    add("int-sym",    15, sample=True)
    add("int-nonsym", 15, sample=True)

    df = pd.DataFrame(rows)
    # L0 is the stored string, unmodified — rebuilding it from a template would
    # risk a one-character drift and break the replication the abort gate rests on.
    df["l0_prompt"] = df["problem_latex"]     # as stored, for the drift check
    # The bare matrix, frozen alongside it. L1-L4 supply their own
    # instructions and take nothing from the bank but this block, so it's
    # stored once here rather than re-sliced at every render.
    df["matrix_latex"] = df["problem_latex"].map(problem_statement)
    return df


# ─────────────────────────────────────────────────────────────────────────
#  Verify
# ─────────────────────────────────────────────────────────────────────────
def spec_recipe() -> dict[str, list[str]]:
    """The pool membership rule, re-implemented independently of build().

    A second, minimal implementation rather than a refactor of the first:
    build() carries extra machinery (collision handling, prompt assembly,
    the L0 tail append) that could in principle perturb which ids end up in
    a pool. This function does only:

      irr-sym     C_5x5_eig_007..036, all 30, no sampling
      irr-nonsym  every other row, symmetry == non-symmetric
                  AND spectrum_type == irrational, all 30
      int-sym     every other row, symmetry == symmetric
                  AND spectrum_type == integer, then
                  random.seed(42); random.sample(sorted_ids, 15)
      int-nonsym  the same with non-symmetric, fresh seed before the draw

    If the two disagree, the pools are wrong and the build stops.
    """
    _require_source()
    meta = pd.read_csv(SOURCE_CSV)
    meta = meta[~meta["problem_id"].isin(IRR_SYM_IDS)]

    def ids_for(symmetry: str, spectrum: str) -> list[str]:
        m = meta[(meta.symmetry.map(_norm_symmetry) == _norm_symmetry(symmetry))
                 & (meta.spectrum_type.str.strip().str.lower() == spectrum)]
        return sorted(m.problem_id)

    out = {"irr-sym": [f"C_5x5_eig_{i:03d}" for i in range(7, 37)],
           "irr-nonsym": ids_for("non-symmetric", "irrational")}

    for pool, symmetry in (("int-sym", "symmetric"), ("int-nonsym", "non-symmetric")):
        sorted_ids = ids_for(symmetry, "integer")
        random.seed(SEED)                       # fresh seed before EACH draw
        out[pool] = sorted(random.sample(sorted_ids, 15))
    return out


def verify(df: pd.DataFrame) -> int:
    fails = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal fails
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            fails += 1

    counts = Counter(df.pool_id)
    check(dict(counts) == POOL_SIZES, f"pool sizes {dict(counts)} == {POOL_SIZES}")
    check(len(df) == 90, f"90 rows total (got {len(df)})")
    check(df.problem_id.nunique() == len(df), "no id appears in two pools")

    r2 = sorted(df[df.pool_id == "irr-sym"].problem_id)
    check(r2 == [f"C_5x5_eig_{i:03d}" for i in range(7, 37)],
          "irr-sym is exactly C_5x5_eig_007..036 (all 30, no sampling)")

    # The strongest check: the membership rule, re-derived independently.
    spec = spec_recipe()
    for pool in POOL_SIZES:
        got = sorted(df[df.pool_id == pool].problem_id)
        ok = got == spec[pool]
        check(ok, f"{pool} matches the recipe exactly ({len(got)} ids)")
        if not ok:
            print(f"        only in build : {sorted(set(got) - set(spec[pool]))}")
            print(f"        only in spec  : {sorted(set(spec[pool]) - set(got))}")

    # Each pool must hold one symmetry class and one spectrum type only,
    # with nothing mixed in.
    for pool, g in df.groupby("pool_id"):
        homogeneous = g.symmetry.nunique() == 1 and g.spectrum_type.nunique() == 1
        check(homogeneous,
              f"{pool} is one class only: {sorted(set(zip(g.symmetry, g.spectrum_type)))}")

    # Every row must carry the exact published L0 wrapper, or scaffold_levels
    # .problem_statement() cannot slice the matrix out for L1-L4.
    from scaffold_levels import problem_statement
    try:
        for t in df.l0_prompt:
            problem_statement(t)
        check(True, "all 90 L0 prompts have the expected head/tail wrapper")
    except ValueError as e:
        check(False, f"L0 wrapper: {e}")

    # The oracle check for the tail-append: the 30 benchmark rows carry the
    # real published wrapper, so every appended extension prompt must match
    # their exact head and tail, and must re-extract to its own matrix
    # unchanged.
    bench_rows = df[df.source == "baseline"].l0_prompt.tolist()
    head_ok = all(t.startswith(L0_HEAD) for t in bench_rows)
    tail_ok = all(t.rstrip().endswith(L0_TAIL) for t in bench_rows)
    check(head_ok and tail_ok,
          f"the {len(bench_rows)} benchmark rows define the published L0 wrapper")

    ext = df[df.source == "extension"]
    check(all(t.startswith(L0_HEAD) for t in ext.l0_prompt),
          f"all {len(ext)} extension L0 prompts carry the same head")
    check(all(t.rstrip().endswith(L0_TAIL) for t in ext.l0_prompt),
          f"all {len(ext)} extension L0 prompts carry the appended tail")

    # Replication check: every level, L0 included, is built from its own
    # template plus the matrix, so the built L0 must come back
    # byte-identical to the string the published run actually sent —
    # otherwise the abort gate would compare a rebuilt prompt against a
    # published number.
    from scaffold_levels import render as _render
    bench = df[df.source == "baseline"]
    drift = [p for p, m, orig in zip(bench.problem_id, bench.matrix_latex,
                                     bench.problem_latex)
             if _render("L0", m) != orig]
    check(not drift,
          f"built L0 is byte-identical to the published string for all "
          f"{len(bench)} benchmark rows"
          + (f" — DRIFT on {drift[:3]}" if drift else ""))

    # Round-trip: strip the wrapper back off and the matrix must be unchanged.
    bad = [p for p, l0, m in zip(df.problem_id, df.l0_prompt, df.matrix_latex)
           if problem_statement(l0) != m]
    check(not bad, f"matrix round-trips out of every L0 prompt (0 mismatched)")

    # The matrix is the only thing L1-L4 take from the bank, so it must be a
    # real bmatrix and must carry none of the L0 instruction wording.
    check(all("\\begin{bmatrix}" in m and "\\end{bmatrix}" in m
              for m in df.matrix_latex),
          "every stored matrix_latex is a bmatrix block")
    leaked = [p for p, m in zip(df.problem_id, df.matrix_latex)
              if L0_HEAD in m or L0_TAIL in m or "boxed" in m]
    check(not leaked, "no L0 instruction text leaked into matrix_latex")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Required, with no default, in both modes — --check still calls
    # verify(), which re-reads eig_extension_metadata.csv to re-derive the
    # membership rule independently of the frozen file.
    ap.add_argument("--source", required=True,
                    help="Directory holding scaffold_source.csv. Normally "
                         "<projects-root>/source. No default — see the module "
                         "header for why.")
    ap.add_argument("--output", default=OUTPUT_CSV,
                    help=f"Frozen pool file (default: {OUTPUT_CSV}). Stays "
                         f"beside the code on purpose: the pools are frozen "
                         f"once for the whole experiment.")
    ap.add_argument("--check", action="store_true",
                    help="verify the existing CSV instead of rebuilding it")
    args = ap.parse_args()

    set_source(args.source)

    if args.check:
        if not os.path.exists(args.output):
            raise SystemExit(f"No frozen pool file at {args.output}")
        df = pd.read_csv(args.output)
        print(f"Checking {args.output}")
        print(f"  against source {SOURCE_DIR}")
    else:
        df = build()
        print("Built pools from:")
        print(f"  {SOURCE_DIR}")

    fails = verify(df)

    if not args.check:
        if fails:
            print("\nVerification failed — NOT writing the pool file.")
            return 1
        df.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}  ({len(df)} problems)")
        for pool, n in POOL_SIZES.items():
            print(f"  {pool:<12} {n}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
