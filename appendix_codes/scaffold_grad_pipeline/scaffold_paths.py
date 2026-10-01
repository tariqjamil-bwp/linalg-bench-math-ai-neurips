#!/usr/bin/env python3
# Module:    scaffold_paths.py
# Version:   2.0
"""
scaffold_paths.py — where run output lives. One source of truth for the layout.

    <projects-root>/                    <- named by --projects-root
        └── eigenvalue/                  <- named by --subcat
            ├── Claude-4.5-Sonnet/
            │   ├── run1_abort/          the L0 replication gate
            │   │   ├── Scaffold_Claude-4.5-Sonnet_run1_abort_response.jsonl
            │   │   └── RUN.txt
            │   ├── run2_full/           all six levels, all four pools
            │   └── run3_l4_refetch/     a top-up after a short run
            ├── Gemini-3.1-Pro/
            ├── Qwen3-235B/
            ├── coding/                  the two step-coding worksheets
            └── deliverables/            the three final output files, flat

This mirrors the sibling IBP pipeline's path module and the main benchmark
pipeline's {subcat}/{model} shape, so the experiments stay consistent to
operate.

There is no default root and no environment-variable fallback: configure()
must be called explicitly with a path from the command line, so a run can
never silently write into the wrong tree. Every path below is a function,
not a module constant, so nothing can be bound before configure() runs —
calling any of them first raises. configure() also rejects an empty,
missing, or non-directory root, and a subcat this tree does not serve,
before anything is created.

Each run gets its own run{n} directory, never reused. A short or partial
run stays on disk rather than being overwritten; the deliverables builder
reads every run together and dedups by (model, level, pool, problem,
repeat), preferring a record that carries an actual response. The run
number is allocated by the code, not chosen by hand.

Every file a run writes shares one stem, Scaffold_{model}_run{n}[_{tag}],
so the file stays self-identifying once it leaves the tree (email
attachment, shared folder, ticket), not just inside it. The optional tag
keeps two builds inside one run directory apart — e.g. the abort-gate cell
is tagged 'abort' so its output can't collide with the full grid's.

scaffold_pools.csv and scaffold_abort_passed.json are experiment-global,
single-copy files kept beside the code rather than exposed as command-line
paths: the pools are frozen once for all runs, and the abort marker is the
one gate every run is checked against, so there is deliberately no way to
point at more than one of either.
"""

from __future__ import annotations

import os
import re

EXPERIMENT = "Scaffold"

# The subcategories this tree serves. One, today. Kept as a set rather than a
# bare string so a wrong --subcat fails at argument-parsing time instead of
# quietly creating projects/eigenvlaue/ next to projects/eigenvalue/.
# scaffold_pools.py and scaffold_grade.py both already assert "eigenvalue" on
# the data; this is the earliest of the three checks.
SUBCATS = {"eigenvalue"}
DEFAULT_SUBCAT = "eigenvalue"

# The suffix appended to the stem. Named here rather than spelled out at each
# call site so every script agrees on it exactly. Not a path, so it stays a
# plain constant.
RESPONSE = "_response.jsonl"

SOURCE_CSV_NAME = "scaffold_source.csv"

_RUN_DIR = re.compile(r"^run(\d+)(?:_.*)?$")

_ROOT: str | None = None
_SUBCAT: str | None = None


# ── configuration ────────────────────────────────────────────────────────────

def configure(projects_root: str | None, subcat: str = DEFAULT_SUBCAT) -> None:
    """Bind the output tree. Call once, from main(), before any path is used.

    projects_root must already exist. Creating it on demand would defeat the
    purpose: a typo would then produce a new empty tree that looks like a fresh
    experiment instead of an error.
    """
    global _ROOT, _SUBCAT

    if projects_root is None or not str(projects_root).strip():
        raise SystemExit(
            "--projects-root is empty.\n"
            "  There is no default. Name the tree explicitly, e.g.\n"
            "    --projects-root ../projects")

    root = os.path.abspath(os.path.expanduser(str(projects_root).strip()))
    if not os.path.exists(root):
        raise SystemExit(
            f"--projects-root does not exist: {root}\n"
            "  It is not created on demand — a typo would otherwise look like a\n"
            "  fresh experiment rather than a mistake.")
    if not os.path.isdir(root):
        raise SystemExit(f"--projects-root is not a directory: {root}")

    sub = str(subcat).strip()
    if sub not in SUBCATS:
        raise SystemExit(
            f"--subcat {sub!r} is not served by this tree.\n"
            f"  Known: {', '.join(sorted(SUBCATS))}")

    _ROOT, _SUBCAT = root, sub


def is_configured() -> bool:
    return _ROOT is not None


def _root() -> str:
    if _ROOT is None:
        raise RuntimeError(
            "scaffold_paths is not configured — call configure(--projects-root).\n"
            "  There is deliberately no default: a path is never guessed.")
    return _ROOT


def subcat() -> str:
    _root()
    return _SUBCAT


# ── the tree ─────────────────────────────────────────────────────────────────

def projects() -> str:
    """The root the operator named."""
    return _root()


def subcat_dir() -> str:
    """projects/{subcat} — everything a run writes lives under here."""
    return os.path.join(_root(), subcat())


def source() -> str:
    """projects/source — inputs, shared by every subcat of this tree."""
    return os.path.join(_root(), "source")


def deliverables() -> str:
    return os.path.join(subcat_dir(), "deliverables")


def coding() -> str:
    return os.path.join(subcat_dir(), "coding")


def source_csv() -> str:
    """scaffold_source.csv — all four 5x5 cells, one row per problem."""
    return os.path.join(source(), SOURCE_CSV_NAME)


def results_glob() -> str:
    """Every response file any run of this subcat has written.

    The deliverables builder globs this, so adding a run directory is all it
    takes to fold a top-up into the analysis.
    """
    return os.path.join(subcat_dir(), "*", "run*", "*" + RESPONSE)


def model_dir(model: str) -> str:
    return os.path.join(subcat_dir(), model)


# ── run directories ──────────────────────────────────────────────────────────

def existing_runs(model: str) -> list[tuple[int, str]]:
    """(number, path) for every run directory of this model, lowest first."""
    d = model_dir(model)
    if not os.path.isdir(d):
        return []
    out = []
    for name in os.listdir(d):
        m = _RUN_DIR.match(name)
        if m and os.path.isdir(os.path.join(d, name)):
            out.append((int(m.group(1)), os.path.join(d, name)))
    return sorted(out)


def next_run_number(model: str) -> int:
    runs = existing_runs(model)
    return (runs[-1][0] + 1) if runs else 1


def run_number_of(path: str) -> int | None:
    """The run number encoded in a directory name, or None if it is not one.

    Returning None rather than raising is deliberate: --output may point
    anywhere, and a scratch directory should still produce usable filenames.
    """
    m = _RUN_DIR.match(os.path.basename(os.path.normpath(path)))
    return int(m.group(1)) if m else None


def run_dir(model: str, run: int | None = None, label: str | None = None,
            create: bool = True) -> str:
    """Resolve projects/{subcat}/{model}/run{n}[_{label}].

    With run=None the next free number is taken, so a run is never numbered
    by hand. An explicit number that already exists is returned as-is, to
    resume a run in progress.
    """
    if run is None:
        run = next_run_number(model)
    for n, path in existing_runs(model):
        if n == run:
            return path
    name = f"run{run}" + (f"_{_slug(label)}" if label else "")
    path = os.path.join(model_dir(model), name)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()


# ── filenames ────────────────────────────────────────────────────────────────

def stem(model: str, outdir: str, tag: str | None = None) -> str:
    """Scaffold_{model}_run{n}[_{tag}] — the stem every file of a run shares.

    The run number is read back out of the directory rather than passed in,
    so build, fetch, and parse always derive the same name from the same
    source instead of separately-tracked values that could drift apart.
    """
    parts = [EXPERIMENT, model]
    run = run_number_of(outdir)
    if run is not None:
        parts.append(f"run{run}")
    if tag:
        parts.append(_slug(tag))
    return "_".join(parts)


def response_name(model: str, outdir: str, tag: str | None = None) -> str:
    """Scaffold_{model}_run{n}[_{tag}]_response.jsonl"""
    return stem(model, outdir, tag) + RESPONSE


# ── reporting ────────────────────────────────────────────────────────────────

def print_paths(source_path: str | None = None, run_directory: str | None = None,
                n_rows: int | None = None, source_label: str = "source") -> None:
    """The resolved absolute paths, printed before the first API call.

    Echoed to the screen rather than left to RUN.txt alone, since RUN.txt
    is only read after the fact.
    """
    print("\nPATHS")
    print(f"  projects root   {projects()}")
    print(f"  subcat          {subcat()}")
    if source_path:
        rows = f"   ({n_rows} rows)" if n_rows is not None else ""
        print(f"  {source_label:<14}  {os.path.abspath(source_path)}{rows}")
    if run_directory:
        state = "existing — resuming" if os.path.isdir(run_directory) and \
            os.listdir(run_directory) else "new"
        print(f"  run directory   {os.path.abspath(run_directory)}   ({state})")
    print()


def write_manifest(path: str, **fields) -> str:
    """A RUN.txt beside the data saying what this run was and why.

    Written at build time, when the reason is known — a directory name
    alone doesn't say whether a later run was a top-up or a redo.
    """
    manifest = os.path.join(path, "RUN.txt")
    width = max(len(k) for k in fields)
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(f"{os.path.basename(path)}\n")
        f.write("=" * len(os.path.basename(path)) + "\n\n")
        for k, v in fields.items():
            f.write(f"{k.replace('_', ' '):<{width}}  {v}\n")
    return manifest
