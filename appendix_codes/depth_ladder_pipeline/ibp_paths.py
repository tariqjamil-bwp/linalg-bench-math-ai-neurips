#!/usr/bin/env python3
# Module:    ibp_paths.py
# Version:   2.0
"""
ibp_paths.py — where run output lives. One source of truth for the layout.
This module never writes inside the code tree; every path below is rooted
at whatever directory --projects-root names.

    <projects-root>/                    <- named by --projects-root
        └── ibp/                         <- named by --subcat
            ├── Llama-3.3-70B/
            │   ├── run1_full/
            │   │   ├── IBP_Llama-3.3-70B_run1_response.jsonl
            │   │   ├── IBP_Llama-3.3-70B_run1.log
            │   │   └── RUN.txt
            │   └── run2_topup/          a second pass after a short run
            ├── Claude-4.5-Sonnet/
            ├── Gemini-3.1-Pro/
            ├── Qwen3-235B/
            └── deliverables/            the three deliverable files, flat

This mirrors the scaffold-gradient pipeline's own paths module deliberately,
and both mirror the main pipeline's {subcat}/{model} shape. A layout that
differs between them is a layout that gets confused between them.

THERE IS NO DEFAULT ROOT.  THIS IS THE POINT OF THE MODULE.
───────────────────────────────────────────────────────────
A default computed from an environment variable (e.g.
`os.environ.get("IBP_PROJECTS") or <default>`) lets a run start with no path
in the command at all — and an *empty* override falls through to the
default in silence, sending a real run into the wrong tree with nothing on
screen to say so.

A smart default is still a guess, and a guess that is usually right is the
hardest kind to catch. So:

  * there is no default root and no environment variable. configure() must be
    called, from main(), with a value that came off the command line;
  * every path is a FUNCTION, not a module constant, so a caller cannot bind a
    value at import time and then use it after the fact. Calling one before
    configure() raises;
  * configure() rejects an empty, blank, missing, or non-directory root, and a
    subcat this tree does not serve, before anything is created.

The main pipeline reaches the same place by practice rather than by code — its
documented commands always spell out --input and --output, and its stage 2/3
scripts mark them required=True. This module enforces what that practice
assumes.

WHY ONE DIRECTORY PER RUN
─────────────────────────
Numbers are monotonic and never reused, so a second attempt cannot land on
the first one's files. That matters more than tidiness: it makes a partial
run safe to keep. run1 that came back short and run2 that fetched the rest
both stay on disk; the deliverables builder reads every run together.

The run NUMBER is still allocated by the code, inside the root the operator
named. Hand-numbering runs is how a partial run's files get silently
overwritten; the operator says where the tree is, the code says which number
is free, and both are printed before the first API call.
"""

from __future__ import annotations

import os
import re

EXPERIMENT = "IBP"

# The subcategories this tree serves. One, today. Kept as a set rather than a
# bare string so a wrong --subcat fails here, at argument-parsing time, instead
# of quietly creating projects/ipb/ next to projects/ibp/.
# subcat_config_ibp.SUBCAT_CONFIGS raises for anything else as well; this is the
# earlier of the two checks.
SUBCATS = {"ibp"}
DEFAULT_SUBCAT = "ibp"

RESPONSE = "_response.jsonl"

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
            "ibp_paths is not configured — call configure(--projects-root).\n"
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


def results_glob() -> str:
    """Every response file any run of this subcat has written, both lanes.

    The deliverables builder globs this, so adding a run directory is all it
    takes to fold a top-up into the analysis. The write side and this read side
    must agree exactly — they did not once, and a finished run would have
    produced empty deliverables with no error. verify_run_layout.py asserts the
    agreement with both real modules rather than with fixtures.
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

    With run=None the next free number is taken, so a re-run never has to be
    numbered by hand — getting that wrong is how a partial run's files get
    silently overwritten. An explicit number that already exists is returned
    as-is: that is how you resume a run you are in the middle of.
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

def run_stem(model: str, outdir: str) -> str:
    """IBP_{model}_run{n} — the stem every file of one run shares.

    Inside the tree the experiment and model are already in the path, so this
    looks redundant. It is not redundant once a file leaves the tree — a bare
    'Claude-4.5-Sonnet_results.jsonl' names neither which of the two
    experiments produced it nor which attempt.

    The subcat is not added: the IBP_ prefix already maps one-to-one onto it,
    and a longer name buys nothing.

    The run number is read back out of the directory name rather than passed
    in, so it cannot drift from the directory the file sits in. A directory
    that is not a run directory simply yields a stem without a number — scratch
    runs keep working, with no second code path.
    """
    parts = [EXPERIMENT, model]
    run = run_number_of(outdir)
    if run is not None:
        parts.append(f"run{run}")
    return "_".join(parts)


def response_name(model: str, outdir: str) -> str:
    """IBP_{model}_run{n}_response.jsonl"""
    return run_stem(model, outdir) + RESPONSE


# ── reporting ────────────────────────────────────────────────────────────────

def print_paths(bank_path: str | None = None, run_directory: str | None = None,
                n_rows: int | None = None) -> None:
    """The resolved absolute paths, printed before the first API call.

    RUN.txt records these too, but it's read afterwards — a run that writes
    into the wrong tree is invisible unless something says so on screen
    while it's happening.
    """
    print("\nPATHS")
    print(f"  projects root   {projects()}")
    print(f"  subcat          {subcat()}")
    if bank_path:
        rows = f"   ({n_rows} rows)" if n_rows is not None else ""
        print(f"  bank            {os.path.abspath(bank_path)}{rows}")
    if run_directory:
        state = "existing — resuming" if os.path.isdir(run_directory) and \
            os.listdir(run_directory) else "new"
        print(f"  run directory   {os.path.abspath(run_directory)}   ({state})")
    print()


def write_manifest(path: str, **fields) -> str:
    """A RUN.txt beside the data saying what this run was and why.

    Written at build time, when the answer is known. Six weeks later the
    directory name alone will not say whether run2 was a top-up, a re-run after
    a bad fetch, or an experiment that was abandoned.
    """
    manifest = os.path.join(path, "RUN.txt")
    width = max(len(k) for k in fields)
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(f"{os.path.basename(path)}\n")
        f.write("=" * len(os.path.basename(path)) + "\n\n")
        for k, v in fields.items():
            f.write(f"{k.replace('_', ' '):<{width}}  {v}\n")
    return manifest
