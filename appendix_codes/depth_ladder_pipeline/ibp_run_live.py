#!/usr/bin/env python3
# Module:    ibp_run_live.py
# Version:   3.0
"""
================================================================================
IBP Depth Ladder · Stage 1 — Inference
================================================================================

Title:       0-shot inference for the IBP recursion-depth benchmark
Version:     3.0 (depth-ladder fork of the LinAlg-Bench inference stage)

PURPOSE:
    Zero-shot inference over the IBP Depth-Ladder bank — 340 SymPy-verified
    integrals of x^n·trig(bx), n = 0..16, 20 per level — at temperature 0.
    n sets the number of required IBP applications (n+1), giving a
    dimension-free recursion-depth axis.

    Key features:
      • Prompt = the bank's problem_latex, sent VERBATIM (no assembly)
      • One call per problem per run; a second repeat is a second run,
        tagged with --variant (rep1, rep2, ...)
      • Parallel API calls with per-provider rate limiting
      • Adaptive token step-up on truncation (ceiling = min_tokens × 3)
      • Loop detection and infinite repetition trimming
      • Deduplication by (question_id, variant), variant = rep1/rep2/...
      • Atomic per-call result saves with timestamp-based deduplication

WHAT THIS STAGE DOES NOT DO — GRADING
    It writes raw responses. The `correct` field is a coarse string-equality
    tally for run monitoring ONLY — it is not the result. Grading is symbolic,
    never string matching, and lives in a separate stage with a locked rule:

        ibp_grade_depth_ladder.py   -> symbolic pass/fail per response
        ibp_build_deliverables.py   -> the three deliverable files

    Grading is symbolic only — equivalent answers are judged by
    differentiation against the integrand, never by a paid adjudication call.

PARALLELIZATION:
    • ThreadPoolExecutor: concurrent API calls (--max-workers, default 8)
    • Semaphore-based rate limiting, separate semaphores per provider
    • Thread-safe JSONL: atomic writes with file locks prevent interleaving

TRANSPORT (inference_llm.py):
    • Backend auto-detection: OpenRouter / OpenAI / GenAI / generic OpenAI-compat
    • Temperature is hard-coded to 0.0 on every backend
    • Adaptive token step-up: truncations retry with increased max_tokens
    • finish_reason: "stop" | "length" | "loop_trimmed" | None

USAGE:
    No path has a default. --projects-root and --bank are both required, so
    output is never written somewhere the operator did not name.

    # Dry run — preview prompts, no API calls (lands in run{n}_dryrun/)
    python3 ibp_run_live.py --projects-root ../projects \
        --bank ../projects/source/ibp_depth_ladder_problems.csv \
        --model Llama-3.3-70B --limit 5 --dry-run

    # Full run for one model (340 problems, one call each = 340 calls)
    python3 ibp_run_live.py --projects-root ../projects \
        --bank ../projects/source/ibp_depth_ladder_problems.csv \
        --model Llama-3.3-70B

    # A second repeat is a second run, tagged rep2
    python3 ibp_run_live.py --projects-root ../projects \
        --bank ../projects/source/ibp_depth_ladder_problems.csv \
        --model Llama-3.3-70B --variant rep2

    # Resume an interrupted run (skips completed (id, variant) pairs).
    # --output is REQUIRED here: without it the next free run number is taken
    # and there would be nothing in the new directory to resume from.
    python3 ibp_run_live.py --projects-root ../projects \
        --bank ../projects/source/ibp_depth_ladder_problems.csv \
        --model Llama-3.3-70B \
        --resume all --output ../projects/ibp/Llama-3.3-70B/run1_full

INPUT CONTRACT:
    Reads ibp_depth_ladder_problems.csv as shipped — no prep step, no copy.
        Problem_ID     -> id  (lower-cased and aliased automatically)
        problem_latex  -> the complete user turn, sent verbatim (REQUIRED)
        answer_latex   -> ground truth
        n, trig, b     -> unused here; the grader reads them from the bank

OUTPUT:
    {output}/{stem}_response.jsonl — per-call records:
        • question_id       : Problem_ID from the bank
        • variant           : repeat index — "rep1", "rep2", ...
        • model             : model id string as reported by the provider
        • user_prompt       : the prompt sent (== problem_latex, byte-for-byte)
        • response          : full model response text
        • extracted_answer  : first \\boxed{} content (monitoring only)
        • correct           : extracted == ground_truth (monitoring only)
        • timestamp, finish_reason, tokens_used, latency_ms, error

    {output}/{stem}_summary.csv    — per-problem tally
    {output}/{stem}_response.xlsx  — formatted workbook, one row per call
    {output}/{stem}.log            — this run's log

  where stem is IBP_{model}_run{n} — see run_stem(). Every file one run writes
  shares it, so a file stays identifiable after it is copied out of the tree.

ENVIRONMENT:
    Key name comes from each model's api_key_env in models.py:
    OPENROUTER_API_KEY (Llama, Qwen), ANTHROPIC_API_KEY (Claude),
    GEMINI_API_KEY (Gemini).

================================================================================
"""

import os
import re
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import heapq
from threading import Lock, Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

# models.py, inference_llm.py and textguard.py live in ../common/, shared with
# other pipelines in this repo. Added to sys.path by location rather than a
# symlink, so the import resolves the same way on every OS and after a plain
# git clone or zip download.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from inference_llm import InferenceClient
from textguard import assert_clean
from subcat_config_ibp import SUBCAT_CONFIGS as get_subcat_config
from models import MODELS, get_model_names
try:
    from json_repair import loads as json_repair_loads
except ImportError:
    json_repair_loads = None

# NOTE: no grading import here on purpose. Inference writes raw responses only;
# grading is a separate stage (ibp_grade_depth_ladder.py) so the rule stays
# locked and auditable.

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ═════════════════════════════════════════════════════════════════════════
# 1. MODEL CONFIG
# ═════════════════════════════════════════════════════════════════════════

# Each record's `variant` carries its repeat index (rep1, rep2, ...), set from
# --variant. Reporting code discovers the set from the data rather than
# assuming it.


# ═════════════════════════════════════════════════════════════════════════
# 4. IBP CONFIG — the one subcategory this pipeline serves
# ═════════════════════════════════════════════════════════════════════════
# This tree runs the IBP Depth-Ladder experiment only. subcat_config_ibp.
# SUBCAT_CONFIGS raises KeyError for anything but "ibp".
#
# The system prompt matches the earlier small-model IBP runs, so depth-ladder
# results stay comparable to them.

_BASE_SYSTEM = (
    "You are a precise mathematical assistant. "
    "Show all computation steps clearly. "
)

IBP_SYSTEM_PROMPT = assert_clean(
    _BASE_SYSTEM + "Always put your final simplified antiderivative inside \\boxed{}.",
    "IBP_SYSTEM_PROMPT")

# ceiling = min_tokens × TOKEN_MULT (adaptive step-up upper bound)
IBP_TOKEN_MULT = 3


# ═════════════════════════════════════════════════════════════════════════
# 5. PROMPT BUILDER
# ═════════════════════════════════════════════════════════════════════════

def build_standard_prompt(problem_latex: str) -> str:
    """Return the user turn: the bank's problem_latex, byte-for-byte.

    The IBP Depth-Ladder bank stores the COMPLETE prompt in problem_latex —
    task line, integral, and the \\boxed{} directive already assembled by the
    generator. The model must receive exactly that text, so this deliberately
    does nothing to it: no wrapper sentence, no task-label concatenation, no
    boxed-hint fallback, no strip.

    Anything that edits the prompt here silently invalidates the experiment,
    which is why there is no branch left that could. The guard does not edit —
    it refuses, so a corrupted \\boxed{} stops the run instead of being sent.
    """
    return assert_clean(problem_latex, "the IBP user prompt")


# ═════════════════════════════════════════════════════════════════════════
# 6. ANSWER EXTRACTION
# ═════════════════════════════════════════════════════════════════════════
# ground_truth_fn and extract_answer_fn are provided per-subcat by
# subcat_config.get_config(subcat) — no duplication here.

def count_boxed(text: str) -> int:
    """Count \\boxed{} occurrences — multiples indicate hedging behavior."""
    if not text:
        return 0
    return len(re.findall(r'\\boxed\{', text))


# ─────────────────────────────────────────────────────────────────────────
#  Output file naming
# ─────────────────────────────────────────────────────────────────────────
# The suffix constant is a plain string and safe to import by name. Every
# actual PATH is reached through P.<fn>() instead, because ibp_paths has no
# default root: a from-import would bind at import time, before --projects-root
# has been seen, which is exactly the class of mistake this redesign removes.
import ibp_paths as P                                          # noqa: E402
from ibp_paths import (                                        # noqa: E402
    RESPONSE, response_name, run_dir, run_number_of, run_stem, write_manifest,
)


# Loop detection, loop trimming, adaptive tokens, finish_reason:
# all handled inside InferenceClient (inference_llm.py)


# ═════════════════════════════════════════════════════════════════════════
# 7. DATA LOADING
# ═════════════════════════════════════════════════════════════════════════

# This experiment has no judge stage and no error taxonomy — only
# pass/parse_fail — so there are no Error_Tag-style columns to load.


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column name variants to standard internal names."""
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    mapping = {
        # Problem_ID lower-cases to problem_id and resolves here, so the
        # depth-ladder bank is consumed as shipped, with no renamed copy.
        'id': ['question_id', 'problem_id', 'problemid', 'qid', 'idx'],
        # 'problem_latex' has NO alias — it must be named literally, because it
        # is sent to the model verbatim. Aliasing another column into it would
        # silently change the prompt text the experiment is measuring.
        'answer_latex': ['gt', 'ground_truth'],
        'subcategory': ['subcat', 'category', 'type'],
    }

    for std_name, aliases in mapping.items():
        if std_name in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                df.rename(columns={alias: std_name}, inplace=True)
                break

    return df


def _detect_file_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return 'excel'
    elif ext == '.csv':
        return 'csv'
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_dataset(input_path: str) -> pd.DataFrame:
    """
    Load a problem bank from CSV or Excel, normalise its column names, and
    attach the ground-truth column.

    Standard internal names:
      id, problem_latex, answer_latex

    problem_latex is the complete user turn and is sent verbatim. There is no
    input-mode detection and no prompt assembly — see build_standard_prompt().

    Adds one column: ground_truth (see the ground-truth note at the end of
    this function). Ground truth is read from the bank, never recomputed, so
    this function makes no SymPy calls and needs no subcat config.
    """
    file_type = _detect_file_type(input_path)

    if file_type == 'csv':
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    df = _normalize_columns(df)

    # problem_latex is the only column whose absence is fatal — it IS the prompt.
    if 'problem_latex' not in df.columns:
        raise ValueError(
            f"Input {input_path} has no 'problem_latex' column. That column is "
            f"sent to the model verbatim as the user turn; without it there is "
            f"no prompt. Found: {list(df.columns)}"
        )

    for col in ['id', 'answer_latex']:
        if col not in df.columns:
            print(f"  [WARN] Column '{col}' not found in input — inference may fail")

    # GROUND TRUTH = answer_latex, read straight from the bank. The generators
    # always write it (sp.latex of the simplified SymPy result), so there is
    # nothing to re-derive — no SymPy at load time.
    # Empty answer_latex falls back to answer_sympy.
    _fallback = df['answer_sympy'] if 'answer_sympy' in df.columns else ''
    df["ground_truth"] = (df["answer_latex"]
                          .replace(r'^\s*$', pd.NA, regex=True)
                          .fillna(_fallback).fillna('').astype(str))
    return df


# ─────────────────────────────────────────────────────────────────────────
#  Depth-slice selection
# ─────────────────────────────────────────────────────────────────────────
# The planned run is not sliced: one run per model covers the whole
# 340-problem ladder in one pass.
#
# --n-range has NO ROLE in the planned run. Omit it and the whole ladder is
# built. It exists for one job: re-collecting a level that came back short
# or stalled, without hand-cutting a CSV. The deliverables builder folds the
# re-collected run in by globbing, with nothing renamed.
#
# The block is a FILTER on the bank, never a separate bank file. One input
# file for the whole run means no two slices can drift apart.

def parse_n_range(text: str | None) -> tuple[int, int] | None:
    """'0-1' or '16' -> (0, 1) / (16, 16). None means the whole ladder.

    Rejects a reversed or non-numeric range outright: a silently empty slice
    would build zero requests and look like a finished block.
    """
    if text is None or not str(text).strip():
        return None
    raw = str(text).strip()
    try:
        if "-" in raw:
            lo_s, hi_s = raw.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
        else:
            lo = hi = int(raw)
    except ValueError:
        raise SystemExit(
            f"--n-range {raw!r} is not a range. Use LO-HI (e.g. 0-1) or a "
            f"single level (e.g. 16).")
    if lo > hi:
        raise SystemExit(f"--n-range {raw!r} is reversed: {lo} > {hi}.")
    return lo, hi


def n_range_label(n_range: tuple[int, int] | None) -> str:
    """The run-directory label for a block: n00_01, n16, or 'full'."""
    if n_range is None:
        return "full"
    lo, hi = n_range
    return f"n{lo:02d}" if lo == hi else f"n{lo:02d}_{hi:02d}"


def filter_by_n(df: pd.DataFrame, n_range: tuple[int, int] | None):
    """Rows whose depth n falls in the block. Empty is a hard error."""
    if n_range is None:
        return df
    if "n" not in df.columns:
        raise SystemExit(
            "--n-range was given but the bank has no 'n' column, so the depth "
            f"block cannot be selected. Found: {list(df.columns)}")
    lo, hi = n_range
    sel = df[(df["n"].astype(int) >= lo) & (df["n"].astype(int) <= hi)]
    if sel.empty:
        raise SystemExit(
            f"--n-range {lo}-{hi} selected 0 problems from the bank "
            f"(it holds n={int(df['n'].min())}..{int(df['n'].max())}).\n"
            f"  Refusing to build an empty block — it would look like a "
            f"finished one.")
    return sel


def load_completed(results_file: str, resume_mode: str = "all") -> set:
    """
    Load set of completed (qid, variant) tuples.

    Uses last-record-wins: if the JSONL has multiple records for the same
    (qid, variant) pair (from retries or interrupted runs), the LAST record's
    state determines whether the task is considered complete.

    resume_mode:
      "all"      → skip all existing records (default --resume)
      "failures" → skip only records with correct!=None, redo correct==None
    """
    if not os.path.exists(results_file):
        return set()

    last_record = {}
    with open(results_file) as f:
        for line in f:
            try:
                r = json.loads(line)
                key = (r["question_id"], r["variant"])
                last_record[key] = r
            except (json.JSONDecodeError, KeyError):
                pass

    if resume_mode == "all":
        return set(last_record.keys())

    completed = set()
    for key, r in last_record.items():
        if r.get("correct") is not None:
            completed.add(key)
    return completed


def save_result(result: Dict, results_file: str, file_lock: Lock = None):
    """Save result to single _results.jsonl — all outcomes including None."""
    if file_lock:
        with file_lock:
            _save_result_unsafe(result, results_file)
    else:
        _save_result_unsafe(result, results_file)


def _save_result_unsafe(result: Dict, results_file: str):
    with open(results_file, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def deduplicate_results(results_file: str):
    """
    Deduplicate results by (question_id, variant).
    Strategy:
      1. Prefer records with correct==True
      2. If no True record, keep latest by timestamp
      3. Remove all other responses for the same (id, variant) pair

    Uses pure Python (no pandas) to avoid NaN corruption of None values.
    Overwrites JSONL file with deduplicated records.
    """
    if not os.path.exists(results_file):
        return

    # Load all results
    all_results = []
    with open(results_file) as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not all_results:
        return

    # Group by (question_id, variant)
    groups = {}
    for r in all_results:
        key = (r["question_id"], r["variant"])
        groups.setdefault(key, []).append(r)

    # Pick best record per group
    deduped = []
    for key, records in groups.items():
        # Sort: correct first (True > False > None), then latest timestamp
        def sort_key(r):
            correct_rank = {True: 0, False: 1, None: 2}.get(r.get("correct"), 2)
            ts = r.get("timestamp", "")
            return (correct_rank, ts)
        records.sort(key=sort_key)
        deduped.append(records[0])

    # Sort by (question_id, variant) for consistent ordering
    deduped.sort(key=lambda r: (r["question_id"], r["variant"]))

    # Write back
    with open(results_file, "w") as f:
        for r in deduped:
            f.write(json.dumps(r, default=str) + "\n")

    removed = len(all_results) - len(deduped)
    if removed > 0:
        logging.info(f"  Deduplicated {len(all_results)} → {len(deduped)} records (removed {removed} duplicates, preferred correct==True)")


# ═════════════════════════════════════════════════════════════════════════
# 7b. PARALLEL EXECUTION
# ═════════════════════════════════════════════════════════════════════════

class ParallelExecutor:
    """Execute InferenceClient calls in parallel while maintaining result ordering.

    Features:
      • Heap-buffered flush: results written to JSONL in (task_idx) order
      • Per-provider rate limiting: separate semaphores for OpenRouter, OpenAI, GenAI
      • Task timeout: individual tasks killed after `task_timeout` seconds
    """

    # Per-provider concurrency limits (conservative to avoid 429s)
    PROVIDER_LIMITS = {
        "openrouter": 5,
        "openai":     3,
        "genai":      4,
        "default":    4,
    }

    def __init__(self, client: InferenceClient, results_file: str,
                 extract_fn,
                 max_workers: int = 8, rate_limit: float = 1.0,
                 max_tokens: int = 8192, ceiling: int = 16384,
                 task_timeout: int = 600):
        self.client       = client
        self.results_file = results_file
        self.extract_fn   = extract_fn
        self.max_workers  = max_workers
        self.rate_limit   = rate_limit
        self.max_tokens   = max_tokens
        self.ceiling      = ceiling
        self.task_timeout = task_timeout
        self.call_count   = 0
        self.call_lock    = Lock()
        self.file_lock    = Lock()
        self.buffer_lock  = Lock()
        self.result_buffer = []   # Min-heap: (task_idx, result)
        self.next_flush_idx  = 0

        # Per-provider rate-limiting semaphores
        provider = self._get_provider(client.model_id)
        provider_limit = self.PROVIDER_LIMITS.get(provider, self.PROVIDER_LIMITS["default"])
        effective_workers = min(max_workers, provider_limit)
        self.rate_semaphore = Semaphore(effective_workers)
        logging.info(f"  Rate limit: {provider} (semaphore={effective_workers}, provider_limit={provider_limit})")

    @staticmethod
    def _get_provider(model_id: str) -> str:
        """Determine provider from model ID for rate limiting."""
        mid = model_id.lower()
        if "gemini" in mid:
            return "genai"
        elif "gpt" in mid or "o1" in mid or "o3" in mid:
            return "openai"
        else:
            return "openrouter"

    def _flush_buffer(self):
        """Flush contiguous results to disk in task_idx order.
        NOTE: Caller must hold buffer_lock — this method does not acquire it
        to avoid deadlock when called from within 'with self.buffer_lock'.

        On resume, task_idx values start from 0 again (not contiguous with
        previous run). We flush any record whose task_idx is strictly less
        than the smallest unflushed index in the buffer, ensuring results
        are written in order without gaps blocking the flush.
        """
        while self.result_buffer:
            idx = self.result_buffer[0][0]
            # Flush if this is the next expected index, OR if we've already
            # passed it (resume gap: old run had indices 0..49, new run has
            # 0..19 — we flush immediately since no older indices are pending)
            if idx == self.next_flush_idx or idx < self.next_flush_idx:
                _, result = heapq.heappop(self.result_buffer)
                save_result(result, self.results_file, self.file_lock)
                # Only advance next_flush_idx for indices >= current position
                if idx >= self.next_flush_idx:
                    self.next_flush_idx = idx + 1
            else:
                break  # gap — wait for missing index

    def _execute_task(self, task_idx: int, qid: str,
                     prompt: str, gt: str, problem_latex: str,
                     answer_latex: str, subcat: str, variant: str = "rep1"):
        try:
            # Build prompt
            full_prompt = prompt
            # subcat is always "ibp" here; there is no infer fallback branch.

            logging.info(f"  Task {task_idx}: {qid} [{variant}]")

            # Call LLM
            cr = self.client.call(
                user_prompt=full_prompt,
                qid=qid,
                max_tokens=self.max_tokens,
                ceiling=self.ceiling,
            )

            with self.call_lock:
                self.call_count += 1

            # Coarse monitoring only — NOT the grade. The authoritative result
            # is symbolic and comes from ibp_grade_depth_ladder.py; string
            # matching is not used for grading.
            extracted = self.extract_fn(cr.text) if cr.text else None

            # final_prompt: the literal payload structure actually sent to the
            # model API (provider-dependent shape — see inference_llm.py
            # _call_openai_compat / _call_genai). Not a guess: mirrors exactly
            # what those two methods construct from system_prompt/user_prompt.
            if getattr(self.client, "_backend", None) == "genai":
                final_prompt = [self.client.system_prompt, full_prompt]
            else:
                final_prompt = [
                    {"role": "system", "content": self.client.system_prompt},
                    {"role": "user",   "content": full_prompt},
                ]

            result = {
                "question_id":          qid,
                # Carries the repeat index (rep1/rep2/...). The dedup key is
                # (question_id, variant), so this is what stops repeats of the
                # same problem being collapsed into a single record.
                "variant":              variant,
                "model":                self.client.model_id,
                # provider: the upstream host that actually served this call,
                # read off the response — not the pin we requested. A pin that
                # failed to hold is therefore visible here, and
                # ibp_build_deliverables.py refuses to build if a model's rows
                # ever show more than one.
                # model_version: the resolved exact version string written to
                # results_raw.csv (e.g. "anthropic/claude-4.5-sonnet-20250929"),
                # which "model" above does not carry — that is only the slug
                # we asked for.
                "provider":             cr.provider,
                "model_version":        cr.model_version,
                # Marks a record that was never sent to a provider. A dry-run
                # response is placeholder text, not an empty string, so without
                # this flag ibp_build_deliverables.py grades it as a real
                # answer and moves the accuracy for its depth rung. The flag
                # travels IN the record because --output can put a dry run in
                # any directory, so the directory name cannot be the only
                # defence.
                "dry_run":              bool(self.client.dry_run),
                "system_prompt":        self.client.system_prompt,
                "user_prompt":          full_prompt,
                "prompt_message":       final_prompt,
                "prompt_length":        len(prompt),
                "prompt_tokens_approx": len(prompt) // 4,
                "response":             cr.text,
                "extracted_answer":     extracted,
                "boxed_count":          count_boxed(cr.text) if cr.text else 0,
                "correct":              (extracted == gt) if extracted is not None else None,
                "ground_truth":         gt,
                "problem_latex":        problem_latex,
                "answer_latex":         answer_latex,
                "latency_ms":           cr.latency_ms,
                "tokens_used":          cr.tokens_used,
                "finish_reason":        cr.finish_reason,
                "error":                cr.error,
                "timestamp":            datetime.now().isoformat(),
            }

            # Buffer result for ordered flush
            with self.buffer_lock:
                heapq.heappush(self.result_buffer, (task_idx, result))
                self._flush_buffer()

            print(f"✓ Completed {qid}", flush=True)

        except Exception as e:
            print(f"✗ Task failed {qid}: {e}", flush=True)
            raise

    def run(self, tasks_list: List[tuple], pbar_callback=None):
        """Execute all tasks in parallel, preserving result order via heap buffer.

        Task timeout: individual tasks are given `task_timeout` seconds to complete.
        If exceeded, a TIMEOUT is logged and the task is skipped.
        NOTE: Python threads cannot be forcibly killed — the timed-out thread
        continues running in the background and may still write its result to
        JSONL when it eventually completes. The deduplicate_results() call at
        the end of run_pipeline cleans up any such late arrivals.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for task in tasks_list:
                # task = (task_idx, qid, prompt, gt, problem_latex, answer_latex, subcat, variant)
                task_idx, qid, prompt, gt, problem_latex, answer_latex, row_subcat, variant = task
                future = executor.submit(self._execute_task, task_idx, qid, prompt, gt,
                                         problem_latex, answer_latex, row_subcat, variant)
                futures[future] = (task_idx, qid)

            for future in as_completed(futures):
                task_idx, qid = futures[future]
                try:
                    future.result(timeout=self.task_timeout)
                except FutureTimeoutError:
                    logging.error(f"⏱ TIMEOUT {qid} after {self.task_timeout}s")
                except Exception as e:
                    logging.error(f"Failed {qid}: {e}")
                if pbar_callback:
                    pbar_callback(1)

        # Final flush for any remaining buffered results
        time.sleep(0.5)
        with self.buffer_lock:
            while self.result_buffer:
                idx, result = heapq.heappop(self.result_buffer)
                save_result(result, self.results_file, self.file_lock)
                self.next_flush_idx += 1


# ═════════════════════════════════════════════════════════════════════════
# 8. QUICK RESULTS SUMMARY (printed to stdout immediately after API calls)
# ═════════════════════════════════════════════════════════════════════════

def print_results_summary(results_file: str):
    """Print a quick summary of results to stdout after API calls complete."""
    if not os.path.exists(results_file):
        return

    all_results = []
    try:
        with open(results_file) as f:
            for line in f:
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception:
        return

    if not all_results:
        return

    # WHAT THIS BLOCK IS, AND WHAT IT IS NOT
    # ───────────────────────────────────────
    # Two things here were inherited from the LinAlg pipeline this file was
    # forked from, and both were wrong for the depth ladder:
    #
    #  1. It printed "Correct: n/N" from the record's `correct` field, which is
    #     a naive equality test between the extracted string and the stored
    #     answer. Grading for this experiment is SYMBOLIC and lives in
    #     ibp_grade_depth_ladder.py, deliberately outside both lanes so a
    #     grader change costs a re-parse and never a re-submit. The two
    #     disagree often — an observed real call printed "Correct: 0/1" here
    #     and was graded PASS by the real grader. A wrong accuracy on screen
    #     mid-run is worse than none.
    #
    #  2. It split by qid.split('_')[-2] as a "subcategory". IBP ids look like
    #     IBP_n05_sin_b01_00, so that yields "b01" — the b-parameter. This tree
    #     serves exactly one subcategory (ibp); the axis that means anything
    #     here is DEPTH n, which is what the experiment varies.
    #
    # So this now reports extraction reach by depth and says plainly that it is
    # not the grade.
    depths: dict[str, dict] = {}
    n_extracted = n_trunc = n_err = 0

    for record in all_results:
        qid = record.get('question_id', '')
        # IBP_n05_sin_b01_00 -> "n05". Falls back to the whole id rather than
        # inventing a bucket, so a malformed id is visible instead of silently
        # joining someone else's row.
        parts = qid.split('_')
        depth = parts[1] if len(parts) >= 2 and parts[1].startswith('n') else qid

        d = depths.setdefault(depth, {"extracted": 0, "total": 0})
        d["total"] += 1
        if record.get('extracted_answer') is not None:
            d["extracted"] += 1
            n_extracted += 1
        if record.get('finish_reason') == 'length':
            n_trunc += 1
        if record.get('error'):
            n_err += 1

    total = len(all_results)
    pct = 100 * n_extracted / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  RUN SUMMARY — NOT THE GRADE")
    print(f"{'='*60}")
    print(f"  Total records:      {total}")
    print(f"  Answer extracted:   {n_extracted}/{total} ({pct:.1f}%)")
    print(f"  Truncated (length): {n_trunc}")
    print(f"  Errors:             {n_err}")
    print(f"\n  Extraction reach by depth:")
    for depth in sorted(depths.keys()):
        s = depths[depth]
        p = 100 * s["extracted"] / s["total"] if s["total"] > 0 else 0
        print(f"    {depth:10} {s['extracted']}/{s['total']:<3} ({p:5.1f}%)")
    print()
    print(f"  Accuracy is NOT reported here. Symbolic grading is a separate,")
    print(f"  auditable step — run ibp_build_deliverables.py for the verdict.")
    print(f"{'='*60}\n")


# ═════════════════════════════════════════════════════════════════════════
# 9. ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

def run_analysis(results_file: str, model_name: str, output_dir: str):
    """Per-question summary table, one column group per repeat.

    Run monitoring only. There is no error-type breakdown here any more: that
    came from a Stage-2/3 judge, which this experiment does not have. Only
    pass and parse_fail are tracked.
    """

    all_results = []
    with open(results_file) as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    # Build per-question summary
    questions = {}
    for r in all_results:
        qid = r["question_id"]
        if qid not in questions:
            questions[qid] = {"id": qid, "ground_truth": r.get("ground_truth")}
        v = r["variant"]
        # Preserve None vs False distinction — None means extraction failure, not wrong
        raw_correct = r.get("correct")
        questions[qid][f"{v}_correct"] = bool(raw_correct) if raw_correct is not None else None
        questions[qid][f"{v}_answer"] = r.get("extracted_answer")
        questions[qid][f"{v}_boxed_count"] = r.get("boxed_count", 0)
        questions[qid][f"{v}_tokens"] = r.get("tokens_used", 0)

    summary_df = pd.DataFrame(questions.values())

    # ── Accuracy by repeat ───────────────────────────────────────
    # Variants are discovered from the data (rep1, rep2, ...) rather than a
    # fixed list: each repeat is a separate run tagged by --variant.
    # NOTE this is a coarse extraction-equality tally for run monitoring only.
    # The authoritative grade is symbolic and comes from
    # ibp_grade_depth_ladder.py.
    variants = sorted({r["variant"] for r in all_results})

    logging.info("\n" + "═" * 60)
    logging.info("  EXTRACTION-MATCH TALLY BY REPEAT (not the graded result)")
    logging.info("═" * 60)
    logging.info(f"  {'Repeat':<25} {'Match':>7} {'Total':>7} {'Rate':>10}")
    logging.info("  " + "─" * 55)

    for v in variants:
        col = f"{v}_correct"
        if col not in summary_df.columns:
            continue
        subset  = summary_df[summary_df[col].notna()]
        correct = int(subset[col].fillna(False).astype(bool).sum())
        total   = len(subset)
        acc     = correct / total * 100 if total else 0
        logging.info(f"  {v:<25} {correct:>7} {total:>7} {acc:>9.1f}%")

    logging.info("═" * 60)

    # ── Accuracy note ────────────────────────────────────────────
    logging.info("\n  No scaffolding required.\n")

    # ── Save summary CSV ─────────────────────────────────────────
    summary_csv = os.path.join(output_dir,
                               run_stem(model_name, output_dir) + "_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    logging.info(f"\n  Saved: {summary_csv}")

    return summary_df


# ═════════════════════════════════════════════════════════════════════════
# 10. MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════

def run_pipeline(args):

    # ── Bind the output tree FIRST ───────────────────────────────
    # Before anything reads a path. ibp_paths has no default root, so every
    # path accessor raises until this line has run.
    P.configure(args.projects_root, args.subcat)

    # ── Set seed for reproducibility ─────────────────────────────
    if args.seed is not None:
        import random
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)

    # ── Load dataset first to detect subcategory ─────────────────
    file_type = _detect_file_type(args.input)
    if file_type == 'csv':
        df_raw = pd.read_csv(args.input)
    else:
        df_raw = pd.read_excel(args.input)
    df_raw = _normalize_columns(df_raw)

    # ── Subcat comes from --subcat, validated twice ──────────────
    # No detection: the bank has no subcategory column and never needs one.
    # P.configure() has already rejected anything outside ibp_paths.SUBCATS,
    # and subcat_config_ibp.SUBCAT_CONFIGS raises for anything but "ibp" as a
    # second, independent check.
    subcat        = P.subcat()
    system_prompt = IBP_SYSTEM_PROMPT
    extract_fn    = get_subcat_config(subcat).extract_answer_fn

    # ── Model config ─────────────────────────────────────────────
    model_cfg = MODELS[args.model].copy()
    if args.api_key_env:
        model_cfg["api_key_env"] = args.api_key_env
    if args.api_base:
        model_cfg["api_base"] = args.api_base

    min_tokens  = model_cfg.get("min_tokens", 8192)
    max_tokens  = args.max_tokens if args.max_tokens else min_tokens
    token_mult  = IBP_TOKEN_MULT
    ceiling     = max_tokens * token_mult

    # The adaptive step-up cannot exceed what the PINNED endpoint will emit.
    # A provider that caps a model below the requested budget would otherwise
    # make the step-up retry the same wall and pay for the same cut-off answer
    # more than once. Clamping both numbers to the cap makes the truncation
    # visible on the first call instead of the last.
    provider_cap = model_cfg.get("max_output_tokens")
    if provider_cap:
        if max_tokens > provider_cap:
            logging.warning(
                f"  [CAP] {args.model} requested {max_tokens} tokens but the "
                f"pinned endpoint emits at most {provider_cap}. Clamping.")
            max_tokens = provider_cap
        ceiling = min(ceiling, provider_cap)
        if ceiling <= max_tokens:
            logging.warning(
                f"  [CAP] no step-up headroom: start and ceiling are both "
                f"{max_tokens}. A truncated response CANNOT be recovered on "
                f"this endpoint — finish_reason 'length' is final.")

    # ── Output paths ─────────────────────────────────────────────
    # run_dir() gives <projects-root>/{subcat}/{model}/run{n}, under the root
    # named on the command line, with the next free run number. A second pass
    # therefore cannot land on the first one's files, and the deliverables glob
    # finds it with no argument to remember. --output still overrides for a
    # one-off.
    #
    # The number, and only the number, is chosen by the code. Hand-numbering
    # runs is how a partial run's files get silently overwritten; the operator
    # names the tree and the code names the run, and both are printed below
    # before any call.
    # The depth block also names the run directory, so two blocks of the same
    # model cannot land on each other and the ladder's progress is readable
    # so a re-collected level can never land on the main run.
    n_range = parse_n_range(args.n_range)
    block   = n_range_label(n_range)

    if args.output:
        output_dir = args.output
        if args.dry_run:
            output_dir += "_dryrun"
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = run_dir(args.model,
                             label="dryrun" if args.dry_run else block)
    stem         = run_stem(args.model, output_dir)
    results_file = os.path.join(output_dir, stem + RESPONSE)

    # Say where everything is BEFORE the first call, not in RUN.txt afterwards
    # — a run that writes into the wrong tree is invisible if nothing on
    # screen names the tree while it's happening. Printed before the log
    # handler is attached, so "new" still means an empty directory.
    P.print_paths(bank_path=args.input, run_directory=output_dir,
                  n_rows=len(df_raw))
    if n_range is not None:
        lo, hi = n_range
        span = f"n={lo}" if lo == hi else f"n={lo}-{hi}"
        print(f"  depth block     {span}   "
              f"({len(filter_by_n(df_raw, n_range))} problems, "
              f"1 call each)\n")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            # The log shares the run's stem too, so it stays attached to its
            # own run rather than being overwritten by the next one.
            logging.FileHandler(os.path.join(output_dir, stem + ".log")),
            logging.StreamHandler(),
        ],
    )

    logging.info("=" * 60)
    logging.info(f"  LinAlg-Bench · Unified Inference")
    logging.info(f"  Subcat: {subcat}")
    logging.info(f"  Model:  {args.model}")
    logging.info(f"  Mode:   {'DRY RUN' if args.dry_run else 'LIVE'}")
    logging.info(f"  Tokens: {max_tokens} × {token_mult} = ceiling {ceiling}")
    logging.info(f"  Output: {output_dir}")
    logging.info("=" * 60)

    # Written BEFORE the first call, not after the run finishes. A run that is
    # interrupted — or abandoned — is exactly the one whose directory nobody
    # will be able to identify later.
    write_manifest(
        output_dir,
        experiment="IBP Depth Ladder",
        lane="live",
        projects_root=P.projects(),
        model=args.model,
        # The registry key is model_id, not model. Reading the wrong one wrote
        # a blank line rather than failing, which is the manifest's whole
        # weakness: nobody checks a field they never see populated.
        model_slug=model_cfg.get("model_id", "(not in registry)"),
        subcat=subcat,
        bank=os.path.abspath(args.input),
        # Recorded so a directory says which rung of the ladder it holds
        # without opening the data.
        n_block=block if n_range else "all levels (n=0-16)",
        variant=args.variant or "rep1",
        limit=args.limit if getattr(args, "limit", None) else "all",
        max_tokens=f"{max_tokens} (ceiling {ceiling})",
        dry_run=args.dry_run,
        resume=args.resume or "no",
        started=datetime.now().isoformat(timespec="seconds"),
    )

    # ── API key validation (fail fast with clear message) ────────
    if not args.dry_run:
        api_key_env = model_cfg.get("api_key_env", "")
        api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        if not api_key:
            raise ValueError(
                f"API key is empty for model '{args.model}'!\n"
                f"  Environment variable: {api_key_env}\n"
                f"  Set it in .env with {api_key_env}=your-key"
            )

    # ── Load data ────────────────────────────────────────────────
    df = load_dataset(args.input)
    # The depth block, applied before --limit, so "20 problems in this level"
    # stays true however the run is sliced.
    df = filter_by_n(df, n_range)
    if n_range is not None:
        logging.info(f"  Block:        n={n_range[0]}"
                     f"{'' if n_range[0] == n_range[1] else f'-{n_range[1]}'} "
                     f"({len(df)} problems of this level)")

    logging.info("  Prompt:       problem_latex, verbatim (no assembly)")

    # ── Build InferenceClient ────────────────────────────────────
    client = InferenceClient(
        model_cfg=model_cfg,
        system_prompt=system_prompt,
        dry_run=args.dry_run,
    )

    # Pre-resume dedup: clean stale duplicates BEFORE reading completed set
    # so load_completed sees the authoritative last-record state
    if args.resume and results_file and os.path.exists(results_file):
        deduplicate_results(results_file)

    completed = load_completed(results_file, args.resume) if args.resume else set()

    logging.info(f"  Questions:    {len(df)}")
    resume_info = ""
    if args.resume:
        resume_info = f" ({args.resume} mode)"
    logging.info(f"  Resumed:      {len(completed)} completed calls{resume_info}")
    logging.info(f"  Total calls:  {len(df)} planned "
                 f"({len(df)} problems × 1 call each)\n")

    # ── Build task list ──────────────────────────────────────────
    logging.info("  Building task list...")
    tasks_list = []
    task_idx   = 0

    # --limit counts PROBLEMS, and this run makes one call per problem, so
    # --limit 5 is exactly 5 calls.
    df_sel = df.head(args.limit) if args.limit and args.limit > 0 else df

    for _, row in df_sel.iterrows():
        qid          = str(row["id"]).strip()
        gt           = row["ground_truth"]
        stored_latex = row["problem_latex"]
        prompt       = build_standard_prompt(stored_latex)

        # One variant per run. --variant names it; default rep1.
        variants = [args.variant] if args.variant else ["rep1"]
        for variant in variants:
            if (qid, variant) in completed:
                continue
            tasks_list.append(
                (task_idx, qid, prompt, gt, stored_latex, row["answer_latex"], subcat, variant)
            )
            task_idx += 1

    # ── Execute in parallel ──────────────────────────────────────
    max_workers = args.max_workers  # concurrent API calls (--max-workers, default 8)
    logging.info(f"  Executing {len(tasks_list)} calls ({max_workers} workers)...\n")

    parallel_exec = ParallelExecutor(
        client=client,
        results_file=results_file,
        extract_fn=extract_fn,
        max_workers=max_workers,
        rate_limit=1.0,  # hardcoded default
        max_tokens=max_tokens,
        ceiling=ceiling,
    )

    if len(tasks_list) > 0:
        pbar = tqdm(total=len(tasks_list), desc="API Calls", unit="call")
        parallel_exec.run(tasks_list, pbar_callback=lambda n: pbar.update(n))
        pbar.close()
    else:
        logging.info("  No new tasks (all completed, --resume in effect)")

    # ── Print quick results summary to stdout ─────────────────────
    print_results_summary(results_file)

    # ── Deduplicate results (if --resume created duplicates) ─────
    deduplicate_results(results_file)

    # ── Export workbook ──────────────────────────────────────────
    _export_results(results_file, output_dir, args.model, subcat)

    # Grading is deliberately NOT done here. Grading is symbolic only, done in
    # a separate, auditable stage with a locked rule: run
    # ibp_grade_depth_ladder.py over the results JSONL. Equivalent answers are
    # judged by differentiation against the integrand, never by a paid
    # adjudication call.

    # ── Analysis ─────────────────────────────────────────────────
    logging.info("\n" + "─" * 60)
    logging.info("  Running analysis...")
    run_analysis(results_file, args.model, output_dir)

    # Try to run summarize.py if it exists
    try:
        import summarize as sm
        sm.update_accuracy_file(args.input, args.model, results_file)
    except ImportError:
        pass  # summarize.py not required

    logging.info(f"\n  Raw results:     {results_file}")
    if len(tasks_list) > 0:
        logging.info(f"  Total API calls: {parallel_exec.call_count}")
    else:
        logging.info(f"  (Analyzed existing results, no new API calls)")



def _export_results(results_file: str, output_dir: str, model_name: str, subcat: str):
    """Export every record to a formatted Excel workbook, one row per call.

    Grading here is symbolic and runs from the results JSONL
    (ibp_grade_depth_ladder.py); this workbook is a raw record export.
    """
    MODEL_MAP = {cfg['model_id']: name for name, cfg in MODELS.items()}

    records = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        return

    # Deduplicate: keep last per (question_id, variant). variant carries the
    # repeat index, so each repeat survives as its own row.
    deduped, order = {}, {}
    for i, r in enumerate(records):
        key = (r['question_id'], r.get("variant", "rep1"))
        deduped[key] = r
        order[key] = i

    _export_results_excel(deduped, order, output_dir, model_name, subcat, MODEL_MAP)


def _export_results_excel(deduped: dict, order: dict, output_dir: str,
                           model_name: str, subcat: str, model_map: dict):
    """Export ALL records to professional-grade Excel with formatted columns.

    Produces a comprehensive results workbook with:
      • All records (correct + incorrect, all variants)
      • Computed columns (response_correct, completion_status, Correct/FAIL)
      • Summary sheet with per-variant accuracy breakdown
    """
    # Excel rejects the C0 control characters, and openpyxl raises
    # IllegalCharacterError on the first one it meets — a single bad character
    # anywhere in the data would abort the whole final export step.
    #
    # Only the workbook copy is sanitised; the JSONL stays byte-exact and
    # remains the record of truth. Every substitute is exactly one character
    # wide, so a string keeps the same length in both copies and any character
    # offset computed against the JSONL still lands on the same position in the
    # Excel cell.
    _XL_ILLEGAL = {c: '�' for c in range(0x00, 0x20)
                   if c not in (0x09, 0x0A, 0x0D)}   # keep tab, LF, CR
    _XL_ILLEGAL[0x7F] = '�'
    _XL_ILLEGAL[0x08] = '␈'                     # visible marker for the \boxed{} corruption

    def _xl_safe(v):
        """Excel-safe view of a value; non-strings and clean strings pass through."""
        return v.translate(_XL_ILLEGAL) if isinstance(v, str) else v

    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    except ImportError:
        logging.warning("  openpyxl not available — skipping Excel export")
        return

    # Sort records by original order
    all_records = [deduped[k] for k in sorted(deduped, key=lambda k: order[k])]

    if not all_records:
        return

    MODEL_MAP = model_map

    # Build rows with all columns
    COLUMNS = [
        'Problem_ID', 'Model', 'Subcat', 'Variant', 'Correct',
        'response_correct', 'completion_status', 'extracted_answer',
        'ground_truth', 'answer_latex', 'problem_latex', 'response',
        'system_prompt', 'user_prompt', 'prompt_message',
        'format', 'boxed_count', 'tokens_used', 'latency_ms',
        'finish_reason', 'error', 'timestamp',
    ]

    def classify_status(r):
        extracted = r.get('extracted_answer')
        response = r.get('response', '') or ''
        if extracted is not None and extracted != '':
            return 'COMPLETE'
        if not response.strip():
            return 'API_ERROR'
        if len(response) >= 32760:
            return 'HARD_TRUNCATION'
        last = response[-400:]
        clean = bool(re.search(r'[.!?)\]}]\s*$', last)) or bool(
            re.search(r'(?:thus|therefore|in conclusion|hope this)', last, re.IGNORECASE))
        return 'ABANDONMENT' if clean else 'SOFT_TRUNCATION'

    rows = []
    for r in all_records:
        correct_val = r.get('correct')
        if correct_val is None:
            correct_label = 'EXTRACT_FAIL'
        elif correct_val:
            correct_label = 'CORRECT'
        else:
            correct_label = 'FAIL'

        rows.append({
            'Problem_ID':       r.get('question_id', ''),
            'Model':            MODEL_MAP.get(r.get('model', ''), r.get('model', '')),
            'Subcat':           subcat,
            'Variant':          r.get('variant', ''),
            'Correct':          correct_label,
            'response_correct': correct_val if correct_val is not None else False,
            'completion_status': classify_status(r),
            'extracted_answer': str(r.get('extracted_answer', '')) if r.get('extracted_answer') is not None else '',
            'ground_truth':     str(r.get('ground_truth', '')),
            'answer_latex':     r.get('answer_latex', ''),
            'problem_latex':    r.get('problem_latex', ''),
            'response':         r.get('response', ''),
            'system_prompt':    r.get('system_prompt', ''),
            'user_prompt':      r.get('user_prompt', ''),
            'prompt_message':   json.dumps(r['prompt_message']) if r.get('prompt_message') is not None else '',
            'format':           r.get('format', 'latex'),
            'boxed_count':      r.get('boxed_count', 0),
            'tokens_used':      r.get('tokens_used', 0),
            'latency_ms':       r.get('latency_ms', 0),
            'finish_reason':    r.get('finish_reason', ''),
            'error':            r.get('error', ''),
            'timestamp':        r.get('timestamp', ''),
        })

    # ── Create workbook ──────────────────────────────────────────────
    wb = openpyxl.Workbook()

    # Styles
    TITLE_FONT = Font(name='Calibri', bold=True, size=14, color='1F3864')
    HEADER_FONT = Font(name='Calibri', bold=True, size=11, color='FFFFFF')
    HEADER_FILL = PatternFill('solid', fgColor='1F3864')
    DATA_FONT = Font(name='Calibri', size=10)
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin'),
    )
    CORRECT_FILL = PatternFill('solid', fgColor='C6EFCE')
    FAIL_FILL = PatternFill('solid', fgColor='FFC7CE')
    OTHER_FILL = PatternFill('solid', fgColor='FFEB9C')
    ALT_ROW = PatternFill('solid', fgColor='F2F2F2')

    # ── Sheet 1: All Results ─────────────────────────────────────────
    ws = wb.active
    ws.title = 'All_Results'
    ws.sheet_properties.tabColor = '1F3864'

    # Title
    ws.merge_cells('A1:T1')
    ws['A1'].value = f'Results: {model_name} — {subcat} '
    ws['A1'].font = TITLE_FONT
    ws['A1'].alignment = Alignment(horizontal='left', vertical='center')
    ws.row_dimensions[1].height = 28

    # Headers
    for c, col in enumerate(COLUMNS, 1):
        cell = ws.cell(row=2, column=c, value=col)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = thin_border

    # Data rows
    for i, row_data in enumerate(rows):
        r_num = i + 3
        for c, col in enumerate(COLUMNS, 1):
            val = row_data.get(col, '')
            cell = ws.cell(row=r_num, column=c, value=_xl_safe(val))
            cell.font = DATA_FONT
            cell.border = thin_border
            cell.alignment = Alignment(vertical='top', wrap_text=True)
            if i % 2 == 1:
                cell.fill = ALT_ROW

        # Color the "Correct" column
        status = row_data.get('Correct', '')
        cell_c = ws.cell(row=r_num, column=5)
        if status == 'CORRECT':
            cell_c.fill = CORRECT_FILL
        elif status == 'FAIL':
            cell_c.fill = FAIL_FILL
        else:
            cell_c.fill = OTHER_FILL

    # Column widths
    col_widths = {
        'A': 22, 'B': 18, 'C': 12, 'D': 22, 'E': 14,
        'F': 16, 'G': 18, 'H': 16, 'I': 14, 'J': 30,
        'K': 40, 'L': 60, 'M': 10, 'N': 12, 'O': 12,
        'P': 12, 'Q': 14, 'R': 14, 'S': 22,
    }
    for letter, width in col_widths.items():
        ws.column_dimensions[letter].width = width

    # Row heights (data rows)
    for r_num in range(3, 3 + len(rows)):
        ws.row_dimensions[r_num].height = 60

    ws.freeze_panes = 'C3'
    ws.auto_filter.ref = f'A2:T{2 + len(rows)}'

    # ── Sheet 2: Summary ──────────────────────────────────────────────
    ws2 = wb.create_sheet('Summary')
    ws2.sheet_properties.tabColor = '59A14F'
    ws2.column_dimensions['A'].width = 30
    ws2.column_dimensions['B'].width = 14
    ws2.column_dimensions['C'].width = 14

    ws2.merge_cells('A1:C1')
    ws2['A1'].value = f'Summary — {model_name} ({subcat})'
    ws2['A1'].font = TITLE_FONT
    ws2.row_dimensions[1].height = 28

    # Stats
    total = len(rows)
    correct_count = sum(1 for r in rows if r['Correct'] == 'CORRECT')
    fail_count = sum(1 for r in rows if r['Correct'] == 'FAIL')
    other_count = sum(1 for r in rows if r['Correct'] not in ('CORRECT', 'FAIL'))

    # Per-variant breakdown
    variant_stats = {}
    for r in rows:
        v = r['Variant']
        if v not in variant_stats:
            variant_stats[v] = {'total': 0, 'correct': 0}
        variant_stats[v]['total'] += 1
        if r['Correct'] == 'CORRECT':
            variant_stats[v]['correct'] += 1

    r = 3
    ws2.merge_cells(f'A{r}:C{r}')
    ws2.cell(row=r, column=1, value='Overall').font = Font(name='Calibri', bold=True, size=12, color='1F3864')
    r = 4
    for h, w in [('Metric', 30), ('Count', 14), ('Percentage', 14)]:
        ws2.cell(row=r, column=['Metric', 'Count', 'Percentage'].index(h) + 1, value=h)
        ws2.cell(row=r, column=['Metric', 'Count', 'Percentage'].index(h) + 1).font = HEADER_FONT
        ws2.cell(row=r, column=['Metric', 'Count', 'Percentage'].index(h) + 1).fill = HEADER_FILL
        ws2.cell(row=r, column=['Metric', 'Count', 'Percentage'].index(h) + 1).border = thin_border
        ws2.cell(row=r, column=['Metric', 'Count', 'Percentage'].index(h) + 1).alignment = Alignment(horizontal='center')

    metrics = [
        ('Total Records', total, 1.0),
        ('Correct', correct_count, correct_count / total if total else 0),
        ('Failed', fail_count, fail_count / total if total else 0),
        ('Other (extract fail)', other_count, other_count / total if total else 0),
    ]
    for i, (metric, cnt, pct) in enumerate(metrics):
        r += 1
        ws2.cell(row=r, column=1, value=metric).font = DATA_FONT
        ws2.cell(row=r, column=2, value=cnt).font = DATA_FONT
        ws2.cell(row=r, column=2).number_format = '#,##0'
        ws2.cell(row=r, column=3, value=pct).font = DATA_FONT
        ws2.cell(row=r, column=3).number_format = '0.0%'
        for c in range(1, 4):
            ws2.cell(row=r, column=c).border = thin_border
            ws2.cell(row=r, column=c).alignment = Alignment(horizontal='center')
            if i % 2 == 0:
                ws2.cell(row=r, column=c).fill = ALT_ROW

    # Variant breakdown
    r += 2
    ws2.merge_cells(f'A{r}:C{r}')
    ws2.cell(row=r, column=1, value='By Variant').font = Font(name='Calibri', bold=True, size=12, color='1F3864')
    r += 1
    for h in ['Variant', 'Correct/Total', 'Accuracy']:
        ws2.cell(row=r, column=['Variant', 'Correct/Total', 'Accuracy'].index(h) + 1, value=h)
        ws2.cell(row=r, column=['Variant', 'Correct/Total', 'Accuracy'].index(h) + 1).font = HEADER_FONT
        ws2.cell(row=r, column=['Variant', 'Correct/Total', 'Accuracy'].index(h) + 1).fill = HEADER_FILL
        ws2.cell(row=r, column=['Variant', 'Correct/Total', 'Accuracy'].index(h) + 1).border = thin_border
        ws2.cell(row=r, column=['Variant', 'Correct/Total', 'Accuracy'].index(h) + 1).alignment = Alignment(horizontal='center')

    for i, (v, stats) in enumerate(variant_stats.items()):
        r += 1
        acc = stats['correct'] / stats['total'] if stats['total'] else 0
        ws2.cell(row=r, column=1, value=v).font = DATA_FONT
        ws2.cell(row=r, column=2, value=f"{stats['correct']}/{stats['total']}").font = DATA_FONT
        ws2.cell(row=r, column=3, value=acc).font = DATA_FONT
        ws2.cell(row=r, column=3).number_format = '0.0%'
        for c in range(1, 4):
            ws2.cell(row=r, column=c).border = thin_border
            ws2.cell(row=r, column=c).alignment = Alignment(horizontal='center')
            if i % 2 == 0:
                ws2.cell(row=r, column=c).fill = ALT_ROW

    # Save
    # Save Excel with same base name as JSONL
    xlsx_path = os.path.join(output_dir,
                             run_stem(model_name, output_dir) + '_response.xlsx')
    wb.save(xlsx_path)
    logging.info(f"  Generated Excel output: {xlsx_path} ({total} records)")



# ═════════════════════════════════════════════════════════════════════════
# 11. CLI
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="IBP Depth Ladder — 0-shot inference over the 340-problem bank",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Backends (auto-detected from model config):
  OpenRouter — Llama-3.3-70B, Qwen3-235B
  Anthropic  — Claude-4.5-Sonnet, via api.anthropic.com
  GenAI      — Gemini-3.1-Pro, via the google-genai SDK

Examples:
  python3 ibp_run_live.py --projects-root ../projects \\
      --bank ../projects/source/ibp_depth_ladder_problems.csv \\
      --model Llama-3.3-70B --limit 5 --dry-run

  python3 ibp_run_live.py --projects-root ../projects \\
      --bank ../projects/source/ibp_depth_ladder_problems.csv \\
      --model Claude-4.5-Sonnet --variant rep2
        """,
    )
    parser.add_argument(
        "--model", default="Llama-3.3-70B",
        choices=list(MODELS.keys()),
        help="Model name from MODELS registry (default: Llama-3.3-70B)",
    )
    # Every path is on the command line and none has a guessed default — see
    # ibp_paths.py for why.
    parser.add_argument(
        "--projects-root", required=True,
        help="Output tree, e.g. ../projects. Must already exist. Required: "
             "there is no default, so output is never written somewhere you "
             "did not name.",
    )
    parser.add_argument(
        "--subcat", default=P.DEFAULT_SUBCAT, choices=sorted(P.SUBCATS),
        help=f"Subcategory segment of the path, <projects-root>/{{subcat}}/... "
             f"(default: {P.DEFAULT_SUBCAT}). Matches the main pipeline's "
             f"{{subcat}}/{{model}} layout.",
    )
    parser.add_argument(
        "--bank", "--input", dest="input", required=True,
        help="Input bank (.csv or .xlsx), e.g. "
             "../projects/source/ibp_depth_ladder_problems.csv. Required. "
             "--input is kept as an alias for the main pipeline's spelling.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Override the run directory. Default: "
             "<projects-root>/{subcat}/{model}/run{n}_full, taking the next "
             "free run number so a second pass cannot overwrite the first "
             "one's files. The number is allocated by the code because "
             "hand-numbering runs is how a partial run's files get silently "
             "overwritten; the resolved directory is printed before the "
             "first API call.",
    )
    parser.add_argument(
        "--n-range", default=None,
        help='RECOVERY ONLY - omit it for the planned run, which builds the whole 340-problem ladder in one submission. Pass a level (e.g. 7) or a LO-HI range (e.g. 4-6) to re-collect a level that came back short, without hand-cutting a CSV. The block names the run directory (run{k}_n07) and is recorded in the meta and RUN.txt, so a top-up is never mistaken for the main run.',
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview prompts without making API calls")
    parser.add_argument(
        "--resume", nargs='?', const='all', default=None,
        choices=['all', 'failures'],
        help="Resume: 'all' skip all records, 'failures' redo extraction failures")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: None)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Starting token budget (default: model min_tokens from config)")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Concurrent API calls (default: 8). Throughput is "
                             "max_workers/mean_latency; long high-n IBP responses "
                             "average ~130s, so 8 workers gives only ~0.06 calls/s. "
                             "Dispatch is separately paced at 1.0 call/s, which does "
                             "not bind until roughly 130 workers.")
    parser.add_argument("--api-base", default=None,
                        help="Override API base URL")
    parser.add_argument("--api-key-env", default=None,
                        help="Override environment variable name for API key")
    # One call per problem per run; the repeat is the RUN (run1_full /
    # run2_full), tagged with --variant below.
    parser.add_argument("-n", "--limit", type=int, default=None,
                        help="Limit number of PROBLEMS to process (default: all). "
                             "One call each, so --limit 5 = 5 calls.")
    parser.add_argument("--variant", default=None,
                        help="Repeat tag for this run (default rep1). The repeat IS the "
                             "run: tag separate run folders by hand, run1 -> --variant rep1, "
                             "run2 -> --variant rep2.")

    args = parser.parse_args()

    try:
        run_pipeline(args)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()


