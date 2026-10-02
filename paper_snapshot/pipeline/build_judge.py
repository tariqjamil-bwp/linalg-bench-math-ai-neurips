#!/usr/bin/env python3
"""
================================================================================
LinAlg-Bench · Stage 2 — Forensic Error Labeler (label_judge.py)
================================================================================

Title:       First-pass error classification via Gemini AI for all subcategories
Version:     1.0

PURPOSE:
    Analyzes model response failures from Stage 0 (0_preprocess_results.py) to
    identify and classify errors using the master error taxonomy. Acts as a
    forensic auditor:
      • Independently computes correct solution (100% accurate for all subcats)
      • Traces response step-by-step to pinpoint FIRST erroneous value
      • Classifies root cause using error taxonomy
      • Generates detailed forensic observations

    Processes enriched failures JSONL (output of Stage 0), sends each failure
    to Gemini for error classification, writes results to CSV ready for Stage 3
    (validate_judge.py for second-pass verification).

    Execution: Sequential for-loop (no ThreadPoolExecutor) for simplicity and
    easier debugging. Includes fallback mechanism for quota/429 errors.

PREREQUISITES:
    • 0_preprocess_results.py must have run (creates {subcat}_failures.jsonl)
    • subcat_config.py must be available (ground_truth, error taxonomy)
    • judge_prompts module must be available (system + user prompts per subcat)
    • Gemini API keys configured

SUPPORTED SUBCATEGORIES:
    det, eig, inv (reference), rank, nullity, mult, pow, vec, trans, trace

JUDGE PROMPTS:
    Each subcategory has a custom judge prompt defining:
      • Error taxonomy and subtypes
      • Verification rules (how to detect each error type)
      • Output format (JSON structure with error_tag, confidence, etc.)

    Prompts are loaded from judge_prompts/{subcat}.py files.

USAGE EXAMPLES:
    # Dry-run: preview without API calls
    python pipeline/label_judge.py \\
        --results data/rank/rank_failures.jsonl \\
        --output  data/rank/rank_judge_labels.csv \\
        --subcat  rank \\
        --judge-llm-id gemini-3.1-pro-preview \\
        --dry-run

    # Test mode: use flash-lite (cheap, testing)
    python pipeline/label_judge.py \\
        --results data/rank/rank_failures.jsonl \\
        --output  data/rank/rank_judge_labels.csv \\
        --subcat  rank \\
        --judge-llm-id gemini-3.1-pro-preview

    # Production: use pro model (accurate, slower)
    python pipeline/label_judge.py \\
        --results data/rank/rank_failures.jsonl \\
        --output  data/rank/rank_judge_labels.csv \\
        --subcat  rank \\
        --judge-llm-id gemini-3.1-pro-preview

    # Custom limit (for testing)
    python pipeline/label_judge.py \\
        --results data/rank/rank_failures.jsonl \\
        --output  data/rank/rank_judge_labels.csv \\
        --subcat  rank \\
        --judge-llm-id gemini-3.1-pro-preview \\
        --limit 5

OUTPUT:
    data/{subcat}/{subcat}_judge_labels.csv columns:
        • Problem_ID                : Unique problem identifier
        • Model                     : LLM model name
        • Subcat                    : Subcategory (rank, nullity, etc.)
        • Error_Tag                 : Primary error classification
        • Sign_Subtype              : Subtype if Error_Tag=SIGN_ERROR (optional)
        • Hallucination_Subtype     : Subtype if Error_Tag=HALLUCINATION (optional)
        • Solution_Strategy         : Strategy attempted by model (if applicable)
        • First_Error_Step          : Step number where error first occurs
        • First_Error_Description   : Details of first erroneous value
        • Proposed_Novel_Tag        : NEW tag if unrecognized (OTHER_UNMAPPED)
        • Maps_Closest_To           : Closest standard tag if OTHER_UNMAPPED
        • Confidence                : HIGH | MEDIUM | LOW

GEMINI API:
    SDK: google-genai (NOT google.generativeai — deprecated)
    Model: pass via --judge-llm-id (e.g. gemini-3.1-flash-lite-preview for testing,
           gemini-3.1-pro-preview for production)
    Temperature: 0.0 (deterministic)
    Max tokens: 16384
    Retry policy: 5 attempts with flat 60-second wait (for Gemini quota recovery)

ENVIRONMENT:
    export GEMINI_API_KEY="your-primary-key"    # required

================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from json_repair import repair_json
    HAS_JSONREPAIR = True
except ImportError:
    HAS_JSONREPAIR = False
    repair_json = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_ERROR_PATTERN = re.compile(
    r"httpsconnectionpool|max\s+retries\s+exceeded|connection\s*(refused|timeout|error)|"
    r"name\s+resolution\s+error|"
    r"rate\s+limit|quota\s+exceeded|"
    r"\bapi\b(?:\W+\w+){0,3}\W+\berror\b|\berror\b(?:\W+\w+){0,3}\W+\bapi\b",
    re.IGNORECASE
)

from subcat_config import SUBCAT_CONFIGS, SubcatJudgeConfig
from judge_prompts import BUILD_PROMPTS
from judge_llm import define_clients, call_llm


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CSV_FIELDNAMES = [
    "Problem_ID",
    "Model",
    "Subcat",
    "Error_Tag",
    "Sign_Subtype",
    "Hallucination_Subtype",
    "Solution_Strategy",
    "First_Error_Step",
    "First_Error_Description",
    "Proposed_Novel_Tag",
    "Maps_Closest_To",
    "Confidence",
    "build_judge_llm",
    "needs_review",
]

EMPTY_LABEL = {f: "" for f in CSV_FIELDNAMES}


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────

def load_all_failures(results_path: str, filter_subcat: str | None = None) -> list[dict]:
    """
    Load enriched JSONL (from 0_preprocess_results.py).
    All records in this file are failures (response_correct=False).
    Optionally filter to a specific subcat if the JSONL mixes multiple.
    """
    failures = []
    skipped  = 0

    with open(results_path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"  [WARN] Line {lineno}: invalid JSON — {exc}  (skipped)",
                      file=sys.stderr)
                skipped += 1
                continue

            # Map CLI --subcat arg to data Subcat value
            from subcat_config import SUBCAT_ARG as _SUBCAT_ARG
            filter_subcat_mapped = _SUBCAT_ARG.get(filter_subcat, filter_subcat)

            if filter_subcat_mapped and rec.get("Subcat", "") != filter_subcat_mapped:
                skipped += 1
                continue

            failures.append(rec)

    if skipped:
        print(f"  [INFO] Skipped {skipped} records (parse errors or wrong subcat)")
    return failures


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING — 3-layer defence
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_response(raw: str) -> dict:
    """
    Parse Gemini's text response into a dict.
    Layers:
      1. Strip markdown code fences
      2. json.loads()
      3. json_repair fallback
      4. Unwrap list → first element
    Raises ValueError if all layers fail.
    """
    # Layer 1 — strip markdown fences
    text = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)

    # Layer 2 — standard parse
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Layer 3 — json_repair
        if HAS_JSONREPAIR and repair_json:
            try:
                result = json.loads(repair_json(text))
            except Exception as exc:
                raise ValueError(f"json_repair also failed: {exc}\nRaw: {raw[:300]}")
        else:
            raise ValueError(f"json.loads failed and json_repair not installed.\nRaw: {raw[:300]}")

    # Layer 4 — unwrap array
    if isinstance(result, list):
        if len(result) == 0:
            raise ValueError("Gemini returned empty JSON array")
        result = result[0]

    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object, got {type(result)}: {raw[:200]}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify(
    record: dict,
    primary_client,
    model_id: str,
    judge_llm_id: str,
    config: SubcatJudgeConfig,
    dry_run: bool = False,
) -> dict:
    """
    Classify a single failure record using the judge LLM.
    Checks completion_status before calling the API.
    Returns a CSV row dict matching CSV_FIELDNAMES.
    """
    problem_id        = record.get("Problem_ID", record.get("question_id", "?"))
    model_name        = record.get("Model", "?")
    subcat            = record.get("Subcat", config.subcat)
    completion_status = record.get("completion_status", "COMPLETE")
    response_text     = record.get("response", "[no response stored]")

# ── Check response for API error patterns and update status ─────────
    response_text_check = str(record.get("response", "")).strip()
    if completion_status != "API_ERROR" and API_ERROR_PATTERN.search(response_text_check):
        completion_status = "API_ERROR"

    # ── Skip if no response to classify ───────────────────────────────────
    response_text = str(record.get("response", "")).strip().lower()
    is_no_response = (
        completion_status == "API_ERROR" or
        completion_status == "NO_RESPONSE" or
        response_text in ("nan", "none", "") or
        not response_text
    )
    if is_no_response:
        print(f"  [SKIP] {problem_id} — {completion_status or 'NO_RESPONSE'}", flush=True)
        label = {**EMPTY_LABEL}
        label["Problem_ID"]              = problem_id
        label["Model"]                   = model_name
        label["Subcat"]                  = subcat
        label["Error_Tag"]               = "SKIP"
        desc = "API_ERROR" if completion_status == "API_ERROR" else "NO_RESPONSE"
        label["First_Error_Description"] = f"{desc} — no response to classify"
        label["Confidence"]              = "ZERO"
        label["build_judge_llm"]         = judge_llm_id
        return label

    # ── Build user prompt ───────────────────────────────────────────────
    user_prompt = config.judge_user_template.format(
        matrix_latex=record.get("problem_latex", "[matrix LaTeX not available]"),
        ground_truth=record.get("ground_truth", "unknown"),
        extracted_answer=record.get("extracted_answer", "none"),
        response=record.get("response", "[no response stored]"),
    )

    # ── Dry run ──────────────────────────────────────────────────────────
    if dry_run:
        print(f"\n{'─'*60}")
        print(f"DRY RUN — {problem_id}  model={model_name}  status={completion_status}")
        print(f"  gt={record.get('ground_truth')}  "
              f"extracted={record.get('extracted_answer')}  "
              f"response_len={len(response_text)}")
        print(f"  Matrix LaTeX: {'OK' if record.get('problem_latex') else 'MISSING'}")
        print(f"  Prompt preview:\n{user_prompt[:400]}...")
        label = {**EMPTY_LABEL}
        label["Problem_ID"]              = problem_id
        label["Model"]                   = model_name
        label["Subcat"]                  = subcat
        label["Error_Tag"]               = "DRY_RUN"
        label["First_Error_Description"] = "dry run — no API call made"
        label["build_judge_llm"]         = judge_llm_id
        return label

    # ── Call judge LLM ───────────────────────────────────────────────────
    try:
        raw, finish_reason = call_llm(
            config.judge_system_prompt, user_prompt,
            model_id, primary_client,
        )
    except RuntimeError as exc:
        print(f"  [FAIL] {problem_id} — {exc}", file=sys.stderr)
        label = {**EMPTY_LABEL}
        label["Problem_ID"]              = problem_id
        label["Model"]                   = model_name
        label["Subcat"]                  = subcat
        label["Error_Tag"]               = "UNKNOWN"
        label["First_Error_Description"] = f"API failed: {exc}"
        label["Confidence"]              = "ZERO"
        label["build_judge_llm"]         = judge_llm_id
        return label

    # ── Parse response ───────────────────────────────────────────────────
    try:
        result = parse_json_response(raw)
    except ValueError as exc:
        print(f"  [JSON_ERROR] {problem_id} — {exc}", file=sys.stderr)
        label = {**EMPTY_LABEL}
        label["Problem_ID"]              = problem_id
        label["Model"]                   = model_name
        label["Subcat"]                  = subcat
        label["Error_Tag"]               = "OTHER_UNMAPPED"
        label["First_Error_Description"] = f"JSON parse failed: {str(exc)[:100]}"
        label["Confidence"]              = "ZERO"
        label["build_judge_llm"]         = judge_llm_id
        label["needs_review"]            = "YES"
        return label

    # Validate error_tag
    error_tag = str(result.get("error_tag", "")).strip()
    if error_tag not in config.valid_error_tags:
        print(f"  [WARN] Unknown error_tag {error_tag!r} → OTHER_UNMAPPED",
              file=sys.stderr)
        error_tag = "OTHER_UNMAPPED"

    # Validate subtypes against taxonomy
    sign_subtype   = str(result.get("sign_subtype", "")).strip()
    halluc_subtype = str(result.get("hallucination_subtype", "")).strip()
    valid_sign     = config.valid_subtypes.get("SIGN_ERROR", [])
    valid_halluc   = config.valid_subtypes.get("HALLUCINATION", [])
    if sign_subtype and valid_sign and sign_subtype not in valid_sign:
        print(f"  [WARN] Unknown sign_subtype {sign_subtype!r} → cleared",
              file=sys.stderr)
        sign_subtype = ""
    if halluc_subtype and valid_halluc and halluc_subtype not in valid_halluc:
        print(f"  [WARN] Unknown hallucination_subtype {halluc_subtype!r} → cleared",
              file=sys.stderr)
        halluc_subtype = ""

    # If judge itself was truncated, downgrade confidence
    confidence = str(result.get("confidence", "")).strip()
    if finish_reason == "length":
        print(f"  [WARN] {problem_id} — judge response truncated, confidence → LOW",
              file=sys.stderr)
        confidence = "LOW"

    return {
        "Problem_ID":              problem_id,
        "Model":                   model_name,
        "Subcat":                  subcat,
        "Error_Tag":               error_tag,
        "Sign_Subtype":            sign_subtype,
        "Hallucination_Subtype":   halluc_subtype,
        "Solution_Strategy":       result.get("solution_strategy", ""),
        "First_Error_Step":        result.get("first_error_step", ""),
        "First_Error_Description": result.get("first_error_description", ""),
        "Proposed_Novel_Tag":      result.get("proposed_novel_tag", ""),
        "Maps_Closest_To":         result.get("maps_closest_to", ""),
        "Confidence":              confidence,
        "build_judge_llm":         judge_llm_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify LLM failures with Gemini forensic judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/label_judge.py \\
      --results data/rank/rank_failures.jsonl \\
      --output  data/rank/rank_judge_labels.csv \\
      --subcat  rank --dry-run

  python pipeline/label_judge.py \\
      --results data/rank/rank_failures.jsonl \\
      --output  data/rank/rank_judge_labels.csv \\
      --subcat  rank

  python pipeline/label_judge.py \\
      --results data/rank/rank_failures.jsonl \\
      --output  data/rank/rank_judge_labels.csv \\
      --subcat  rank --judge-llm-id gemini-3.1-pro-preview
        """,
    )
    parser.add_argument("--results",  required=True,
                        help="Failures JSONL from 0_preprocess_results.py (e.g. data/rank/rank_failures.jsonl)")
    parser.add_argument("--output",   required=True,
                        help="Output CSV path for judge labels")
    parser.add_argument("--subcat",   required=True,
                        help="Subcategory to classify (e.g. rank)")
    parser.add_argument("--judge-llm-id", required=True,
                        help="Judge LLM model ID (e.g. gemini-3.1-pro-preview)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print prompts without making API calls")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Process only first N records (useful for testing)")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip records already present in output CSV and append new rows")
    args = parser.parse_args()

    # ── Validate subcat & attach prompts ─────────────────────────────────
    try:
        config = SUBCAT_CONFIGS(args.subcat)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    prompts = BUILD_PROMPTS.get(args.subcat)
    if not prompts or not prompts.get("system", "").strip():
        print(f"ERROR: No prompts defined for subcat {args.subcat!r} — write them in judge_prompts/{args.subcat}.py",
              file=sys.stderr)
        sys.exit(1)
    config.judge_system_prompt = prompts["system"]
    config.judge_user_template = prompts["user_template"]

    # ── Print header ──────────────────────────────────────────────────────
    print(f"{'═'*60}")
    print(f"  pipeline/label_judge.py")
    print(f"  Subcat      : {args.subcat}")
    print(f"  Model       : {args.judge_llm_id}")
    print(f"  Results     : {args.results}")
    print(f"  Output      : {args.output}")
    if args.dry_run:
        print(f"  Mode        : DRY RUN")
    if args.resume:
        print(f"  Mode        : RESUME (skip already-done rows)")
    if args.limit:
        print(f"  Limit       : {args.limit} records")
    print(f"{'═'*60}\n")

    # ── Setup LLM clients ─────────────────────────────────────────────────
    primary_client  = None

    if not args.dry_run:
        try:
            primary_client = define_clients(args.judge_llm_id)
        except (ImportError, EnvironmentError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        print()

    # ── Load failures ─────────────────────────────────────────────────────
    print(f"Loading failures from: {args.results}", flush=True)
    failures = load_all_failures(args.results, filter_subcat=args.subcat)
    if args.limit:
        failures = failures[:args.limit]
        print(f"Failures to classify  : {len(failures)} (limited to {args.limit})\n", flush=True)
    else:
        print(f"Failures to classify  : {len(failures)}\n", flush=True)

    if not failures:
        print("No failures found. Has 0_preprocess_results.py been run?")
        return

    # ── Classify (sequential) ─────────────────────────────────────────────
    out_path = Path(args.output)
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Resume: load already-processed keys ──────────────────────────────
    # Skip rows that are truly done (Confidence != ZERO).
    # Re-process rows with Confidence=ZERO (API failure / no response).
    done_keys: set[tuple] = set()
    if args.resume and out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as fexist:
            for row in csv.DictReader(fexist):
                pid_done = row.get("Problem_ID", "")
                mod_done = row.get("Model", "")
                conf     = str(row.get("Confidence", "")).strip().upper()
                if conf != "ZERO":
                    done_keys.add((pid_done, mod_done))
        print(f"  [RESUME] {len(done_keys)} already-done rows found in {out_path.name}",
              flush=True)

    labels: list[dict] = []

    if args.dry_run:
        for record in failures:
            label = classify(record, primary_client,
                             args.judge_llm_id, args.judge_llm_id, config, dry_run=True)
            labels.append(label)
    else:
        write_mode = "a" if args.resume and out_path.exists() else "w"
        with out_path.open(write_mode, newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=CSV_FIELDNAMES)
            if write_mode == "w":
                writer.writeheader()

            skipped = 0
            for i, record in enumerate(failures):
                pid        = record.get("Problem_ID", record.get("question_id", "?"))
                model_name = record.get("Model", "?")

                if (pid, model_name) in done_keys:
                    skipped += 1
                    continue

                label = classify(record, primary_client,
                                 args.judge_llm_id, args.judge_llm_id, config, dry_run=False)
                writer.writerow(label)
                fcsv.flush()
                labels.append(label)

                tag  = label.get("Error_Tag", "?")
                conf = label.get("Confidence", "?")
                done_so_far = len(labels) + skipped
                print(f"  [{done_so_far:03d}/{len(failures)}] {pid:<25} {model_name:<18} {tag:<20} conf={conf}",
                      flush=True)

                if i < len(failures) - 1:
                    time.sleep(0.5)

            if skipped:
                print(f"\n  [RESUME] Skipped {skipped} already-done records.", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────
    from collections import Counter
    tag_counts = Counter(l["Error_Tag"] for l in labels)

    print(f"\n{'═'*60}")
    print(f"  CLASSIFICATION COMPLETE")
    print(f"  Total classified : {len(labels)}")
    print(f"\n  Error tag distribution:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        pct = count / len(labels) * 100 if labels else 0
        print(f"    {tag:<25} {count:>4}  ({pct:.1f}%)")
    if not args.dry_run:
        print(f"\n  Output written to: {args.output}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
