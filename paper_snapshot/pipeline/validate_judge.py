#!/usr/bin/env python3
"""
================================================================================
LinAlg-Bench · Stage 3 — Judge Validator (validate_judge.py)
================================================================================

Title:       Second-pass validation of error classifications via Gemini
Version:     1.0

PURPOSE:
    Performs second-pass verification of error classifications from Stage 2
    (pipeline/label_judge.py). Acts as a senior auditor with targeted,
    tag-specific verification prompts for each classification.

    Workflow:
      1. Reads judge labels CSV (junior judge output)
      2. Loads corresponding failures JSONL (enriched responses + ground truth)
      3. For each classification, builds targeted verification prompt
      4. Calls Gemini (pro model for accuracy) to confirm or correct
      5. Writes validated CSV with validation status and optional corrections

    Output indicates:
      • TRUE: Junior judge classification confirmed
      • FALSE: Classification incorrect; writes correction
      • NEEDS_REVIEW: Ambiguous; flagged for manual review
      • TRUNCATED: Response was truncated; classification unreliable

    Result feeds into statistical analysis and error tag distribution reports.

PREREQUISITES:
    • 0_preprocess_results.py must have run (creates {subcat}_failures.jsonl)
    • pipeline/label_judge.py must have run (creates {subcat}_judge_labels.csv)
    • subcat_config.py must be available (error taxonomy)
    • Validation prompts must be in judge_prompts module

SUPPORTED SUBCATEGORIES:
    det, eig, inv (reference), rank, nullity, mult, pow, vec, trans, trace

VALIDATION APPROACH:
    For each classification in judge_labels.csv:
      • Extract model response + ground truth from failures JSONL
      • Build subcat-specific verification prompt with the classification to verify
      • Ask Gemini: "Is this classification correct? Why or why not?"
      • Parse response: validated=TRUE/FALSE/NEEDS_REVIEW
      • If FALSE: include corrected_tag and reason
      • Write one row per input classification (append validation columns)

USAGE EXAMPLES:
    # Dry-run: preview without API calls
    python pipeline/validate_judge.py \\
        --judge   data/rank/rank_judge_labels.csv \\
        --results data/rank/rank_failures.jsonl \\
        --output  data/rank/rank_judge_validated.csv \\
        --subcat  rank \\
        --judge-llm-id gemini-3.1-pro-preview \\
        --dry-run

    # Test mode: use flash-lite (cheaper)
    python pipeline/validate_judge.py \\
        --judge   data/rank/rank_judge_labels.csv \\
        --results data/rank/rank_failures.jsonl \\
        --output  data/rank/rank_judge_validated.csv \\
        --subcat  rank \\
        --judge-llm-id gemini-3.1-pro-preview

    # Production: use pro model (more accurate)
    python pipeline/validate_judge.py \\
        --judge   data/rank/rank_judge_labels.csv \\
        --results data/rank/rank_failures.jsonl \\
        --output  data/rank/rank_judge_validated.csv \\
        --subcat  rank \\
        --judge-llm-id gemini-3.1-pro-preview

    # Resume interrupted validation (skip already-validated rows)
    python pipeline/validate_judge.py \\
        --judge   data/rank/rank_judge_labels.csv \\
        --results data/rank/rank_failures.jsonl \\
        --output  data/rank/rank_judge_validated.csv \\
        --subcat  rank \\
        --resume

OUTPUT:
    data/{subcat}/{subcat}_judge_validated.csv columns:
        • Problem_ID                  : Unique problem identifier
        • Model, Subcat, Error_Tag    : From input judge CSV
        • validated                   : TRUE | FALSE | NEEDS_REVIEW | TRUNCATED
        • corrected_tag               : Classification if validated=FALSE (or original if TRUE)
        • corrected_sign_subtype      : SIGN_ERROR subtype if applicable (corrected)
        • corrected_halluc_subtype    : HALLUCINATION subtype if applicable (corrected)
        • proposed_novel_tag          : NEW tag if corrected_tag=OTHER_UNMAPPED
        • maps_closest_to             : Closest standard tag if OTHER_UNMAPPED
        • forensic_observation        : Validator's detailed findings
        • validation_confidence       : HIGH | MEDIUM | LOW

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

from subcat_config import SUBCAT_CONFIGS, ALL_PRIMARY_TAGS, SUBCAT_ARG
from judge_prompts import VALIDATE_PROMPTS
from judge_llm import define_clients, call_llm


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Original judge fields passed through unchanged
JUDGE_FIELDNAMES = [
    "Problem_ID", "Model", "Subcat",
    "Error_Tag", "Sign_Subtype", "Hallucination_Subtype",
    "Solution_Strategy", "First_Error_Step", "First_Error_Description",
    "Proposed_Novel_Tag", "Maps_Closest_To", "Confidence",
]

# Fields added by the validator
VALIDATE_FIELDNAMES = [
    "validated",
    "corrected_tag",
    "corrected_sign_subtype",
    "corrected_hallucination_subtype",
    "forensic_observation",
    "validation_confidence",
    "validate_judge_llm",
    "needs_review",
]

ALL_FIELDNAMES = JUDGE_FIELDNAMES + VALIDATE_FIELDNAMES

EMPTY_VALIDATE = {f: "" for f in VALIDATE_FIELDNAMES}

VALID_PRIMARY_TAGS = set(ALL_PRIMARY_TAGS)


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────

def load_judge_labels(path: str) -> list[dict]:
    """Load judge labels CSV from Stage 2."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def load_failures_index(jsonl_path: str) -> dict[tuple, dict]:
    """
    Load failures JSONL and index by (Problem_ID, Model).
    Returns dict: (Problem_ID, Model) → {response, problem_latex, ground_truth, ...}
    """
    index = {}
    with open(jsonl_path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"  [WARN] JSONL line {lineno}: {exc} (skipped)", file=sys.stderr)
                continue
            pid   = str(rec.get("Problem_ID", rec.get("question_id", ""))).strip()
            model = str(rec.get("Model", "")).strip()
            if pid and model:
                index[(pid, model)] = {
                    "response":           rec.get("response", ""),
                    "problem_latex":      rec.get("problem_latex", ""),
                    "ground_truth":       rec.get("ground_truth", ""),
                    "completion_status":  rec.get("completion_status", "COMPLETE"),
                }
    return index


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING — 3-layer defence (same as Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_response(raw: str) -> dict:
    text = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        if HAS_JSONREPAIR and repair_json:
            try:
                result = json.loads(repair_json(text))
            except Exception as exc:
                raise ValueError(f"json_repair also failed: {exc}\nRaw: {raw[:300]}")
        else:
            raise ValueError(f"json.loads failed and json_repair not installed.\nRaw: {raw[:300]}")

    if isinstance(result, list):
        if not result:
            raise ValueError("Gemini returned empty JSON array")
        result = result[0]

    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object, got {type(result)}: {raw[:200]}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SUBTYPE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _validate_subtypes(p_tag: str, result: dict, config) -> tuple[str, str]:
    """
    Extract and validate corrected sign/hallucination subtypes from validator result.
    Returns (corrected_sign_subtype, corrected_hallucination_subtype).
    """
    sign_sub   = str(result.get("sign_subtype",          "") or "").strip()
    halluc_sub = str(result.get("hallucination_subtype", "") or "").strip()

    valid_sign   = config.valid_subtypes.get("SIGN_ERROR",    [])
    valid_halluc = config.valid_subtypes.get("HALLUCINATION", [])

    if sign_sub and valid_sign and sign_sub not in valid_sign:
        print(f"  [WARN] Unknown corrected sign_subtype {sign_sub!r} → cleared",
              file=sys.stderr)
        sign_sub = ""
    if halluc_sub and valid_halluc and halluc_sub not in valid_halluc:
        print(f"  [WARN] Unknown corrected hallucination_subtype {halluc_sub!r} → cleared",
              file=sys.stderr)
        halluc_sub = ""

    # Only populate the subtype relevant to the corrected tag
    if p_tag != "SIGN_ERROR":
        sign_sub = ""
    if p_tag != "HALLUCINATION":
        halluc_sub = ""

    return sign_sub, halluc_sub


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATE ONE RECORD
# ─────────────────────────────────────────────────────────────────────────────

def validate(
    judge_row: dict,
    response_data: dict,
    primary_client,
    model_id: str,
    validate_llm_id: str,
    system_prompt: str,
    build_user_prompt,        # callable: (row, response, matrix_latex) → str
    config,
    dry_run: bool = False,
) -> dict:
    """
    Call Gemini to validate a single junior judge classification.
    Returns a dict of VALIDATE_FIELDNAMES.
    """
    response          = response_data.get("response", "")
    matrix_latex      = response_data.get("problem_latex", "[matrix not available]")
    completion_status = response_data.get("completion_status", "COMPLETE")

    pid   = judge_row.get("Problem_ID", "?")
    model = judge_row.get("Model", "?")
    tag   = judge_row.get("Error_Tag", "UNKNOWN")
    sub   = judge_row.get("Sign_Subtype", "") or judge_row.get("Hallucination_Subtype", "") or ""

    if not response:
        return {
            **EMPTY_VALIDATE,
            "validated":            "NEEDS_REVIEW",
            "forensic_observation": "No model response found in failures JSONL",
            "validate_judge_llm":   validate_llm_id,
        }

    response_text = str(response or "").strip().lower()
    is_no_response = (
        completion_status == "API_ERROR" or
        completion_status == "NO_RESPONSE" or
        response_text in ("nan", "none", "") or
        not response_text
    )
    if is_no_response:
        print(f"  [SKIP] {pid} — {completion_status or 'NO_RESPONSE'}", flush=True)
        return {
            **EMPTY_VALIDATE,
            "validated":            "SKIP",
            "forensic_observation": f"{completion_status or 'NO_RESPONSE'} — no response to validate",
            "validate_judge_llm":   validate_llm_id,
        }

    user_prompt = build_user_prompt(judge_row, response, matrix_latex)

    # ── Dry run ──────────────────────────────────────────────────────────
    if dry_run:
        print(f"\n{'─'*60}")
        print(f"DRY RUN — {pid}  model={model}  tag={tag}/{sub}  status={completion_status}")
        print(f"  Response: {len(response)} chars  "
              f"Matrix: {'OK' if matrix_latex != '[matrix not available]' else 'MISSING'}")
        print(f"  Prompt preview:\n{user_prompt[:400]}...")
        return {**EMPTY_VALIDATE, "validated": "DRY_RUN", "validate_judge_llm": validate_llm_id}

    # ── Call judge LLM ───────────────────────────────────────────────────
    try:
        raw, finish_reason = call_llm(
            system_prompt, user_prompt,
            model_id, primary_client,
        )
    except RuntimeError as exc:
        print(f"  [FAIL] {pid} — {exc}", file=sys.stderr)
        return {
            **EMPTY_VALIDATE,
            "validated":            "NEEDS_REVIEW",
            "forensic_observation": f"API failed: {exc}",
            "validate_judge_llm":   validate_llm_id,
        }

    if True:  # scope block replacing the retry for-loop

        try:
            result = parse_json_response(raw)
        except ValueError as exc:
            print(f"  [JSON_ERROR] {pid} — {exc}", file=sys.stderr)
            # Return default validation with error flag
            return {
                "Problem_ID":              pid,
                "Model":                   model,
                "Subcat":                  judge_row.get("Subcat", "?"),
                "validated":               False,
                "primary_tag":             "OTHER_UNMAPPED",
                "forensic_observation":    f"JSON parse failed: {str(exc)[:100]}",
                "confidence":              "ZERO",
                "validate_judge_llm":      validate_llm_id,
                "needs_review":            "YES",
            }

        verified  = result.get("verified")
        p_tag     = str(result.get("primary_tag", "")).strip()
        obs       = str(result.get("forensic_observation", "")).strip()
        conf      = str(result.get("confidence", "")).strip()
        novel     = str(result.get("proposed_novel_tag", "")).strip()
        closest   = str(result.get("maps_closest_to", "")).strip()

        # Auto-map invented tags
        if p_tag not in VALID_PRIMARY_TAGS:
            novel = p_tag or novel
            p_tag = "OTHER_UNMAPPED"
            obs   = f"[Auto-mapped invented tag '{novel}'] " + obs

        corrected_sign, corrected_halluc = _validate_subtypes(p_tag, result, config)

        # If judge itself was truncated, downgrade confidence
        if finish_reason == "length":
            print(f"  [WARN] {pid} — judge response truncated, confidence → LOW",
                  file=sys.stderr)
            conf = "LOW"

        base = {
            "forensic_observation":  obs,
            "validation_confidence": conf,
            "validate_judge_llm":    validate_llm_id,
        }

        if verified is True:
            return {
                **EMPTY_VALIDATE,
                "validated":                      "TRUE",
                "corrected_sign_subtype":         "",
                "corrected_hallucination_subtype": "",
                **base,
            }
        elif verified is False:
            return {
                "validated":                       "FALSE",
                "corrected_tag":                   p_tag,
                "corrected_sign_subtype":          corrected_sign,
                "corrected_hallucination_subtype": corrected_halluc,
                **base,
            }
        else:  # null / ambiguous
            return {
                **EMPTY_VALIDATE,
                "validated":      "NEEDS_REVIEW",
                **base,
            }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Senior validation judge — verify Stage 2 judge classifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/validate_judge.py \\
      --judge   data/rank/rank_judge_labels.csv \\
      --results data/rank/rank_failures.jsonl \\
      --output  data/rank/rank_judge_validated.csv \\
      --subcat  rank --dry-run

  python pipeline/validate_judge.py \\
      --judge   data/rank/rank_judge_labels.csv \\
      --results data/rank/rank_failures.jsonl \\
      --output  data/rank/rank_judge_validated.csv \\
      --subcat  rank --resume
        """,
    )
    parser.add_argument("--judge",    required=True,
                        help="Judge labels CSV from pipeline/label_judge.py")
    parser.add_argument("--results",  required=True,
                        help="Failures JSONL from 0_preprocess_results.py")
    parser.add_argument("--output",   required=True,
                        help="Output validated CSV path")
    parser.add_argument("--subcat",   required=True,
                        help="Subcategory (e.g. rank)")
    parser.add_argument("--judge-llm-id", required=True,
                        help="Judge LLM model ID (e.g. gemini-3.1-pro-preview)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print prompts without making API calls")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Process only first N records (useful for testing)")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip already-validated rows and append new rows")
    args = parser.parse_args()

    # ── Load subcat config ────────────────────────────────────────────────
    try:
        config = SUBCAT_CONFIGS(args.subcat)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Load validate prompts ─────────────────────────────────────────────
    vprompts = VALIDATE_PROMPTS.get(args.subcat)
    if not vprompts or not vprompts.get("system", "").strip():
        print(f"ERROR: No validate prompts for subcat {args.subcat!r} — "
              f"write VALIDATE_SYSTEM in judge_prompts/{args.subcat}.py",
              file=sys.stderr)
        sys.exit(1)

    system_prompt    = vprompts["system"]
    build_user_prompt = vprompts.get("build_user_prompt")
    if build_user_prompt is None:
        print(f"ERROR: build_validate_user_prompt() not defined in "
              f"judge_prompts/{args.subcat}.py", file=sys.stderr)
        sys.exit(1)

    # ── Print header ──────────────────────────────────────────────────────
    print(f"{'═'*60}")
    print(f"  pipeline/validate_judge.py")
    print(f"  Subcat      : {args.subcat}")
    print(f"  Model       : {args.judge_llm_id}")
    print(f"  Judge CSV   : {args.judge}")
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

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading judge labels  : {args.judge}", flush=True)
    judge_rows = load_judge_labels(args.judge)

    # Map CLI --subcat arg to canonical data Subcat value
    args.subcat = SUBCAT_ARG.get(args.subcat, args.subcat)

    # Filter to subcat (in case the CSV mixes subcats)
    judge_rows = [r for r in judge_rows if r.get("Subcat", "") == args.subcat]

    print(f"Loading failures JSONL: {args.results}", flush=True)
    failures_index = load_failures_index(args.results)

    if args.limit:
        judge_rows = judge_rows[:args.limit]

    print(f"Records to validate   : {len(judge_rows)}", flush=True)
    print(f"Failures indexed      : {len(failures_index)}", flush=True)

    missing = [
        r for r in judge_rows
        if (r.get("Problem_ID", ""), r.get("Model", "")) not in failures_index
    ]
    if missing:
        print(f"  [WARN] {len(missing)} judge rows have no matching JSONL entry "
              f"(will produce NEEDS_REVIEW)", flush=True)
    print()

    if not judge_rows:
        print("No judge rows found. Has pipeline/label_judge.py been run?")
        return

    # ── Resume: load already-processed keys ──────────────────────────────
    # Skip rows that are truly done (validated != NEEDS_REVIEW/SKIP and confidence != ZERO).
    # Re-process rows with ZERO confidence or NEEDS_REVIEW/SKIP status.
    done_keys: set[tuple] = set()
    out_path = Path(args.output)
    if args.resume and out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as fexist:
            for row in csv.DictReader(fexist):
                pid_done = row.get("Problem_ID", "")
                mod_done = row.get("Model", "")
                conf     = str(row.get("validation_confidence", "")).strip().upper()
                val_stat = str(row.get("validated", "")).strip().upper()
                if conf not in ("ZERO", "") and val_stat not in ("NEEDS_REVIEW", "SKIP"):
                    done_keys.add((pid_done, mod_done))
        print(f"  [RESUME] {len(done_keys)} already-validated rows found in {out_path.name}\n",
              flush=True)

    # ── Validate (sequential) ─────────────────────────────────────────────
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    write_mode = "a" if args.resume and out_path.exists() else "w"

    def _run(write_fh=None):
        writer = None
        if write_fh:
            writer = csv.DictWriter(write_fh, fieldnames=ALL_FIELDNAMES)
            if write_mode == "w":
                writer.writeheader()

        skipped = 0
        for i, judge_row in enumerate(judge_rows):
            pid   = judge_row.get("Problem_ID", "?")
            model = judge_row.get("Model", "?")

            if (pid, model) in done_keys:
                skipped += 1
                continue

            # Error_Tag=SKIP means the model had no response — skip Stage 3 entirely.
            if judge_row.get("Error_Tag", "") == "SKIP":
                print(f"  [SKIP] {pid:<25} {model:<18} — no response", flush=True)
                continue

            response_data = failures_index.get((pid, model), {})
            val_result    = validate(
                judge_row, response_data,
                primary_client,
                args.judge_llm_id, args.judge_llm_id,
                system_prompt, build_user_prompt,
                config, dry_run=args.dry_run,
            )

            out_row = {**{f: judge_row.get(f, "") for f in JUDGE_FIELDNAMES}, **val_result}
            results.append(out_row)

            if writer:
                writer.writerow(out_row)
                fcsv.flush()

            tag      = judge_row.get("Error_Tag", "?")
            verified = val_result.get("validated", "?")
            corr     = val_result.get("corrected_tag", "")
            status   = f"{verified}" + (f" → {corr}" if corr else "")
            done_so_far = len(results) + skipped
            print(f"  [{done_so_far:03d}/{len(judge_rows)}] {pid:<25} {model:<18} {tag:<20} {status}",
                  flush=True)

            if not args.dry_run and i < len(judge_rows) - 1:
                time.sleep(0.5)

        if skipped:
            print(f"\n  [RESUME] Skipped {skipped} already-validated records.", flush=True)

    if args.dry_run:
        _run(write_fh=None)
    else:
        with out_path.open(write_mode, newline="", encoding="utf-8") as fcsv:
            _run(write_fh=fcsv)

    # ── Summary ───────────────────────────────────────────────────────────
    from collections import Counter
    validated_counts = Counter(r["validated"]     for r in results)
    tag_counts       = Counter(r["Error_Tag"]     for r in results)
    corrected        = [r for r in results if r.get("corrected_tag")]

    print(f"\n{'═'*60}")
    print(f"  VALIDATION COMPLETE")
    print(f"  Total validated  : {len(results)}")
    print(f"\n  Validation outcomes:")
    for status, cnt in sorted(validated_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(results) * 100 if results else 0
        print(f"    {status:<15} {cnt:>4}  ({pct:.1f}%)")
    if corrected:
        print(f"\n  Corrected tags ({len(corrected)} records):")
        for r in corrected:
            print(f"    {r['Problem_ID']:<30} {r['Error_Tag']} → {r['corrected_tag']}")
    print(f"\n  Original tag distribution:")
    for tag, cnt in sorted(tag_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(results) * 100 if results else 0
        print(f"    {tag:<25} {cnt:>4}  ({pct:.1f}%)")
    if not args.dry_run:
        print(f"\n  Output written to: {args.output}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
