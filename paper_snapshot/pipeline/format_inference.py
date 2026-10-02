"""
================================================================================
LinAlg-Bench · Format Inference Pipeline  (format_inference_pipeline.py)
================================================================================

Changes in this revision
────────────────────────
1. STRICT subcat enforcement
   • Rows with null / NaN / empty / sentinel subcategory are DROPPED before
     any task is built.  A warning lists how many were dropped and why.
   • _normalise_subcat() raises ValueError for anything not in the known set
     so bad data never silently falls through to _DEFAULT_SYSTEM.

2. LAST-100-TOKEN TAIL saved unconditionally
   • response_tail  = last 100 "tokens" (whitespace-split words) of the raw
     model response, stored in a new OUTPUT column.
   • Saved regardless of whether \boxed{} extraction succeeded or failed.
   • Useful for post-hoc inspection without storing the full response twice.

3. Robust pipeline
   • All DataFrame operations are guarded (fillna → cast → strip).
   • CSV reader skips malformed resume rows gracefully.
   • Thread-safe heap drain on timeout / error.
   • Ceiling is capped at model max_ceiling.
   • logging.basicConfig uses force=True to avoid silent no-ops.

4. Flexible input schema  (NEW in this revision)
   • Accepts either 'question_id' or 'id' for the question identifier.
   • Accepts either 'format_type' or 'format' for the format column.
   • If 'instruction' is missing, builds it from 'problem_text' +
     'problem_representation' (typo 'problem_represntation' also tolerated).
   • If 'question_id' lacks a format suffix (_latex / _ascii / _list),
     the corresponding format is appended:
         C_4x4_det + latex  →  C_4x4_det_latex
         C_4x4_det + ascii  →  C_4x4_det_ascii
         C_4x4_det + list   →  C_4x4_det_list

5. is_correct column  (NEW in this revision)
   • The previous boolean 'correct' column is now named 'is_correct'.
   • The response's \\boxed{} extraction ('extracted_answer') is written
     immediately next to the ground-truth column ('answer_latex'), and
     'is_correct' holds the True/False/blank match flag for that pair.

6. Subcategory aliasing  (NEW in this revision)
   • SUBCAT_ALIASES maps full-name inputs (determinant, eigenvalues,
     multiplication, transpose, matrix_power, matrix_vector, inverse, …)
     to canonical short codes (det, eig, mult, trans, pow, matvec, inv, …)
     BEFORE the strict KNOWN_SUBCATS check.
   • The bareword 'null' was removed from the sentinel set so that the
     legitimate null-space subcategory is no longer silently dropped.

7. Structured ground-truth extraction  (NEW in this revision)
   • Every output row now also carries:
       ground_scalar     – scalar pulled out of answer_latex
                             ("\\text{nullity}(A) = 0" → "0")
       ground_matrix     – matrix pulled out of answer_latex
                             ("\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}"
                              → [["1","2"],["3","4"]])
       extracted_scalar  – same extraction applied to the model's \\boxed{}
       extracted_matrix  – same extraction applied to the model's \\boxed{}
   • answers_match() now tries scalar → matrix → string in order, so
     numeric and structural matches succeed across LaTeX / list / text
     formats (e.g. "-2" matches "-2.0", "\\det(A) = -2", and a bare "-2").

Bug-fixes applied
─────────────────
BUG-1  Duplicate --tail-tokens argument definition; CLI override was commented
       out so the flag was parsed but never applied.  Fixed: single definition,
       override applied before pipeline runs.
BUG-2  _classify() inside load_and_validate_dataset had a broken float-NaN
       guard — non-float NaN values fell through to the sentinel branch and
       were miscounted.  Fixed: explicit pd.isna() check.
BUG-3  Wrong model ID for Claude-4.5-Sonnet: "claude-sonnet-4.5" → correct
       OpenRouter slug is "claude-sonnet-4-5" (hyphens, not dots).
BUG-4  Gemini-3.0-Pro had empty api_base ("") causing silent connection
       failures.  Fixed: set to Google GenAI base URL; key name corrected.
BUG-5  _flush_heap used <= instead of == for the heap drain condition, keeping
       almost all rows buffered until _drain_heap.  The _min_pending advance
       loop also had an off-by-one.  Fixed: proper sequential drain logic.
BUG-6  last_n_tokens() captured _TAIL_TOKENS at definition time as a default
       arg — CLI override would not propagate.  Fixed: callers pass n
       explicitly; _TAIL_TOKENS is a mutable module-level int.
BUG-7  _print_summary compared nullable series with == True, mishandling None.
       Fixed: fillna(False) before summing.
================================================================================
"""
from __future__ import annotations
import os
import re
import csv
import math
import time
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
import argparse
import threading
import heapq
from datetime import datetime
from typing import Optional, List, Dict, Set
from concurrent.futures import (
    ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError,
)

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):          # type: ignore[misc]
        return x

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TOKEN_STEP_UP    = 8_192   # tokens added per adaptive retry on V4 truncation
LOOP_MIN_LINES   = 100     # minimum total lines before loop check fires
LOOP_THRESHOLD   = 0.5     # fraction of second-half dominated by one repeated line


# ─────────────────────────────────────────────────────────────────────────────
# RETURN TYPE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CallResult:
    text:          Optional[str]   = None
    finish_reason: Optional[str]   = None   # "stop" | "length" | "loop_trimmed" | None
    tokens_used:   int             = 0
    latency_ms:    float           = 0.0
    error:         Optional[str]   = None


# ─────────────────────────────────────────────────────────────────────────────
# LOOP DETECTION / TRIMMING  (module-level helpers)
# ─────────────────────────────────────────────────────────────────────────────

def _is_loop(text: str, variant: str) -> bool:
    """Return True if the response looks like an infinite repetition loop."""
    if variant != "V4_scratch_baseline":
        return False
    if not text:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < LOOP_MIN_LINES:
        return False
    second_half = lines[len(lines) // 2:]
    top_count = Counter(second_half).most_common(1)[0][1]
    return (top_count / len(second_half)) > LOOP_THRESHOLD


def _trim_loop(text: str) -> str:
    """
    Deduplicate repeated lines in the second half of the response.
    Appends a sentinel so downstream analysis can identify trimmed records.
    """
    all_lines = text.splitlines()
    mid = len(all_lines) // 2
    seen: set[str] = set()
    deduped: list[str] = []
    for line in all_lines[mid:]:
        key = line.strip()
        if key not in seen:
            seen.add(key)
            deduped.append(line)
    return "\n".join(all_lines[:mid] + deduped) + "\n[INFINITE LOOP DETECTED AT END]"


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class InferenceClient:
    """
    Single-model inference client.

    Parameters
    ──────────
    model_cfg     dict  — entry from MODELS registry:
                          {"model_id", "api_base", "api_key_env", "min_tokens"}
    system_prompt str   — locked system prompt (varies by subcat)
    dry_run       bool  — if True, skip API calls and return placeholder text
    retries       int   — max API attempts per call (default 3)
    """

    def __init__(
        self,
        model_cfg:     dict,
        system_prompt: str,
        dry_run:       bool = False,
        retries:       int  = 3,
    ):
        self.model_id      = model_cfg["model_id"]
        self.api_base      = model_cfg.get("api_base", "")
        self.system_prompt = system_prompt
        self.dry_run       = dry_run
        self.retries       = retries

        self._client   = None
        self._fallback = None           # GenAI only

        if not dry_run:
            api_key_env = model_cfg.get("api_key_env", "")
            api_key     = os.environ.get(api_key_env, "") if api_key_env else ""
            self._backend = self._detect_backend()
            self._init_client(api_key)
            logging.info(f"  InferenceClient: {self.model_id} via {self._backend}")
        else:
            self._backend = "dry"

    # ── backend detection ─────────────────────────────────────────────────

    def _detect_backend(self) -> str:
        if "openai.com" in self.api_base:
            return "openai"
        elif "openrouter" in self.api_base:
            return "openrouter"
        elif self.model_id.lower().startswith("gemini"):
            return "genai"
        else:
            return "openai_compat"      # generic OpenAI-compatible endpoint

    def _init_client(self, api_key: str) -> None:
        if self._backend == "genai":
            try:
                from google import genai
            except ImportError:
                raise ImportError("pip install google-genai")
            primary_key = os.environ.get("GEMINI_API_KEY", "")
            if not primary_key:
                raise EnvironmentError("GEMINI_API_KEY not set")
            self._client = genai.Client(api_key=primary_key)
            fallback_key = os.environ.get("GOOGLE_API_KEY2", "")
            if fallback_key:
                self._fallback = genai.Client(api_key=fallback_key)
        else:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("pip install openai")
            if not api_key:
                logging.warning(f"  No API key resolved for {self.model_id} — calls will fail")
            self._client = OpenAI(base_url=self.api_base, api_key=api_key or "no-key")

    # ── public call ───────────────────────────────────────────────────────

    def call(
        self,
        user_prompt:       str,
        qid:               str,
        variant:           str,
        max_tokens:        int           = 8_192,
        ceiling:           Optional[int] = None,
    ) -> CallResult:
        """
        Call the model and return a CallResult.

        Parameters
        ──────────
        user_prompt   str  — the user-turn prompt
        qid           str  — question ID (for logging only)
        variant       str  — variant name, e.g. "V4_scratch_baseline"
        max_tokens    int  — starting token budget
        ceiling       int  — hard upper limit for adaptive step-up
                             (defaults to 2 × max_tokens)
        """
        if self.dry_run:
            return CallResult(
                text=f"[DRY RUN — {variant}]\nPrompt length: {len(user_prompt)} chars",
                finish_reason=None,
            )

        is_v4    = (variant == "V4_scratch_baseline")
        ceiling  = ceiling or (max_tokens * 2)
        current  = max_tokens

        for attempt in range(self.retries):
            try:
                t0 = time.time()
                text, finish_reason, tokens_used = self._dispatch(user_prompt, current)
                latency_ms = round((time.time() - t0) * 1000, 1)

                # ── Loop detection (check before adaptive retry) ──────────
                if finish_reason == "length" and _is_loop(text, variant):
                    text          = _trim_loop(text)
                    finish_reason = "loop_trimmed"
                    logging.warning(
                        f"  ∞ LOOP {qid}/{variant} | tokens={tokens_used} | trimmed"
                    )

                # ── Adaptive step-up (V4 truncation only) ─────────────────
                if finish_reason == "length":
                    if is_v4 and current < ceiling:
                        new = min(current + TOKEN_STEP_UP, ceiling)
                        logging.warning(
                            f"  ⚠ TRUNCATED {qid}/{variant} | tokens={tokens_used} "
                            f"| step-up {current}→{new}"
                        )
                        current = new
                        continue        # retry with more tokens
                    elif is_v4:
                        logging.warning(
                            f"  ✗ TRUNCATED {qid}/{variant} | tokens={tokens_used} "
                            f"| ceiling {ceiling} reached"
                        )
                    else:
                        logging.warning(
                            f"  ✗ TRUNCATED {qid}/{variant} | tokens={tokens_used} "
                            f"| no retry for non-V4 variants"
                        )

                return CallResult(
                    text=text,
                    finish_reason=finish_reason,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                )

            except Exception as exc:
                logging.warning(
                    f"  API error {qid}/{variant} attempt {attempt + 1}/{self.retries}: {exc}"
                )
                if attempt < self.retries - 1:
                    time.sleep(2 ** (attempt + 1))

        return CallResult(error=f"All {self.retries} retries failed")

    # ── internal dispatch ─────────────────────────────────────────────────

    def _dispatch(
        self, user_prompt: str, max_tokens: int
    ) -> tuple[str, str, int]:
        """Returns (text, finish_reason, tokens_used)."""
        if self._backend == "genai":
            return self._call_genai(user_prompt, max_tokens)
        return self._call_openai_compat(user_prompt, max_tokens)

    def _call_openai_compat(
        self, user_prompt: str, max_tokens: int
    ) -> tuple[str, str, int]:
        """OpenAI-compatible call — handles OpenRouter and OpenAI direct.

        Model quirks handled here:
          gpt-5.x / o1 / o3 — max_completion_tokens instead of max_tokens
        """
        mid = self.model_id.lower()
        uses_completion_tokens = "gpt-5" in mid or mid in ("o1", "o1-mini", "o1-preview", "o3", "o3-mini")

        kwargs: dict = dict(
            model=self.model_id,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
        )

        if uses_completion_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        resp          = self._client.chat.completions.create(**kwargs)
        text          = resp.choices[0].message.content or ""
        finish_reason = resp.choices[0].finish_reason or "stop"
        tokens_used   = resp.usage.total_tokens if resp.usage else 0
        return text, finish_reason, tokens_used

    def _call_genai(
        self, user_prompt: str, max_tokens: int
    ) -> tuple[str, str, int]:
        """Google GenAI call with primary/fallback client."""
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=max_tokens,
        )

        def _invoke(client) -> tuple[str, str, int]:
            resp          = client.models.generate_content(
                model=self.model_id,
                contents=[self.system_prompt, user_prompt],
                config=config,
            )
            text          = resp.text.strip()
            fr            = resp.candidates[0].finish_reason
            finish_reason = "length" if fr.name == "MAX_TOKENS" else "stop"
            tokens_used   = getattr(
                getattr(resp, "usage_metadata", None), "total_token_count", 0
            ) or 0
            return text, finish_reason, tokens_used

        try:
            return _invoke(self._client)
        except Exception as api_error:
            err = str(api_error).lower()
            if ("quota" in err or "exhausted" in err or "429" in err) and self._fallback:
                logging.info("  [INFO] Quota/rate-limit — switching to GenAI fallback key")
                return _invoke(self._fallback)
            raise

# ═══════════════════════════════════════════════════════════════════════════
# 0.  KNOWN SUBCATEGORIES  (strict allow-list)  +  ALIASES  +  FORMAT SUFFIXES
# ═══════════════════════════════════════════════════════════════════════════

#: Every CANONICAL subcategory key that the pipeline accepts.
#: Rows whose normalised subcat is NOT in this set are dropped at load time.
KNOWN_SUBCATS: Set[str] = {
    "det", "eig", "rank", "null", "nullity",
    "mult", "pow", "pow2", "vec", "matvec",
    "trans", "trace", "inv",
}

#: Maps any input subcategory string (full name OR short code) to its
#: canonical short code in KNOWN_SUBCATS.  The dataset can use either form
#: and rows are reclassified before the strict KNOWN_SUBCATS check.
#:
#: Notes
#: ─────
#: • 'matrix_power' is mapped to 'pow'.  If your dataset distinguishes A^2
#:   specifically with a 'matrix_power_2' / 'square' label, add the alias
#:   here pointing to 'pow2'.
#: • 'null' (null-space) and 'nullity' (its dimension) are kept distinct.
SUBCAT_ALIASES: Dict[str, str] = {
    # full names → canonical short codes
    "determinant":           "det",
    "eigenvalue":            "eig",
    "eigenvalues":           "eig",
    "multiplication":        "mult",
    "matrix_multiplication": "mult",
    "matrix-multiplication": "mult",
    "matrix_vector":         "matvec",
    "matrix-vector":         "matvec",
    "matrix_vector_product": "matvec",
    "transpose":             "trans",
    "matrix_power":          "pow",
    "matrix-power":          "pow",
    "power":                 "pow",
    "inverse":               "inv",
    "null_space":            "null",
    "nullspace":             "null",
    "vector":                "vec",
    # short codes (identity mappings for already-canonical inputs)
    "det":     "det",
    "eig":     "eig",
    "rank":    "rank",
    "null":    "null",
    "nullity": "nullity",
    "mult":    "mult",
    "pow":     "pow",
    "pow2":    "pow2",
    "vec":     "vec",
    "matvec":  "matvec",
    "trans":   "trans",
    "trace":   "trace",
    "inv":     "inv",
}

#: Strings that mean "missing".  Note that the bareword 'null' is NOT here:
#: 'null' is a legitimate subcategory key (null-space).  Treating it as a
#: sentinel was a pre-existing bug that silently dropped null-space rows.
SENTINEL_VALUES: Set[str] = {"", "nan", "none", "n/a", "na", "unknown"}

#: Format suffixes used to detect / construct fully-qualified question_ids.
#: Example fully-qualified ids:  C_4x4_det_latex, C_4x4_det_ascii, C_4x4_det_list.
KNOWN_FORMAT_SUFFIXES: Set[str] = {"latex", "ascii", "list"}

# Sanity: every alias value must be a real canonical subcat.
_bad_aliases = {v for v in SUBCAT_ALIASES.values() if v not in KNOWN_SUBCATS}
if _bad_aliases:                            # pragma: no cover
    raise RuntimeError(
        f"BUG: SUBCAT_ALIASES point to non-canonical values: {_bad_aliases}"
    )

# FIX-BUG-6: mutable module-level int so CLI override propagates everywhere.
_TAIL_TOKENS: int = 100


# ═══════════════════════════════════════════════════════════════════════════
# 1.  MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

MODELS: Dict[str, dict] = {
    "Qwen2.5-72B": {
        "model_id":    "qwen/qwen-2.5-72b-instruct",
        "api_base":    "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_tokens":  8192,
        "max_ceiling": 32768,
    },
    "Llama-3.3-70B": {
        "model_id":    "meta-llama/llama-3.3-70b-instruct",
        "api_base":    "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_tokens":  8192,
        "max_ceiling": 32768,
    },
    "GPT-4o": {
        "model_id":    "gpt-4o",
        "api_base":    "https://api.openai.com/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_tokens":  8192,
        "max_ceiling": 32768,
    },
    # FIX-BUG-3: corrected OpenRouter slug — hyphens not dots.
    "Claude-4.5-Sonnet": {
        "model_id":    "anthropic/claude-sonnet-4-5",
        "api_base":    "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_tokens":  8192,
        "max_ceiling": 32768,
    },
    "Mistral-Large": {
        "model_id":    "mistralai/mistral-large-2512",
        "api_base":    "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_tokens":  8192,
        "max_ceiling": 32768,
    },
    "DeepSeek-V3": {
        "model_id":    "deepseek/deepseek-v3.2",
        "api_base":    "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_tokens":  16384,
        "max_ceiling": 65536,
    },
    "GPT-5.2": {
        "model_id":    "gpt-5.2",
        "api_base":    "https://api.openai.com/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_tokens":  16384,
        "max_ceiling": 65536,
    },
    "Qwen3-235B": {
        "model_id":    "qwen/qwen3-235b-a22b-2507",
        "api_base":    "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_tokens":  16384,
        "max_ceiling": 65536,
    },
    "OpenAI-o1": {
        "model_id":    "o1",
        "api_base":    "https://api.openai.com/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_tokens":  8192,
        "max_ceiling": 32768,
    },
    "OpenAI-o1-OR": {
        "model_id":    "openai/o1",
        "api_base":    "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_tokens":  8192,
        "max_ceiling": 32768,
    },
    # FIX-BUG-4: api_base was empty string — set to Google GenAI REST endpoint.
    "Gemini-3.0-Pro": {
        "model_id":    "gemini-3.0-pro-preview",
        "api_base":    "https://generativelanguage.googleapis.com/v1beta",
        "api_key_env": "GEMINI_API_KEY",
        "min_tokens":  8192,
        "max_ceiling": 32768,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# 2.  SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

_BASE_SYSTEM = (
    "You are a precise mathematical assistant. "
    "Show all computation steps clearly. "
)

_SUBCAT_SYSTEM: Dict[str, str] = {
    "det":     _BASE_SYSTEM + "Always put your final numerical answer inside \\boxed{}.",
    "eig":     _BASE_SYSTEM + "Always put your final answers inside \\boxed{}.",
    "rank":    _BASE_SYSTEM + "Always put your final integer answer inside \\boxed{}.",
    "null":    _BASE_SYSTEM + "Always put your final integer answer inside \\boxed{}.",
    "nullity": _BASE_SYSTEM + "Always put your final integer answer inside \\boxed{}.",
    "mult":    _BASE_SYSTEM + "Always put your final matrix answer inside \\boxed{}.",
    "pow":     _BASE_SYSTEM + "Always put your final matrix answer inside \\boxed{}.",
    "pow2":    _BASE_SYSTEM + "Always put your final matrix answer inside \\boxed{}.",
    "vec":     _BASE_SYSTEM + "Always put your final vector answer inside \\boxed{}.",
    "matvec":  _BASE_SYSTEM + "Always put your final vector answer inside \\boxed{}.",
    "trans":   _BASE_SYSTEM + "Always put your final matrix answer inside \\boxed{}.",
    "trace":   _BASE_SYSTEM + "Always put your final numerical answer inside \\boxed{}.",
    "inv":     _BASE_SYSTEM + "Always put your final matrix answer inside \\boxed{}.",
}

# Verify compile-time consistency: every KNOWN_SUBCAT must have a prompt
_missing_prompts = KNOWN_SUBCATS - set(_SUBCAT_SYSTEM)
if _missing_prompts:                        # pragma: no cover
    raise RuntimeError(
        f"BUG: KNOWN_SUBCATS entries have no system prompt: {_missing_prompts}"
    )


def _normalise_subcat(raw) -> Optional[str]:
    """
    Strictly normalise a subcategory value.

    Resolution order:
        1. NaN / None        → None
        2. SENTINEL_VALUES   → None (e.g. "", "nan", "n/a")
        3. SUBCAT_ALIASES    → canonical short code (e.g. "determinant" → "det")
        4. unrecognised      → None

    Returns the canonical short code if the input maps to a key in
    KNOWN_SUBCATS, otherwise None.  Callers must decide what to do with
    None (the pipeline drops those rows).
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None

    cleaned = str(raw).strip().lower()

    if cleaned in SENTINEL_VALUES:
        return None

    canonical = SUBCAT_ALIASES.get(cleaned)
    if canonical is None or canonical not in KNOWN_SUBCATS:
        return None

    return canonical


def _system_for_subcat(subcat: str) -> str:
    """
    Return the system prompt for a KNOWN subcat key.
    Raises KeyError if an invalid key is passed — this is intentional so
    bad data never silently reaches the API.
    """
    try:
        return _SUBCAT_SYSTEM[subcat]
    except KeyError:
        raise KeyError(
            f"'{subcat}' is not a recognised subcategory. "
            f"Valid keys: {sorted(KNOWN_SUBCATS)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3.  ANSWER EXTRACTION  +  LAST-N-TOKEN TAIL  +  QID NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════

def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the LAST \\boxed{...} value from *text*.
    Handles arbitrarily nested braces.
    Returns None when no \\boxed{} is present.
    """
    if not text:
        return None

    positions = [m.start() for m in re.finditer(r'\\boxed\{', text)]
    if not positions:
        return None

    start = positions[-1] + len(r'\boxed{')
    depth, idx = 1, start
    while idx < len(text) and depth > 0:
        if text[idx] == '{':
            depth += 1
        elif text[idx] == '}':
            depth -= 1
        idx += 1

    extracted = text[start: idx - 1].strip() if depth == 0 else text[start:idx].strip()
    return extracted if extracted else None


# FIX-BUG-6: removed default arg that captured the module constant at
# definition time; callers must now pass n explicitly so that CLI overrides
# to _TAIL_TOKENS propagate correctly.
def last_n_tokens(text: str, n: int) -> str:
    """
    Return the last *n* whitespace-split tokens of *text* joined by spaces.

    This is a cheap approximation of "last N tokens" that avoids importing
    a full tokeniser.  Suitable for post-hoc inspection of model tails.
    Stored unconditionally — regardless of whether extraction succeeded.
    """
    if not text:
        return ""
    words = text.split()
    return " ".join(words[-n:]) if len(words) > n else " ".join(words)


def _qid_has_format_suffix(qid: str) -> bool:
    """Return True if *qid* already ends with one of the KNOWN_FORMAT_SUFFIXES."""
    qid_l = str(qid).strip().lower()
    return any(qid_l.endswith("_" + suf) for suf in KNOWN_FORMAT_SUFFIXES)


def _ensure_qid_has_format(qid: str, fmt: str) -> str:
    """
    Ensure *qid* ends with the appropriate format suffix.

    Examples
    ────────
        ("C_4x4_det",        "latex") → "C_4x4_det_latex"
        ("C_4x4_det",        "ascii") → "C_4x4_det_ascii"
        ("C_4x4_det",        "list")  → "C_4x4_det_list"
        ("C_4x4_det_latex",  "latex") → "C_4x4_det_latex"   (untouched)
        ("C_4x4_det_ascii",  "list")  → "C_4x4_det_ascii"   (already suffixed)

    If *fmt* is empty / unknown and *qid* is unsuffixed, the qid is returned
    as-is (the strict subcat / required-field checks elsewhere will catch it).
    """
    qid_clean = str(qid).strip()
    fmt_clean = str(fmt).strip().lower()

    if not qid_clean:
        return qid_clean

    if _qid_has_format_suffix(qid_clean):
        return qid_clean

    if fmt_clean:
        return f"{qid_clean}_{fmt_clean}"

    return qid_clean


# ═══════════════════════════════════════════════════════════════════════════
# 3b. STRUCTURED ANSWER EXTRACTION  (scalar / matrix from raw LaTeX)
# ═══════════════════════════════════════════════════════════════════════════
#
# Goal
# ────
# Convert raw LaTeX/text answers into normalised, comparable forms so that
# correctness checking is robust across formats.  Examples:
#
#   "0"                                       → scalar "0"
#   "-2"                                      → scalar "-2"
#   "\text{nullity}(A) = 0"                   → scalar "0"
#   "\det(A) = -2"                            → scalar "-2"
#   "\begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}"  → matrix [['1','2'],['3','4']]
#   "[[1, 2], [3, 4]]"                        → matrix [['1','2'],['3','4']]
#
# Cells are kept as STRINGS so we can preserve exact precision (e.g. '1/2').
# Numeric coercion happens only at comparison time inside _scalars_equal.
#
# These helpers are used both:
#   • on the ground truth ('answer_latex')          → ground_scalar / ground_matrix
#   • on the model's \boxed{} content               → extracted_scalar / extracted_matrix
# ---------------------------------------------------------------------------

# Markers that indicate "this string is a matrix, not a scalar".
_MATRIX_MARKERS = (
    r"\begin{matrix}", r"\begin{pmatrix}", r"\begin{bmatrix}",
    r"\begin{vmatrix}", r"\begin{Vmatrix}", r"\begin{smallmatrix}",
)


def extract_scalar(text) -> Optional[str]:
    """
    Try to extract a single scalar value from a LaTeX or plain-text answer.

    Returns the scalar as a STRING (preserving the original numeric token,
    so '0.5' stays '0.5' and '-2' stays '-2').  Returns None if the input
    is empty, looks matrix-shaped, or contains no extractable number.

    Strategy
    ────────
    1. Reject obvious matrix shapes (LaTeX pmatrix/bmatrix/…, or '[[...]]').
    2. If '=' appears, take the right-hand side (handles "\\text{rank}(A) = 3").
    3. Strip surrounding $, {, } and whitespace.
    4. Try a direct float() — if it parses, return as-is.
    5. Otherwise scan for a single numeric token; if exactly one is found,
       return it.  Multiple numbers → ambiguous → None.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    low = s.lower()
    if any(m in s for m in _MATRIX_MARKERS) or "[[" in s or "]]" in s:
        return None
    if "\\\\" in s or "&" in s:        # row separator / col separator → matrix-y
        return None

    # Take the RHS of the LAST '=' if present
    if "=" in s:
        s = s.rsplit("=", 1)[1]

    # Trim wrappers like $...$, \(...\), {...}
    s = s.strip()
    s = s.strip("$").strip()
    s = re.sub(r"^\\\(|\\\)$", "", s).strip()
    s = s.strip("{}").strip()

    # Direct numeric parse
    try:
        float(s)
        return s
    except (ValueError, TypeError):
        pass

    # Single-number scan
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if len(nums) == 1:
        return nums[0]
    return None


def extract_matrix(text) -> Optional[List[List[str]]]:
    """
    Extract a 2D matrix from a LaTeX or Python-list representation.

    Returns a list of rows where each cell is the raw STRING (so we can
    compare numerically with tolerance OR fall back to text).  Returns
    None if no matrix shape is detected.

    Supported shapes
    ────────────────
        \\begin{pmatrix}1 & 2 \\\\ 3 & 4\\end{pmatrix}  (also bmatrix/vmatrix/…)
        \\begin{bmatrix}1\\\\2\\\\3\\end{bmatrix}        (column vector)
        [[1, 2], [3, 4]]                                (Python list of lists)
        [1, 2, 3]                                       (treated as column vector)
    """
    if text is None:
        return None
    s = str(text)
    if not s.strip():
        return None

    # ── LaTeX matrix environments ────────────────────────────────────────
    m = re.search(
        r"\\begin\{([bpvV]?)matrix\}(.*?)\\end\{\1matrix\}",
        s, re.DOTALL,
    )
    if m:
        body = m.group(2)
        rows_raw = re.split(r"\\\\", body)
        result: List[List[str]] = []
        for r in rows_raw:
            r = r.strip()
            if not r:
                continue
            cells = [c.strip() for c in r.split("&")]
            result.append(cells)
        return result if result else None

    # ── Python-style nested list ─────────────────────────────────────────
    m = re.search(r"\[\s*\[.*?\]\s*\]", s, re.DOTALL)
    if m:
        try:
            import ast
            val = ast.literal_eval(m.group(0))
            if isinstance(val, list) and val and all(isinstance(r, list) for r in val):
                return [[str(c) for c in row] for row in val]
        except (ValueError, SyntaxError):
            pass

    # ── flat list as column vector  e.g. "[1, 2, 3]" ────────────────────
    m = re.search(r"\[[^\[\]]+\]", s)
    if m:
        try:
            import ast
            val = ast.literal_eval(m.group(0))
            if (
                isinstance(val, list) and val
                and all(isinstance(c, (int, float)) for c in val)
            ):
                return [[str(c)] for c in val]
        except (ValueError, SyntaxError):
            pass

    return None


def _scalars_equal(a, b, tol: float = 1e-9) -> bool:
    """Numeric-tolerant equality with a string-equality fallback."""
    try:
        return abs(float(a) - float(b)) <= tol
    except (ValueError, TypeError):
        return _normalise_answer(a) == _normalise_answer(b)


def _matrices_equal(a: List[List[str]], b: List[List[str]]) -> bool:
    """Shape-then-cell equality using _scalars_equal per cell."""
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for ca, cb in zip(ra, rb):
            if not _scalars_equal(ca, cb):
                return False
    return True


def _format_matrix(matrix: Optional[List[List[str]]]) -> str:
    """Stable string repr for CSV storage; '' for None."""
    if not matrix:
        return ""
    # Compact JSON-ish: [["1","2"],["3","4"]]  — keeps quotes around cells
    # so spreadsheet readers don't mangle expressions like '1/2'.
    return "[" + ",".join(
        "[" + ",".join(f'"{c}"' for c in row) + "]" for row in matrix
    ) + "]"


# ═══════════════════════════════════════════════════════════════════════════
# 4.  GROUND-TRUTH COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def _normalise_answer(val) -> str:
    if val is None:
        return ""
    return re.sub(r'\s+', ' ', str(val).strip().lower())


def answers_match(extracted: Optional[str], answer_latex: str) -> Optional[bool]:
    """
    Multi-strategy correctness check.

    Returns
    ───────
        True  – matched
        False – did not match
        None  – extracted is None (no \\boxed{} found in response)

    Order of attempts
    ─────────────────
    1. SCALAR  : if both ground truth and extracted parse to a single number,
                 compare numerically with tolerance (so "-2" == "-2.0").
    2. MATRIX  : if both parse to a 2D matrix (LaTeX pmatrix/bmatrix/… or
                 nested list), compare cell-by-cell with _scalars_equal.
    3. STRING  : fallback — whitespace-collapsed, case-insensitive equality.

    The structured layers handle the common LaTeX-vs-list-vs-text mismatches
    that pure string equality would miss; the string fallback preserves
    backwards compatibility for unusual answers (fractions, symbolic, …).
    """
    if extracted is None:
        return None

    # 1. scalar
    g_scalar = extract_scalar(answer_latex)
    e_scalar = extract_scalar(extracted)
    if g_scalar is not None and e_scalar is not None:
        return _scalars_equal(e_scalar, g_scalar)

    # 2. matrix
    g_matrix = extract_matrix(answer_latex)
    e_matrix = extract_matrix(extracted)
    if g_matrix is not None and e_matrix is not None:
        return _matrices_equal(e_matrix, g_matrix)

    # 3. string fallback
    return _normalise_answer(extracted) == _normalise_answer(answer_latex)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  CSV I/O
# ═══════════════════════════════════════════════════════════════════════════
#
# Column-order note:
#   answer_latex      ← raw ground-truth value (LaTeX, list, …)
#   ground_scalar     ← scalar pulled out of answer_latex (e.g. "0", "-2")
#   ground_matrix     ← matrix pulled out of answer_latex (e.g. [["1","2"],…])
#   extracted_answer  ← raw \boxed{...} content from the model
#   extracted_scalar  ← scalar pulled out of extracted_answer
#   extracted_matrix  ← matrix pulled out of extracted_answer
#   is_correct        ← True / False / blank match flag
#
# Each "raw" column sits beside its structured sidecars, so a reviewer can
# eyeball ground-vs-extracted scalars / matrices without parsing LaTeX.
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS: List[str] = [
    "question_id",
    "format_type",
    "subcategory",
    "model_name",
    "instruction",
    "response",
    "response_tail",
    "answer_latex",        # raw ground truth
    "ground_scalar",       # scalar form of ground truth
    "ground_matrix",       # matrix form of ground truth
    "extracted_answer",    # raw boxed output from the model
    "extracted_scalar",    # scalar form of boxed output
    "extracted_matrix",    # matrix form of boxed output
    "is_correct",          # boolean correctness flag
    "finish_reason",
    "tokens_used",
    "latency_ms",
    "timestamp",
    "error",
]


def _ensure_output_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def _ensure_csv(path: str) -> None:
    """Create the CSV with a header row if it does not already exist."""
    _ensure_output_dir(path)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS).writeheader()


def _serialise_correct(value) -> str:
    """Bool / None → unambiguous CSV string."""
    if value is True:
        return "True"
    if value is False:
        return "False"
    return ""


def _parse_correct(value) -> Optional[bool]:
    s = str(value).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return None


def _append_row(row: dict, path: str, lock: threading.Lock) -> None:
    safe = dict(row)
    safe["is_correct"] = _serialise_correct(safe.get("is_correct"))
    # Sanitise any non-UTF-8 bytes that may appear in model responses
    for key in ("response", "response_tail", "error"):
        if isinstance(safe.get(key), str):
            safe[key] = safe[key].encode("utf-8", errors="replace").decode("utf-8")
    with lock:
        with open(path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(
                fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore"
            ).writerow(safe)


def load_completed_triples(path: str) -> Set[tuple]:
    """Return set of (question_id, format_type, model_name) already written."""
    if not os.path.exists(path):
        return set()
    done: Set[tuple] = set()
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                qid = row.get("question_id", "").strip()
                fmt = row.get("format_type", "").strip()
                mdl = row.get("model_name", "").strip()
                if qid and fmt and mdl:
                    done.add((qid, fmt, mdl))
    except Exception as exc:
        logging.warning(f"  Partial resume state loaded — {exc}")
    return done


# ═══════════════════════════════════════════════════════════════════════════
# 6.  DATASET LOADING  +  STRICT SUBCAT FILTERING  +  COLUMN ALIASES
# ═══════════════════════════════════════════════════════════════════════════

def _apply_input_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise heterogeneous input schemas into the canonical column names
    expected by the rest of the pipeline.

    Rules
    ─────
    • 'id'                   → 'question_id'    (only if 'question_id' absent)
    • 'format'               → 'format_type'    (only if 'format_type' absent)
    • 'instruction' missing  → built from 'problem_text' + 'problem_representation'
                               (also tolerates the typo 'problem_represntation')

    The function mutates and returns *df* for convenience.
    """
    # ── question_id alias ────────────────────────────────────────────────
    if "question_id" not in df.columns and "id" in df.columns:
        df["question_id"] = df["id"]
        logging.info("  Schema alias: 'id' → 'question_id'")

    # ── format_type alias ────────────────────────────────────────────────
    if "format_type" not in df.columns and "format" in df.columns:
        df["format_type"] = df["format"]
        logging.info("  Schema alias: 'format' → 'format_type'")

    # ── instruction synthesised from problem_text + problem_representation ─
    if "instruction" not in df.columns:
        # tolerate the misspelling 'problem_represntation'
        repr_col = None
        if "problem_representation" in df.columns:
            repr_col = "problem_representation"
        elif "problem_represntation" in df.columns:
            repr_col = "problem_represntation"

        if "problem_text" in df.columns and repr_col is not None:
            pt = df["problem_text"].fillna("").astype(str).str.strip()
            pr = df[repr_col].fillna("").astype(str).str.strip()
            # Join with a blank line; collapse if either part is empty.
            joined = (pt + "\n\n" + pr).str.strip()
            df["instruction"] = joined
            logging.info(
                f"  Schema alias: built 'instruction' from "
                f"'problem_text' + '{repr_col}'"
            )

    return df


def load_and_validate_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the expanded CSV and apply STRICT subcategory filtering.

    Rules
    ─────
    • 'id' / 'question_id' and 'format' / 'format_type' are accepted aliases.
    • If 'instruction' is missing, it is built from
      'problem_text' + 'problem_representation'.
    • Rows where 'subcategory' is null, NaN, empty, a sentinel string
      ("nan", "none", "null", …), or not in KNOWN_SUBCATS are DROPPED.
    • A detailed warning is emitted showing counts per rejection reason.
    • 'question_id' is normalised to include a format suffix
      (_latex / _ascii / _list) when missing one, e.g.
          C_4x4_det + latex → C_4x4_det_latex.
    • The returned DataFrame is guaranteed to have only rows whose
      normalised 'subcategory' is in KNOWN_SUBCATS.

    Raises
    ──────
    FileNotFoundError  – input file absent
    ValueError         – required columns missing, or zero valid rows remain
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    # Stop pandas from converting string sentinels like "null" / "NaN" /
    # "None" to actual NaN — those are kept as raw strings so our own
    # SENTINEL_VALUES / SUBCAT_ALIASES logic can handle them per column.
    # Without this, the bareword 'null' (a legitimate null-space subcat key)
    # would be silently dropped before _classify ever saw it.
    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )
    df.columns = df.columns.str.strip().str.lower()

    # ── apply schema aliases BEFORE the required-columns check ───────────
    df = _apply_input_aliases(df)

    required = {"question_id", "format_type", "instruction", "answer_latex"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Expanded CSV is missing required columns: {missing}.  "
            f"Accepted aliases: id↔question_id, format↔format_type, "
            f"instruction↔(problem_text + problem_representation)."
        )

    if "subcategory" not in df.columns:
        logging.warning(
            "  'subcategory' column absent from input CSV — "
            "ALL rows will be dropped (strict mode)."
        )
        df["subcategory"] = float("nan")

    total_rows = len(df)

    reasons: Dict[str, int] = {
        "null/NaN":       0,
        "empty/sentinel": 0,
        "unrecognised":   0,
    }
    aliased: Dict[str, str] = {}   # raw → canonical, for logging examples

    # FIX-BUG-2: use pd.isna() for reliable NaN detection regardless of
    # dtype; avoids the broken isinstance(raw, float) outer guard that
    # caused non-float NaN values to fall through to the sentinel branch.
    #
    # Alias-aware classification: full names (e.g. "determinant",
    # "eigenvalues", "transpose") are mapped to canonical short codes
    # via SUBCAT_ALIASES BEFORE the KNOWN_SUBCATS membership check.
    def _classify(raw) -> Optional[str]:
        """Map raw cell → canonical short code or None (recording reason)."""
        if pd.isna(raw):
            reasons["null/NaN"] += 1
            return None

        cleaned = str(raw).strip().lower()

        if cleaned in SENTINEL_VALUES:
            reasons["empty/sentinel"] += 1
            return None

        canonical = SUBCAT_ALIASES.get(cleaned)
        if canonical is None or canonical not in KNOWN_SUBCATS:
            reasons["unrecognised"] += 1
            return None

        if cleaned != canonical:
            aliased.setdefault(cleaned, canonical)

        return canonical

    df["_subcat_clean"] = df["subcategory"].apply(_classify)

    valid_mask = df["_subcat_clean"].notna()
    dropped_df = df[~valid_mask]
    valid_df   = df[valid_mask].copy()

    if aliased:
        sample = list(aliased.items())[:6]
        logging.info(f"  Subcat aliases applied (sample): {sample}")

    n_dropped = len(dropped_df)
    if n_dropped:
        logging.warning(
            f"  STRICT SUBCAT FILTER: dropped {n_dropped}/{total_rows} rows  "
            f"(null/NaN={reasons['null/NaN']}, "
            f"empty/sentinel={reasons['empty/sentinel']}, "
            f"unrecognised={reasons['unrecognised']})"
        )
        bad_vals = (
            dropped_df["subcategory"]
            .dropna()
            .unique()
            .tolist()[:10]
        )
        if bad_vals:
            logging.warning(f"  Sample of dropped subcat values: {bad_vals}")

    if valid_df.empty:
        raise ValueError(
            "No rows remain after strict subcategory filtering. "
            f"Valid subcategories are: {sorted(KNOWN_SUBCATS)}"
        )

    valid_df["subcategory"] = valid_df["_subcat_clean"]
    valid_df = valid_df.drop(columns=["_subcat_clean"])

    for col in ["question_id", "format_type", "instruction", "answer_latex"]:
        valid_df[col] = valid_df[col].fillna("").astype(str).str.strip()

    # ── normalise question_id to include format suffix ───────────────────
    pre_qids = valid_df["question_id"].copy()
    valid_df["question_id"] = valid_df.apply(
        lambda r: _ensure_qid_has_format(r["question_id"], r["format_type"]),
        axis=1,
    )
    n_qid_changed = int((pre_qids != valid_df["question_id"]).sum())
    if n_qid_changed:
        sample_changes = list(
            zip(pre_qids[pre_qids != valid_df["question_id"]].head(3).tolist(),
                valid_df["question_id"][pre_qids != valid_df["question_id"]].head(3).tolist())
        )
        logging.info(
            f"  Question-id format suffix appended on {n_qid_changed} rows "
            f"(sample: {sample_changes})"
        )

    logging.info(
        f"  Dataset loaded: {total_rows} total rows → "
        f"{len(valid_df)} valid after strict subcat filter"
    )
    return valid_df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 7.  TASK BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_tasks(
    df: pd.DataFrame,
    model_name: str,
    completed: Set[tuple],
) -> List[dict]:
    """
    Convert validated DataFrame rows into executor task dicts.

    Pre-conditions (guaranteed by load_and_validate_dataset):
      • df["subcategory"] is always a clean key in KNOWN_SUBCATS.
      • df["question_id"] always carries a format suffix (_latex/_ascii/_list)
        when format_type was a known suffix value.
      • No NaN in question_id / format_type / instruction / answer_latex.
    """
    tasks: List[dict] = []
    skipped = 0

    for pos, (_, row) in enumerate(df.iterrows()):
        qid          = row["question_id"]
        fmt          = row["format_type"]
        subcat       = row["subcategory"]
        instruction  = row["instruction"]
        answer_latex = row["answer_latex"]

        if not qid or not fmt or not subcat or not instruction:
            logging.warning(
                f"  Row {pos}: skipping — empty required field "
                f"(qid={qid!r}, fmt={fmt!r}, subcat={subcat!r})"
            )
            skipped += 1
            continue

        if (qid, fmt, model_name) in completed:
            skipped += 1
            continue

        tasks.append({
            "task_idx":      pos,
            "question_id":   qid,
            "format_type":   fmt,
            "subcategory":   subcat,
            "instruction":   instruction,
            "answer_latex":  answer_latex,
            "system_prompt": _system_for_subcat(subcat),
        })

    if skipped:
        logging.info(f"  Skipped {skipped} rows (completed or empty fields).")

    return tasks


# ═══════════════════════════════════════════════════════════════════════════
# 8.  PARALLEL EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════

class FormatInferenceExecutor:
    """
    Thread-pool executor for one model / subcategory group.

    Heap-ordered writes ensure the output CSV row order mirrors task order
    even when tasks complete out of sequence.
    """

    PROVIDER_LIMITS: Dict[str, int] = {
        "openrouter": 5,
        "openai":     3,
        "genai":      4,
        "default":    4,
    }

    def __init__(
        self,
        client,
        model_name: str,
        output_csv: str,
        csv_lock: threading.Lock,
        max_workers: int = 6,
        rate_limit: float = 1.0,
        max_tokens: int = 8192,
        ceiling: int = 16384,
        task_timeout: int = 600,
    ):
        self.client       = client
        self.model_name   = model_name
        self.output_csv   = output_csv
        self.csv_lock     = csv_lock
        self.max_workers  = max_workers
        self.rate_limit   = rate_limit
        self.max_tokens   = max_tokens
        self.ceiling      = ceiling
        self.task_timeout = task_timeout

        self.call_count   = 0
        self._call_lock   = threading.Lock()
        self._buf_lock    = threading.Lock()
        self._heap: list  = []
        self._next_write  = 0          # FIX-BUG-5: replaces _min_pending + _written set

        provider  = self._detect_provider(client.model_id)
        sem_limit = min(
            max_workers,
            self.PROVIDER_LIMITS.get(provider, self.PROVIDER_LIMITS["default"]),
        )
        self._sem = threading.Semaphore(sem_limit)
        logging.info(
            f"    [{model_name}] provider={provider}  "
            f"concurrency={sem_limit}/{max_workers}"
        )

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _detect_provider(model_id: str) -> str:
        mid = str(model_id).lower()
        if "gemini" in mid:
            return "genai"
        if any(x in mid for x in ("gpt", "/o1", "o1-", "o3")):
            return "openai"
        return "openrouter"

    # FIX-BUG-5: rewritten heap flush.
    # Old code used `heap[0][0] <= _min_pending` which only ever flushed
    # task_idx=0 immediately; all others stayed buffered until _drain_heap.
    # The correct logic is: flush the heap while the smallest buffered
    # task_idx equals the next expected write index (_next_write).
    def _flush_heap(self) -> None:
        """Flush heap entries that arrive in order (call with _buf_lock held)."""
        while self._heap and self._heap[0][0] == self._next_write:
            _, row = heapq.heappop(self._heap)
            _append_row(row, self.output_csv, self.csv_lock)
            self._next_write += 1

    def _drain_heap(self) -> None:
        """Unconditional drain — call after thread pool exits."""
        with self._buf_lock:
            while self._heap:
                _, row = heapq.heappop(self._heap)
                _append_row(row, self.output_csv, self.csv_lock)

    # ── single task ──────────────────────────────────────────────────────

    def _run_task(self, task: dict) -> None:
        task_idx     = task["task_idx"]
        question_id  = task["question_id"]
        format_type  = task["format_type"]
        subcat       = task["subcategory"]
        instruction  = task["instruction"]
        answer_latex = task["answer_latex"]

        response_text = ""
        finish_reason = ""
        tokens_used   = 0
        latency_ms    = 0
        error_msg     = ""
        extracted     = None
        is_correct    = None

        try:
            with self._sem:
                time.sleep(self.rate_limit)
                cr = self.client.call(
                    user_prompt=instruction,
                    qid=question_id,
                    variant=f"format_{format_type}",
                    max_tokens=self.max_tokens,
                    ceiling=self.ceiling,
                )

            with self._call_lock:
                self.call_count += 1

            response_text = cr.text or ""
            finish_reason = cr.finish_reason or ""
            tokens_used   = cr.tokens_used or 0
            latency_ms    = cr.latency_ms or 0
            error_msg     = cr.error or ""

            # ── boxed extraction + structured forms + ground-truth match ──
            extracted  = extract_boxed_answer(response_text)
            is_correct = answers_match(extracted, answer_latex)

        except Exception as exc:
            error_msg = str(exc)
            finish_reason = "error"
            logging.error(
                f"  Task failed  {question_id}/{format_type}: {exc}",
                exc_info=True,
            )

        # Structured sidecars — computed regardless of whether the API call
        # succeeded.  When extraction failed, extracted is None and the
        # extracted_* sidecars come out empty, while the ground_* sidecars
        # are still populated from answer_latex for inspection.
        g_scalar = extract_scalar(answer_latex) or ""
        g_matrix = _format_matrix(extract_matrix(answer_latex))
        if extracted is not None:
            e_scalar = extract_scalar(extracted) or ""
            e_matrix = _format_matrix(extract_matrix(extracted))
        else:
            e_scalar = ""
            e_matrix = ""

        # FIX-BUG-6: pass _TAIL_TOKENS explicitly so CLI override propagates.
        # Column order mirrors OUTPUT_COLUMNS:
        #   answer_latex → ground_scalar → ground_matrix
        #   → extracted_answer → extracted_scalar → extracted_matrix
        #   → is_correct
        row: dict = {
            "question_id":      question_id,
            "format_type":      format_type,
            "subcategory":      subcat,
            "model_name":       self.model_name,
            "instruction":      instruction,
            "response":         response_text,
            "response_tail":    last_n_tokens(response_text, _TAIL_TOKENS),
            "answer_latex":     answer_latex,        # raw ground truth
            "ground_scalar":    g_scalar,            # scalar form of GT
            "ground_matrix":    g_matrix,            # matrix form of GT
            "extracted_answer": extracted or "",     # raw boxed output
            "extracted_scalar": e_scalar,            # scalar form of boxed
            "extracted_matrix": e_matrix,            # matrix form of boxed
            "is_correct":       is_correct,          # boolean (or None)
            "finish_reason":    finish_reason,
            "tokens_used":      tokens_used,
            "latency_ms":       latency_ms,
            "timestamp":        datetime.now().isoformat(),
            "error":            error_msg,
        }

        with self._buf_lock:
            heapq.heappush(self._heap, (task_idx, row))
            self._flush_heap()

        logging.info(
            f"  ✓ {self.model_name}  [{subcat}]  {question_id}  "
            f"fmt={format_type}  is_correct={is_correct}  "
            f"tail_words={len(response_text.split())}"
        )

    # ── public entry point ────────────────────────────────────────────────

    def run(self, tasks: List[dict], pbar=None) -> None:
        if not tasks:
            return
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._run_task, t): t for t in tasks}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    fut.result(timeout=self.task_timeout)
                except FutureTimeoutError:
                    logging.error(
                        f"  TIMEOUT  {t['question_id']}/{t['format_type']}"
                    )
                except Exception as exc:
                    logging.error(
                        f"  ERROR    {t['question_id']}/{t['format_type']}: {exc}"
                    )
                finally:
                    if pbar:
                        pbar.update(1)

        self._drain_heap()


# ═══════════════════════════════════════════════════════════════════════════
# 9.  MODEL RUNNER  (one model, all subcats)
# ═══════════════════════════════════════════════════════════════════════════

def run_model(
    *,
    model_name: str,
    model_cfg: dict,
    tasks: List[dict],
    output_path: str,
    csv_lock: threading.Lock,
    args: argparse.Namespace,
    InferenceClient,                # type: ignore[type-arg]
) -> int:
    """
    Execute all tasks for *model_name*, grouped by subcategory so each group
    gets the correct system prompt.  Returns total API calls made.
    """
    if not tasks:
        logging.info("  Nothing to do (all complete or zero tasks).")
        return 0

    min_tokens  = model_cfg.get("min_tokens", 8192)
    max_tokens  = args.max_tokens or min_tokens
    raw_ceiling = max_tokens * 3
    max_ceiling = model_cfg.get("max_ceiling", 65536)
    ceiling     = min(raw_ceiling, max_ceiling)

    logging.info(
        f"  Token budget: max_tokens={max_tokens}  "
        f"ceiling={ceiling} (model cap={max_ceiling})"
    )

    groups: Dict[str, List[dict]] = {}
    for t in tasks:
        groups.setdefault(t["subcategory"], []).append(t)

    total_calls = 0

    for subcat, sc_tasks in sorted(groups.items()):
        system_prompt = _system_for_subcat(subcat)
        logging.info(f"  ├─ subcat '{subcat}': {len(sc_tasks)} tasks")

        client = InferenceClient(
            model_cfg=model_cfg,
            system_prompt=system_prompt,
            dry_run=False,
        )
        executor = FormatInferenceExecutor(
            client=client,
            model_name=model_name,
            output_csv=output_path,
            csv_lock=csv_lock,
            max_workers=args.max_workers,
            rate_limit=args.rate_limit,
            max_tokens=max_tokens,
            ceiling=ceiling,
        )
        pbar = tqdm(
            total=len(sc_tasks),
            desc=f"{model_name}/{subcat}",
            unit="call",
        )
        executor.run(sc_tasks, pbar=pbar)
        pbar.close()
        total_calls += executor.call_count

    return total_calls


# ═══════════════════════════════════════════════════════════════════════════
# 10. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace) -> None:

    if args.models:
        unknown = [m for m in args.models if m not in MODELS]
        if unknown:
            raise ValueError(
                f"Unknown model(s): {unknown}. "
                f"Available: {list(MODELS.keys())}"
            )
        model_keys = args.models
    else:
        model_keys = list(MODELS.keys())

    if not args.dry_run:
        missing_keys = [
            (m, MODELS[m]["api_key_env"])
            for m in model_keys
            if MODELS[m].get("api_key_env")
            and not os.environ.get(MODELS[m]["api_key_env"])
        ]
        if missing_keys:
            lines = "\n".join(f"  {m}: ${env}" for m, env in missing_keys)
            raise EnvironmentError(
                "Missing API keys:\n" + lines +
                "\nExport them or add to .env"
            )

    df = load_and_validate_dataset(args.input)

    _ensure_output_dir(args.output)
    _ensure_csv(args.output)
    csv_lock = threading.Lock()

    completed: Set[tuple] = set()
    if args.resume:
        completed = load_completed_triples(args.output)
        logging.info(f"  Resume: {len(completed):,} triples already done")

    total_written = 0

    for model_name in model_keys:
        model_cfg = MODELS[model_name].copy()

        logging.info(f"\n{'═'*62}")
        logging.info(f"  Model : {model_name}  ({model_cfg['model_id']})")
        logging.info(f"{'═'*62}")

        tasks = build_tasks(df, model_name, completed)
        logging.info(f"  Tasks pending : {len(tasks):,}")

        if args.dry_run:
            _dry_run_preview(model_name, tasks)
            continue

        total_written += run_model(
            model_name=model_name,
            model_cfg=model_cfg,
            tasks=tasks,
            output_path=args.output,
            csv_lock=csv_lock,
            args=args,
            InferenceClient=InferenceClient,
        )

    _print_summary(args.output, model_keys)
    logging.info(f"\n  Master CSV    : {args.output}")
    logging.info(f"  New API calls : {total_written:,}")


# ═══════════════════════════════════════════════════════════════════════════
# 11. HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _dry_run_preview(model_name: str, tasks: List[dict], n: int = 3) -> None:
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  DRY RUN · {model_name} · {min(n, len(tasks))}/{len(tasks)} tasks shown")
    print(sep)
    for t in tasks[:n]:
        # FIX-BUG-6: pass n explicitly.
        tail = last_n_tokens(t["instruction"], 20)
        print(f"\n  question_id   : {t['question_id']}")
        print(f"  format_type   : {t['format_type']}")
        print(f"  subcategory   : {t['subcategory']}")
        print(f"  system_prompt : {t['system_prompt'][:80]}…")
        print(f"  instruction   : …{tail}")
    if not tasks:
        print("  (no tasks to preview)")
    print(f"{sep}\n")


def _print_summary(output_path: str, model_keys: List[str]) -> None:
    """Print per-model × per-format × per-subcat accuracy table."""
    if not os.path.exists(output_path):
        return
    try:
        df = pd.read_csv(output_path, dtype=str)
    except Exception as exc:
        logging.warning(f"  Cannot read output for summary: {exc}")
        return
    if df.empty:
        return

    # Tolerate legacy CSVs that still use the old 'correct' column name.
    if "is_correct" in df.columns:
        correct_src = df["is_correct"]
    elif "correct" in df.columns:
        correct_src = df["correct"]
    else:
        correct_src = pd.Series([None] * len(df), index=df.index)
    df["_correct"] = correct_src.apply(_parse_correct)

    w = 62
    print(f"\n{'='*w}")
    print("  RESULTS SUMMARY")
    print(f"{'='*w}")
    print(
        f"  {'Model':<20} {'Subcat':<9} {'Format':<8} "
        f"{'OK':>6} {'N':>6} {'Acc':>8}"
    )
    print(f"  {'─'*(w-2)}")

    for m in model_keys:
        msub = df[df["model_name"] == m]
        if msub.empty:
            continue
        for sc in sorted(msub["subcategory"].dropna().unique()):
            ssub = msub[msub["subcategory"] == sc]
            for fmt in sorted(ssub["format_type"].dropna().unique()):
                fsub  = ssub[ssub["format_type"] == fmt]
                total = len(fsub)
                # FIX-BUG-7: fillna(False) before summing to handle None/pd.NA
                # correctly; avoids the == True comparison on a nullable series.
                correct = int(fsub["_correct"].fillna(False).sum())
                pct     = 100.0 * correct / total if total else 0.0
                print(
                    f"  {m:<20} {sc:<9} {fmt:<8} "
                    f"{correct:>6} {total:>6} {pct:>7.1f}%"
                )

    print(f"{'='*w}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 12. CLI
# ═══════════════════════════════════════════════════════════════════════════

def _setup_logging(args: argparse.Namespace) -> None:
    handlers: list = [logging.StreamHandler()]
    if args.log_file:
        log_dir = os.path.dirname(os.path.abspath(args.log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def main() -> None:
    # FIX-BUG-1: declare global FIRST so the assignment lower in this
    # function (after argparse) takes effect — Python forbids 'global X'
    # to appear after any reference to X within the same function body.
    global _TAIL_TOKENS

    parser = argparse.ArgumentParser(
        description="LinAlg-Bench · Format Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python format_inference_pipeline.py \\
      --input  results/big_expanded_dataset.csv \\
      --output results/master_responses.csv

  python format_inference_pipeline.py \\
      --input  results/big_expanded_dataset.csv \\
      --output results/master_responses.csv \\
      --models GPT-4o DeepSeek-V3 --max-workers 4

  python format_inference_pipeline.py \\
      --input  results/big_expanded_dataset.csv \\
      --output results/master_responses.csv \\
      --resume --dry-run
        """,
    )

    parser.add_argument("--input",       required=True,
                        help="Expanded dataset CSV (from expand_dataset.py)")
    parser.add_argument("--output",      required=True,
                        help="Master output CSV (appended if exists)")
    parser.add_argument("--models",      nargs="+", default=None,
                        choices=list(MODELS.keys()),
                        help="Models to run (default: all)")
    # FIX-BUG-1: single definition of --tail-tokens (duplicate removed).
    parser.add_argument("--tail-tokens", type=int, default=_TAIL_TOKENS,
                        help=f"Words to keep in response_tail (default: {_TAIL_TOKENS})")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip already-completed (qid, fmt, model) triples")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Preview prompts only — no API calls")
    parser.add_argument("--max-workers", type=int,   default=6,
                        help="Parallel workers per model (default: 6)")
    parser.add_argument("--rate-limit",  type=float, default=1.0,
                        help="Seconds sleep per worker before each call (default: 1.0)")
    parser.add_argument("--max-tokens",  type=int,   default=None,
                        help="Token budget override (default: model min_tokens)")
    parser.add_argument("--log-file",    default=None,
                        help="Optional log file path")

    args = parser.parse_args()

    # FIX-BUG-1: apply CLI override to the module-level constant so that
    # last_n_tokens() calls throughout the pipeline use the correct value.
    _TAIL_TOKENS = args.tail_tokens

    _setup_logging(args)

    logging.info("=" * 62)
    logging.info("  LinAlg-Bench · Format Inference Pipeline")
    logging.info(f"  Input        : {args.input}")
    logging.info(f"  Output       : {args.output}")
    logging.info(f"  Models       : {args.models or 'ALL'}")
    logging.info(f"  Mode         : {'DRY RUN' if args.dry_run else 'LIVE'}")
    logging.info(f"  Resume       : {args.resume}")
    logging.info(f"  Tail tokens  : {args.tail_tokens}")
    logging.info(f"  Known subcats: {sorted(KNOWN_SUBCATS)}")
    logging.info(f"  Known formats: {sorted(KNOWN_FORMAT_SUFFIXES)}")
    logging.info("=" * 62)

    try:
        run_pipeline(args)
    except Exception as exc:
        logging.exception(f"FATAL: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()