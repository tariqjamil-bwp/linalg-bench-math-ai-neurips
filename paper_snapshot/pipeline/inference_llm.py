#!/usr/bin/env python3
"""
inference_llm.py — LLM transport layer for inference scripts.

Provides unified inference client that auto-detects backend and handles common
patterns: retry logic, token step-up on truncation, loop detection, result
formatting. Abstraction layer shields pipeline scripts from API differences.

PUBLIC INTERFACE
────────────────
    client = InferenceClient(model_cfg, system_prompt, dry_run=False)
    result = client.call(user_prompt, qid, variant, max_tokens, ceiling)
    # Returns: CallResult dataclass with text, finish_reason, tokens_used, latency_ms, error

BACKEND AUTO-DETECTION
──────────────────────
    Backend inferred from model_cfg["api_base"]:
      *.openai.com    → OpenAI (GPT-4o, GPT-5.2, o1)
      openrouter.ai   → OpenRouter (Qwen, Llama, Claude, DeepSeek, Mistral)
      (no api_base)   → Google GenAI (Gemini models)

    API keys loaded from environment:
      - model_cfg["api_key_env"] specifies which env var to read
      - For GenAI: uses GEMINI_API_KEY

FEATURES
─────────────────────────────
    Retry policy:
      3 attempts with exponential backoff (2s, 4s) on transient API errors.
      Fatal errors (auth, invalid model) raised immediately.

    Result formatting:
      Trims response, records finish_reason and token usage, measures latency.
      Handles backend-specific response structures (OpenAI vs GenAI vs OpenRouter).

CallResult SCHEMA
─────────────────
    text          : str | None     — model response text (trimmed)
    finish_reason : str | None     — "stop" | "length" | "loop_trimmed" | None
    tokens_used   : int            — total tokens in completion
    latency_ms    : float          — wall-clock time for API call (ms)
    error         : str | None     — error message if all retries exhausted

USAGE EXAMPLES
──────────────
    from inference_llm import InferenceClient, CallResult

    # OpenAI model
    model_cfg = {
        "model_id": "gpt-4o",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "min_tokens": 8192,
    }
    client = InferenceClient(model_cfg, system_prompt="You are a mathematician.")
    result = client.call(
        user_prompt="Solve: 2+2",
        qid="q123",
        variant="standard",
        max_tokens=8192,
        ceiling=16384,
    )
    if result.error:
        print(f"Failed: {result.error}")
    else:
        print(f"Response: {result.text}")
        print(f"Finish reason: {result.finish_reason}")

    # Dry-run mode (no API calls)
    client_dry = InferenceClient(model_cfg, system_prompt, dry_run=True)
    result_dry = client_dry.call(...)  # returns dummy CallResult
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TOKEN_STEP_UP    = 8_192   # tokens added per adaptive retry on truncation
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

def _is_loop(text: str) -> bool:
    """Return True if the response looks like an infinite repetition loop."""
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
        user_prompt: str,
        qid:         str,
        max_tokens:  int           = 8_192,
        ceiling:     Optional[int] = None,
    ) -> CallResult:
        """
        Call the model and return a CallResult.

        Parameters
        ──────────
        user_prompt   str  — the user-turn prompt
        qid           str  — question ID (for logging only)
        max_tokens    int  — starting token budget
        ceiling       int  — hard upper limit for adaptive step-up
                             (defaults to 2 × max_tokens)
        """
        if self.dry_run:
            return CallResult(
                text=f"[DRY RUN]\nPrompt length: {len(user_prompt)} chars",
                finish_reason=None,
            )

        ceiling  = ceiling or (max_tokens * 2)
        current  = max_tokens

        for attempt in range(self.retries):
            try:
                t0 = time.time()
                text, finish_reason, tokens_used = self._dispatch(user_prompt, current)
                latency_ms = round((time.time() - t0) * 1000, 1)

                # ── Loop detection ───────────────────────────
                if finish_reason == "length" and _is_loop(text):
                    text          = _trim_loop(text)
                    finish_reason = "loop_trimmed"
                    logging.warning(
                        f"  ∞ LOOP {qid} | tokens={tokens_used} | trimmed"
                    )

                # ── Adaptive step-up on truncation ───────────
                if finish_reason == "length":
                    if current < ceiling:
                        new = min(current + TOKEN_STEP_UP, ceiling)
                        logging.warning(
                            f"  ⚠ TRUNCATED {qid} | tokens={tokens_used} "
                            f"| step-up {current}→{new}"
                        )
                        current = new
                        continue        # retry with more tokens
                    else:
                        logging.warning(
                            f"  ✗ TRUNCATED {qid} | tokens={tokens_used} "
                            f"| ceiling {ceiling} reached"
                        )

                return CallResult(
                    text=text,
                    finish_reason=finish_reason,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                )

            except Exception as exc:
                logging.warning(
                    f"  API error {qid} attempt {attempt + 1}/{self.retries}: {exc}"
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
        is_qwen = "qwen" in mid

        kwargs: dict = dict(
            model=self.model_id,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
        )

        # Qwen models on OpenRouter: avoid Novita provider (returns empty responses)
        if is_qwen:
            kwargs["extra_body"] = {
                "provider": {"order": ["Together", "DeepInfra", "Fireworks"]}
            }

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
            raise
