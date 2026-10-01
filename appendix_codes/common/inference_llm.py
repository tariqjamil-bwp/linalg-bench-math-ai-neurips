#!/usr/bin/env python3
# Module:    inference_llm.py
# Version:   v2
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
      *.openai.com     → OpenAI (GPT-4o, GPT-5.2, o1)
      openrouter.ai    → OpenRouter (Qwen, Llama, Claude, DeepSeek, Mistral)
      *.anthropic.com  → native Anthropic (Claude, direct provider call)
      (no api_base)    → Google GenAI (Gemini models)

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
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(dotenv_path=find_dotenv(), override=True)
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TOKEN_STEP_UP    = 8_192   # tokens added per adaptive retry on truncation
LOOP_MIN_LINES   = 100     # minimum total lines before loop check fires
LOOP_THRESHOLD   = 0.5     # fraction of second-half dominated by one repeated line

# Rate-limit / overload errors need a much longer backoff than a generic
# transient error — a flat 2s/4s does nothing against a provider that's
# overloaded for tens of seconds. Detected by matching the exception text
# (the OpenAI SDK surfaces provider errors as plain Exception, not a typed
# RateLimitError, when routed through OpenRouter).
RATE_LIMIT_PATTERN  = re.compile(
    r'429|rate.?limit|rate-limited|overloaded|engine_overloaded|too many requests',
    re.IGNORECASE,
)
RATE_LIMIT_BACKOFF  = [15, 45]   # seconds, indexed by attempt (vs. 2/4 for other errors)

# OpenRouter surfaces which upstream provider actually handled (or rejected) a
# request inside the error body's 'provider_name' field, including ones it
# fell back to via 'previous_errors'. A provider that fails repeatedly is put
# on a temporary, self-expiring cooldown (not a permanent allow/deny list —
# which provider is reliable varies model-to-model and shifts over time, so
# the cooldown is generic and reactive rather than hardcoded per model).
PROVIDER_NAME_PATTERN      = re.compile(r"'provider_name':\s*'([^']+)'")
PROVIDER_FAIL_THRESHOLD    = 2    # failures before a provider is blacklisted
PROVIDER_BLACKLIST_SECONDS = 60   # cooldown duration

# Fires when the blacklist has ruled out every provider OpenRouter has for
# this model — a 404 with no provider_name to attribute it to. Treating it as
# a blank attempt would keep the cooldown spinning, so it is surfaced.
ALL_PROVIDERS_IGNORED_PATTERN = re.compile(r"All providers have been ignored", re.IGNORECASE)


def _is_rate_limit_error(exc: Exception) -> bool:
    return bool(RATE_LIMIT_PATTERN.search(str(exc)))


def _named_providers(exc: Exception) -> set[str]:
    return set(PROVIDER_NAME_PATTERN.findall(str(exc)))


def _all_providers_ignored(exc: Exception) -> bool:
    return bool(ALL_PROVIDERS_IGNORED_PATTERN.search(str(exc)))


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
    # Which upstream provider actually served the call, and the exact model
    # version string it reports. Both come straight off the response, never
    # from the request — the point is to detect a provider pin that did NOT
    # hold, so echoing back what we asked for would defeat the purpose.
    # provider is None for backends that expose no such field (OpenAI direct);
    # native Gemini reports the literal "google-ai-studio-native".
    provider:      Optional[str]   = None
    model_version: Optional[str]   = None


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
    retries       int   — max API attempts per call (default 4)
    rate_limit    float — max calls/sec, globally paced across all callers
                          sharing this client instance (e.g. every worker
                          thread in a ParallelExecutor); <=0 disables pacing
                          (default 1.0). Ignored if pace_levels is given.
    pace_levels   list[float] | None — seconds between calls, ascending. If
                          given, pacing is adaptive instead of static: starts
                          at the first level and escalates one step at a time
                          (see _escalate_pace) each time an attempt fails with
                          a rate-limit/overload-shaped error, capping at the
                          last level. Never de-escalates within this client's
                          lifetime (default None — static rate_limit applies).
    """

    def __init__(
        self,
        model_cfg:     dict,
        system_prompt: str,
        dry_run:       bool                       = False,
        retries:       int                        = 4,
        rate_limit:    float                      = 1.0,
        pace_levels:   Optional[list[float]]       = None,
    ):
        self.model_id      = model_cfg["model_id"]
        self.api_base      = model_cfg.get("api_base", "")
        self.system_prompt = system_prompt
        self.dry_run       = dry_run
        self.retries       = retries
        self.rate_limit    = rate_limit
        # Hard provider restriction (see models.py). When set, this model may
        # only ever be served by this one provider — the cooldown blacklist and
        # the per-call local_exclude are both bypassed, because silently
        # rerouting to a different provider is the failure this prevents.
        self.provider_pin  = model_cfg.get("provider_pin")

        # Call pacing: a global gate so calls dispatched by concurrent callers
        # sharing this client are spaced out, regardless of caller count.
        self._pace_lock     = Lock()
        self._last_call_ts  = 0.0
        self._pace_levels   = list(pace_levels) if pace_levels else []
        self._adaptive_pace = bool(self._pace_levels)
        self._pace_idx      = 0

        # Provider cooldown: a provider that fails PROVIDER_FAIL_THRESHOLD
        # times is excluded from routing for PROVIDER_BLACKLIST_SECONDS, then
        # automatically becomes eligible again.
        self._provider_lock            = Lock()
        self._provider_fail_count      = {}   # provider_name -> int
        self._provider_blacklist_until = {}   # provider_name -> epoch seconds

        self._client   = None

        if not dry_run:
            api_key_env = model_cfg.get("api_key_env", "")
            api_key     = os.environ.get(api_key_env, "") if api_key_env else ""
            self._backend = self._detect_backend()
            self._init_client(api_key, api_key_env)
            logging.info(f"  InferenceClient: {self.model_id} via {self._backend}")
        else:
            self._backend = "dry"

    # ── call pacing ──────────────────────────────────────────────────────

    def _pace(self) -> None:
        """Block until at least one interval has passed since the last call
        was dispatched. Adaptive mode uses the current
        `_pace_levels[_pace_idx]` (seconds); static mode uses 1/rate_limit,
        with rate_limit <= 0 disabling pacing entirely."""
        if self._adaptive_pace:
            min_interval = self._pace_levels[self._pace_idx]
        elif self.rate_limit <= 0:
            return
        else:
            min_interval = 1.0 / self.rate_limit
        with self._pace_lock:
            now  = time.time()
            wait = self._last_call_ts + min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_call_ts = time.time()

    def _escalate_pace(self) -> None:
        """Step adaptive pacing up to the next (slower) level. No-op in
        static mode or once already at the slowest level. Called whenever an
        attempt fails with a rate-limit/overload-shaped error."""
        if not self._adaptive_pace:
            return
        with self._pace_lock:
            if self._pace_idx < len(self._pace_levels) - 1:
                self._pace_idx += 1
                logging.warning(
                    f"  ⏫ pacing escalated to {self._pace_levels[self._pace_idx]}s "
                    f"between calls (rate-limit error observed)"
                )

    # ── provider cooldown ─────────────────────────────────────────────────

    def _record_provider_failure(self, exc: Exception) -> None:
        """Bump the failure count for every provider named in exc; once a
        provider hits PROVIDER_FAIL_THRESHOLD, blacklist it for
        PROVIDER_BLACKLIST_SECONDS and reset its count (so it takes a fresh
        run of failures after the cooldown expires to blacklist it again)."""
        named = _named_providers(exc)
        if not named:
            return
        now = time.time()
        with self._provider_lock:
            for p in named:
                self._provider_fail_count[p] = self._provider_fail_count.get(p, 0) + 1
                if self._provider_fail_count[p] >= PROVIDER_FAIL_THRESHOLD:
                    self._provider_blacklist_until[p] = now + PROVIDER_BLACKLIST_SECONDS
                    self._provider_fail_count[p] = 0
                    logging.warning(
                        f"  ⊘ provider '{p}' blacklisted for {PROVIDER_BLACKLIST_SECONDS}s "
                        f"({PROVIDER_FAIL_THRESHOLD} failures observed)"
                    )

    def _blacklisted_providers(self) -> list[str]:
        """Providers currently within their cooldown."""
        now = time.time()
        with self._provider_lock:
            return sorted(p for p, until in self._provider_blacklist_until.items() if until > now)

    def _clear_provider_blacklist(self) -> None:
        """Drop every active cooldown. Called when OpenRouter reports no
        eligible provider remains (a 404 with no provider_name to attribute
        it to) — evidence the current blacklist has emptied the pool, so it's
        better to route freely again than guarantee another 404. Providers
        still genuinely bad will simply fail and get re-blacklisted."""
        with self._provider_lock:
            if self._provider_blacklist_until:
                logging.warning(
                    f"  ⊘ blacklist {sorted(self._provider_blacklist_until)} left no "
                    f"eligible provider — clearing cooldowns"
                )
            self._provider_blacklist_until.clear()

    # ── backend detection ─────────────────────────────────────────────────

    def _detect_backend(self) -> str:
        if "openai.com" in self.api_base:
            return "openai"
        elif "openrouter" in self.api_base:
            return "openrouter"
        elif "anthropic.com" in self.api_base:
            return "anthropic"
        elif self.model_id.lower().startswith("gemini"):
            return "genai"
        else:
            return "openai_compat"      # generic OpenAI-compatible endpoint

    def _init_client(self, api_key: str, api_key_env: str = "") -> None:
        if self._backend == "genai":
            try:
                from google import genai
            except ImportError:
                raise ImportError("pip install google-genai")
            primary_key = os.environ.get("GEMINI_API_KEY", "")
            if not primary_key:
                raise EnvironmentError("GEMINI_API_KEY not set")
            self._client = genai.Client(api_key=primary_key)
        elif self._backend == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError("pip install anthropic")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY not set")
            self._client = Anthropic(api_key=api_key)
        else:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("pip install openai")
            if not api_key:
                raise EnvironmentError(
                    f"{api_key_env or 'API key'} not set for {self.model_id} "
                    f"(set it in the repo root's .env)"
                )
            self._client = OpenAI(base_url=self.api_base, api_key=api_key)

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
        self._pace()
        # Providers that failed on an earlier attempt of THIS id are excluded
        # on the next attempt of this same id immediately — no reason to
        # retry a provider that just failed for this exact call, regardless
        # of whether it's crossed the cross-call blacklist threshold yet.
        local_exclude: set[str] = set()

        for attempt in range(self.retries):
            try:
                t0 = time.time()
                text, finish_reason, tokens_used, provider, model_version = \
                    self._dispatch(user_prompt, current, local_exclude)
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

                # Case-insensitive on purpose. OpenRouter PINS are lowercase
                # slugs ("together", "alibaba") but the provider it reports
                # back on the response is display-cased ("Together",
                # "Alibaba"). Exact equality would therefore fire on every
                # correct call, flagging calls that were all served by the
                # pinned provider as violations.
                #
                # That is worse than noise. At full scale it is ~1,360 false
                # alarms across Llama and Qwen, and an alarm that fires on
                # every correct call is one nobody reads — so a REAL
                # substitution would scroll past unnoticed, which is the exact
                # failure this guard exists to catch.
                #
                # Only case and surrounding space are forgiven. together ->
                # deepinfra still fires. The record keeps `provider` verbatim,
                # so the deliverables show precisely what the API returned.
                if (self.provider_pin and provider
                        and provider.strip().casefold()
                        != self.provider_pin.strip().casefold()):
                    # Should be unreachable: allow_fallbacks=False means
                    # OpenRouter errors rather than substituting. Logged loudly
                    # anyway, because a silent substitution is the one failure
                    # that would invalidate the run without leaving a trace.
                    # The record still keeps the real provider — the guard in
                    # ibp_build_deliverables.py catches it at build time.
                    logging.error(
                        f"  ‼ PROVIDER PIN VIOLATED {qid}: asked for "
                        f"'{self.provider_pin}', served by '{provider}'"
                    )

                return CallResult(
                    text=text,
                    finish_reason=finish_reason,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    provider=provider,
                    model_version=model_version,
                )

            except Exception as exc:
                logging.warning(
                    f"  API error {qid} attempt {attempt + 1}/{self.retries}: {exc}"
                )
                is_rate_limit = _is_rate_limit_error(exc)
                if is_rate_limit:
                    self._escalate_pace()
                if self.provider_pin:
                    # A pinned model has exactly one legal provider, so there is
                    # nothing to route around: blacklisting it would only turn
                    # every subsequent attempt into a guaranteed "all providers
                    # ignored" 404. Retry the same provider and let it recover.
                    pass
                elif _all_providers_ignored(exc):
                    self._clear_provider_blacklist()
                    local_exclude.clear()
                else:
                    self._record_provider_failure(exc)
                    local_exclude |= _named_providers(exc)
                if attempt < self.retries - 1:
                    # Jitter (+0-50%) so concurrent workers that failed together
                    # (e.g. all 5 hitting an overloaded provider at once) don't
                    # all retry at the exact same instant and re-trigger the
                    # same overload as a synchronized burst.
                    if is_rate_limit:
                        base    = RATE_LIMIT_BACKOFF[min(attempt, len(RATE_LIMIT_BACKOFF) - 1)]
                        backoff = base + random.uniform(0, base * 0.5)
                        logging.warning(f"  ⏳ rate-limit/overload detected for {qid} — "
                                       f"backing off {backoff:.1f}s (base {base}s, vs {2**(attempt+1)}s default)")
                    else:
                        base    = 2 ** (attempt + 1)
                        backoff = base + random.uniform(0, base * 0.5)
                    time.sleep(backoff)

        return CallResult(error=f"All {self.retries} retries failed")

    # ── internal dispatch ─────────────────────────────────────────────────

    def _dispatch(
        self, user_prompt: str, max_tokens: int, local_exclude: set[str] | None = None
    ) -> tuple[str, str, int, Optional[str], Optional[str]]:
        """Returns (text, finish_reason, tokens_used, provider, model_version)."""
        if self._backend == "genai":
            return self._call_genai(user_prompt, max_tokens)
        elif self._backend == "anthropic":
            return self._call_anthropic(user_prompt, max_tokens)
        return self._call_openai_compat(user_prompt, max_tokens, local_exclude)

    def _call_openai_compat(
        self, user_prompt: str, max_tokens: int, local_exclude: set[str] | None = None
    ) -> tuple[str, str, int, Optional[str], Optional[str]]:
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

        if self.provider_pin and self._backend == "openrouter":
            # PINNED: exactly one provider is permitted, and allow_fallbacks is
            # off so OpenRouter errors instead of substituting another host.
            #
            # The exclusion layers below are deliberately NOT applied here. They
            # exist to route around a flaky provider, but for a pinned model
            # that "helpful" reroute is the exact failure being prevented — a
            # run that silently switches from Together's build of Llama-3.3-70B
            # to DeepInfra's fp8 one mixes quantizations mid-experiment. With a
            # pin, a failing provider must surface as an error to be retried or
            # investigated, never as a quiet substitution.
            kwargs["extra_body"] = {
                "provider": {"only": [self.provider_pin], "allow_fallbacks": False}
            }
        else:
            # UNPINNED: OpenRouter routes freely. Two layers exclude a provider
            # temporarily — local_exclude (this id's own prior attempts, reset
            # per call) and the cross-call cooldown blacklist (see
            # _record_provider_failure / _blacklisted_providers).
            exclude = self._blacklisted_providers()
            exclude = sorted(set(exclude) | (local_exclude or set()))
            if exclude and self._backend == "openrouter":
                kwargs["extra_body"] = {"provider": {"ignore": exclude}}

        if uses_completion_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        resp          = self._client.chat.completions.create(**kwargs)
        text          = resp.choices[0].message.content or ""
        finish_reason = resp.choices[0].finish_reason or "stop"
        tokens_used   = resp.usage.total_tokens if resp.usage else 0
        # OpenRouter returns a top-level "provider"; OpenAI direct does not, so
        # getattr rather than attribute access. resp.model is the resolved
        # version string actually served (e.g. "anthropic/claude-4.5-sonnet-20250929"),
        # distinct from self.model_id, which is only what we asked for.
        provider      = getattr(resp, "provider", None)
        model_version = getattr(resp, "model", None) or self.model_id
        return text, finish_reason, tokens_used, provider, model_version

    def _call_genai(
        self, user_prompt: str, max_tokens: int
    ) -> tuple[str, str, int, Optional[str], Optional[str]]:
        """Google GenAI call with primary/fallback client.

        A direct provider call has no routing layer, so there is no provider
        field to read back. The literal "google-ai-studio-native" is recorded
        instead, so every record across every backend carries provenance in the
        same column and a blank provider always means a genuine bug.
        """
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=max_tokens,
        )

        def _invoke(client) -> tuple[str, str, int, Optional[str], Optional[str]]:
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
            model_version = getattr(resp, "model_version", None) or self.model_id
            return text, finish_reason, tokens_used, "google-ai-studio-native", model_version

        try:
            return _invoke(self._client)
        except Exception as api_error:
            raise

    def _call_anthropic(
        self, user_prompt: str, max_tokens: int
    ) -> tuple[str, str, int, Optional[str], Optional[str]]:
        """Native Anthropic Messages API call.

        A direct provider call has no routing layer, so — same convention as
        _call_genai — the literal "anthropic" is recorded as provider rather
        than leaving the field blank.
        """
        resp = self._client.messages.create(
            model=self.model_id,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        text          = "".join(
            block.text for block in resp.content if block.type == "text"
        ).strip()
        finish_reason = "length" if resp.stop_reason == "max_tokens" else "stop"
        tokens_used   = (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
        model_version = resp.model or self.model_id
        return text, finish_reason, tokens_used, "anthropic", model_version
