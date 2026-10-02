#!/usr/bin/env python3
"""
judge_llm.py — LLM transport layer for judge scripts (Stage 2 & 3).

Thin abstraction over Gemini API for error classification and validation.
Provides two public functions:

  define_clients(model_id)
    → Initialize and return a single client
    → Handles env var loading and client setup

  call_llm(system, user, model_id, client)
    → Make a single API call with retry logic
    → Returns (raw_text, finish_reason)
    → finish_reason: "stop" | "length" | error message

BACKEND SELECTION
─────────────────
  Auto-selected by model_id prefix:
    gemini-*  → Google GenAI SDK (GEMINI_API_KEY)
    (others)  → OpenRouter (OPENROUTER_API_KEY)

  Note: google.generativeai (old SDK) is BANNED — always use google-genai.

CONFIGURATION
──────────────
  Temperature: 0.0 (deterministic output for reproducibility)
    Exception: Qwen models use temperature=0.6 (required for reasoning output)
  Max tokens: 16384 (Gemini/non-Qwen); 4096 (Qwen)
  Model selection: flash-lite (testing/cheap) or pro (production/accurate)

RETRY POLICY
─────────────
  5 attempts with flat 60-second wait between retries:
    Attempt 1: immediate
    Attempt 2–5: wait 60 seconds, then retry
    (60s chosen for Gemini quota/503 recovery — longer than typical backoff)

  After all retries exhausted: raises RuntimeError.

USAGE EXAMPLES
──────────────
  from judge_llm import define_clients, call_llm

  # Initialize client
  client = define_clients("gemini-3.1-flash-lite-preview")

  # Make API call
  raw_text, finish_reason = call_llm(
      system_prompt="You are a mathematics expert.",
      user_prompt="Classify this error: ...",
      model_id="gemini-3.1-flash-lite-preview",
      client=client,
  )

  if finish_reason == "stop":
      print("Response:", raw_text)
  elif finish_reason == "length":
      print("Response truncated (exceeded max_tokens)")
  else:
      print("Error:", finish_reason)
"""

from __future__ import annotations

import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ─────────────────────────────────────────────────────────────────────────────
# CLIENT INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def define_clients(model_id: str):
    """
    Initialise and return the LLM client.

    gemini-* — requires GEMINI_API_KEY.
    other    — requires OPENROUTER_API_KEY.
    """
    if model_id.startswith("gemini"):
        try:
            from google import genai
        except ImportError:
            raise ImportError("pip install google-genai")

        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise EnvironmentError("export GEMINI_API_KEY=your-gemini-api-key")

        client = genai.Client(api_key=key)
        print("LLM ready: google-genai (GEMINI_API_KEY)", flush=True)
        return client

    else:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise EnvironmentError("export OPENROUTER_API_KEY=your-openrouter-key")

        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
        print(f"LLM ready: openai-compatible via OpenRouter (model={model_id})", flush=True)
        return client


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    client,
) -> tuple[str, str]:
    """
    Call the LLM and return (raw_text, finish_reason).
    finish_reason is normalised to "stop" or "length" across all backends.
    Retries 5 times with 60-second flat wait between attempts.
    Raises RuntimeError after all retries are exhausted.
    """
    for attempt in range(5):
        try:
            return _dispatch(system_prompt, user_prompt, model_id, client)
        except Exception as exc:
            if attempt < 4:
                wait = 60  # 1 minute between retries for 503/high-demand
                print(f"  [RETRY {attempt+1}/5] {exc}  (wait {wait}s)", flush=True)
                time.sleep(wait)
            else:
                raise RuntimeError(f"All retries exhausted: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL DISPATCH
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch(
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    client,
) -> tuple[str, str]:
    if model_id.startswith("gemini"):
        return _call_genai(system_prompt, user_prompt, model_id, client)
    else:
        return _call_openai(system_prompt, user_prompt, model_id, client)


def _call_genai(
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    client,
) -> tuple[str, str]:
    from google.genai import types

    cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=16384)

    resp = client.models.generate_content(
        model=model_id,
        contents=[system_prompt, user_prompt],
        config=cfg,
    )
    raw = resp.text.strip()
    fr = resp.candidates[0].finish_reason
    finish_reason = "length" if fr.name == "MAX_TOKENS" else "stop"
    return raw, finish_reason


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    client,
) -> tuple[str, str]:
    is_qwen = model_id.lower().startswith("qwen")
    # Qwen requires temperature >= 0.6 for reasoning output (per official docs)
    # All other models use 0.0 for deterministic reproducibility
    temperature = 0.6 if is_qwen else 0.0

    kwargs: dict = dict(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=4096 if is_qwen else 16384,
    )

    if is_qwen:
        kwargs["extra_body"] = {"reasoning_effort": "medium"}

    resp = client.chat.completions.create(**kwargs)
    raw = resp.choices[0].message.content.strip()
    finish_reason = resp.choices[0].finish_reason or "stop"
    return raw, finish_reason
