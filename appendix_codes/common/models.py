# Module:    models.py
# Version:   1.0
"""
Model registry for LinAlg-Bench / IBP inference. Every model is called live
(synchronous call through inference_llm_v2) — Llama and Qwen through
OpenRouter, Claude and Gemini through their native APIs.

Every entry has:
  model_id      — API model identifier (OpenRouter / native provider format)
  api_base      — API base URL
  api_key_env   — environment variable holding the API key
  min_tokens    — minimum token budget for inference

The four IBP Depth Ladder entries at the bottom carry two extra keys. They
are optional everywhere: the five small-model entries above omit them and are
unaffected.

  provider_pin    Provider slug this model must be served by. When set, the
                  client sends {"provider": {"only": [slug],
                  "allow_fallbacks": false}} and the failure-cooldown
                  blacklist is bypassed — see inference_llm_v2's
                  _call_openai_compat. Absent means route freely.
  max_output_tokens
                  The provider's own output ceiling for THIS entry, on the
                  pinned endpoint — not a budget we chose.

WHY THE PIN EXISTS
──────────────────
OpenRouter's live endpoint list for an open-weights model carries several
endpoints at different quantizations (e.g. meta-llama/llama-3.3-70b-instruct
has fp8, bf16 and fp16 builds from multiple hosts). Unpinned, a single run
would silently sample several different builds of "the same" model, which is
exactly the confound the pin guards against. qwen3-235b-a22b-2507 has the
same problem, with alibaba one endpoint among many. Every record stores the
provider that actually served it, so a pin that fails to hold is visible in
the data rather than invisible.
"""

MODELS = {
    # ─────────────────────────────────────────────────────────────────────
    #  IBP Depth Ladder — the four study models, all called live.
    #
    #  min_tokens is 16384 for all four — the STARTING budget for the
    #  adaptive step-up. max_output_tokens is the provider's hard ceiling on
    #  the pinned endpoint.
    # ─────────────────────────────────────────────────────────────────────

    # SINGLE PIN — exactly one provider, allow_fallbacks stays false, so an
    # outage surfaces as an error rather than a silent switch to a different
    # quantization. Pinned to DeepInfra, whose endpoint reports a 16384
    # completion ceiling, so start == ceiling and a truncation there is final
    # and visible on the first call (no step-up room). Together was tried
    # first but rejected: its endpoint caps completions at 2048 tokens, too
    # low for the deep (n=14-16) derivations this study needs.
    "Llama-3.3-70B": {
        "model_id":          "meta-llama/llama-3.3-70b-instruct",
        "api_base":          "https://openrouter.ai/api/v1",
        "api_key_env":       "OPENROUTER_API_KEY",
        "min_tokens":        16384,
        "max_output_tokens": 16384,
        "provider_pin":      "deepinfra",
    },
    # qwen3-235b-a22b-2507, not the 04-28 original: the original caps
    # max_completion_tokens at 8192, which risks truncating the deep (n=14-16)
    # derivations with no step-up to recover, and costs 5x more. The
    # -thinking-2507 variant is excluded by this study's hidden-CoT rule.
    "Qwen3-235B": {
        "model_id":     "qwen/qwen3-235b-a22b-2507",
        "api_base":     "https://openrouter.ai/api/v1",
        "api_key_env":  "OPENROUTER_API_KEY",
        "min_tokens":   16384,
        # Alibaba's own endpoint caps at 32768 (others on this model go far
        # higher, but the provider is pinned to alibaba, so this is the real cap).
        "max_output_tokens": 32768,
        "provider_pin": "alibaba",
    },
    # Native Anthropic API (api.anthropic.com), not OpenRouter — a direct
    # provider call has no routing to pin, so provider_pin is absent and
    # provenance is recorded as the literal "anthropic".
    "Claude-4.5-Sonnet": {
        # Dated snapshot, not the claude-sonnet-4-5 alias: the run must be
        # reproducible even after Anthropic repoints the alias.
        "model_id":          "claude-sonnet-4-5-20250929",
        "api_base":          "https://api.anthropic.com",
        "api_key_env":       "ANTHROPIC_API_KEY",
        "min_tokens":        16384,
        # Sonnet 4.5 output ceiling.
        "max_output_tokens": 64000,
    },
    # Native Gemini API, not OpenRouter — through the google-genai SDK
    # (inference_llm_v2's "genai" backend), not an OpenAI-compatible
    # endpoint. No api_base: absence is what inference_llm_v2._detect_backend()
    # reads as "route through genai". A direct provider call has no routing
    # to pin, so provider_pin is absent and provenance is recorded as the
    # literal "google-ai-studio-native".
    "Gemini-3.1-Pro": {
        "model_id":          "gemini-3.1-pro-preview",
        "api_key_env":       "GEMINI_API_KEY",
        "min_tokens":        16384,
        # outputTokenLimit from the v1beta models list.
        "max_output_tokens": 65536,
    },
}


def get_model_names() -> list[str]:
    """Return list of all registered model display names."""
    return list(MODELS.keys())


def get_model_config(name: str) -> dict | None:
    """Return config dict for a model, or None if not found."""
    return MODELS.get(name)


def get_max_output_tokens(name: str, default: int = 16384) -> int:
    """The provider's own output ceiling for this entry.

    The cap can sit below min_tokens for an endpoint whose provider limit is
    tighter than the adaptive start. That is a real provider limit, not a bug,
    so it is returned as-is; the caller decides whether to clamp its budget or
    change the pin.
    """
    cfg = MODELS.get(name)
    if cfg is None:
        raise KeyError(f"Unknown model '{name}'")
    cap = cfg.get("max_output_tokens", default)
    return cap
