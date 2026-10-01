"""
Model registry for LinAlg-Bench inference.

Each entry has:
  model_id      — API model identifier (OpenRouter / OpenAI / GenAI format)
  api_base      — API base URL
  api_key_env   — environment variable holding the API key
  min_tokens    — minimum token budget for inference
"""

MODELS = {
    "Qwen2.5-72B": {
        "model_id":    "qwen/qwen-2.5-72b-instruct",
        "api_base":    "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_tokens":  8192,
    },
    "Llama-3.3-70B": {
        "model_id":    "meta-llama/llama-3.3-70b-instruct",
        "api_base":    "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_tokens":  8192,
    },
    "GPT-4o": {
        "model_id":    "gpt-4o",
        "api_base":    "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "min_tokens":  8192,
    },
    "Claude-4.5-Sonnet": {
        "model_id":    "anthropic/claude-sonnet-4.5",
        "api_base":    "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_tokens":  8192,
    },
    "Mistral-Large": {
        "model_id":    "mistralai/mistral-large-2512",
        "api_base":    "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_tokens":  8192,
    },
    "DeepSeek-V3": {
        "model_id":    "deepseek/deepseek-v3.2",
        "api_base":    "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_tokens":  16384,
    },
    "GPT-5.2": {
        "model_id":    "gpt-5.2",
        "api_base":    "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "min_tokens":  16384,
    },
    "Qwen3-235B": {
        "model_id":    "qwen/qwen3-235b-a22b-2507",
        "api_base":    "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "min_tokens":  16384,
    },
    "OpenAI-o1": {
        "model_id":    "o1",
        "api_base":    "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "min_tokens":  16384,
    },
    "Gemini-3.1-Pro": {
        "model_id":    "gemini-3.1-pro-preview",
        "api_base":    "https://generativelanguage.googleapis.com/v1beta",
        "api_key_env": "GEMINI_API_KEY",
        "min_tokens":  16384,
    },
}


def get_model_names() -> list[str]:
    """Return list of all registered model display names."""
    return list(MODELS.keys())


def get_model_config(name: str) -> dict | None:
    """Return config dict for a model, or None if not found."""
    return MODELS.get(name)
