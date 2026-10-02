#!/usr/bin/env python3
"""
judge_prompts/__init__.py — Aggregates per-subcategory prompt modules.

Central registry for all judge prompts (build and validate stages).
Loads individual subcat prompt modules and exposes two dicts:

  BUILD_PROMPTS   : Dict[subcat] → system + user_template for Stage 2 (judge)
  VALIDATE_PROMPTS: Dict[subcat] → system + builder for Stage 3 (validator)

Subcategory prompt files live in judge_prompts/{subcat}.py and define:
  BUILD_SYSTEM_TEMPLATE        : System prompt for error classification
  BUILD_USER_TEMPLATE          : User prompt template with placeholders
  VALIDATE_SYSTEM_TEMPLATE     : System prompt for verification
  build_validate_user_prompt() : Function to build validation prompt

USAGE
─────
  from judge_prompts import BUILD_PROMPTS, VALIDATE_PROMPTS

  # Get classification prompts for rank
  build_system = BUILD_PROMPTS["rank"]["system"]
  build_template = BUILD_PROMPTS["rank"]["user_template"]
  user_prompt = build_template.format(
      matrix_latex=latex_str,
      ground_truth=gt,
      extracted_answer=extracted,
      response=resp,
  )

  # Get validation prompts
  validate_system = VALIDATE_PROMPTS["rank"]["system"]
  build_func = VALIDATE_PROMPTS["rank"]["build_user_prompt"]
  validate_prompt = build_func(judge_classification, model_response, ...)

ADDING A NEW SUBCAT
───────────────────
  1. Create judge_prompts/<subcat>.py with:
     - BUILD_SYSTEM_TEMPLATE (str)
     - BUILD_USER_TEMPLATE (str with format placeholders)
     - VALIDATE_SYSTEM_TEMPLATE (str)
     - build_validate_user_prompt(classification, response, ...) function

  2. Add import at top of this file:
     from . import <subcat>

  3. Add to _SUBCATS dict:
     "<subcat>": <subcat>,

  Done! Scripts automatically discover and register the new prompts.

SUPPORTED SUBCATEGORIES
───────────────────────
  rank, nullity, matrix_vector, multiplication, matrix_power, transpose,
  det, eigen, trace
"""

from . import (
    rank,
    nullity,
    matrix_vector,
    multiplication,
    matrix_power,
    transpose,
    det,
    eigen,
    trace,
)

_SUBCATS = {
    # long-form keys
    "rank":           rank,
    "nullity":        nullity,
    "matrix_vector":  matrix_vector,
    "multiplication": multiplication,
    "matrix_power":   matrix_power,
    "transpose":      transpose,
    "det":            det,
    "eigen":          eigen,
    # short-code aliases
    "null":           nullity,
    "vec":            matrix_vector,
    "mult":           multiplication,
    "pow":            matrix_power,
    "trans":          transpose,
    "eig":            eigen,
    "trace":          trace,
}

BUILD_PROMPTS: dict[str, dict[str, str]] = {
    name: {
        "system":        mod.BUILD_SYSTEM_TEMPLATE,
        "user_template": mod.BUILD_USER_TEMPLATE,
    }
    for name, mod in _SUBCATS.items()
}

def _get_validate_fn(mod):
    fn = getattr(mod, "build_validate_user_prompt", None)
    if fn is None:
        raise AttributeError(
            f"judge_prompts/{mod.__name__.split('.')[-1]}.py is missing build_validate_user_prompt"
        )
    return fn

VALIDATE_PROMPTS: dict[str, dict] = {
    name: {
        "system":            mod.VALIDATE_SYSTEM_TEMPLATE,
        "build_user_prompt": _get_validate_fn(mod),
    }
    for name, mod in _SUBCATS.items()
}
