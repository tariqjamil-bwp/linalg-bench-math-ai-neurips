# Changelog

Differences between the pipeline that produced the published results (v1) and the
code in this repository (v2). Files not listed are unchanged.

Grading-relevant changes are marked **[affects grading]**. There is exactly one:
the eigenvalue rounding tolerance. Everything else changes how a run is executed,
configured, or recorded, not whether an answer is judged correct.

---

## v2.0 — `main_pipeline/subcat_config.py`

**[affects grading] Eigenvalues are rounded to 3 decimal places before comparison.**
`EIG_DECIMAL_PLACES = 3` is applied in `compute_eigenvalue_from_answer()` and in
both extraction paths of `extract_eigenvalue_from_response()`. v1 compared exact
floats, so a response reporting `10.960` against a ground truth of `10.96017`
graded as a failure despite agreeing to the stated tolerance.

All other subcategories (det, rank, nullity, mult, pow, vec, trans, trace) are
unchanged.

---

## v2.0 — `main_pipeline/inference.py`

### Concurrency and rate limiting

- **The per-provider cap now sizes the thread pool.** v1 computed
  `effective_workers = min(max_workers, provider_limit)` and used it only to size
  a `Semaphore` that nothing acquired, while `ThreadPoolExecutor` ran at the
  uncapped `max_workers`. The cap is now applied to the executor itself and the
  unused semaphore is gone.
- **Added `_pace()`**, a global call-rate throttle shared by all worker threads.
  Set `--rate-limit 0` to disable.
- **`--max-workers` and `--rate-limit` are now flags.** Both were hardcoded (`8`
  and `1.0`) with `# hardcoded default` comments; the defaults are unchanged.

### Dataset validation

- **A dataset missing `problem_text` now stops the run instead of printing a
  warning.** A row states its task either in `problem_text` or in instruction
  text embedded in `problem_latex`:
  - Rows with **neither** abort the run — their prompt would name no operation,
    so the model cannot know what to compute. `--force` overrides.
  - Rows missing only `problem_text` print the instruction that will be sent and
    wait for the operator to type `YES` or `NO`. A 120 s timeout or EOF
    continues, so an unattended run does not stall.
- The whole file is scanned before `--limit` is applied, so a deficient row
  outside the sampled range is still caught.

### Configuration

- **`--api-base` and `--api-key-env` were removed.** Endpoint and key-environment
  variable now come only from the `MODELS` registry, so a run cannot be pointed at
  an endpoint the published results were not produced on. Adding a model means
  editing `main_pipeline/models.py`.
- **`.env` is loaded with `find_dotenv()` and `override=True`**, so the file is
  found from any working directory and its values win over a stale exported shell
  variable.

### Recording

- **Every record carries `full_prompt`**, the exact `[system, user]` message list
  sent to the model. Downstream stages cannot reconstruct this: the prompt builder
  evolves, so re-deriving it later yields the current prompt rather than a
  transcript of the original call. It appears in:
  - `results.jsonl` — `"full_prompt"`, a list of dicts
  - `{model}_{subcat}_failures.jsonl` — same
  - `results.xlsx` — a `full_prompt` column, JSON-encoded, inserted at **column
    L** after `problem_latex`. Every column from the old L onward shifts one
    letter right, so the sheet now runs A–T rather than A–S.
- **`rerun_equivalence_check()` skips records that already carry an
  `equivalence_check` verdict.** Re-checking them cannot change the outcome, since
  neither the response nor `answer_latex` has changed.
- The equivalence-check prompt now states the 3-decimal-place eigenvalue
  tolerance, so the LLM checker and the code extractor apply one rule.

---

## v2.0 — `main_pipeline/inference_llm.py`

- **The `temperature` parameter is omitted for the o1/o3 series**, which rejects
  it outright. v1 sent `temperature=0.0` unconditionally. `gpt-5.x` accepts
  temperature and is unaffected — it shares v1's separate
  `max_completion_tokens` handling, which is unchanged.
- **`.env` is loaded with `find_dotenv()` and `override=True`** (as above).
