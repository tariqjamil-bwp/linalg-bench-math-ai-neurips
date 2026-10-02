# LinAlg-Bench

**LinAlg-Bench** is a diagnostic benchmark for evaluating LLM reasoning on linear algebra problems across 9 operation types and 3 matrix sizes (3×3, 4×4, 5×5). Beyond accuracy measurement, it includes a three-stage forensic pipeline that classifies the root cause of every model failure into a 13-category error taxonomy.

---

## Setup

**1. Install uv** (if not already installed)

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

**2. Create environment and install dependencies**

```bash
git clone https://github.com/tariqjamil-bwp/linalg-bench-math-ai-neurips.git
cd linalg-bench-math-ai-neurips

uv venv                  # creates .venv/
uv sync                  # installs all dependencies from pyproject.toml
```

**3. Configure API keys**

```bash
cp .env.example .env
# Edit .env and add:
#   OPENROUTER_API_KEY=...   # Qwen, Llama, Claude, Mistral, DeepSeek
#   OPENAI_API_KEY=...       # GPT-4o, GPT-5.2, o1
#   GEMINI_API_KEY=...       # Gemini + all judge calls (Stages 2–3)
```

All commands run from the repo root with `uv run python`.

---

## Stage 1: Inference (`pipeline/inference.py`)

Run zero-shot inference via CLI:

```bash
# Dry run (preview prompts, no API calls)
uv run python pipeline/inference.py \
    --input data/linalg_bench_3x3.csv \
    --model DeepSeek-V3 \
    --dry-run

# Full run
uv run python pipeline/inference.py \
    --input data/linalg_bench_3x3.csv \
    --model DeepSeek-V3

# Resume interrupted run
uv run python pipeline/inference.py \
    --input data/linalg_bench_3x3.csv \
    --model DeepSeek-V3 \
    --resume all
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | Llama-3.3-70B | Model name from registry |
| `--input` | required | Benchmark CSV |
| `--output` | `data/output/{Model}/` | Output directory |
| `--resume` | none | `all` or `failures` |
| `--limit` | none | Process only first N problems |
| `--dry-run` | False | Preview prompts without API calls |

Output: `{model}_results.jsonl`, `{model}_summary.csv`, `{model}_failures.jsonl`.

---

## Stage 2: Build Judge (`pipeline/build_judge.py`)

Classify each failure into the error taxonomy via CLI:

```bash
uv run python pipeline/build_judge.py \
    --results data/output/{Model}/{Model}_failures.jsonl \
    --output  data/output/{Model}/judge/{subcat}_judge_labels.csv \
    --subcat  det \
    --judge-llm-id gemini-3.1-pro-preview

# Cheap test
uv run python pipeline/build_judge.py \
    --results data/output/{Model}/{Model}_failures.jsonl \
    --output  data/output/{Model}/judge/{subcat}_judge_labels.csv \
    --subcat  det \
    --judge-llm-id gemini-3.1-flash-lite-preview \
    --dry-run --limit 2
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--results` | required | Failures JSONL from Stage 1 |
| `--output` | required | Output CSV path |
| `--subcat` | required | Subcategory (det, rank, eig, etc.) |
| `--judge-llm-id` | `gemini-3.1-pro-preview` | Judge model |
| `--resume` | False | Skip already-processed rows |
| `--limit` | none | Process only first N records |
| `--dry-run` | False | Print prompts without API calls |

---

## Stage 3: Validate Judge (`pipeline/validate_judge.py`)

Second independent pass verifies or corrects Stage 2 classifications:

```bash
uv run python pipeline/validate_judge.py \
    --judge   data/output/{Model}/judge/{subcat}_judge_labels.csv \
    --results data/output/{Model}/{Model}_failures.jsonl \
    --output  data/output/{Model}/judge/{subcat}_judge_validated.csv \
    --subcat  det \
    --judge-llm-id gemini-3.1-pro-preview
```

Output `validated` values: `TRUE` · `FALSE` (see `corrected_tag`) · `NEEDS_REVIEW` · `TRUNCATED`

**FINAL_TAG rule**: if `validated == FALSE` and `corrected_tag` non-empty → use `corrected_tag`; otherwise → use `Error_Tag`.

---

## Format Sensitivity (`pipeline/format_inference.py`)

Runs format variant experiments to test how prompt/answer formatting affects model accuracy.

```bash
uv run python pipeline/format_inference.py \
    --input data/format_sensitivity/input_data/linalg_bench_3x3_formats.csv \
    --model DeepSeek-V3 \
    --output data/format_sensitivity/results/ \
    --dry-run
```

Evaluation: `pipeline/format_eval.py` scores format variant responses against ground truth.

---

## Data Files

```
data/
  linalg_bench_3x3.csv      # 220 benchmark problems (3×3)
  linalg_bench_4x4.csv      # 220 benchmark problems (4×4)
  linalg_bench_5x5.csv      # 220 benchmark problems (5×5)
  output/                   # All outputs go here
    {ModelName}/
      {ModelName}_results.jsonl        # Stage 1 results
      {ModelName}_summary.csv       # Accuracy summary
      {ModelName}_failures.jsonl    # Failures for Stage 2
      judge/
        {subcat}_judge_labels.csv   # Stage 2 output
        {subcat}_judge_validated.csv # Stage 3 output
```

All pipeline output is written to `data/output/{Model}/` — this is the central location forResults, failures JSONL, and judge CSVs.

---

## Environment Variables

| Variable | Required for |
|----------|-------------|
| `OPENROUTER_API_KEY` | Qwen, Llama, Claude, Mistral, DeepSeek |
| `OPENAI_API_KEY` | GPT-4o, GPT-5.2, o1 |
| `GEMINI_API_KEY` | Gemini-3.0-Pro; all judge calls (Stages 2–3) |

---

## Citation

```bibtex
@inproceedings{agarwal2026linalgbench,
  title     = {LinAlg-Bench: A Benchmark Exposing Structural Failure Modes in LLM Linear
               Algebra -- Where Models Stop Computing and Start Hallucinating},
  author    = {Agarwal, Shradha and Jamil, Tariq and Rajbhar, Deepak},
  booktitle = {NeurIPS 2026 Workshop on Mathematical Reasoning and AI (MATH-AI)},
  year      = {2026}
}
```

---

## License

CC BY 4.0 — see `LICENSE`.
