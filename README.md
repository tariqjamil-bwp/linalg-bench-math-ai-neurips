# LinAlg-Bench

**LinAlg-Bench** is a diagnostic benchmark for evaluating LLM reasoning on linear algebra problems across 9 operation types and 3 matrix sizes (3×3, 4×4, 5×5). Beyond accuracy measurement, it includes a three-stage forensic pipeline that classifies the root cause of every model failure into the paper's fourteen-tag error taxonomy (ten primary tags; 13 of the 14 tags appear in the released annotations — one tag was never assigned by the judges).

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

**2. Download the code and install dependencies**

```bash
git clone https://github.com/shradhautk/linalg-bench-math-ai-neurips.git
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
#   ANTHROPIC_API_KEY=...    # appendix_codes/ only — depth ladder + scaffold gradient
```

Stages 1–3 run from the repo root with `uv run python`. The `appendix_codes/` pipelines are the
exception — each runs from its own directory (see below).

---

## Stage 1: Inference (`main_pipeline/inference.py`)

Run zero-shot inference via CLI:

```bash
# Dry run (preview prompts, no API calls)
uv run python main_pipeline/inference.py \
    --input data/linalg_bench_3x3.csv \
    --model DeepSeek-V3 \
    --dry-run

# Full run
uv run python main_pipeline/inference.py \
    --input data/linalg_bench_3x3.csv \
    --model DeepSeek-V3

# Resume interrupted run
uv run python main_pipeline/inference.py \
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

## Stage 2: Build Judge (`main_pipeline/build_judge.py`)

Classify each failure into the error taxonomy via CLI:

```bash
uv run python main_pipeline/build_judge.py \
    --results data/output/{Model}/{Model}_failures.jsonl \
    --output  data/output/{Model}/judge/{subcat}_judge_labels.csv \
    --subcat  det \
    --judge-llm-id gemini-3.1-pro-preview

# Cheap test
uv run python main_pipeline/build_judge.py \
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

## Stage 3: Validate Judge (`main_pipeline/validate_judge.py`)

Second independent pass verifies or corrects Stage 2 classifications:

```bash
uv run python main_pipeline/validate_judge.py \
    --judge   data/output/{Model}/judge/{subcat}_judge_labels.csv \
    --results data/output/{Model}/{Model}_failures.jsonl \
    --output  data/output/{Model}/judge/{subcat}_judge_validated.csv \
    --subcat  det \
    --judge-llm-id gemini-3.1-pro-preview
```

Output `validated` values: `TRUE` · `FALSE` (see `corrected_tag`) · `NEEDS_REVIEW` · `SKIP`

**FINAL_TAG rule**: if `validated == FALSE` and `corrected_tag` non-empty → use `corrected_tag`; otherwise → use `Error_Tag`.

---

## Format Sensitivity (`main_pipeline/format_inference.py`)

Runs format variant experiments to test how prompt/answer formatting affects model accuracy.

```bash
uv run python main_pipeline/format_inference.py \
    --input  <expanded_formats.csv> \
    --model  DeepSeek-V3 \
    --output data/format_sensitivity/results/ \
    --dry-run
```

**The expanded format-variant CSV is not bundled.** It is derived from the benchmark CSVs by
expanding each problem across format variants, and must provide these columns (case-insensitive):

| Column | Accepted aliases |
|--------|------------------|
| `question_id` | `id` |
| `format_type` | `format` |
| `instruction` | `problem_text` + `problem_representation` |
| `answer_latex` | — |
| `subcategory` | required, else all rows are dropped in strict mode |

Evaluation: `main_pipeline/format_eval.py` scores format variant responses against ground truth.

---

## Appendix Experiments (`appendix_codes/`)

Three self-contained pipelines for extended experiments beyond the MATH-AI paper (results included in the extended dataset release). Each ships its own run guide with the full
command sequence, so those steps are not duplicated here:

| Experiment | Directory | Run guide |
|------------|-----------|-----------|
| Op-count control | `appendix_codes/opcount_control/` | `_RUN_GUIDE.txt` |
| Depth ladder (IBP) | `appendix_codes/depth_ladder_pipeline/` | `_RUN_GUIDE.txt` |
| Scaffold gradient | `appendix_codes/scaffold_grad_pipeline/` | `_RUN_GUIDE.txt` |

Unlike Stages 1–3, **run these from inside their own directory** — each writes to a relative
`results/` path, created on first run:

```bash
cd appendix_codes/opcount_control
python3 run_opcount_experiment.py --model claude-4.5-sonnet --k 6 11
```

Shared model registry and transport live in `appendix_codes/common/`. All three read the same `.env`
at the repo root. The depth-ladder and scaffold-gradient pipelines call Claude on its native
Anthropic API and additionally need `ANTHROPIC_API_KEY`; op-count control runs entirely through
OpenRouter.

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

`paper_snapshot/` holds the frozen 660-problem baseline and results exactly as reported in the MATH-AI 2026 paper.

All pipeline output is written to `data/output/{Model}/` — this is the central location for results, failures JSONL, and judge CSVs.

---

## Environment Variables

| Variable | Required for |
|----------|-------------|
| `OPENROUTER_API_KEY` | Qwen, Llama, Claude, Mistral, DeepSeek |
| `OPENAI_API_KEY` | GPT-4o, GPT-5.2, o1 |
| `GEMINI_API_KEY` | Gemini-3.1-Pro; all judge calls (Stages 2–3) |
| `ANTHROPIC_API_KEY` | `appendix_codes/` only — Claude on its native API for the depth-ladder and scaffold-gradient experiments |

---

## Citation

```bibtex
@inproceedings{agarwal2026linalgbench,
  title     = {LinAlg-Bench: A Benchmark Exposing Structural Failure Modes in LLM Linear
               Algebra -- Where Models Stop Computing and Start Hallucinating},
  author    = {Agarwal, Shradha and J., Tariq and Rajbhar, Deepak},
  booktitle = {NeurIPS 2026 Workshop on Mathematical Reasoning and AI (MATH-AI)},
  year      = {2026}
}
```

---

## License

CC BY 4.0 — see `LICENSE`.
