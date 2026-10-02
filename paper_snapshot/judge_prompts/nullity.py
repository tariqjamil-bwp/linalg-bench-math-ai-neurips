"""
prompts/nullity.py — Build-judge and validate-judge prompts for the NULLITY subcategory.

Exports:
  BUILD_SYSTEM          → system prompt for label_judge.py
  BUILD_USER_TEMPLATE   → user template for label_judge.py
  VALIDATE_SYSTEM       → system prompt for validate_judge.py
  VALIDATE_USER_TEMPLATE→ user template for validate_judge.py
"""

from string import Template
from .taxonomy_reference_card import (
    TRUNCATION_PRECHECK,
    MAGNITUDE_RULE,
    ADVERSARIAL_FRAMING,
    ANSWER_CHECK_STEP0,
)

TAG_DEFINITIONS = """
───────────────────────────────────────────────────────────────────────────
GENERATION_TRUNCATION
  The response ends mid-expression or mid-computation with no final answer.
  No mathematical error is classifiable — the computation was never finished.
  DETECT: Response ends at "= " or mid-word. No \\boxed{} present.
  Use this before attempting any other classification.

───────────────────────────────────────────────────────────────────────────
FORMATTING_MISMATCH
  The model's mathematics is 100% correct but the final answer is presented
  incorrectly (wrong box format, missing required structure from the output
  specification).
  DETECT: Every intermediate (row operations, pivots, nullity subtraction)
  matches ground truth. Only the final boxed presentation fails the formatting
  requirement.
  CRITICAL: Do NOT use if any intermediate value is wrong.
  NOTE: Rare for nullity since the answer is a single integer.

───────────────────────────────────────────────────────────────────────────
SIGN_ERROR
  The first wrong value has the correct absolute magnitude but wrong sign.
  See MAGNITUDE RULE above — do not use ARITHMETIC for sign flips.
  Choose exactly one subtype:

  Product_Sign_Error
    A product of two terms has correct magnitude but wrong sign.
    Applies in ANY algorithm — row reduction, dot product, scalar multiplication.
    The multiplier k is correct in magnitude but the sign of the scaled row
    is flipped.
    DETECT: |product| = |correct|, sign differs in a multiply step.
    Nullity example: should compute R2 = R2 − 3·R1 but the model formed
    the product as +3·R1 instead of −3·R1, corrupting the pivot count.

  Operation_Direction
    Model added where it should have subtracted, or vice versa.
    DETECT: Accumulation operator +/− applied in wrong direction.
    Nullity example: Model eliminates a(2,1) by adding instead of subtracting
    the scaled pivot row, making the entry non-zero when it should be 0.

  Rule_Interference
    A negative matrix entry causes the model to mishandle the sign —
    double-applying or cancelling the negative from that entry.
    DETECT: Entry is negative; model treats it as positive in computation.
    Nullity example: Entry a(i,j) = −k; during R_p = R_p − m·R_q the model
    computes −m·(−k) incorrectly (e.g., treating it as −m·k), corrupting
    the resulting row and downstream pivot count.

  Parametric_Sign_Flip
    A sign error in a parameterized null-space solution. The free variable
    coefficient has the correct magnitude but wrong sign in the final answer.
    DETECT: Nullity problem; parameterized vector entries have flipped signs
    relative to ground truth. |model value| = |correct value|, sign differs.
    Nullity example: −3·(−2) computed as −6 instead of +6 during elimination.

  Silent_Sign_Flip
    BEHAVIORAL DEFINITION — do not infer intent:
    A sign-valued output appears in the response AND zero intermediate
    computation is shown for that specific step.
    DETECT: Surrounding steps have shown working. This one step has only
    a stated row entry with the wrong sign, no supporting computation.
    Do NOT use if any computation is shown for that step.
    MAGNITUDE RULE applies: |wrong| must equal |correct|; if magnitudes
    differ and no working is shown, classify as ARITHMETIC.
    Nullity example: Rows 1–3 correct and fully shown; row 4 states an
    entry with correct magnitude but wrong sign, no computation shown.

───────────────────────────────────────────────────────────────────────────
ARITHMETIC
  The first wrong value has the correct sign but wrong absolute magnitude.
  Method correct, signs tracked correctly, but a numerical calculation
  produced the wrong number. This includes errors in the final
  nullity = n − rank subtraction (correct rank, correct n, wrong result).
  See MAGNITUDE RULE — do not use for sign flips.
  DETECT: Wrong products or sums where sign is correct but |value| is wrong.
  Nullity example: Model computes 3·(−4) = −10 instead of −12, or correctly
  finds rank = 3 for a 5-column matrix but then writes nullity = 5 − 3 = 1
  instead of 2.

───────────────────────────────────────────────────────────────────────────
HALLUCINATION
  The model does not produce a wrong intermediate value by computation.
  Instead it fabricates, abandons, or invents. Choose one subtype:

  Complete_Collapse
    Model explicitly abandons row reduction with a meta-statement.
    Response is short (under 3000 characters). Writes phrases like
    "due to the complexity", "we can observe that", "the nullity is clearly"
    and outputs a fabricated integer.
    DETECT: Short response + explicit meta-statement + no REF computed.

  Teleological_Zeroing
    Model shows arithmetic for a row operation but writes [0 0 … 0] as the
    result despite the arithmetic not supporting it. The zero row is asserted
    to inflate the nullity count, not derived from the computation.
    DETECT: Arithmetic shown → result should be non-zero → model writes
    all zeros anyway.

  Premature_Assertion
    Model begins row reduction, completes only partial elimination, then
    states a nullity as if the computation were complete.
    DETECT: Medium-length response, partial row operations, then sudden
    nullity claim without finishing.

  Silent_Omission
    Model skips computation blocks silently with no meta-statement.
    Response continues as if skipped steps were completed.
    NOTE: Do NOT use for truncated responses — use GENERATION_TRUNCATION.
    DETECT: Long response + "after elimination..." + no row matrices shown.

  Ungrounded_Guess
    Essentially no working shown. Model outputs a final nullity with
    minimal or no supporting computation.
    DETECT: Very short response, no row operations shown, final integer
    presented as if obvious.

  Spontaneous_Insertion
    Model completes a correct computation chain then inserts a fabricated
    pivot value, row entry, or nullity claim with no mathematical origin
    in prior steps.
    DETECT: All row operations up to the insertion point are correct.
    A large or arbitrary value appears in a row or pivot that cannot be
    traced to any prior row operation.

───────────────────────────────────────────────────────────────────────────
INPUT_TRANSCRIPTION
  The model miscopied an entry from the ORIGINAL INPUT matrix.
  Error occurs before any arithmetic — the value was never computed,
  just read from the problem statement incorrectly.
  DETECT: Model's row operation uses a value that does not match the
  corresponding entry in the given matrix A. Method is correct, arithmetic
  on stated values is correct, but the source value was wrong.

  SCOPE CHECK (mandatory after finding a miscopied entry):
  Check ALL adjacent entries in the same row and column.
  If the model replaced an entire row or column with fabricated data,
  do NOT use INPUT_TRANSCRIPTION — escalate to HALLUCINATION
  (Silent_Omission or Spontaneous_Insertion).

───────────────────────────────────────────────────────────────────────────
CARRY_DOWN_ERROR
  The model correctly computed an intermediate value at step N and stated
  it correctly, but then miscopied that same value at step N+1 when
  carrying it forward. A line-to-line copy error, not an arithmetic error.
  DETECT: Row R_i stated correctly after elimination at step N. Same row
  used with a changed entry at step N+1 with no new computation applied.
  CRITICAL: The FIRST occurrence must be genuinely correct. If the first
  occurrence is also wrong, classify as ARITHMETIC or SIGN_ERROR instead.
  Boundary with MEMORY_LOSS: CARRY_DOWN_ERROR = N→N+1 (no intervening
  steps); MEMORY_LOSS = value recalled after ≥1 unrelated intervening steps.

───────────────────────────────────────────────────────────────────────────
METHOD_FAIL
  The model uses a fundamentally wrong algorithm from the very start.
  DETECT: First 3–4 steps do not follow any valid method for nullity.
  CRITICAL: A valid alternative algorithm is NOT a METHOD_FAIL.
  Only use METHOD_FAIL if the algorithm itself is incorrect AND applied
  incorrectly — not merely different from the expected approach.
  Examples: counting non-zero rows of the ORIGINAL matrix and calling that
  the nullity, applying rank-nullity theorem with n = number of rows instead
  of columns, computing the determinant and claiming nullity = 0 or n,
  confusing nullity with rank, computing trace instead of nullity.

───────────────────────────────────────────────────────────────────────────
MEMORY_LOSS
  The model correctly computed and stated a value at step N, then recalled
  a different wrong value for the same quantity at step M > N.
  DETECT: Row R2 stated correctly as [0, 5, −2, 3, 1] at step 4, but
  the model uses [0, 5, −2, 3, 0] for that same row at step 7 without any
  further operation applied to it. Also applies if correct rank was stated
  earlier but a different rank value is used when computing nullity = n − rank.
  CRITICAL: If the first occurrence is also wrong, classify as ARITHMETIC
  or SIGN_ERROR, not MEMORY_LOSS.
  Boundary with CARRY_DOWN_ERROR: MEMORY_LOSS = recalled after ≥1 unrelated
  intervening steps; CARRY_DOWN_ERROR = N→N+1 immediate copy.

───────────────────────────────────────────────────────────────────────────
OTHER_UNMAPPED
  Use ONLY if the error genuinely cannot be mapped to any tag above after
  careful analysis. Populate proposed_novel_tag and maps_closest_to.
  This is not a default — exhaust all other options first.

───────────────────────────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────────────────────────────────────
# BUILD — SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

BUILD_SYSTEM_TEMPLATE = Template("""You are a forensic mathematics auditor for an AI research paper
benchmarking LLMs on matrix nullity computation.

You will be given a matrix A and a model's FAILED attempt to compute its nullity.

Recall: nullity(A) = n − rank(A), where n is the number of columns.
Equivalently, nullity equals the number of free variables in the RREF of A.

The model may have used any of these strategies:

  STRATEGY 1 — GAUSSIAN ELIMINATION (Row Echelon Form)
    The model reduces A to row echelon form (REF), counts the pivot rows to get
    rank, then computes nullity = n − rank.
    Failure mode: arithmetic or sign errors in row operations corrupt the pivot
    count, causing wrong rank and therefore wrong nullity.

  STRATEGY 2 — REDUCED ROW ECHELON FORM (RREF / Gauss-Jordan)
    The model fully reduces A to RREF and directly counts the free variables
    (columns without pivots) to read off the nullity.
    Failure mode: back-substitution errors or pivot normalisation errors cause
    incorrect identification of free variables.

  STRATEGY 3 — RANK-NULLITY THEOREM (explicit two-step)
    The model explicitly computes rank(A) by any means, then separately applies
    nullity = n − rank(A) as a second step.
    Failure mode: (a) rank computed wrong, or (b) rank correct but wrong n used
    in the subtraction (e.g., uses number of rows instead of columns).

  STRATEGY 4 — HEURISTIC / VISUAL INSPECTION
    The model guesses nullity from surface features (zero rows, apparent
    dependencies) without formal row reduction.
    Failure mode: misses non-obvious linear dependencies; hallucinates nullity
    without grounded computation.

${TRUNCATION_PRECHECK}

YOUR TASK HAS THREE STEPS:

STEP 1 — COMPUTE THE CORRECT NULLITY YOURSELF
  You are a 100% accurate nullity calculator. Perform row reduction on A to
  row echelon form. Count the pivot rows to determine rank(A), then compute
  nullity = n − rank(A) where n is the number of columns of A.
  Do not reveal these intermediate values in your response.

STEP 2 — IDENTIFY THE STRATEGY THE MODEL USED
  Based on the model's response, decide which of the four strategies above it
  attempted. Record this as "solution_strategy" in your output.

STEP 3 — SCAN THE MODEL RESPONSE FOR THE FIRST WRONG VALUE
  Read the model response step by step from beginning to end.
  Compare every intermediate value (row operation result, pivot entry, row
  after elimination, free variable identification, final subtraction) against
  your correct computation.
  Find the FIRST step where the model's value diverges from correct.

  CRITICAL RULE: Classify by first point of failure, not final symptom.
  If the model makes a sign error in a row operation at step 4 that cascades
  to a wrong pivot count and therefore wrong nullity, the correct label is
  SIGN_ERROR, not ARITHMETIC or METHOD_FAIL.

${MAGNITUDE_RULE}

STEP 4 — CLASSIFY THE FIRST WRONG VALUE
  Use exactly one of these tags:

${TAG_DEFINITIONS}


RESPOND IN EXACTLY THIS JSON FORMAT AND NOTHING ELSE.

For SIGN_ERROR:
{
  "error_tag": "SIGN_ERROR",
  "sign_subtype": "Product_Sign_Error",
  "hallucination_subtype": "",
  "solution_strategy": "GAUSSIAN_ELIMINATION",
  "first_error_step": "Step 3 — eliminating a(2,1)",
  "first_error_description": "Model formed +3·R1 instead of −3·R1. Entry (2,1) should be 0 but model got 6, corrupting pivot count and therefore nullity. |6|=|−6| confirms sign error.",
  "proposed_novel_tag": "",
  "maps_closest_to": "",
  "confidence": "HIGH"
}

For HALLUCINATION:
{
  "error_tag": "HALLUCINATION",
  "sign_subtype": "",
  "hallucination_subtype": "Complete_Collapse",
  "solution_strategy": "GAUSSIAN_ELIMINATION",
  "first_error_step": "Step 2 — after completing only R1 elimination",
  "first_error_description": "Model wrote due to the complexity of this matrix we can observe the nullity is clearly 2 without completing row reduction.",
  "proposed_novel_tag": "",
  "maps_closest_to": "",
  "confidence": "HIGH"
}

For all other tags:
{
  "error_tag": "ARITHMETIC",
  "sign_subtype": "",
  "hallucination_subtype": "",
  "solution_strategy": "RANK_NULLITY_THEOREM",
  "first_error_step": "Final step — applying nullity = n − rank",
  "first_error_description": "Model correctly computed rank = 3 for a 5-column matrix but then wrote nullity = 5 − 3 = 1 instead of 2. |1|≠|2| confirms arithmetic error.",
  "proposed_novel_tag": "",
  "maps_closest_to": "",
  "confidence": "HIGH"
}

Confidence levels:
  HIGH   — you can point to a specific step and specific wrong value
  MEDIUM — fairly confident but response is ambiguous in places
  LOW    — response is very short or garbled, you are inferring
""").substitute(
    TRUNCATION_PRECHECK=TRUNCATION_PRECHECK,
    MAGNITUDE_RULE=MAGNITUDE_RULE,
    TAG_DEFINITIONS=TAG_DEFINITIONS,
)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD — USER TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

BUILD_USER_TEMPLATE = """\
Matrix A:
{matrix_latex}

Correct nullity: {ground_truth}
Model's extracted nullity answer: {extracted_answer}

Model's full response:
---
{response}
---

Perform the three-step audit described in the system prompt.
Find the FIRST wrong value and classify it.
Respond with only the JSON object.
"""


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATE — SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

VALIDATE_SYSTEM_TEMPLATE = Template("""

You will be given:
  1. A matrix A
  2. A model's failed response attempting to compute its nullity
  3. A classification made by a junior judge
  4. A targeted verification question for that error type

${ADVERSARIAL_FRAMING}

${ANSWER_CHECK_STEP0}

YOUR PROCESS:
  Step 1: Compute the correct nullity yourself. Perform row reduction on A to
          row echelon form (REF). Count the pivot rows to get rank(A), then
          compute nullity = n − rank(A). You are the ground truth oracle.
  Step 2: Read the model response step by step from beginning to end.
          Compare every intermediate value (row operation result, pivot entry,
          row after elimination, free variable count, final subtraction) against
          your correct computation. Find the FIRST step where the model diverges.
  Step 3: Answer the verification question with specific evidence quoted
          directly from the response. Confirm or deny the classification.

FIRST-ERROR PRINCIPLE:
  Classify by first point of failure, not final symptom.
  A sign error in a row operation at step 4 that cascades to a wrong pivot
  count and wrong nullity is SIGN_ERROR, not ARITHMETIC or METHOD_FAIL.

${MAGNITUDE_RULE}

STRICT TAXONOMY — primary_tag MUST be exactly one of these ten strings:
  GENERATION_TRUNCATION | FORMATTING_MISMATCH | SIGN_ERROR | ARITHMETIC |
  HALLUCINATION | INPUT_TRANSCRIPTION | CARRY_DOWN_ERROR | METHOD_FAIL |
  MEMORY_LOSS | OTHER_UNMAPPED

  Do NOT invent new primary tags. If error genuinely does not fit the
  ten primary tags, use OTHER_UNMAPPED and populate proposed_novel_tag
  and maps_closest_to.

SIGN_ERROR subtypes (required when primary_tag = SIGN_ERROR):
  Product_Sign_Error   - product has correct magnitude, wrong sign in multiply step
  Operation_Direction  - added scaled row where subtraction required (or vice versa)
  Rule_Interference    - negative entry triggers wrong sign behaviour during scaling
  Parametric_Sign_Flip - sign error in parameterized null-space solution
  Silent_Sign_Flip     - wrong-sign output with zero working shown for that step;
                         MAGNITUDE RULE applies: |wrong| must equal |correct|

HALLUCINATION subtypes (required when primary_tag = HALLUCINATION):
  Complete_Collapse     - explicit abandonment phrase, fewer than 2 rows reduced
  Teleological_Zeroing  - row written as [0…0] despite arithmetic not supporting it
  Premature_Assertion   - partial row reduction then nullity claimed as complete
  Silent_Omission       - skips computation blocks silently, no meta-statement
  Ungrounded_Guess      - essentially no working shown, just a final integer
  Spontaneous_Insertion - fabricated value inserted after a correct computation chain

RESPOND IN EXACTLY THIS JSON FORMAT AND NOTHING ELSE:
{
  "verified":              true,
  "primary_tag":           "INPUT_TRANSCRIPTION",
  "sign_subtype":          "",
  "hallucination_subtype": "",
  "proposed_novel_tag":    "",
  "maps_closest_to":       "",
  "forensic_observation":  "Confirmed: model miscopied a(2,3) = -3 as +3 when forming R2 at step 2, corrupting the pivot count and nullity result",
  "error_scope":           "isolated",
  "affected_positions":    ["a(2,3) at step 2"],
  "confidence":            "HIGH"
}

verified=true  - classification confirmed
verified=false - classification wrong, primary_tag is your corrected tag
verified=null  - insufficient evidence (truncated or genuinely ambiguous)

When verified=false, primary_tag must be your corrected tag, not the original.
When verified=false and primary_tag = SIGN_ERROR, populate sign_subtype.
When verified=false and primary_tag = HALLUCINATION, populate hallucination_subtype.
forensic_observation always required — at least one sentence of specific evidence.
confidence: HIGH=specific step+value identified, MEDIUM=fairly certain, LOW=inferring.

error_scope:          "isolated"    = one position wrong
                      "systematic"  = same error pattern across ≥2 positions
                                      sharing the same tag and subtype
affected_positions:   list the specific steps, rows, or entries where the
                      error manifests (e.g. ["R2 step 3", "R3 step 5"])
""").substitute(
    MAGNITUDE_RULE=MAGNITUDE_RULE,
    ADVERSARIAL_FRAMING=ADVERSARIAL_FRAMING,
    ANSWER_CHECK_STEP0=ANSWER_CHECK_STEP0,
)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATE — VERIFICATION QUESTION BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_verification_question(
    tag: str,
    sign_sub: str,
    halluc_sub: str,
    strategy: str,
    step: str,
    desc: str,
) -> str:
    """
    Return a targeted verification question for the senior validator based on
    the junior judge's classification tag and active subtype.
    """

    # ── GENERATION_TRUNCATION ─────────────────────────────────────────────
    if tag == "GENERATION_TRUNCATION":
        return """The junior judge classified this as GENERATION_TRUNCATION — the response
ends mid-expression or mid-computation with no final nullity answer.

Verification question:
  (1) Confirm the response has no final answer in \\boxed{} or equivalent.
  (2) Quote the last ~50 characters of the response. Does it end mid-sentence,
      mid-expression (e.g. "= "), or mid-row operation?
  (3) Does the response contain any complete nullity claim anywhere, even if
      stated before the apparent cut-off?

CRITICAL: If a complete nullity claim exists anywhere in the response, this is
NOT GENERATION_TRUNCATION. Reclassify based on whether that claim is correct
or fabricated. Truncation requires no final classifiable answer."""

    # ── FORMATTING_MISMATCH ───────────────────────────────────────────────
    elif tag == "FORMATTING_MISMATCH":
        return """The junior judge classified this as FORMATTING_MISMATCH — the model's
nullity computation is mathematically correct but the final answer format is wrong.

Verification question:
  (1) Verify every intermediate value (row operations, pivots, nullity
      subtraction) matches your correct computation.
  (2) Identify the specific formatting failure (missing \\boxed{}, answer
      stated as sentence, etc.).
  (3) Confirm the mathematical nullity value itself matches the ground truth.

CRITICAL: Do NOT confirm FORMATTING_MISMATCH if any intermediate value is
wrong. If the nullity value itself is wrong, this is a mathematical error
(ARITHMETIC, HALLUCINATION, etc.) — set verified=false and reclassify.
NOTE: This tag is rare for nullity since the answer is a single integer."""

    # ── INPUT_TRANSCRIPTION ───────────────────────────────────────────────
    elif tag == "INPUT_TRANSCRIPTION":
        return """The junior judge classified this as INPUT_TRANSCRIPTION — the model
miscopied an entry from the original input matrix A into a row operation.

Verification question:
Identify the EXACT cell that was miscopied. State:
  (1) Row and column in the original matrix A (e.g., a(2,3)),
  (2) Correct value at that position in A,
  (3) Value the model actually used in the row operation instead.

SCOPE CHECK (mandatory):
Check ALL adjacent entries in the same row and column of the model's
written-out submatrix. If the model replaced an entire row or column with
fabricated data (not just one cell), reclassify to HALLUCINATION
(Silent_Omission or Spontaneous_Insertion) — this exceeds the scope of
a single-cell transcription error.

CRITICAL: You must find both values directly in the response text.
If you cannot identify a specific miscopied entry with both the correct and
wrong value present, set verified=false and give your corrected tag.
Do not accept vague claims — exact row, column, and both values required."""

    # ── CARRY_DOWN_ERROR ──────────────────────────────────────────────────
    elif tag == "CARRY_DOWN_ERROR":
        return """The junior judge classified this as CARRY_DOWN_ERROR — the model
correctly stated a row or entry at step N but miscopied it at step N+1
when carrying it forward into the next operation.

Verification question:
  (1) Quote the correctly-stated row or entry at step N (with step label).
  (2) Quote the changed form at step N+1 (must be adjacent — no intervening
      unrelated steps between N and N+1).
  (3) Confirm the first occurrence matches your correct row reduction.
  (4) Confirm there is no new row operation between N and N+1 that would
      legitimately change the row.

CRITICAL: If the first occurrence is itself wrong, this is ARITHMETIC or
SIGN_ERROR — not CARRY_DOWN_ERROR. Set verified=false and reclassify.
If the two occurrences are separated by ≥1 unrelated intervening steps,
reclassify as MEMORY_LOSS instead."""

    # ── ARITHMETIC ────────────────────────────────────────────────────────
    elif tag == "ARITHMETIC":
        return """The junior judge classified this as ARITHMETIC — correct method and
correct signs, but wrong magnitude in a numerical calculation or in the
final nullity = n − rank subtraction.

Verification question:
Find the specific step where the wrong number first appeared.
  (1) Were ALL signs in that operation correct going in?
  (2) Apply the MAGNITUDE RULE: is |wrong value| ≠ |correct value|?
  (3) Was the operation a legitimate step at that point?
      (Include: correct rank but wrong n − rank result.)

CRITICAL: Apply MAGNITUDE RULE strictly. If |wrong| = |correct|, the root
error is SIGN_ERROR not ARITHMETIC — set verified=false and correct the tag.
If any sign was wrong BEFORE or DURING that computation, also reclassify as
SIGN_ERROR. Also check: if the model used wrong n (rows instead of columns)
in the final subtraction, reclassify as METHOD_FAIL.
Only set verified=true if signs were genuinely correct AND magnitude failed."""

    # ── SIGN_ERROR subtypes ───────────────────────────────────────────────
    elif tag == "SIGN_ERROR" and sign_sub == "Product_Sign_Error":
        return f"""The junior judge classified this as SIGN_ERROR/Product_Sign_Error —
a product of two terms in a row operation has correct magnitude but wrong sign.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
Find the exact row operation where this occurred. State:
  (1) Which rows were involved (e.g., R3 = R3 − k·R1),
  (2) The correct multiplier k and the sign the product should carry,
  (3) The sign the model actually used on the product,
  (4) The specific entry that became wrong as a result.

CRITICAL: Apply MAGNITUDE RULE. |product| must be correct — only the sign
is flipped. If |wrong| ≠ |correct|, reclassify as ARITHMETIC.
If the model used addition instead of subtraction as the operation itself,
reclassify as Operation_Direction."""

    elif tag == "SIGN_ERROR" and sign_sub == "Operation_Direction":
        return f"""The junior judge classified this as SIGN_ERROR/Operation_Direction —
the model added a scaled row where it should have subtracted (or vice versa),
making the target entry non-zero when it should be 0.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
Find the exact row operation. State:
  (1) Which entry was being eliminated (row, column),
  (2) The correct operation (e.g., R2 = R2 − 3·R1),
  (3) What the model wrote instead (e.g., R2 = R2 + 3·R1),
  (4) The resulting entry value vs the correct value of 0.

CRITICAL: Confirm the target entry should be 0 after correct elimination.
If |product| is correct but only the multiplier sign was flipped independently,
reclassify as Product_Sign_Error."""

    elif tag == "SIGN_ERROR" and sign_sub == "Rule_Interference":
        return f"""The junior judge classified this as SIGN_ERROR/Rule_Interference —
a negative matrix entry triggered wrong behaviour during a row operation.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Which matrix entry caused the interference? State row and column.
  (2) What is its correct value in A (must be negative)?
  (3) Show the row operation where the sign was mishandled
      (e.g., −m·(−k) computed as −m·k instead of +m·k).
  (4) What did the model compute vs the correct result?

CRITICAL: The entry must be negative and the error must be in how the model
handles that negative during scaling — not a misread at copy time (that is
INPUT_TRANSCRIPTION). If the model simply wrote the wrong operation direction
(+ instead of −), reclassify as Operation_Direction."""

    elif tag == "SIGN_ERROR" and sign_sub == "Parametric_Sign_Flip":
        return f"""The junior judge classified this as SIGN_ERROR/Parametric_Sign_Flip —
a sign error in a parameterized null-space solution. The free variable
coefficient has correct magnitude but wrong sign.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Identify the free variable and its coefficient in the model's answer.
  (2) Compare with the ground truth parameterized solution.
  (3) Confirm the coefficient magnitude is correct — only the sign differs.
  (4) Apply MAGNITUDE RULE: confirm |model value| = |correct value|.

CRITICAL: Nullity problems only. The parameterized solution must have
correct structure (correct number of free variables, correct basis vectors)
— only the sign on a parameter coefficient should be wrong."""

    elif tag == "SIGN_ERROR" and sign_sub == "Silent_Sign_Flip":
        return f"""The junior judge classified this as SIGN_ERROR/Silent_Sign_Flip —
BEHAVIORAL DEFINITION: a sign-valued output appears in the response AND
zero intermediate computation is shown for that specific step.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Identify the last step that was fully correct and shown (row, entry, value).
  (2) Identify the first entry with the wrong sign — state row, column,
      correct value, and model's value.
  (3) Confirm there is NO intermediate computation shown for that entry
      between the last correct step and the wrong result.
  (4) Apply MAGNITUDE RULE: confirm |wrong| = |correct| (magnitudes equal,
      sign flipped). If magnitudes differ, classify as ARITHMETIC instead.

CRITICAL: If working IS shown for that step, do not classify as Silent_Sign_Flip.
Identify the actual operation and reclassify (Product_Sign_Error,
Operation_Direction, Rule_Interference, or Parametric_Sign_Flip)."""

    elif tag == "SIGN_ERROR":
        return """The junior judge classified this as SIGN_ERROR but no subtype was given.

Verification question:
Find the first wrong value. Apply MAGNITUDE RULE:
  — If |wrong| ≠ |correct|: reclassify as ARITHMETIC (not SIGN_ERROR).
  — If |wrong| = |correct|, sign differs: confirm SIGN_ERROR, then identify subtype:

  Product_Sign_Error    — product has correct magnitude, sign flipped in multiply step
  Operation_Direction   — added where should subtract (or vice versa)
  Rule_Interference     — negative entry triggers wrong sign during scaling
  Parametric_Sign_Flip  — sign error in parameterized null-space solution
  Silent_Sign_Flip      — wrong-sign output with no working shown; |wrong|=|correct|

Populate sign_subtype in your response."""

    # ── HALLUCINATION subtypes ────────────────────────────────────────────
    elif tag == "HALLUCINATION" and halluc_sub == "Teleological_Zeroing":
        return """The junior judge classified this as HALLUCINATION/Teleological_Zeroing —
the model showed arithmetic for a row operation but wrote [0 0 … 0] as the
result despite the computation not supporting it.

Verification question:
  (1) Which row was written as all zeros?
  (2) Reproduce the row operation the model stated using correct matrix values.
      What should the result be?
  (3) Does the arithmetic shown (if any) actually yield zero?

CRITICAL: If the model's arithmetic IS correct and genuinely yields zero,
this is not Teleological_Zeroing — that is a correct step. The defining
feature is that the arithmetic does NOT yield zero but the model writes zero
anyway. If the model skips all working for that row, use Silent_Omission."""

    elif tag == "HALLUCINATION" and halluc_sub == "Complete_Collapse":
        return """The junior judge classified this as HALLUCINATION/Complete_Collapse —
the model explicitly abandoned row reduction with a meta-statement and
fewer than 2 rows were actually reduced.

Verification question:
  (1) Quote the exact phrase where the model abandoned computation
      (e.g., "due to the complexity", "the nullity is clearly").
  (2) How many rows of the matrix did the model actually reduce before stopping?
      This must be fewer than 2 for Complete_Collapse.
  (3) Did the model show any valid elementary row operations?

CRITICAL: If the model reduced 2 or more rows before stopping, reclassify
as Premature_Assertion. If no meta-statement exists and computation is just
absent, reclassify as Silent_Omission or Ungrounded_Guess."""

    elif tag == "HALLUCINATION" and halluc_sub == "Premature_Assertion":
        return """The junior judge classified this as HALLUCINATION/Premature_Assertion —
the model began row reduction, completed only partial elimination, then
stated a nullity as if the computation were complete.

Verification question:
  (1) Which rows did the model actually reduce? List them.
  (2) Which rows remain unreduced when the nullity is claimed?
  (3) Quote the line where the nullity is asserted prematurely.

CRITICAL: At least one valid row operation must have been performed for this
to be Premature_Assertion. If no row operations exist, use Ungrounded_Guess.
If the model explicitly says the computation is too complex and stops,
reclassify as Complete_Collapse."""

    elif tag == "HALLUCINATION" and halluc_sub == "Silent_Omission":
        return """The junior judge classified this as HALLUCINATION/Silent_Omission —
the model produced a full-length response but silently skipped computation
blocks — row operations claimed without showing resulting row entries.

Verification question:
  (1) Which row reduction steps are absent (no resulting matrix rows shown)?
  (2) Is there any meta-statement or explanation for the skip?
  (3) Does the response jump from setup directly to a nullity conclusion?
  (4) Does the response end with a complete final answer, or is it truncated?

CRITICAL: If the response is truncated (no final answer), reclassify as
GENERATION_TRUNCATION. If there IS a meta-statement, reclassify as
Complete_Collapse. If very short with essentially no working, use Ungrounded_Guess."""

    elif tag == "HALLUCINATION" and halluc_sub == "Ungrounded_Guess":
        return """The junior judge classified this as HALLUCINATION/Ungrounded_Guess —
essentially no row reduction working shown; model states nullity with no
meaningful derivation.

Verification question:
Confirm the response contains no meaningful row reduction — no elementary
row operations, no pivot identification, no free variable counting.
  (1) Are there any valid row operations (R_i = R_i ± k·R_j) in the response?
  (2) Are any intermediate matrices shown after row operations?

CRITICAL: If any row operation exists, even incomplete, use Silent_Omission
or Premature_Assertion instead. Ungrounded_Guess is reserved for responses
that are essentially pure assertion with zero working."""

    elif tag == "HALLUCINATION" and halluc_sub == "Spontaneous_Insertion":
        return """The junior judge classified this as HALLUCINATION/Spontaneous_Insertion —
the model completed a correct computation chain up to a point, then inserted
a fabricated pivot value, row entry, or nullity claim with no mathematical
origin in the prior steps.

Verification question:
  (1) Identify the last row/entry that was correctly derived per your
      computation — state the step and value.
  (2) Identify the inserted value that has no prior basis — state exactly
      what the model wrote.
  (3) Confirm the inserted value cannot be derived from any preceding row
      operation shown in the response.

CRITICAL: If the inserted value can be traced to a preceding computation
(even a wrong one), this is ARITHMETIC or SIGN_ERROR, not Spontaneous_Insertion.
Spontaneous_Insertion requires a value with NO computational origin."""

    elif tag == "HALLUCINATION":
        return """The junior judge classified this as HALLUCINATION but no subtype was given.

Determine which subtype applies:
  Complete_Collapse     — explicit abandonment phrase + fewer than 2 rows reduced
  Teleological_Zeroing  — row written as [0…0] despite arithmetic not supporting it
  Premature_Assertion   — partial reduction then nullity claimed as complete
  Silent_Omission       — computation blocks silently missing, no meta-statement
  Ungrounded_Guess      — essentially no row operations shown, just a final integer
  Spontaneous_Insertion — fabricated value inserted after a correct computation chain

Populate hallucination_subtype in your response."""

    # ── MEMORY_LOSS ───────────────────────────────────────────────────────
    elif tag == "MEMORY_LOSS":
        return """The junior judge classified this as MEMORY_LOSS — the model computed
a correct intermediate value in an earlier step and stated it, then used
a different wrong value for that same quantity in a later step (recalled
after ≥1 unrelated intervening steps).

Verification question:
  (1) What value was stated correctly at the early step?
      Give step description and exact value (e.g., "R2 after elimination = [0, 3, -1, 2]",
      or "rank = 3 stated at step 7").
  (2) What value did the model use for that same row/entry/rank at the later step?
      Give step description and the wrong value used.
  (3) Was the FIRST occurrence genuinely correct per your computation?
  (4) Are the two occurrences separated by ≥1 unrelated intervening steps?

CRITICAL: If the first occurrence is also wrong, the root error is ARITHMETIC
or SIGN_ERROR — set verified=false and correct the tag. This includes recalling
the correct rank but using a different value when computing nullity = n − rank.
If the two occurrences are adjacent (N→N+1, no intervening steps), reclassify
as CARRY_DOWN_ERROR instead."""

    # ── METHOD_FAIL ───────────────────────────────────────────────────────
    elif tag == "METHOD_FAIL":
        return """The junior judge classified this as METHOD_FAIL — the model never
performed legitimate nullity computation.

Verification question:
Describe what method the model used in the first 3–4 steps.
Confirm it fundamentally differs from valid nullity computation.

Examples that qualify as METHOD_FAIL:
  - Counting non-zero rows of the ORIGINAL (unreduced) matrix and calling that the nullity
  - Using n = number of rows instead of columns in nullity = n − rank
  - Computing the determinant and claiming nullity = 0 or nullity = n
  - Confusing nullity with rank (reporting rank as nullity)
  - Applying column operations when row operations were needed

CRITICAL: If the model set up legitimate row operations but made errors
inside them, classify by the actual error type (ARITHMETIC, SIGN_ERROR, etc.)
and set verified=false. METHOD_FAIL is only for wrong algorithm from step 1,
or correct row reduction followed by fundamentally wrong final step
(e.g., using n = number of rows instead of columns)."""

    # ── OTHER_UNMAPPED / UNKNOWN ──────────────────────────────────────────
    else:
        return """The junior judge could not classify this error (UNKNOWN or OTHER_UNMAPPED).

Fresh classification — ignore the previous tag entirely.

Read the model's nullity computation from the beginning.
Find the first step where the model's value diverges from correct computation.

Classify using exactly one of the ten valid tags:
  GENERATION_TRUNCATION | FORMATTING_MISMATCH | SIGN_ERROR | ARITHMETIC |
  HALLUCINATION | INPUT_TRANSCRIPTION | CARRY_DOWN_ERROR | METHOD_FAIL |
  MEMORY_LOSS | OTHER_UNMAPPED

For SIGN_ERROR, first apply MAGNITUDE RULE (|wrong|=|correct| required),
then populate sign_subtype:
  Product_Sign_Error   | Operation_Direction  | Rule_Interference |
  Parametric_Sign_Flip | Silent_Sign_Flip

For HALLUCINATION, populate hallucination_subtype:
  Complete_Collapse     | Teleological_Zeroing | Premature_Assertion |
  Silent_Omission       | Ungrounded_Guess     | Spontaneous_Insertion

Use OTHER_UNMAPPED only if genuinely impossible to map after careful analysis.
If OTHER_UNMAPPED, populate proposed_novel_tag (e.g., "FREE_VARIABLE_COUNT_ERROR")
and maps_closest_to with the nearest valid tag."""


def build_validate_user_prompt(row: dict, response: str, matrix_latex: str) -> str:
    """
    Build the full user message for the senior validation judge.
    Combines the junior judge's classification summary with a targeted
    verification question derived from the error tag and subtype.
    """
    tag        = str(row.get("Error_Tag", "")            or "UNKNOWN").strip()
    sign_sub   = str(row.get("Sign_Subtype", "")         or "").strip()
    halluc_sub = str(row.get("Hallucination_Subtype", "") or "").strip()
    strategy   = str(row.get("Solution_Strategy", "")    or "").strip()
    step       = str(row.get("First_Error_Step", "")     or "").strip()
    desc       = str(row.get("First_Error_Description", "") or "").strip()

    active_subtype = sign_sub or halluc_sub or "(none)"

    vq = _build_verification_question(tag, sign_sub, halluc_sub, strategy, step, desc)

    return f"""Matrix A:
{matrix_latex}

Junior judge classification:
  Error Tag         : {tag}
  Active Subtype    : {active_subtype}
  Solution Strategy : {strategy or '(not specified)'}
  First error step  : {step or '(not specified)'}
  Description       : {desc or '(not specified)'}

Model's full response:
---
{response}
---

{vq}"""
#############################################################################################
# END OF FILE
#############################################################################################

