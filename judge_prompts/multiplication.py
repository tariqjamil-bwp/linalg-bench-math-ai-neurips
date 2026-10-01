"""
prompts/multiplication.py — Build-judge and validate-judge prompts for the MULTIPLICATION subcategory.

Exports:
  BUILD_SYSTEM          → system prompt for label_judge.py
  BUILD_USER_TEMPLATE   → user template for label_judge.py
  VALIDATE_SYSTEM       → system prompt for validate_judge.py
  build_validate_user_prompt(row, response, matrix_latex) → str
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
  DETECT: Response ends at "= " or mid-word. No complete matrix result present.
  Use this before attempting any other classification.

───────────────────────────────────────────────────────────────────────────
FORMATTING_MISMATCH
  The model's mathematics is 100% correct but the final answer is presented
  incorrectly (wrong matrix notation, missing required structure from the
  output specification).
  DETECT: Every dot product term and every entry matches ground truth.
  Only the final matrix presentation fails the formatting requirement.
  CRITICAL: Do NOT use if any entry value is wrong.

───────────────────────────────────────────────────────────────────────────
SIGN_ERROR
  The first wrong value has the correct absolute magnitude but wrong sign.
  See MAGNITUDE RULE above — do not use ARITHMETIC for sign flips.
  Choose exactly one subtype:

  Product_Sign_Error
    A product of two terms has correct magnitude but wrong sign.
    Applies in ANY algorithm — dot product terms, scalar multiplication.
    The individual product a·b is correct in magnitude but the sign is flipped.
    DETECT: |product| = |correct|, sign differs in a multiply step.
    Multiplication example: A_{i2}·B_{2j} = (−3)(4) should give −12 but
    model writes +12 — magnitude correct, sign flipped.

  Operation_Direction
    Model added a term where it should have subtracted, or vice versa,
    when accumulating the dot product sum.
    DETECT: Accumulation operator +/− applied in wrong direction in the sum.
    Multiplication example: running sum should subtract a term but model adds
    it, making the partial sum wrong when it should reduce.

  Rule_Interference
    A negative matrix entry causes the model to mishandle the sign —
    double-applying or cancelling the negative from that entry.
    DETECT: Entry is negative; model treats it as positive in computation.
    Multiplication example: Entry a = −k; model computes a·b as −(k·b)
    instead of correctly as (−k)·b, effectively treating a as +k.

  Double_Negative_Trap
    Two or more negatives met in a product or chain and the model failed
    to resolve them (e.g. −(−x) written as −x instead of +x).
    DETECT: Expression with two negatives; model produces wrong sign on
    the result while the magnitude is correct.
    Multiplication example: A_{ij} = −3, B_{jk} = −4; model computes
    (−3)·(−4) = −12 instead of +12.

  Silent_Sign_Flip
    BEHAVIORAL DEFINITION — do not infer intent:
    A sign-valued output appears in the response AND zero intermediate
    computation is shown for that specific entry.
    DETECT: Surrounding entries correct and fully shown with dot products.
    This one entry has only a stated value with the wrong sign, no dot
    product terms shown.
    Do NOT use if any dot product terms are shown for that entry.
    MAGNITUDE RULE applies: |wrong| must equal |correct|; if magnitudes
    differ and no working is shown, classify as ARITHMETIC.

───────────────────────────────────────────────────────────────────────────
ARITHMETIC
  The first wrong value has the correct sign but wrong absolute magnitude.
  Method correct, signs tracked correctly, but a numerical calculation
  produced the wrong number.
  See MAGNITUDE RULE — do not use for sign flips.
  DETECT: Wrong products or sums where sign is correct but |value| is wrong.
  Multiplication example: Model computes (−3)·4 = −10 instead of −12, or
  accumulates −6 + 16 − 12 − 12 = −4 instead of −14.

───────────────────────────────────────────────────────────────────────────
HALLUCINATION
  The model does not produce a wrong intermediate dot product value.
  Instead it fabricates, abandons, or invents. Choose one subtype:

  Complete_Collapse
    Model explicitly abandons computation with a meta-statement and fewer
    than 2 rows of AB have been computed. Short response.
    DETECT: Short response + explicit meta-statement ("due to the complexity",
    "we can observe") + no valid dot products computed.

  Teleological_Zeroing
    Model shows dot product terms for an entry but writes 0 as the result
    despite the terms not summing to zero. Zero is asserted, not derived.
    DETECT: Dot product terms shown → sum ≠ 0 → model writes 0 anyway.

  Premature_Assertion
    Model computes some rows of AB correctly then states the full result
    matrix as if complete without computing remaining rows.
    DETECT: Medium-length response, correct partial rows, then sudden full
    matrix claim without computing remaining entries.

  Silent_Omission
    Model produces a full-length response but silently skips dot product
    computations. Entries claimed without showing the sum of terms.
    NOTE: Do NOT use for truncated responses — use GENERATION_TRUNCATION.
    DETECT: Long response + "following the same process..." + no dot
    product terms shown for multiple entries.

  Ungrounded_Guess
    Essentially no working shown. Model outputs AB with minimal or no
    supporting dot product computation.
    DETECT: Very short response, no A_{ik}·B_{kj} terms shown, final
    matrix presented as if obvious.

  Spontaneous_Insertion
    Model completes a correct sequence of dot products then inserts a
    fabricated entry value with no mathematical origin in prior steps.
    DETECT: All dot products up to the insertion point are correct.
    An arbitrary entry value appears that cannot be traced to any prior
    dot product computation.

───────────────────────────────────────────────────────────────────────────
INPUT_TRANSCRIPTION
  The model miscopied an entry from the ORIGINAL INPUT matrices A or B.
  Error occurs before any arithmetic — the value was never computed,
  just read from the problem statement incorrectly.
  DETECT: Model's dot product uses a value that does not match the
  corresponding entry in the given matrices A or B. Method is correct,
  arithmetic on stated values is correct, but the source value was wrong.

  SCOPE CHECK (mandatory after finding a miscopied entry):
  Check ALL adjacent entries in the same row/column of the model's written
  submatrix. If the model replaced an entire row or column with fabricated
  data, do NOT use INPUT_TRANSCRIPTION — escalate to HALLUCINATION
  (Silent_Omission or Spontaneous_Insertion).

───────────────────────────────────────────────────────────────────────────
CARRY_DOWN_ERROR
  The model correctly computed a dot product term or entry at step N and
  stated it correctly, but then miscopied that same value at step N+1 when
  carrying it forward. A line-to-line copy error, not an arithmetic error.
  DETECT: Entry (AB)_{ij} or partial sum stated correctly at step N. Same
  value written differently at step N+1 with no new computation applied.
  CRITICAL: The FIRST occurrence must be genuinely correct. If the first
  occurrence is also wrong, classify as ARITHMETIC or SIGN_ERROR instead.
  Boundary with MEMORY_LOSS: CARRY_DOWN_ERROR = N→N+1 (no intervening
  steps); MEMORY_LOSS = value recalled after ≥1 unrelated intervening steps.

───────────────────────────────────────────────────────────────────────────
METHOD_FAIL
  The model uses a fundamentally wrong algorithm from the very start.
  DETECT: First 3–4 steps do not follow any valid matrix multiplication method.
  CRITICAL: A valid alternative algorithm is NOT a METHOD_FAIL.
  Only use METHOD_FAIL if the algorithm itself is incorrect AND applied
  incorrectly — not merely different from the expected approach.

  (base — no subtype)
    Model never performs row·column dot products. Examples: computing
    element-wise product (Hadamard) instead of matrix product, computing
    det(A)·det(B) as the answer, applying row reduction to find AB.
    DETECT: No row·column dot products performed; wrong procedure from step 1.

  Operand_Confusion
    Model computes valid dot products but on the wrong operand arrangement:
    BA instead of AB; rows of B dotted with columns of A throughout.
    Dot products are structurally correct — only the row/column pairing
    is inverted.
    DETECT: Dot products valid in form; result equals BA not AB.
    Disambiguate from base METHOD_FAIL: Operand_Confusion has valid dot
    products — only operand arrangement wrong. Base METHOD_FAIL covers
    truly wrong algorithms (e.g. Hadamard product, determinant).

───────────────────────────────────────────────────────────────────────────
MEMORY_LOSS
  The model correctly computed and stated a dot product entry at step N,
  then recalled a different wrong value for that same entry at step M > N.
  DETECT: Entry (AB)_{ij} stated correctly at one point; wrong value used
  for same entry later without intervening computation.
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
benchmarking LLMs on matrix multiplication computation.

You will be given two matrices A and B and a model's FAILED attempt to compute AB.

The model may have used any of these strategies:

  STRATEGY 1 — STANDARD DOT PRODUCT
    The model computes each entry (AB)_{ij} as the dot product of row i of A
    with column j of B: sum over k of A_{ik} * B_{kj}.
    Failure mode: arithmetic or sign errors in individual dot product terms
    accumulate across entries, corrupting rows or columns of the result.

  STRATEGY 2 — BLOCK / PARTITIONED MULTIPLICATION
    The model partitions A and B into submatrices and applies block
    multiplication rules to compute AB in sections.
    Failure mode: incorrect block boundaries or wrong block combination
    rules cause systematic errors across multiple entries.

  STRATEGY 3 — HEURISTIC / VISUAL INSPECTION
    The model guesses entries from surface features without computing
    individual dot products, or applies shortcuts without justification.
    Failure mode: fabricated entries with no supporting computation.

${TRUNCATION_PRECHECK}

YOUR TASK HAS THREE STEPS:

STEP 1 — COMPUTE THE CORRECT PRODUCT YOURSELF
  You are a 100% accurate matrix multiplication calculator. Compute AB
  independently by taking the dot product of each row of A with each column
  of B. Do not reveal these intermediate values in your response.

STEP 2 — IDENTIFY THE STRATEGY THE MODEL USED
  Based on the model's response, decide which of the three strategies above
  it attempted. Record this as "solution_strategy" in your output.

STEP 3 — SCAN THE MODEL RESPONSE FOR THE FIRST WRONG VALUE
  Read the model response step by step from beginning to end.
  Compare every computed entry (AB)_{ij} and every dot product term against
  your correct computation.
  Find the FIRST step where the model's value diverges from correct.

  CRITICAL RULE: Classify by first point of failure, not final symptom.
  If the model makes a sign error in a dot product term at entry (1,2) and
  this propagates to a wrong final matrix, the correct label is SIGN_ERROR,
  not ARITHMETIC.

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
  "solution_strategy": "DOT_PRODUCT",
  "first_error_step": "Entry (1,2) — dot product term A_{12}·B_{22}",
  "first_error_description": "Model computed (−3)·(4) = +12 instead of −12. |12|=|−12| confirms sign error.",
  "proposed_novel_tag": "",
  "maps_closest_to": "",
  "confidence": "HIGH"
}

For HALLUCINATION:
{
  "error_tag": "HALLUCINATION",
  "sign_subtype": "",
  "hallucination_subtype": "Premature_Assertion",
  "solution_strategy": "DOT_PRODUCT",
  "first_error_step": "After computing row 1 of AB",
  "first_error_description": "Model correctly computed row 1 then wrote 'following the same process for remaining rows' and stated the full AB matrix without computing rows 2–5.",
  "proposed_novel_tag": "",
  "maps_closest_to": "",
  "confidence": "HIGH"
}

For METHOD_FAIL/Operand_Confusion:
{
  "error_tag": "METHOD_FAIL",
  "sign_subtype": "",
  "hallucination_subtype": "",
  "solution_strategy": "DOT_PRODUCT",
  "first_error_step": "Entry (1,1) — first dot product",
  "first_error_description": "Model computed row 1 of B dotted with column 1 of A throughout, producing BA instead of AB.",
  "proposed_novel_tag": "Operand_Confusion",
  "maps_closest_to": "",
  "confidence": "HIGH"
}

For all other tags:
{
  "error_tag": "ARITHMETIC",
  "sign_subtype": "",
  "hallucination_subtype": "",
  "solution_strategy": "DOT_PRODUCT",
  "first_error_step": "Entry (1,5) — dot product sum",
  "first_error_description": "Model computed −6+0+16−12−12 = −4 instead of −14. Sign correct, |−4|≠|−14| confirms arithmetic error.",
  "proposed_novel_tag": "",
  "maps_closest_to": "",
  "confidence": "HIGH"
}

Confidence levels:
  HIGH   — you can point to a specific entry and specific wrong value
  MEDIUM — fairly confident but response is ambiguous in places
  LOW    — response is very short or garbled, you are inferring
""").substitute(
    MAGNITUDE_RULE=MAGNITUDE_RULE,
    TAG_DEFINITIONS=TAG_DEFINITIONS,
    TRUNCATION_PRECHECK=TRUNCATION_PRECHECK,
)

# ─────────────────────────────────────────────────────────────────────────────
# BUILD — USER TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

BUILD_USER_TEMPLATE = """\
Matrix A:
{matrix_latex}

Correct AB: {ground_truth}
Model's extracted AB answer: {extracted_answer}

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
  1. Two matrices A and B
  2. A model's failed response attempting to compute AB
  3. A classification made by a junior judge
  4. A targeted verification question for that error type

${ADVERSARIAL_FRAMING}

${ANSWER_CHECK_STEP0}

YOUR PROCESS:
  Step 1: Compute the correct AB yourself by taking the dot product of each
          row of A with each column of B. You are the ground truth oracle.
  Step 2: Read the model response step by step from beginning to end.
          Compare every dot product term and every entry of AB against your
          correct computation. Find the FIRST step where the model diverges.
  Step 3: Answer the verification question with specific evidence quoted
          directly from the response. Confirm or deny the classification.

FIRST-ERROR PRINCIPLE:
  Classify by first point of failure, not final symptom.
  A sign error in a dot product term at entry (1,2) that cascades to a wrong
  final matrix is SIGN_ERROR, not ARITHMETIC or METHOD_FAIL.

${MAGNITUDE_RULE}

STRICT TAXONOMY — primary_tag MUST be exactly one of these ten strings:
  GENERATION_TRUNCATION | FORMATTING_MISMATCH | SIGN_ERROR | ARITHMETIC |
  HALLUCINATION | INPUT_TRANSCRIPTION | CARRY_DOWN_ERROR | METHOD_FAIL |
  MEMORY_LOSS | OTHER_UNMAPPED

  Do NOT invent new primary tags. If error genuinely does not fit the
  ten primary tags, use OTHER_UNMAPPED and populate proposed_novel_tag
  and maps_closest_to.

SIGN_ERROR subtypes (required when primary_tag = SIGN_ERROR):
  Product_Sign_Error   - product has correct magnitude, wrong sign in a multiply step
  Operation_Direction  - added term where subtraction required in accumulation (or vice versa)
  Rule_Interference    - negative entry triggers wrong sign behaviour during multiplication
  Double_Negative_Trap - two negatives met in product chain, model fails to resolve sign
  Silent_Sign_Flip     - wrong-sign entry with no dot product terms shown;
                         MAGNITUDE RULE applies: |wrong| must equal |correct|

HALLUCINATION subtypes (required when primary_tag = HALLUCINATION):
  Complete_Collapse     - explicit abandonment phrase, fewer than 2 rows computed
  Teleological_Zeroing  - entry written as 0 despite dot products not summing to 0
  Premature_Assertion   - partial rows computed then full matrix claimed as complete
  Silent_Omission       - skips dot product blocks silently, no meta-statement
  Ungrounded_Guess      - essentially no dot products shown, just a final matrix
  Spontaneous_Insertion - fabricated entry inserted after a correct computation chain

METHOD_FAIL subtypes:
  (base)              - wrong algorithm entirely (Hadamard, determinant, etc.)
  Operand_Confusion   - valid dot products but BA computed instead of AB

RESPOND IN EXACTLY THIS JSON FORMAT AND NOTHING ELSE:
{
  "verified":              true,
  "primary_tag":           "ARITHMETIC",
  "sign_subtype":          "",
  "hallucination_subtype": "",
  "proposed_novel_tag":    "",
  "maps_closest_to":       "",
  "forensic_observation":  "Confirmed: model computed entry (1,5) as −6+0+16−12−12=−4 instead of −14. Sign correct, magnitude off.",
  "error_scope":           "isolated",
  "affected_positions":    ["entry (1,5)"],
  "confidence":            "HIGH"
}

verified=true  - classification confirmed
verified=false - classification wrong, primary_tag is your corrected tag
verified=null  - insufficient evidence (truncated or genuinely ambiguous)

When verified=false, primary_tag must be your corrected tag, not the original.
When verified=false and primary_tag = SIGN_ERROR, populate sign_subtype.
When verified=false and primary_tag = HALLUCINATION, populate hallucination_subtype.
forensic_observation always required — at least one sentence of specific evidence.
confidence: HIGH=specific entry+value identified, MEDIUM=fairly certain, LOW=inferring.

error_scope:          "isolated"    = one entry/term wrong
                      "systematic"  = same error pattern across ≥2 entries
                                      sharing the same tag and subtype
affected_positions:   list the specific entries or dot product steps where
                      the error manifests (e.g. ["entry (1,2)", "entry (2,3)"])
""").substitute(
    ADVERSARIAL_FRAMING=ADVERSARIAL_FRAMING,
    ANSWER_CHECK_STEP0=ANSWER_CHECK_STEP0,
    MAGNITUDE_RULE=MAGNITUDE_RULE,
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
    """Return a targeted verification question for the senior validator."""

    # ── GENERATION_TRUNCATION ─────────────────────────────────────────────
    if tag == "GENERATION_TRUNCATION":
        return """The junior judge classified this as GENERATION_TRUNCATION — the response
ends mid-expression or mid-computation with no final matrix result.

Verification question:
  (1) Confirm the response has no complete AB matrix in the final answer.
  (2) Quote the last ~50 characters of the response. Does it end mid-sentence,
      mid-expression (e.g. "= "), or mid-dot-product?
  (3) Does the response contain any complete AB claim anywhere, even if
      stated before the apparent cut-off?

CRITICAL: If a complete AB result exists anywhere in the response, this is
NOT GENERATION_TRUNCATION. Reclassify based on whether that result is correct
or fabricated. Truncation requires no final classifiable answer."""

    # ── FORMATTING_MISMATCH ───────────────────────────────────────────────
    elif tag == "FORMATTING_MISMATCH":
        return """The junior judge classified this as FORMATTING_MISMATCH — the model's
matrix multiplication is mathematically correct but the final answer format is wrong.

Verification question:
  (1) Verify every dot product term and every entry of AB matches your
      correct computation.
  (2) Identify the specific formatting failure (wrong matrix notation,
      missing brackets, etc.).
  (3) Confirm every numerical entry value matches the ground truth.

CRITICAL: Do NOT confirm FORMATTING_MISMATCH if any entry value is wrong.
If any entry is wrong, this is a mathematical error — set verified=false
and reclassify."""

    # ── INPUT_TRANSCRIPTION ───────────────────────────────────────────────
    elif tag == "INPUT_TRANSCRIPTION":
        return """The junior judge classified this as INPUT_TRANSCRIPTION — the model
miscopied an entry from the original input matrices A or B into a dot product.

Verification question:
Identify the EXACT entry that was miscopied. State:
  (1) Which matrix (A or B), row and column (e.g., A_{2,3}),
  (2) Correct value at that position,
  (3) Value the model actually used in the dot product instead.

SCOPE CHECK (mandatory):
Check ALL adjacent entries in the same row/column of the model's written
submatrix. If the model replaced an entire row or column with fabricated
data (not just one entry), reclassify to HALLUCINATION (Silent_Omission or
Spontaneous_Insertion) — this exceeds the scope of a single-entry transcription.

CRITICAL: You must find both values directly in the response text.
Exact matrix, row, column, and both values required."""

    # ── CARRY_DOWN_ERROR ──────────────────────────────────────────────────
    elif tag == "CARRY_DOWN_ERROR":
        return """The junior judge classified this as CARRY_DOWN_ERROR — the model
correctly stated a dot product term or entry at step N but miscopied it at
step N+1 when carrying it forward.

Verification question:
  (1) Quote the correctly-stated value at step N (with step label).
  (2) Quote the changed form at step N+1 (must be adjacent — no intervening
      unrelated steps).
  (3) Confirm the first occurrence matches your correct computation.
  (4) Confirm there is no new dot product computation between N and N+1
      that would legitimately change the value.

CRITICAL: If the first occurrence is itself wrong, this is ARITHMETIC or
SIGN_ERROR — not CARRY_DOWN_ERROR. Set verified=false and reclassify.
If separated by ≥1 unrelated intervening steps, reclassify as MEMORY_LOSS."""

    # ── ARITHMETIC ────────────────────────────────────────────────────────
    elif tag == "ARITHMETIC":
        return """The junior judge classified this as ARITHMETIC — correct dot product
method and correct signs, but wrong magnitude in a numerical calculation.

Verification question:
Find the specific entry where the wrong number first appeared.
  (1) Were ALL signs in that dot product correct?
  (2) Apply the MAGNITUDE RULE: is |wrong value| ≠ |correct value|?
  (3) Was the row·column pairing (row of A × column of B) correct?

CRITICAL: Apply MAGNITUDE RULE strictly. If |wrong| = |correct|, the root
error is SIGN_ERROR not ARITHMETIC — set verified=false and correct the tag.
If any sign was wrong BEFORE or DURING that computation, also reclassify as
SIGN_ERROR. Only set verified=true if signs were genuinely correct AND
magnitude failed."""

    # ── SIGN_ERROR subtypes ───────────────────────────────────────────────
    elif tag == "SIGN_ERROR" and sign_sub == "Product_Sign_Error":
        return f"""The junior judge classified this as SIGN_ERROR/Product_Sign_Error —
the model used the correct magnitude for a dot product term but applied
the wrong sign to it.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
Find the exact dot product term where this occurred. State:
  (1) Which entry (AB)_{{ij}} and which term in the dot product (k-th term),
  (2) The correct values of A_{{ik}} and B_{{kj}} and their correct product,
  (3) The sign the model actually used on that product,
  (4) The resulting accumulation error in that entry.

CRITICAL: Apply MAGNITUDE RULE. |product| must be correct — only the sign
is flipped. If |wrong| ≠ |correct|, reclassify as ARITHMETIC.
If the model used the wrong entry values entirely, reclassify as
INPUT_TRANSCRIPTION."""

    elif tag == "SIGN_ERROR" and sign_sub == "Operation_Direction":
        return f"""The junior judge classified this as SIGN_ERROR/Operation_Direction —
the model added a dot product term where it should have subtracted (or vice
versa) when accumulating the sum for an entry.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
Find the exact accumulation step where this occurred. State:
  (1) Which entry (AB)_{{ij}} and which accumulation step,
  (2) The correct operation (e.g., running_sum += A_{{ik}}·B_{{kj}}),
  (3) What the model wrote instead,
  (4) The resulting partial sum vs the correct partial sum.

CRITICAL: Confirm the direction of accumulation was wrong — not just the
sign of the individual term. If the term's own sign was flipped independently,
reclassify as Product_Sign_Error."""

    elif tag == "SIGN_ERROR" and sign_sub == "Rule_Interference":
        return f"""The junior judge classified this as SIGN_ERROR/Rule_Interference —
a negative matrix entry triggered wrong sign behaviour during multiplication,
causing the model to double-apply or cancel the sign.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Which entry caused the interference? State matrix (A or B), row, column.
  (2) What is its correct value (must be negative)?
  (3) Show the multiplication where the sign was mishandled
      (e.g., (−k)·b computed as −(k·b) treating entry as positive).
  (4) What did the model compute vs the correct result?

CRITICAL: The entry must be negative and the error must be in how the model
handles that negative during multiplication — not a misread at copy time
(that is INPUT_TRANSCRIPTION)."""

    elif tag == "SIGN_ERROR" and sign_sub == "Double_Negative_Trap":
        return f"""The junior judge classified this as SIGN_ERROR/Double_Negative_Trap —
two or more negatives met in a product and the model failed to resolve them
correctly (e.g. (−3)·(−4) computed as −12 instead of +12).
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Quote the exact expression containing the two negatives.
  (2) State the correct resolution and the model's resolution.
  (3) Confirm BOTH factors are genuinely negative (from A and B entries).
  (4) Apply MAGNITUDE RULE: confirm |product| is correct — only the sign wrong.

CRITICAL: If only one factor is negative, this is not Double_Negative_Trap.
If |wrong| ≠ |correct|, reclassify as ARITHMETIC."""

    elif tag == "SIGN_ERROR" and sign_sub == "Silent_Sign_Flip":
        return f"""The junior judge classified this as SIGN_ERROR/Silent_Sign_Flip —
BEHAVIORAL DEFINITION: a sign-valued output appears in the response AND
zero intermediate computation is shown for that specific entry.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Identify the last entry that was correct and fully shown with dot products.
  (2) Identify the first entry with the wrong sign — state (i,j),
      correct value, and model's value.
  (3) Confirm there are NO dot product terms shown for that entry.
  (4) Apply MAGNITUDE RULE: confirm |wrong| = |correct| (magnitudes equal,
      sign flipped). If magnitudes differ, classify as ARITHMETIC instead.

CRITICAL: If dot product terms ARE shown for that entry, identify the actual
term that produced the wrong sign and reclassify (Product_Sign_Error,
Operation_Direction, Rule_Interference, or Double_Negative_Trap)."""

    elif tag == "SIGN_ERROR":
        return """The junior judge classified this as SIGN_ERROR but no subtype was given.

Verification question:
Find the first wrong value. Apply MAGNITUDE RULE:
  — If |wrong| ≠ |correct|: reclassify as ARITHMETIC (not SIGN_ERROR).
  — If |wrong| = |correct|, sign differs: confirm SIGN_ERROR, then identify subtype:

  Product_Sign_Error    — correct product magnitude, sign flipped on a term
  Operation_Direction   — added where should subtract in accumulation (or vice versa)
  Rule_Interference     — negative entry triggers wrong sign during multiplication
  Double_Negative_Trap  — two negatives met, sign not correctly resolved
  Silent_Sign_Flip      — wrong-sign entry with no dot product terms shown; |wrong|=|correct|

Populate sign_subtype in your response."""

    # ── HALLUCINATION subtypes ────────────────────────────────────────────
    elif tag == "HALLUCINATION" and halluc_sub == "Teleological_Zeroing":
        return """The junior judge classified this as HALLUCINATION/Teleological_Zeroing —
the model showed dot product terms for an entry but wrote 0 as the result
despite the terms not summing to zero.

Verification question:
  (1) Which entry (AB)_{ij} was written as 0?
  (2) List the dot product terms the model stated for that entry.
      What is their correct sum?
  (3) Does the model show any arithmetic for why the sum is 0?

CRITICAL: If the shown terms DO correctly sum to 0, this is not
Teleological_Zeroing — that is a correct entry. The defining feature is that
the arithmetic shown does NOT support zero but model writes 0 anyway.
If no dot product terms shown, use Silent_Omission."""

    elif tag == "HALLUCINATION" and halluc_sub == "Complete_Collapse":
        return """The junior judge classified this as HALLUCINATION/Complete_Collapse —
the model explicitly abandoned computation with a meta-statement and fewer
than 2 rows of AB were computed.

Verification question:
  (1) Quote the exact phrase where the model abandoned computation.
  (2) How many full rows of AB did the model actually compute before stopping?
      This must be fewer than 2 for Complete_Collapse.
  (3) Did the model show any valid dot products at all?

CRITICAL: If the model computed 2 or more rows before stopping, reclassify
as Premature_Assertion. If no meta-statement and computation is just absent,
reclassify as Silent_Omission or Ungrounded_Guess."""

    elif tag == "HALLUCINATION" and halluc_sub == "Premature_Assertion":
        return """The junior judge classified this as HALLUCINATION/Premature_Assertion —
the model began computing AB, completed only some rows, then stated the
full result matrix as if complete.

Verification question:
  (1) Which rows of AB did the model actually compute with dot products shown?
  (2) Which rows are absent when the full matrix is claimed?
  (3) Quote the line where the full matrix is asserted prematurely.

CRITICAL: At least one valid row of dot products must have been computed.
If no dot products exist, use Ungrounded_Guess.
If the model explicitly says computation is too complex and stops,
reclassify as Complete_Collapse."""

    elif tag == "HALLUCINATION" and halluc_sub == "Silent_Omission":
        return """The junior judge classified this as HALLUCINATION/Silent_Omission —
the model produced a full-length response but silently skipped dot product
computations for multiple entries.

Verification question:
  (1) Which entries have no dot product terms shown?
  (2) Is there any meta-statement or explanation for the skip?
  (3) Does the response jump from partial computation to the full result?
  (4) Does the response end with a complete final answer, or is it truncated?

CRITICAL: If the response is truncated (no final answer), reclassify as
GENERATION_TRUNCATION. If there IS a meta-statement, reclassify as
Complete_Collapse. If very short with no working, use Ungrounded_Guess."""

    elif tag == "HALLUCINATION" and halluc_sub == "Ungrounded_Guess":
        return """The junior judge classified this as HALLUCINATION/Ungrounded_Guess —
essentially no dot product working shown; model states AB with no derivation.

Verification question:
Confirm the response contains no meaningful dot product computation — no
A_{ik}·B_{kj} terms, no partial sums, no row-by-column pairings.
  (1) Are there any valid dot product terms shown for any entry?
  (2) Are any intermediate sums shown?

CRITICAL: If any dot product term exists, even for a single entry, use
Silent_Omission or Premature_Assertion instead. Ungrounded_Guess is reserved
for responses that are essentially pure assertion with zero working."""

    elif tag == "HALLUCINATION" and halluc_sub == "Spontaneous_Insertion":
        return """The junior judge classified this as HALLUCINATION/Spontaneous_Insertion —
the model completed a correct sequence of dot products then inserted a
fabricated entry value with no mathematical origin in prior steps.

Verification question:
  (1) Identify the last entry correctly derived — state (i,j) and value.
  (2) Identify the inserted entry that has no prior basis — state (i,j)
      and what the model wrote.
  (3) Confirm the inserted value cannot be derived from any preceding dot
      product computation shown in the response.

CRITICAL: If the inserted value can be traced to a preceding computation
(even a wrong one), this is ARITHMETIC or SIGN_ERROR, not Spontaneous_Insertion."""

    elif tag == "HALLUCINATION":
        return """The junior judge classified this as HALLUCINATION but no subtype was given.

Determine which subtype applies:
  Complete_Collapse     — explicit abandonment phrase + fewer than 2 rows computed
  Teleological_Zeroing  — entry written as 0 despite dot products not summing to 0
  Premature_Assertion   — partial rows computed then full matrix claimed as complete
  Silent_Omission       — computation blocks silently missing, no meta-statement
  Ungrounded_Guess      — essentially no dot products shown, just a final matrix
  Spontaneous_Insertion — fabricated entry inserted after a correct computation chain

Populate hallucination_subtype in your response."""

    # ── MEMORY_LOSS ───────────────────────────────────────────────────────
    elif tag == "MEMORY_LOSS":
        return """The junior judge classified this as MEMORY_LOSS — the model computed
a correct entry in an earlier step and stated it, then used a different wrong
value for that same entry in a later step (recalled after ≥1 unrelated steps).

Verification question:
  (1) Which entry (AB)_{ij} was stated correctly at the early step?
      Give the step and exact value.
  (2) What value did the model use for that same entry at the later step?
      Give the step and the wrong value used.
  (3) Was the FIRST occurrence genuinely correct per your computation?
  (4) Are the two occurrences separated by ≥1 unrelated intervening steps?

CRITICAL: If the first occurrence is also wrong, the root error is ARITHMETIC
or SIGN_ERROR — set verified=false and correct the tag.
If the two occurrences are adjacent (N→N+1, no intervening steps), reclassify
as CARRY_DOWN_ERROR instead."""

    # ── METHOD_FAIL ───────────────────────────────────────────────────────
    elif tag == "METHOD_FAIL":
        return """The junior judge classified this as METHOD_FAIL — the model never
performed legitimate matrix multiplication (row × column dot products).

Verification question:
Describe what method the model used in the first 3–4 steps.
Confirm it fundamentally differs from standard dot product multiplication.

If the model computed valid dot products but on the wrong operand arrangement
(BA instead of AB), state this explicitly — this is Operand_Confusion.

Examples that qualify as base METHOD_FAIL:
  - Element-wise (Hadamard) multiplication instead of matrix product
  - Computing det(A)·det(B) as the answer
  - Applying row reduction to find AB
  - Confusing matrix multiplication with scalar multiplication

Examples that qualify as Operand_Confusion:
  - All dot products valid in form but using rows of B and columns of A
  - Result equals BA not AB; individual dot products are arithmetically correct

CRITICAL: If the model set up legitimate dot products but made errors inside
them, classify by the actual error type (ARITHMETIC, SIGN_ERROR, etc.)
and set verified=false."""

    # ── OTHER_UNMAPPED / UNKNOWN ──────────────────────────────────────────
    else:
        return """The junior judge could not classify this error (UNKNOWN or OTHER_UNMAPPED).

Fresh classification — ignore the previous tag entirely.

Read the model's matrix multiplication from the beginning.
Find the first step where the model's value diverges from correct AB.

Classify using exactly one of the ten valid tags:
  GENERATION_TRUNCATION | FORMATTING_MISMATCH | SIGN_ERROR | ARITHMETIC |
  HALLUCINATION | INPUT_TRANSCRIPTION | CARRY_DOWN_ERROR | METHOD_FAIL |
  MEMORY_LOSS | OTHER_UNMAPPED

For SIGN_ERROR, first apply MAGNITUDE RULE (|wrong|=|correct| required),
then populate sign_subtype:
  Product_Sign_Error   | Operation_Direction  | Rule_Interference |
  Double_Negative_Trap | Silent_Sign_Flip

For HALLUCINATION, populate hallucination_subtype:
  Complete_Collapse     | Teleological_Zeroing | Premature_Assertion |
  Silent_Omission       | Ungrounded_Guess     | Spontaneous_Insertion

Use OTHER_UNMAPPED only if genuinely impossible to map after careful analysis.
If OTHER_UNMAPPED, populate proposed_novel_tag and maps_closest_to."""


def build_validate_user_prompt(row: dict, response: str, matrix_latex: str) -> str:
    """
    Build the full user message for the senior validation judge.
    Combines the junior judge's classification summary with a targeted
    verification question derived from the error tag and subtype.
    """
    tag        = str(row.get("Error_Tag", "")              or "UNKNOWN").strip()
    sign_sub   = str(row.get("Sign_Subtype", "")           or "").strip()
    halluc_sub = str(row.get("Hallucination_Subtype", "")  or "").strip()
    strategy   = str(row.get("Solution_Strategy", "")      or "").strip()
    step       = str(row.get("First_Error_Step", "")       or "").strip()
    desc       = str(row.get("First_Error_Description", "") or "").strip()

    active_subtype = sign_sub or halluc_sub or "(none)"

    vq = _build_verification_question(tag, sign_sub, halluc_sub, strategy, step, desc)

    return f"""Matrices A and B:
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
####################################################################################
# END OF FILE
####################################################################################

