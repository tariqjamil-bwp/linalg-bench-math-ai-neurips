"""
judge_prompts/det.py — Forensic judge prompts for determinant (5×5) subcategory.

Revised to align with unified taxonomy (taxonomy_reference_card.py + Final signs.pdf).
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
  The model's determinant computation is 100% correct but the final answer
  is presented incorrectly (wrong box format, missing required structure).
  DETECT: Every intermediate (minors, cofactors, final sum) matches ground
  truth. Only the final boxed presentation fails the formatting requirement.
  CRITICAL: Do NOT use if any intermediate value is wrong.

───────────────────────────────────────────────────────────────────────────
SIGN_ERROR
  The first wrong value has the correct absolute magnitude but wrong sign.
  See MAGNITUDE RULE above — do not use ARITHMETIC for sign flips.
  Choose exactly one subtype:

  Product_Sign_Error
    A product of two terms has correct magnitude but wrong sign.
    In det context: a first-row element a_{1j} multiplied by a minor value,
    or a term in a 2×2/3×3 sub-determinant, has correct |product| but
    wrong sign.
    DETECT: |product| = |correct|, sign differs in a multiply step.
    Det example: a_{12} · M_{12} = (−2)·(220) should give −440 but model
    writes +440 — magnitude correct, sign flipped.

  Operation_Direction
    Model added where it should have subtracted, or vice versa, in a
    determinant expansion or sub-determinant accumulation.
    DETECT: Accumulation operator +/− applied in wrong direction.
    Det example: In 2×2 det computation, model writes a·d + b·c instead
    of a·d − b·c, adding where it should subtract.

  Rule_Interference
    A negative matrix entry causes the model to mishandle the sign at the
    final assembly step — treating the negative sign of a negative element
    as a subtraction operator instead of part of the multiplication.
    DETECT: Element a_{1j} = −k; model writes "− k · C_{1j}" in the sum
    treating it as subtract k·C_{1j}. Correct form is "(−k) · C_{1j}".
    This double-applies the negative, flipping the sign of that term.

  Parity_Sign_Error
    The model correctly identifies the cofactor position but applies the
    (−1)^{i+j} parity factor with wrong sign, or applies it twice.
    Closely related to Alternating_Drift but for a single isolated parity
    error rather than a pattern break.
    DETECT: Minor M_{1j} correct in magnitude; cofactor sign (±) is flipped
    on one term without a pattern drift — an isolated parity mistake.

  Double_Negative_Trap
    Two or more negatives met in a product or chain and the model failed
    to resolve them (e.g. −(−x) written as −x instead of +x).
    DETECT: Expression a·d − b·c where b or c is negative; model writes
    the wrong sign for that subtracted term. Both values in the trap must
    be genuinely negative.

  Alternating_Drift
    Model correctly applied (−1)^{1+j} for the first few cofactors but
    then lost the +−+−+ checkerboard pattern. The first wrong value is
    a cofactor with correct magnitude but wrong sign because the pattern
    decayed after 2–3 terms.
    DETECT: Model gets C_11=+M_11, C_12=−M_12, C_13=+M_13 correct, then
    writes C_14=+M_14 instead of −M_14. Or inside a nested 4×4 minor
    expansion, loses the pattern midway.
    Distinguishable from Parity_Sign_Error: Alternating_Drift is a drift
    after a correctly-started pattern (≥2 correct terms); Parity_Sign_Error
    is an isolated single parity error.

  Cofactor_Neglect
    The model computed the minor value correctly but used it directly in
    the final assembly without applying the (−1)^{i+j} parity factor.
    The parity step was simply skipped entirely.
    DETECT: Model computes M_{12} = 220 correctly, then writes
    det(A) = ... + a_{12} · M_{12} instead of ... + a_{12} · (−1) · M_{12}.
    Distinguishable from Parity_Sign_Error: Cofactor_Neglect is zero parity
    application; Parity_Sign_Error is wrong parity applied.

  Silent_Sign_Flip
    BEHAVIORAL DEFINITION — do not infer intent:
    A sign-valued output appears in the response AND zero intermediate
    computation is shown for that specific step.
    DETECT: Surrounding cofactor computations are fully shown. This one
    cofactor or sub-determinant has only a stated value with the wrong sign,
    no supporting computation.
    Do NOT use if any computation is shown for that step.
    MAGNITUDE RULE applies: |wrong| must equal |correct|; if magnitudes
    differ and no working is shown, classify as ARITHMETIC.
    Det example: Model writes "C_13 = 644" with no working shown where the
    correct value is C_13 = −644.

───────────────────────────────────────────────────────────────────────────
ARITHMETIC
  The first wrong value has the correct sign but wrong absolute magnitude.
  Method is correct, signs are tracked correctly at that step, but a
  numerical calculation produced the wrong number.
  See MAGNITUDE RULE — do not use for sign flips.
  DETECT: Wrong products or sums in 2×2 determinant expansions.
  Det example: model computes 7·5 − (−3)·(−4) = 47 instead of 35 − 12 = 23.

───────────────────────────────────────────────────────────────────────────
HALLUCINATION
  The model does not produce a wrong intermediate value by computation.
  Instead it fabricates, abandons, or invents. Choose one subtype:

  Complete_Collapse
    Model explicitly abandons the computation with a meta-statement.
    Response is short (under 5000 characters). Writes phrases like
    "due to the complexity", "we can simplify by", "directly providing
    the answer" and outputs 0 or a fabricated value.
    DETECT: Short response + explicit meta-statement + missing minors.
    Fewer than 3 of the five 4×4 minors were computed before stopping.

  Teleological_Zeroing
    Model produces a full-length response but forces intermediate minors
    or sub-determinants to equal exactly 0 without justification.
    Terms suspiciously cancel perfectly. The model manufactures zeros.
    DETECT: Long response + multiple intermediates = 0 + those zeros
    do not match your correct computation.

  Premature_Assertion
    Model begins cofactor expansion, computes some (but not all) of the
    five 4×4 minors, then states the full determinant as if complete.
    DETECT: Medium-length response, partial minor computations, then sudden
    determinant claim without finishing.

  Silent_Omission
    Model produces a full-length response but silently skips computation
    blocks — minors or cofactors claimed without showing the sub-determinant
    expansion.
    NOTE: Do NOT use for truncated responses — use GENERATION_TRUNCATION.
    DETECT: Long response + "following the same process..." + sub-determinant
    steps missing for one or more minors.

  Ungrounded_Guess
    Essentially no working shown. Model outputs a determinant value with
    minimal or no supporting computation.
    DETECT: Very short response, no minors or cofactors computed, final
    integer presented as if obvious.

  Spontaneous_Insertion
    Model completes a correct cofactor expansion chain up to a point, then
    inserts a fabricated minor or sub-determinant value with no mathematical
    origin in prior steps.
    DETECT: All sub-determinant computations up to the insertion point are
    correct. A large or arbitrary value appears for a minor that cannot be
    derived from the matrix entries.
    Det example: det(A) = −18 + 30 + 24 + 38480 where 38480 has no
    traceable origin in any 4×4 sub-determinant computation.

───────────────────────────────────────────────────────────────────────────
INPUT_TRANSCRIPTION
  The model miscopied an entry from the ORIGINAL INPUT matrix A when
  constructing a submatrix for a minor.
  Error occurs before any arithmetic — the value was never computed,
  just read from the problem statement incorrectly.
  DETECT: Compare the model's stated submatrix entries against the original
  matrix. Find one entry that does not match.

  SCOPE CHECK (mandatory after finding a miscopied entry):
  Check ALL adjacent entries in the same row and column of the model's
  written-out submatrix. If the model replaced an entire row or column with
  fabricated data, do NOT use INPUT_TRANSCRIPTION — escalate to HALLUCINATION
  (Silent_Omission or Spontaneous_Insertion).

───────────────────────────────────────────────────────────────────────────
CARRY_DOWN_ERROR
  The model correctly computed an intermediate minor or cofactor at step N
  and stated it correctly, but then miscopied that same value at step N+1
  when carrying it forward. A line-to-line copy error, not an arithmetic error.
  DETECT: Minor M_{1j} stated correctly after computation at step N. Same
  minor used with a changed value at step N+1 with no new computation applied.
  CRITICAL: The FIRST occurrence must be genuinely correct. If the first
  occurrence is also wrong, classify as ARITHMETIC or SIGN_ERROR instead.
  Boundary with MEMORY_LOSS: CARRY_DOWN_ERROR = N→N+1 (no intervening
  steps); MEMORY_LOSS = value recalled after ≥1 unrelated intervening steps.

───────────────────────────────────────────────────────────────────────────
METHOD_FAIL
  The model uses a fundamentally wrong algorithm from the very start.
  DETECT: First 3–4 steps do not follow any valid determinant method.
  CRITICAL: A valid alternative algorithm is NOT a METHOD_FAIL.
  Row reduction used correctly for determinants is NOT METHOD_FAIL.
  Only use METHOD_FAIL if the algorithm itself is incorrect AND applied
  incorrectly — not merely different from the expected approach.
  Examples: treating det(A) = product of diagonal entries (det shortcut
  valid only for triangular matrices), computing det(A) = trace(A),
  performing row operations without tracking det adjustment (scaling by k
  without dividing det by k), applying Sarrus rule to a 5×5 matrix.

───────────────────────────────────────────────────────────────────────────
MEMORY_LOSS
  The model correctly computed a value at an early step and stated it,
  then used a different wrong value for that same quantity at a later step.
  DETECT: Value V stated correctly at step N, then stated as wrong value
  W at step M where M > N. Example: M_{12} = 220 at step 6, then M_{12} = 202
  at step 18.
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
benchmarking LLMs on 3x3|4x4|5x5 matrix determinant computation.

You will be given a 3x3|4x4|5x5 square matrix A and a model's FAILED attempt to compute its determinant.

The model may have used any of these strategies:

  STRATEGY 1 — COFACTOR EXPANSION (Laplace)
    For a 5x5 matrix, the model expands along a row or column: det(A) = Σ a_{1j} · C_{1j}
    where C_{1j} = (−1)^{1+j} · M_{1j} and M_{1j} is the 4×4 minor.
    Each 4×4 minor is in turn expanded into 3×3 minors, then 2×2 minors.
    Failure mode: sign errors in the (−1)^{i+j} parity, arithmetic errors
    in 2×2 and 3×3 sub-determinants, submatrix misread from input.

  STRATEGY 2 — ROW REDUCTION
    The model applies Gaussian elimination to A, tracking det adjustments
    (row swap → negate det; row scaling by k → divide det by k).
    Failure mode: row operation arithmetic errors, incorrect det adjustment
    for row swaps or scalings, sign errors in multiplier formation.

  STRATEGY 3 — HEURISTIC / VISUAL INSPECTION
    The model guesses the determinant from surface features without systematic
    computation, or applies shortcuts without justification.
    Failure mode: fabricated answer with no supporting derivation.

${TRUNCATION_PRECHECK}

YOUR TASK HAS THREE STEPS:

STEP 1 — COMPUTE THE CORRECT DETERMINANT YOURSELF
  You are a 100% accurate determinant calculator. Compute the determinant of
  the given matrix independently using cofactor expansion along the first row.
  Mentally note all correct intermediate values:
    - The five 4×4 minors M_11 through M_15
    - Their cofactors C_1j = (−1)^{1+j} · M_1j with correct signs
    - Each product a_1j · C_1j
    - The final sum
  These are your ground truth values. Do not reveal them in your response.

STEP 2 — IDENTIFY THE STRATEGY THE MODEL USED
  Based on the model's response, decide which of the three strategies above
  it attempted. Record this as "solution_strategy" in your output.

STEP 3 — SCAN THE MODEL RESPONSE FOR THE FIRST WRONG VALUE
  Read the model response step by step from beginning to end.
  Compare every intermediate value the model states against your correct values.
  Find the FIRST step where the model's value diverges from correct.

  CRITICAL RULE: Classify by first point of failure, not final symptom.
  If the model makes a sign error at step 8 and it cascades into a wrong
  magnitude at step 20, the correct classification is SIGN_ERROR not ARITHMETIC.
  The label always reflects the first point of failure, not the final symptom.

${MAGNITUDE_RULE}

STEP 4 — CLASSIFY THE FIRST WRONG VALUE
  Use exactly one of these tags:

${TAG_DEFINITIONS}


RESPOND IN EXACTLY THIS JSON FORMAT AND NOTHING ELSE.

For SIGN_ERROR:
{
  "error_tag": "SIGN_ERROR",
  "sign_subtype": "Rule_Interference",
  "hallucination_subtype": "",
  "solution_strategy": "COFACTOR_EXPANSION",
  "first_error_step": "Step 16 — final assembly of det(A)",
  "first_error_description": "Element a_{12}=−2; model wrote − 2·C_{12} instead of (−2)·C_{12}, double-applying the negative on that term. |term| correct, sign flipped.",
  "proposed_novel_tag": "",
  "maps_closest_to": "",
  "confidence": "HIGH"
}

For HALLUCINATION:
{
  "error_tag": "HALLUCINATION",
  "sign_subtype": "",
  "hallucination_subtype": "Complete_Collapse",
  "solution_strategy": "COFACTOR_EXPANSION",
  "first_error_step": "Step 8 — after completing only M_11",
  "first_error_description": "Model wrote due to complexity we will simplify and output boxed 0 without computing remaining minors.",
  "proposed_novel_tag": "",
  "maps_closest_to": "",
  "confidence": "HIGH"
}

For all other tags:
{
  "error_tag": "ARITHMETIC",
  "sign_subtype": "",
  "hallucination_subtype": "",
  "solution_strategy": "COFACTOR_EXPANSION",
  "first_error_step": "Step 11 — 2×2 minor inside M_13",
  "first_error_description": "Model computed (−4)·(−1) − (2)·(4) = −12 but correct value is 4 − 8 = −4. Sign correct, magnitude wrong.",
  "proposed_novel_tag": "",
  "maps_closest_to": "",
  "confidence": "MEDIUM"
}

Confidence levels:
  HIGH   — you can point to a specific step and specific wrong value
  MEDIUM — fairly confident but response is ambiguous in places
  LOW    — response is very short or garbled, you are inferring
""").substitute(
    MAGNITUDE_RULE=MAGNITUDE_RULE,
    TAG_DEFINITIONS=TAG_DEFINITIONS,
    TRUNCATION_PRECHECK=TRUNCATION_PRECHECK,
)

BUILD_USER_TEMPLATE = """\
Matrix A:
{matrix_latex}

Correct determinant: {ground_truth}
Model's extracted answer: {extracted_answer}

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
  1. A square matrix A (typical size: 3×3 to 5×5)
  2. A model's failed response attempting to compute its determinant
  3. A classification made by a junior judge
  4. A targeted verification question for that error type

${ADVERSARIAL_FRAMING}

${ANSWER_CHECK_STEP0}

YOUR PROCESS:
  Step 1: Compute the correct determinant yourself using cofactor expansion
          along the first row. Note all correct intermediate values (minors,
          cofactors, products, final sum). You are the ground truth oracle.
  Step 2: Read the model response step by step. Find the FIRST step where
          the model's value diverges from your correct computation.
  Step 3: Answer the verification question with specific evidence from
          the response. Confirm or deny the classification.

FIRST-ERROR PRINCIPLE:
  Classify by first point of failure, not final symptom.
  A sign error at step 8 cascading to wrong magnitude at step 20 is
  SIGN_ERROR not ARITHMETIC.

${MAGNITUDE_RULE}

STRICT TAXONOMY — primary_tag MUST be exactly one of these ten strings:
  GENERATION_TRUNCATION | FORMATTING_MISMATCH | SIGN_ERROR | ARITHMETIC |
  HALLUCINATION | INPUT_TRANSCRIPTION | CARRY_DOWN_ERROR | METHOD_FAIL |
  MEMORY_LOSS | OTHER_UNMAPPED

  Do NOT invent new primary tags. If error genuinely does not fit the
  nine main tags, use OTHER_UNMAPPED and populate proposed_novel_tag
  and maps_closest_to.

SIGN_ERROR subtypes (required when primary_tag = SIGN_ERROR):
  Product_Sign_Error   - product has correct magnitude, wrong sign in a multiply step
  Operation_Direction  - added where subtraction required in accumulation (or vice versa)
  Rule_Interference    - negative element treated as subtraction operator in assembly
  Parity_Sign_Error    - isolated (−1)^{i+j} parity applied with wrong sign on one term
  Double_Negative_Trap - two negatives met in 2×2 sub-det, wrong sign on result
  Alternating_Drift    - +−+−+ pattern correct for ≥2 terms then breaks midway
  Cofactor_Neglect     - minor value used raw without applying (−1)^{i+j} at all
  Silent_Sign_Flip     - wrong-sign output with zero computation shown for that step;
                         MAGNITUDE RULE applies: |wrong| must equal |correct|

HALLUCINATION subtypes (required when primary_tag = HALLUCINATION):
  Complete_Collapse     - explicit abandonment phrase, fewer than 3 minors computed
  Teleological_Zeroing  - forces intermediates to 0, long response
  Premature_Assertion   - partial minors computed then det claimed as complete
  Silent_Omission       - skips computation blocks silently, no meta-statement
  Ungrounded_Guess      - essentially no working shown, just a final number
  Spontaneous_Insertion - fabricated minor/cofactor value after correct computation chain

RESPOND IN EXACTLY THIS JSON FORMAT AND NOTHING ELSE:
{
  "verified":              true,
  "primary_tag":           "INPUT_TRANSCRIPTION",
  "sign_subtype":          "",
  "hallucination_subtype": "",
  "proposed_novel_tag":    "",
  "maps_closest_to":       "",
  "forensic_observation":  "Confirmed: model miscopied a(2,3)=−3 as +3 in submatrix for M_12 at step 6",
  "error_scope":           "isolated",
  "affected_positions":    ["M_12 submatrix, row 2 col 3"],
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
                                      (e.g. all negative-parity cofactors
                                      have flipped sign across the expansion)
affected_positions:   list the specific cofactors, minors, or sub-determinant
                      steps where the error manifests
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
ends mid-expression or mid-computation with no final determinant answer.

Verification question:
  (1) Confirm the response has no final answer in \\boxed{} or equivalent.
  (2) Quote the last ~50 characters of the response. Does it end mid-sentence,
      mid-expression (e.g. "= "), or mid-computation?
  (3) Does the response contain any complete determinant claim anywhere?

CRITICAL: If a complete determinant claim exists anywhere in the response,
this is NOT GENERATION_TRUNCATION. Reclassify based on whether that claim
is correct or fabricated."""

    # ── FORMATTING_MISMATCH ───────────────────────────────────────────────
    elif tag == "FORMATTING_MISMATCH":
        return """The junior judge classified this as FORMATTING_MISMATCH — the model's
determinant computation is mathematically correct but the format is wrong.

Verification question:
  (1) Verify every intermediate (minors, cofactors, final sum) matches your
      correct computation.
  (2) Identify the specific formatting failure (missing \\boxed{}, etc.).
  (3) Confirm the determinant value itself matches the ground truth.

CRITICAL: Do NOT confirm FORMATTING_MISMATCH if any intermediate value is wrong.
If the determinant value itself is wrong, this is a mathematical error —
set verified=false and reclassify."""

    # ── INPUT_TRANSCRIPTION ───────────────────────────────────────────────
    elif tag == "INPUT_TRANSCRIPTION":
        return """The junior judge classified this as INPUT_TRANSCRIPTION — the model
miscopied a matrix entry when constructing a submatrix for a minor.

Verification question:
Identify the EXACT cell that was miscopied. State:
  (1) Row and column in the original 5×5 matrix A,
  (2) Correct value at that position,
  (3) Value the model wrote instead in the submatrix.

SCOPE CHECK (mandatory):
Check ALL adjacent entries in the same row and column of the model's
written-out submatrix. If the model replaced an entire row or column with
fabricated data (not just one cell), reclassify to HALLUCINATION
(Silent_Omission or Spontaneous_Insertion).

CRITICAL: You must find both values directly in the response text.
If you cannot point to a specific cell with specific correct and wrong values,
set verified=false and give your corrected tag. Exact row, column, and
both values required."""

    # ── CARRY_DOWN_ERROR ──────────────────────────────────────────────────
    elif tag == "CARRY_DOWN_ERROR":
        return """The junior judge classified this as CARRY_DOWN_ERROR — the model
correctly stated a minor or cofactor at step N but miscopied it at step N+1
when carrying it forward.

Verification question:
  (1) Quote the correctly-stated value at step N (with step label).
  (2) Quote the changed form at step N+1 (must be adjacent — no intervening
      unrelated steps).
  (3) Confirm the first occurrence matches your correct computation.
  (4) Confirm there is no new sub-determinant computation between N and N+1
      that would legitimately change the value.

CRITICAL: If the first occurrence is itself wrong, this is ARITHMETIC or
SIGN_ERROR — not CARRY_DOWN_ERROR. Set verified=false and reclassify.
If separated by ≥1 unrelated intervening steps, reclassify as MEMORY_LOSS."""

    # ── ARITHMETIC ────────────────────────────────────────────────────────
    elif tag == "ARITHMETIC":
        return """The junior judge classified this as ARITHMETIC — correct method and signs,
but wrong magnitude in a numerical calculation.

Verification question:
Find the specific computation where the wrong number first appeared.
  (1) Were ALL signs in that expression correct going in?
  (2) Apply the MAGNITUDE RULE: is |wrong value| ≠ |correct value|?
  (3) Was the error purely a multiplication or addition mistake with correct signs?

CRITICAL: Apply MAGNITUDE RULE strictly. If |wrong| = |correct|, root error
is SIGN_ERROR — set verified=false and correct the tag. If any sign was wrong
BEFORE or DURING that computation, also reclassify as SIGN_ERROR.
Only set verified=true if signs were genuinely correct AND magnitude failed."""

    # ── SIGN_ERROR subtypes ───────────────────────────────────────────────
    elif tag == "SIGN_ERROR" and sign_sub == "Product_Sign_Error":
        return f"""The junior judge classified this as SIGN_ERROR/Product_Sign_Error —
a product of two terms has correct magnitude but wrong sign.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
Find the exact product where this occurred. State:
  (1) Which terms were multiplied (e.g., a_{{12}} · C_{{12}}),
  (2) Correct values and the sign the product should carry,
  (3) The sign the model actually used,
  (4) The resulting accumulation error.

CRITICAL: Apply MAGNITUDE RULE. |product| must be correct — only sign flipped.
If |wrong| ≠ |correct|, reclassify as ARITHMETIC.
If the model treated a negative element as a subtraction operator (not a
product sign flip), reclassify as Rule_Interference."""

    elif tag == "SIGN_ERROR" and sign_sub == "Operation_Direction":
        return f"""The junior judge classified this as SIGN_ERROR/Operation_Direction —
the model added where it should have subtracted (or vice versa) in a
determinant expansion or sub-determinant computation.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
Find the exact accumulation step. State:
  (1) Which computation (e.g., 2×2 minor, final assembly term),
  (2) The correct operation (e.g., a·d − b·c),
  (3) What the model wrote instead (e.g., a·d + b·c),
  (4) The resulting value vs the correct value.

CRITICAL: Confirm direction of accumulation was wrong — not just the sign
of an individual term. If a term's own sign was flipped, reclassify as
Product_Sign_Error."""

    elif tag == "SIGN_ERROR" and sign_sub == "Rule_Interference":
        return f"""The junior judge classified this as SIGN_ERROR/Rule_Interference —
the model treated a negative first-row element's sign as a subtraction
operator at the final assembly step.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Which first-row element is negative? State position and value.
  (2) Show the exact term as the model wrote it in the final assembly.
  (3) Show how it should have been written: (−k) · C_{{1j}}.

CRITICAL: The element at that position MUST be negative in the original matrix.
If the element is positive, this classification is impossible — set verified=false.
If the model had all cofactor values correct and only mishandled the negative
element in the assembly, this is Rule_Interference.
If the model had a wrong sign on the cofactor itself (not the assembly), use
another subtype."""

    elif tag == "SIGN_ERROR" and sign_sub == "Parity_Sign_Error":
        return f"""The junior judge classified this as SIGN_ERROR/Parity_Sign_Error —
the model correctly identified a cofactor position but applied (−1)^{{i+j}}
with wrong sign on one isolated term.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Which cofactor has the isolated parity error? State position and the
      correct parity (should be + or −).
  (2) What sign did the model apply?
  (3) Confirm this is an ISOLATED parity error (not a drift pattern).
      How many preceding cofactors were correct before this error?

CRITICAL: Distinguish from Alternating_Drift: Parity_Sign_Error is isolated
(fewer than 2 correct preceding terms in the same expansion). Alternating_Drift
requires ≥2 correct terms before the pattern breaks.
If the model simply skipped parity entirely (used raw minor), use Cofactor_Neglect."""

    elif tag == "SIGN_ERROR" and sign_sub == "Double_Negative_Trap":
        return f"""The junior judge classified this as SIGN_ERROR/Double_Negative_Trap —
inside a 2×2 determinant, two negatives met and the model failed to resolve
them (e.g. −8 − (−3) written as −8 − 3 = −11 instead of −8 + 3 = −5).
Junior judge's first error step: {step or '(not specified)'}

Verification question:
Find the exact 2×2 expression where two negatives met and were handled wrong.
  (1) Write out the full expression as the model stated it.
  (2) Confirm BOTH values involved are genuinely negative in the original matrix.
  (3) Show the correct resolution.
  (4) Apply MAGNITUDE RULE: confirm |result| is correct — only the sign wrong.

CRITICAL: If only one value is negative, this cannot be Double_Negative_Trap.
Set verified=false — likely Rule_Interference, Alternating_Drift, or ARITHMETIC."""

    elif tag == "SIGN_ERROR" and sign_sub == "Alternating_Drift":
        return """The junior judge classified this as SIGN_ERROR/Alternating_Drift —
the model lost the cofactor checkerboard pattern midway through expanding
det(A − λI).

Verification question:
First, count how many terms appear in the model's first-row cofactor expansion
(this equals the matrix dimension: 4 terms for a 4×4 matrix, 5 for a 5×5).

List the exact sign the model applied to EACH of those terms in order.
State at which position the alternating pattern first broke.

REQUIRED: List ALL N signs where N is the number of terms in the expansion.
Do NOT require exactly five signs — a 4×4 expansion has four terms, not five.
If you cannot identify all N signs from the response, set verified=false.
If the pattern never broke, set verified=false.
Drift requires the pattern to be correct for at least two terms before breaking.

Example for 4×4: signs should be +, −, +, − for C_11 through C_14.
Example for 5×5: signs should be +, −, +, −, + for C_11 through C_15."""

    elif tag == "SIGN_ERROR" and sign_sub == "Cofactor_Neglect":
        return f"""The junior judge classified this as SIGN_ERROR/Cofactor_Neglect —
the model computed the minor value correctly but used it directly without
applying the (−1)^{{i+j}} parity factor at all.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
Find the exact line where a minor was used as if it were already a cofactor.
State:
  (1) The minor value (magnitude only),
  (2) The correct cofactor value (with parity sign applied),
  (3) What the model actually used.

Confirm the ONLY difference is the missing sign — magnitudes must be identical.

If magnitudes also differ, this is not Cofactor_Neglect.
If model wrote a sign term but got it wrong, reclassify as Alternating_Drift
or Parity_Sign_Error."""

    elif tag == "SIGN_ERROR" and sign_sub == "Silent_Sign_Flip":
        return f"""The junior judge classified this as SIGN_ERROR/Silent_Sign_Flip —
BEHAVIORAL DEFINITION: a sign-valued output appears in the response AND
zero intermediate computation is shown for that specific step.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Quote the exact line where the model stated a sign-related value with
      no computation visible.
  (2) Confirm the correct value has the opposite sign.
  (3) Apply MAGNITUDE RULE: confirm |wrong| = |correct|. If magnitudes differ,
      classify as ARITHMETIC instead.

CRITICAL: If the model showed ANY working at that step, this is not
Silent_Sign_Flip. Identify the actual operation and reclassify accordingly."""

    elif tag == "SIGN_ERROR":
        return """The junior judge classified this as SIGN_ERROR but no subtype was given.

Find the first wrong value. Apply MAGNITUDE RULE:
  — If |wrong| ≠ |correct|: reclassify as ARITHMETIC (not SIGN_ERROR).
  — If |wrong| = |correct|, sign differs: confirm SIGN_ERROR, then identify subtype:

  Product_Sign_Error   — correct product magnitude, wrong sign in multiply step
  Operation_Direction  — added where subtraction required (or vice versa)
  Rule_Interference    — negative element treated as subtraction operator in assembly
  Parity_Sign_Error    — isolated (−1)^{i+j} parity error, not a drift pattern
  Double_Negative_Trap — two negatives in 2×2 det, wrong sign resolved
  Alternating_Drift    — +−+−+ pattern breaks after ≥2 correct terms
  Cofactor_Neglect     — raw minor used without applying (−1)^{i+j} at all
  Silent_Sign_Flip     — sign error stated with zero computation shown; |wrong|=|correct|

Populate sign_subtype in your response."""

    # ── HALLUCINATION subtypes ────────────────────────────────────────────
    elif tag == "HALLUCINATION" and halluc_sub == "Complete_Collapse":
        return """The junior judge classified this as HALLUCINATION/Complete_Collapse —
explicit abandonment with a meta-statement.

Verification question:
  (1) Quote the exact phrase where the model abandoned computation.
  (2) How many of the five 4×4 minors M_11–M_15 did the model actually
      compute before stopping? Must be fewer than 3 for Complete_Collapse.

CRITICAL: If the model computed all five minors, this is not Complete_Collapse.
Consider Teleological_Zeroing, Silent_Omission, or another tag."""

    elif tag == "HALLUCINATION" and halluc_sub == "Teleological_Zeroing":
        return """The junior judge classified this as HALLUCINATION/Teleological_Zeroing —
the model forced intermediate minors or sub-determinants to equal exactly 0.

Verification question:
List THREE intermediate values the model set to zero that should not be zero
per your correct computation. For each: label, model value (0), correct value.

If you cannot name three such cases, the classification is likely wrong.
Reclassify to Complete_Collapse or another tag."""

    elif tag == "HALLUCINATION" and halluc_sub == "Premature_Assertion":
        return """The junior judge classified this as HALLUCINATION/Premature_Assertion —
the model computed some (but not all) of the five 4×4 minors, then stated
the full determinant as if complete.

Verification question:
  (1) Which of the five 4×4 minors M_11–M_15 did the model actually compute?
  (2) Which minors are absent when the determinant is claimed?
  (3) Quote the line where the determinant is asserted prematurely.

CRITICAL: At least one valid minor expansion must have been performed.
If no minors exist, use Ungrounded_Guess.
If explicit abandonment phrase, use Complete_Collapse."""

    elif tag == "HALLUCINATION" and halluc_sub == "Silent_Omission":
        return """The junior judge classified this as HALLUCINATION/Silent_Omission —
the model produced a full-length response but skipped computation blocks
without any meta-statement.

Verification question:
  (1) Which computation steps are missing from the response?
  (2) Is there any meta-statement or explanation for the skip?
  (3) Does the response end with a complete final answer, or is it truncated?

CRITICAL: If the response is truncated (no final answer), reclassify as
GENERATION_TRUNCATION. If there IS a meta-statement, use Complete_Collapse.
If essentially no working exists at all, use Ungrounded_Guess."""

    elif tag == "HALLUCINATION" and halluc_sub == "Ungrounded_Guess":
        return """The junior judge classified this as HALLUCINATION/Ungrounded_Guess —
essentially no working shown, just a final determinant value.

Verification question:
Confirm the response contains no meaningful computation — no minors,
no cofactors, no 2×2 or 3×3 sub-determinants computed.
  (1) Are any intermediate expansions shown?
  (2) Are any sub-determinants computed?

CRITICAL: If any intermediate computation exists, use Silent_Omission instead."""

    elif tag == "HALLUCINATION" and halluc_sub == "Spontaneous_Insertion":
        return """The junior judge classified this as HALLUCINATION/Spontaneous_Insertion —
the model completed a correct cofactor expansion chain up to a point, then
inserted a fabricated minor or sub-determinant value with no mathematical origin.

Verification question:
  (1) Identify the last minor/cofactor correctly computed — state value.
  (2) Identify the inserted value that has no prior basis — state what the
      model wrote and where (e.g., "M_14 = 38480" from nowhere).
  (3) Confirm the inserted value cannot be derived from any sub-expansion
      shown in the response.

CRITICAL: If the inserted value can be traced to a preceding computation
(even a wrong one), this is ARITHMETIC or SIGN_ERROR, not Spontaneous_Insertion."""

    elif tag == "HALLUCINATION":
        return """The junior judge classified this as HALLUCINATION but no subtype given.

Determine which subtype applies:
  Complete_Collapse     — explicit abandonment phrase, < 3 of 5 minors computed
  Teleological_Zeroing  — full response but forces intermediates to 0
  Premature_Assertion   — partial minors computed then det claimed as complete
  Silent_Omission       — skips computation blocks silently, no meta-statement
  Ungrounded_Guess      — essentially no working shown, just a final number
  Spontaneous_Insertion — fabricated minor/value inserted after correct chain

Set the correct hallucination_subtype in your response."""

    # ── MEMORY_LOSS ───────────────────────────────────────────────────────
    elif tag == "MEMORY_LOSS":
        return """The junior judge classified this as MEMORY_LOSS — the model computed
a value correctly at an early step and stated it, then used a different wrong
value for that same quantity at a later step (recalled after ≥1 unrelated steps).

Verification question:
  (1) What value was stated correctly at the early step? Give step and value.
  (2) What value did the model use for that same quantity later?
      Give step and the wrong value.
  (3) Was the FIRST occurrence genuinely correct per your computation?
  (4) Are the two occurrences separated by ≥1 unrelated intervening steps?

CRITICAL: If the first occurrence is also wrong, the error is ARITHMETIC or
SIGN_ERROR — set verified=false and correct the tag.
If the two occurrences are adjacent (N→N+1), reclassify as CARRY_DOWN_ERROR."""

    # ── METHOD_FAIL ───────────────────────────────────────────────────────
    elif tag == "METHOD_FAIL":
        return """The junior judge classified this as METHOD_FAIL — the model never set up
a valid determinant computation from the start.

Verification question:
Describe what method the model used in the first 3–4 steps.
Confirm it fundamentally differs from any valid determinant algorithm.

CRITICAL: Row reduction (Gaussian elimination) is a mathematically valid
method for computing determinants. It is NOT a METHOD_FAIL on its own.
Only classify as METHOD_FAIL if the model ALSO applies its chosen method
incorrectly in a fundamental way — e.g., scales rows without adjusting det,
performs non-elementary operations while claiming det is unchanged.
If the model used row reduction correctly but made a numerical error downstream,
set verified=false and classify by actual error type.
If the model DID set up cofactor expansion but made errors within it,
classify by the type of error — not METHOD_FAIL."""

    # ── OTHER_UNMAPPED / UNKNOWN ──────────────────────────────────────────
    else:
        return """The junior judge could not classify this error (UNKNOWN or OTHER_UNMAPPED).

Fresh classification — ignore the previous tag entirely.

Read the model response from the beginning.
Find the first step where the model's value diverges from correct.

Classify using exactly one of the ten valid tags:
  GENERATION_TRUNCATION | FORMATTING_MISMATCH | SIGN_ERROR | ARITHMETIC |
  HALLUCINATION | INPUT_TRANSCRIPTION | CARRY_DOWN_ERROR | METHOD_FAIL |
  MEMORY_LOSS | OTHER_UNMAPPED

For SIGN_ERROR, first apply MAGNITUDE RULE (|wrong|=|correct| required),
then populate sign_subtype:
  Product_Sign_Error | Operation_Direction | Rule_Interference |
  Parity_Sign_Error  | Double_Negative_Trap | Alternating_Drift |
  Cofactor_Neglect   | Silent_Sign_Flip

For HALLUCINATION, populate hallucination_subtype:
  Complete_Collapse | Teleological_Zeroing | Premature_Assertion |
  Silent_Omission   | Ungrounded_Guess     | Spontaneous_Insertion

Use OTHER_UNMAPPED only if genuinely impossible to map after careful analysis.
If OTHER_UNMAPPED, populate proposed_novel_tag and maps_closest_to.
"""


def build_validate_user_prompt(row: dict, response: str, matrix_latex: str) -> str:
    """
    Build the full user message for the senior validation judge.
    NOTE: reads Sign_Subtype (not the legacy Sign_Error_Types column).
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

##########################################################################################################################################################
# END OF FILE
##########################################################################################################################################################

