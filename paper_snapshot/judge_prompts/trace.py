"""
judge_prompts/trace.py — Build-judge and validate-judge prompts for the TRACE subcategory.

Exports:
  BUILD_SYSTEM_TEMPLATE   → system prompt for label_judge.py
  BUILD_USER_TEMPLATE     → user template for label_judge.py
  VALIDATE_SYSTEM_TEMPLATE→ system prompt for validate_judge.py
  build_validate_user_prompt → function for validate_judge.py
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
  DETECT: Every intermediate (diagonal elements identified, partial sum,
  final sum) matches ground truth. Only the final boxed presentation fails
  the formatting requirement.
  CRITICAL: Do NOT use if any diagonal value or sum is wrong.
  NOTE: Rare for trace since the answer is a single integer.

───────────────────────────────────────────────────────────────────────────
SIGN_ERROR
  The first wrong value has the correct absolute magnitude but wrong sign.
  See MAGNITUDE RULE above — do not use ARITHMETIC for sign flips.
  Choose exactly one subtype:

  Product_Sign_Error
    A product of two terms has correct magnitude but wrong sign.
    In trace context: typically occurs when the model reads a negative
    diagonal entry and applies incorrect sign in the accumulation step.
    DETECT: |value| = |correct|, sign differs in a multiply or copy step.
    Trace example: diagonal entry a(2,2) = −5; model uses +5 in the sum,
    computing tr(A) as if the entry were positive — magnitude correct,
    sign flipped.

  Operation_Direction
    Model added where it should have subtracted, or vice versa, in the
    trace accumulation.
    DETECT: Accumulation operator +/− applied in wrong direction.
    Trace example: model computes partial sum as a(1,1) − a(2,2) instead
    of a(1,1) + a(2,2), subtracting a diagonal entry that should be added.

  Rule_Interference
    A negative diagonal entry causes the model to mishandle the sign —
    double-applying or cancelling the negative from that entry.
    DETECT: Diagonal entry is negative; model treats it as positive in
    the accumulation.
    Trace example: a(3,3) = −7; during summation the model writes
    tr = ... − (−7) = ... − 7 instead of ... + (−7) = ... − 7, or
    otherwise misapplies the sign of the negative entry.

  Double_Negative_Trap
    Two or more negatives met in the accumulation and the model failed
    to resolve them (e.g. subtracting a negative diagonal entry written
    as −(−k) resolved as −k instead of +k).
    DETECT: Expression with two negatives; model produces wrong sign on
    the result while the magnitude is correct.
    Trace example: partial sum ends in −3 and next diagonal is −4;
    model writes −3 − (−4) = −7 instead of −3 + 4 = +1.

  Silent_Sign_Flip
    BEHAVIORAL DEFINITION — do not infer intent:
    A sign-valued output appears in the response AND zero intermediate
    computation is shown for that specific diagonal entry.
    DETECT: Surrounding diagonal entries have shown explicit identification.
    This one entry is stated with the wrong sign, no supporting identification
    or computation shown.
    Do NOT use if any computation is shown for that entry.
    MAGNITUDE RULE applies: |wrong| must equal |correct|; if magnitudes
    differ and no working is shown, classify as ARITHMETIC.
    Trace example: a(1,1), a(2,2), a(3,3) identified correctly and shown;
    a(4,4) stated with correct magnitude but wrong sign, no identification
    shown.

───────────────────────────────────────────────────────────────────────────
ARITHMETIC
  The first wrong value has the correct sign but wrong absolute magnitude.
  Method correct, signs tracked correctly, but a numerical calculation
  produced the wrong number.
  See MAGNITUDE RULE — do not use for sign flips.
  DETECT: Wrong sums or misread diagonal values where the sign of the
  result is correct but the absolute value is wrong.
  Trace example: Model reads a(2,2) = −5 and a(3,3) = 3 correctly but
  computes partial sum as −5 + 3 = −3 instead of −2, or misreads
  a(4,4) = 6 as 8 with correct sign.

───────────────────────────────────────────────────────────────────────────
HALLUCINATION
  The model does not produce a wrong intermediate value by computation.
  Instead it fabricates, abandons, or invents. Choose one subtype:

  Complete_Collapse
    Model explicitly abandons diagonal summation with a meta-statement.
    Response is short (under 3000 characters). Writes phrases like
    "by inspection", "clearly the trace is", "we can observe that"
    and outputs a fabricated integer.
    DETECT: Short response + explicit meta-statement + no diagonal
    elements individually identified.

  Teleological_Zeroing
    Model shows identification of a diagonal entry but writes 0 as its
    value despite the matrix not supporting it. The zero is asserted to
    reach a desired conclusion, not derived.
    DETECT: Diagonal identification shown → entry should be non-zero →
    model writes 0 anyway, often to simplify the final sum.

  Premature_Assertion
    Model begins identifying diagonal elements, reads only a partial
    subset (e.g., only 2 of 4 diagonal entries), then states a trace
    as if complete.
    DETECT: Medium-length response, partial diagonal identification,
    then sudden trace claim without finishing.

  Silent_Omission
    Model skips one or more diagonal entries silently with no meta-
    statement. Response continues as if skipped entries were included.
    NOTE: Do NOT use for truncated responses — use GENERATION_TRUNCATION.
    DETECT: Response shows some diagonal entries then jumps to a final
    sum that cannot be derived from the shown entries alone.

  Ungrounded_Guess
    Essentially no working shown. Model outputs a final trace value with
    minimal or no identification of individual diagonal entries.
    DETECT: Very short response, no diagonal entries explicitly listed,
    final integer presented as if obvious.

  Spontaneous_Insertion
    Model correctly identifies all diagonal entries then inserts a
    fabricated value in the final sum with no mathematical origin in
    prior steps.
    DETECT: All diagonal entries up to the insertion point are correct.
    A different value appears in the final accumulation that cannot be
    traced to any diagonal entry shown.

───────────────────────────────────────────────────────────────────────────
INPUT_TRANSCRIPTION
  The model miscopied a diagonal entry from the ORIGINAL INPUT matrix.
  Error occurs before any arithmetic — the value was never computed,
  just read from the matrix incorrectly.
  DETECT: Model's summation uses a value that does not match the
  corresponding diagonal entry in the given matrix A. Method is correct,
  arithmetic on stated values is correct, but the source value was wrong.

  SCOPE CHECK (mandatory after finding a miscopied entry):
  Check ALL diagonal entries the model listed.
  If the model replaced multiple diagonal entries with fabricated data,
  do NOT use INPUT_TRANSCRIPTION — escalate to HALLUCINATION
  (Silent_Omission or Spontaneous_Insertion).

───────────────────────────────────────────────────────────────────────────
CARRY_DOWN_ERROR
  The model correctly identified a diagonal entry or partial sum at step N
  and stated it correctly, but then miscopied that same value at step N+1
  when carrying it forward. A line-to-line copy error, not an arithmetic
  error.
  DETECT: Diagonal entry a(i,i) or partial sum stated correctly at step N.
  Same value used with a changed number at step N+1 with no new operation
  applied.
  CRITICAL: The FIRST occurrence must be genuinely correct. If the first
  occurrence is also wrong, classify as ARITHMETIC or SIGN_ERROR instead.
  Boundary with MEMORY_LOSS: CARRY_DOWN_ERROR = N→N+1 (no intervening
  steps); MEMORY_LOSS = value recalled after ≥1 unrelated intervening steps.

───────────────────────────────────────────────────────────────────────────
METHOD_FAIL
  The model uses a fundamentally wrong algorithm from the very start.
  DETECT: First 3–4 steps do not follow any valid method for trace
  computation (identifying and summing main diagonal entries).
  CRITICAL: A valid alternative presentation is NOT a METHOD_FAIL.
  Only use METHOD_FAIL if the algorithm itself is incorrect AND applied
  incorrectly — not merely different from the expected approach.
  Examples: summing ALL matrix entries instead of only diagonal entries,
  computing the determinant and presenting it as trace, computing row sums,
  column sums, or the Frobenius norm instead of the diagonal sum.

───────────────────────────────────────────────────────────────────────────
MEMORY_LOSS
  The model correctly identified and stated a diagonal entry or partial
  sum at step N, then recalled a different wrong value for the same
  quantity at step M > N.
  DETECT: a(i,i) stated correctly as −5 during identification, but the
  model uses −3 for that same entry in the final sum without any new
  operation applied to it.
  CRITICAL: If the first occurrence is also wrong, classify as ARITHMETIC
  or SIGN_ERROR, not MEMORY_LOSS.
  Boundary with CARRY_DOWN_ERROR: MEMORY_LOSS = value recalled after ≥1
  unrelated intervening steps; CARRY_DOWN_ERROR = N→N+1 immediate copy.

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
benchmarking LLMs on matrix trace computation.

You will be given a matrix A and a model's FAILED attempt to compute its trace.

The trace of a matrix is the sum of its main diagonal entries:
  tr(A) = a(1,1) + a(2,2) + a(3,3) + a(4,4)  [for a 4×4 matrix]

The model may have used any of these strategies:

  STRATEGY 1 — DIRECT DIAGONAL IDENTIFICATION
    The model explicitly identifies each diagonal entry a(i,i) in order,
    lists them, and sums them.
    Failure mode: sign errors or arithmetic errors when reading or
    accumulating individual diagonal entries.

  STRATEGY 2 — INLINE SUMMATION
    The model reads diagonal entries and accumulates the sum in one pass
    without explicitly listing each entry separately.
    Failure mode: carries an incorrect partial sum forward, or
    misreads a diagonal entry mid-stream.

  STRATEGY 3 — HEURISTIC / VISUAL INSPECTION
    The model estimates or asserts the trace without explicitly identifying
    all diagonal entries.
    Failure mode: fabricates or skips entries; hallucinates trace without
    grounding in the actual diagonal values.

${TRUNCATION_PRECHECK}

YOUR TASK HAS THREE STEPS:

STEP 1 — COMPUTE THE CORRECT TRACE YOURSELF
  You are a 100% accurate trace calculator. Identify each main diagonal
  entry a(1,1), a(2,2), ..., a(n,n) of the given matrix and sum them.
  That is your ground truth.
  Do not reveal these intermediate values in your response.

STEP 2 — IDENTIFY THE STRATEGY THE MODEL USED
  Based on the model's response, decide which of the three strategies above
  it attempted. Record this as "solution_strategy" in your output.

STEP 3 — SCAN THE MODEL RESPONSE FOR THE FIRST WRONG VALUE
  Read the model response step by step from beginning to end.
  Compare every stated diagonal entry and every partial sum against your
  correct computation.
  Find the FIRST step where the model's value diverges from correct.

  CRITICAL RULE: Classify by first point of failure, not final symptom.
  If the model misreads diagonal entry a(2,2) at step 2 and this
  propagates to a wrong final sum, the correct label is INPUT_TRANSCRIPTION
  or SIGN_ERROR, not ARITHMETIC.

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
  "solution_strategy": "DIRECT_DIAGONAL_IDENTIFICATION",
  "first_error_step": "Step 2 — reading a(2,2)",
  "first_error_description": "Diagonal entry a(2,2) = −5 but model used +5 in the sum. |5|=|−5| confirms sign error, not arithmetic.",
  "proposed_novel_tag": "",
  "maps_closest_to": "",
  "confidence": "HIGH"
}

For HALLUCINATION:
{
  "error_tag": "HALLUCINATION",
  "sign_subtype": "",
  "hallucination_subtype": "Complete_Collapse",
  "solution_strategy": "HEURISTIC",
  "first_error_step": "Step 1 — after identifying only a(1,1)",
  "first_error_description": "Model wrote 'by inspection the trace is clearly 4' without identifying all diagonal entries.",
  "proposed_novel_tag": "",
  "maps_closest_to": "",
  "confidence": "HIGH"
}

For all other tags:
{
  "error_tag": "ARITHMETIC",
  "sign_subtype": "",
  "hallucination_subtype": "",
  "solution_strategy": "DIRECT_DIAGONAL_IDENTIFICATION",
  "first_error_step": "Step 4 — final summation",
  "first_error_description": "Model identified all four diagonal entries correctly as 4, 0, 4, −4 but computed 4 + 0 + 4 − 4 = 5 instead of 4. Signs correct, magnitude wrong.",
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


# ─────────────────────────────────────────────────────────────────────────────
# BUILD — USER TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

BUILD_USER_TEMPLATE = """\
Matrix A:
{matrix_latex}

Correct trace: {ground_truth}
Model's extracted trace answer: {extracted_answer}

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
  2. A model's failed response attempting to compute its trace
  3. A classification made by a junior judge
  4. A targeted verification question for that error type

${ADVERSARIAL_FRAMING}

${ANSWER_CHECK_STEP0}

YOUR PROCESS:
  Step 1: Compute the correct trace yourself by identifying each main
          diagonal entry a(1,1), a(2,2), ..., a(n,n) and summing them.
          You are the ground truth oracle.
  Step 2: Read the model response step by step from beginning to end.
          Compare every stated diagonal entry and partial sum against
          your correct computation.
          Find the FIRST step where the model's value diverges from correct.
  Step 3: Answer the verification question with specific evidence quoted
          directly from the response. Confirm or deny the classification.

FIRST-ERROR PRINCIPLE:
  Classify by first point of failure, not final symptom.
  A sign error on diagonal entry a(2,2) at step 2 that cascades to a
  wrong final trace is SIGN_ERROR, not ARITHMETIC.

${MAGNITUDE_RULE}

STRICT TAXONOMY — primary_tag MUST be exactly one of these ten strings:
  GENERATION_TRUNCATION | FORMATTING_MISMATCH | SIGN_ERROR | ARITHMETIC |
  HALLUCINATION | INPUT_TRANSCRIPTION | CARRY_DOWN_ERROR | METHOD_FAIL |
  MEMORY_LOSS | OTHER_UNMAPPED

  Do NOT invent new primary tags. If error genuinely does not fit the
  ten primary tags, use OTHER_UNMAPPED and populate proposed_novel_tag
  and maps_closest_to.

SIGN_ERROR subtypes (required when primary_tag = SIGN_ERROR):
  Product_Sign_Error   - diagonal entry has correct magnitude, wrong sign in accumulation
  Operation_Direction  - added where should subtract in trace accumulation (or vice versa)
  Rule_Interference    - negative diagonal entry triggers wrong sign behaviour during summing
  Double_Negative_Trap - two negatives met in accumulation, model fails to resolve sign
  Silent_Sign_Flip     - wrong-sign diagonal entry with no working shown for that entry;
                         MAGNITUDE RULE applies: |wrong| must equal |correct|

HALLUCINATION subtypes (required when primary_tag = HALLUCINATION):
  Complete_Collapse     - explicit abandonment phrase, fewer than 2 diagonal entries identified
  Teleological_Zeroing  - diagonal entry written as 0 despite matrix not supporting it
  Premature_Assertion   - partial diagonal identification then trace claimed as complete
  Silent_Omission       - skips one or more diagonal entries silently, no meta-statement
  Ungrounded_Guess      - essentially no diagonal identification shown, just a final integer
  Spontaneous_Insertion - fabricated value inserted in sum after correct diagonal identification

RESPOND IN EXACTLY THIS JSON FORMAT AND NOTHING ELSE:
{
  "verified":              true,
  "primary_tag":           "INPUT_TRANSCRIPTION",
  "sign_subtype":          "",
  "hallucination_subtype": "",
  "proposed_novel_tag":    "",
  "maps_closest_to":       "",
  "forensic_observation":  "Confirmed: model misread a(2,2) = −5 as +5 when identifying diagonal entries at step 2",
  "error_scope":           "isolated",
  "affected_positions":    ["a(2,2) at step 2"],
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
                      "systematic"  = same error pattern across ≥2 diagonal positions
                                      sharing the same tag and subtype
                                      (e.g. sign flipped on every negative diagonal
                                      entry across the entire summation)
affected_positions:   list the specific diagonal entries or steps where the
                      error manifests (e.g. ["a(2,2) step 2", "a(4,4) step 4"])
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
    """
    Return a targeted verification question for the senior validator based on
    the junior judge's classification tag and active subtype.
    """

    # ── GENERATION_TRUNCATION ─────────────────────────────────────────────
    if tag == "GENERATION_TRUNCATION":
        return """The junior judge classified this as GENERATION_TRUNCATION — the response
ends mid-expression or mid-computation with no final trace answer.

Verification question:
  (1) Confirm the response has no final answer in \\boxed{} or equivalent.
  (2) Quote the last ~50 characters of the response. Does it end mid-sentence,
      mid-expression (e.g. "= "), or mid-summation?
  (3) Does the response contain any complete trace claim anywhere, even if
      stated before the apparent cut-off?

CRITICAL: If a complete trace claim exists anywhere in the response, this is
NOT GENERATION_TRUNCATION. Reclassify based on whether that claim is correct
or fabricated. Truncation requires no final classifiable answer."""

    # ── FORMATTING_MISMATCH ───────────────────────────────────────────────
    elif tag == "FORMATTING_MISMATCH":
        return """The junior judge classified this as FORMATTING_MISMATCH — the model's
trace computation is mathematically correct but the final answer format is wrong.

Verification question:
  (1) Verify every diagonal entry identified and every partial sum matches
      your correct trace computation.
  (2) Identify the specific formatting failure (missing \\boxed{}, answer
      stated as sentence without numeric form, etc.).
  (3) Confirm the mathematical trace value itself matches the ground truth.

CRITICAL: Do NOT confirm FORMATTING_MISMATCH if any diagonal entry or sum is
wrong. If the trace value itself is wrong, this is a mathematical error
(ARITHMETIC, HALLUCINATION, etc.) — set verified=false and reclassify.
NOTE: This tag is rare for trace since the answer is a single integer."""

    # ── INPUT_TRANSCRIPTION ───────────────────────────────────────────────
    elif tag == "INPUT_TRANSCRIPTION":
        return """The junior judge classified this as INPUT_TRANSCRIPTION — the model
misread a diagonal entry from the original input matrix A.

Verification question:
Identify the EXACT diagonal entry that was misread. State:
  (1) Position in the original matrix A (e.g., a(2,2)),
  (2) Correct value at that diagonal position in A,
  (3) Value the model actually used in its summation instead.

SCOPE CHECK (mandatory):
Check ALL diagonal entries the model listed.
If the model replaced multiple diagonal entries with fabricated data
(not just one cell), reclassify to HALLUCINATION
(Silent_Omission or Spontaneous_Insertion) — this exceeds the scope of
a single-entry transcription error.

CRITICAL: You must find both values directly in the response text.
If you cannot identify a specific misread entry with both the correct and
wrong value present, set verified=false and give your corrected tag.
Do not accept vague claims — exact position and both values required."""

    # ── CARRY_DOWN_ERROR ──────────────────────────────────────────────────
    elif tag == "CARRY_DOWN_ERROR":
        return """The junior judge classified this as CARRY_DOWN_ERROR — the model
correctly stated a diagonal entry or partial sum at step N but miscopied
it at step N+1 when carrying it forward into the next operation.

Verification question:
  (1) Quote the correctly-stated entry or partial sum at step N (with step label).
  (2) Quote the changed form at step N+1 (must be adjacent — no intervening
      unrelated steps between N and N+1).
  (3) Confirm the first occurrence matches your correct trace computation.
  (4) Confirm there is no new operation between N and N+1 that would
      legitimately change the value.

CRITICAL: If the first occurrence is itself wrong, this is ARITHMETIC or
SIGN_ERROR — not CARRY_DOWN_ERROR. Set verified=false and reclassify.
If the two occurrences are separated by ≥1 unrelated intervening steps,
reclassify as MEMORY_LOSS instead."""

    # ── ARITHMETIC ────────────────────────────────────────────────────────
    elif tag == "ARITHMETIC":
        return """The junior judge classified this as ARITHMETIC — correct diagonal
identification and correct signs, but wrong magnitude in a numerical sum.

Verification question:
Find the specific step where the wrong number first appeared.
  (1) Were ALL signs in the diagonal entries and partial sums correct going in?
  (2) Apply the MAGNITUDE RULE: is |wrong value| ≠ |correct value|?
  (3) Were all diagonal entries correctly identified from the matrix?

CRITICAL: Apply MAGNITUDE RULE strictly. If |wrong| = |correct|, the root
error is SIGN_ERROR not ARITHMETIC — set verified=false and correct the tag.
If any diagonal entry was misread with a wrong sign, also reclassify
as SIGN_ERROR. Only set verified=true if signs were genuinely correct AND
magnitude failed."""

    # ── SIGN_ERROR subtypes ───────────────────────────────────────────────
    elif tag == "SIGN_ERROR" and sign_sub == "Product_Sign_Error":
        return f"""The junior judge classified this as SIGN_ERROR/Product_Sign_Error —
a diagonal entry has correct magnitude but wrong sign in the summation.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
Find the exact diagonal entry where this occurred. State:
  (1) Which diagonal position was involved (e.g., a(3,3)),
  (2) The correct value and sign it should carry in the sum,
  (3) The sign the model actually used,
  (4) The specific partial sum that became wrong as a result.

CRITICAL: Apply MAGNITUDE RULE. |entry| must be correct — only the sign
is flipped. If |wrong| ≠ |correct|, reclassify as ARITHMETIC."""

    elif tag == "SIGN_ERROR" and sign_sub == "Operation_Direction":
        return f"""The junior judge classified this as SIGN_ERROR/Operation_Direction —
the model subtracted a diagonal entry where it should have added (or vice versa),
making the partial sum wrong.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
Find the exact accumulation step. State:
  (1) Which diagonal entry was being accumulated (position),
  (2) The correct operation (e.g., partial_sum + a(3,3)),
  (3) What the model wrote instead (e.g., partial_sum − a(3,3)),
  (4) The resulting partial sum value vs the correct value.

CRITICAL: Confirm the diagonal entry itself was correctly identified.
If the entry value itself had the wrong sign, reclassify as Product_Sign_Error."""

    elif tag == "SIGN_ERROR" and sign_sub == "Rule_Interference":
        return f"""The junior judge classified this as SIGN_ERROR/Rule_Interference —
a negative diagonal entry triggered wrong sign behaviour during summation.
The model treated the negative sign as an implicit subtraction command,
double-applying or cancelling it.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Which diagonal entry caused the interference? State position.
  (2) What is its correct value in A (must be negative)?
  (3) Show the accumulation step where the sign was mishandled
      (e.g., tr = ... − (−k) computed as tr = ... − k instead of tr = ... + k).
  (4) What did the model compute vs the correct result?

CRITICAL: The entry must be negative and the error must be in how the model
handles that negative during accumulation — not a misread at copy time (that is
INPUT_TRANSCRIPTION)."""

    elif tag == "SIGN_ERROR" and sign_sub == "Double_Negative_Trap":
        return f"""The junior judge classified this as SIGN_ERROR/Double_Negative_Trap —
two or more negatives met in the trace accumulation and the model failed to
resolve them correctly (e.g. partial_sum = −3 and a(4,4) = −4;
−3 + (−4) computed as −3 − (−4) = +1 instead of −7).
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Quote the exact expression containing the two negatives.
  (2) State the correct resolution and the model's resolution.
  (3) Confirm BOTH operands are genuinely negative in the computation.
  (4) Apply MAGNITUDE RULE: confirm |result| is correct — only the sign wrong.

CRITICAL: If only one operand is negative, this is not Double_Negative_Trap.
If |wrong| ≠ |correct|, reclassify as ARITHMETIC."""

    elif tag == "SIGN_ERROR" and sign_sub == "Silent_Sign_Flip":
        return f"""The junior judge classified this as SIGN_ERROR/Silent_Sign_Flip —
BEHAVIORAL DEFINITION: a sign-valued diagonal entry appears in the response
AND zero intermediate identification is shown for that specific entry.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Identify the last diagonal entry that was fully and correctly identified
      (position and value shown).
  (2) Identify the first entry with the wrong sign — state position,
      correct value, and model's value.
  (3) Confirm there is NO identification or computation shown for that entry
      between the last correct step and the wrong result.
  (4) Apply MAGNITUDE RULE: confirm |wrong| = |correct| (magnitudes equal,
      sign flipped). If magnitudes differ, classify as ARITHMETIC instead.

CRITICAL: If identification IS shown for that entry, do not classify as
Silent_Sign_Flip. Identify the actual step that produced the wrong sign and
reclassify (Product_Sign_Error, Operation_Direction, Rule_Interference, or
Double_Negative_Trap)."""

    elif tag == "SIGN_ERROR":
        return """The junior judge classified this as SIGN_ERROR but no subtype was given.

Verification question:
Find the first wrong value. Apply MAGNITUDE RULE:
  — If |wrong| ≠ |correct|: reclassify as ARITHMETIC (not SIGN_ERROR).
  — If |wrong| = |correct|, sign differs: confirm SIGN_ERROR, then identify subtype:

  Product_Sign_Error    — diagonal entry has correct magnitude, sign flipped in accumulation
  Operation_Direction   — subtracted where should add in trace accumulation (or vice versa)
  Rule_Interference     — negative diagonal entry triggers wrong sign during summing
  Double_Negative_Trap  — two negatives met in accumulation, sign not correctly resolved
  Silent_Sign_Flip      — wrong-sign entry with no identification shown; |wrong|=|correct|

Populate sign_subtype in your response."""

    # ── HALLUCINATION subtypes ────────────────────────────────────────────
    elif tag == "HALLUCINATION" and halluc_sub == "Teleological_Zeroing":
        return """The junior judge classified this as HALLUCINATION/Teleological_Zeroing —
the model showed identification of a diagonal entry but wrote 0 as its value
despite the matrix not supporting it.

Verification question:
  (1) Which diagonal position was written as 0?
  (2) What is the actual value at that diagonal position in A?
  (3) Does the model show any identification for this entry, or does it
      jump straight to 0?

CRITICAL: If the diagonal entry genuinely IS 0 in the matrix, this is not
Teleological_Zeroing — that would be a correct identification. The defining
feature is that the matrix has a non-zero diagonal entry but the model writes
0 anyway. If the model skips all identification, use Silent_Omission."""

    elif tag == "HALLUCINATION" and halluc_sub == "Complete_Collapse":
        return """The junior judge classified this as HALLUCINATION/Complete_Collapse —
the model explicitly abandoned diagonal summation with a meta-statement and
fewer than 2 diagonal entries were actually identified.

Verification question:
  (1) Quote the exact phrase where the model abandoned computation
      (e.g., "by inspection", "clearly the trace is", "we can observe that").
  (2) How many diagonal entries did the model actually identify before stopping?
      This must be fewer than 2 for Complete_Collapse.
  (3) Did the model show any explicit diagonal entry identification?

CRITICAL: If the model identified 2 or more entries before stopping, reclassify
as Premature_Assertion. If no meta-statement exists and computation is just
absent, reclassify as Silent_Omission or Ungrounded_Guess."""

    elif tag == "HALLUCINATION" and halluc_sub == "Premature_Assertion":
        return """The junior judge classified this as HALLUCINATION/Premature_Assertion —
the model began identifying diagonal entries, read only a partial subset,
then stated a trace as if complete.

Verification question:
  (1) Which diagonal entries did the model actually identify? List them.
  (2) Which diagonal entries are missing when the trace is claimed?
  (3) Quote the line where the trace is asserted prematurely.

CRITICAL: At least one valid diagonal entry identification must have been
performed for this to be Premature_Assertion. If no entries are identified,
use Ungrounded_Guess. If the model explicitly says the matrix is too complex
and stops, reclassify as Complete_Collapse."""

    elif tag == "HALLUCINATION" and halluc_sub == "Silent_Omission":
        return """The junior judge classified this as HALLUCINATION/Silent_Omission —
the model produced a response but silently skipped one or more diagonal entries —
their identification is absent without any meta-statement.

Verification question:
  (1) Which diagonal entries are absent from the model's identification?
  (2) Is there any meta-statement or explanation for the skip?
  (3) Does the response jump from partial identification directly to a
      final trace sum that cannot be derived from the shown entries?
  (4) Does the response end with a complete final answer, or is it truncated?

CRITICAL: If the response is truncated (no final answer), reclassify as
GENERATION_TRUNCATION. If there IS a meta-statement like "by inspection",
reclassify as Complete_Collapse.
If very short with essentially no identification, use Ungrounded_Guess."""

    elif tag == "HALLUCINATION" and halluc_sub == "Ungrounded_Guess":
        return """The junior judge classified this as HALLUCINATION/Ungrounded_Guess —
essentially no diagonal entry identification shown; model states trace with no
meaningful derivation.

Verification question:
Confirm the response contains no meaningful diagonal identification — no
explicit listing of a(1,1), a(2,2), etc. with their values from the matrix.
  (1) Are any diagonal entries explicitly identified by position and value?
  (2) Is there any partial summation shown?

CRITICAL: If any diagonal entry is identified explicitly, even one, use
Silent_Omission or Premature_Assertion instead. Ungrounded_Guess is reserved
for responses that are essentially pure assertion with zero identification."""

    elif tag == "HALLUCINATION" and halluc_sub == "Spontaneous_Insertion":
        return """The junior judge classified this as HALLUCINATION/Spontaneous_Insertion —
the model correctly identified all diagonal entries then inserted a fabricated
value in the final sum with no mathematical origin in the prior steps.

Verification question:
  (1) Identify the last diagonal entry that was correctly identified —
      state the position and value.
  (2) Identify the inserted value in the final sum that has no prior basis —
      state exactly what the model wrote vs what the sum of shown entries gives.
  (3) Confirm the inserted value cannot be derived from any diagonal entry
      shown in the response.

CRITICAL: If the inserted value can be traced to a prior step (even a wrong
one), this is ARITHMETIC or SIGN_ERROR, not Spontaneous_Insertion.
Spontaneous_Insertion requires a value with NO computational origin in the
response — pure fabrication after correct identification."""

    elif tag == "HALLUCINATION":
        return """The junior judge classified this as HALLUCINATION but no subtype was given.

Determine which subtype applies:
  Complete_Collapse     — explicit abandonment phrase + fewer than 2 diagonal entries identified
  Teleological_Zeroing  — diagonal entry written as 0 despite matrix having non-zero entry there
  Premature_Assertion   — partial diagonal identification then trace claimed as complete
  Silent_Omission       — one or more diagonal entries silently missing, no meta-statement
  Ungrounded_Guess      — essentially no diagonal identification shown, just a final integer
  Spontaneous_Insertion — fabricated value inserted in sum after correct identification

Populate hallucination_subtype in your response."""

    # ── MEMORY_LOSS ───────────────────────────────────────────────────────
    elif tag == "MEMORY_LOSS":
        return """The junior judge classified this as MEMORY_LOSS — the model correctly
identified and stated a diagonal entry or partial sum in an earlier step,
then used a different wrong value for that same quantity in a later step
(recalled after ≥1 unrelated intervening steps).

Verification question:
  (1) What value was stated correctly at the early step?
      Give step description and exact value (e.g., "a(3,3) identified as −7 at step 3").
  (2) What value did the model use for that same entry or sum at the later step?
      Give step description and the wrong value used.
  (3) Was the FIRST occurrence genuinely correct per your trace computation?
  (4) Are the two occurrences separated by ≥1 unrelated intervening steps?

CRITICAL: If the first occurrence is also wrong, the root error is ARITHMETIC
or SIGN_ERROR — set verified=false and correct the tag.
If the two occurrences are adjacent (N→N+1, no intervening steps), reclassify
as CARRY_DOWN_ERROR instead."""

    # ── METHOD_FAIL ───────────────────────────────────────────────────────
    elif tag == "METHOD_FAIL":
        return """The junior judge classified this as METHOD_FAIL — the model never
performed legitimate diagonal identification and summation.

Verification question:
Describe what method the model used in the first 3–4 steps.
Confirm it fundamentally differs from any valid trace-computation algorithm.

Examples that qualify as METHOD_FAIL:
  - Summing ALL entries of the matrix (not just the main diagonal)
  - Computing the determinant and presenting it as the trace
  - Summing row totals or column totals
  - Computing the Frobenius norm or some other matrix scalar
  - Confusing trace with rank, nullity, or another matrix property

CRITICAL: Directly summing a(1,1) + a(2,2) + ... is the ONLY valid method
for trace — there are no alternative valid algorithms. If the model set up
correct diagonal identification but made errors within the summation, classify
by the actual error type (ARITHMETIC, SIGN_ERROR, etc.) and set verified=false.
METHOD_FAIL is only for wrong algorithm from step 1."""

    # ── OTHER_UNMAPPED / UNKNOWN ──────────────────────────────────────────
    else:
        return """The junior judge could not classify this error (UNKNOWN or OTHER_UNMAPPED).

Fresh classification — ignore the previous tag entirely.

Read the model's trace computation from the beginning.
Find the first step where the model's value diverges from correct diagonal
identification and summation.

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
If OTHER_UNMAPPED, populate proposed_novel_tag (e.g., "DIAGONAL_SKIP_ERROR")
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
