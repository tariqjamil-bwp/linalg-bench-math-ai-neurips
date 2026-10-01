"""
prompts/transpose.py — Build-judge and validate-judge prompts for the TRANSPOSE subcategory.

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
  DETECT: Response ends at "= " or mid-word. No complete Aᵀ matrix present.
  Use this before attempting any other classification.

───────────────────────────────────────────────────────────────────────────
FORMATTING_MISMATCH
  The model's mathematics is 100% correct but the final answer is presented
  incorrectly (wrong matrix notation, missing required structure).
  DETECT: Every entry of Aᵀ matches ground truth. Only the final matrix
  presentation fails the formatting requirement.
  CRITICAL: Do NOT use if any entry value is wrong.

───────────────────────────────────────────────────────────────────────────
SIGN_ERROR
  The first wrong entry has the correct magnitude but wrong sign.
  See MAGNITUDE RULE above — do not use ARITHMETIC for sign flips.
  Choose exactly one subtype:

  Product_Sign_Error
    A product of two terms has correct magnitude but wrong sign, or the
    reflected entry has correct magnitude but wrong sign applied during copy.
    In transpose context: model reflected the correct entry to the correct
    position but wrote the wrong sign on the value.
    DETECT: Correct transposed position (j,i), correct magnitude, wrong sign.
    Transpose example: A_{2,3} = −4; model writes (Aᵀ)_{3,2} = +4.

  Operation_Direction
    Model placed an entry at the wrong position in Aᵀ (wrong row or column
    after the reflection). The value is correct but displacement is wrong.
    DETECT: Entry value is correct but placed at (i,j) instead of (j,i).
    Transpose example: Entry A_{2,3} placed at (2,3) in Aᵀ instead of (3,2).

  Rule_Interference
    A negative entry in A causes the model to mishandle the sign —
    treating the transpose operation as if it also negates negative entries.
    DETECT: Negative entry A_{ij} = −k placed in Aᵀ as +k, as if the
    transpose operation itself negates the entry.
    Transpose example: A_{i,j} = −7; model writes (Aᵀ)_{j,i} = +7, treating
    transpose as sign-flip.

  Double_Negative_Trap
    A negative entry in A is doubly negated during transposition (e.g.
    model writes −(−k) = k instead of preserving −k).
    DETECT: Entry A_{ij} = −k; model writes (Aᵀ)_{ji} = +k, apparently
    applying an extra negation during the copy step.
    Disambiguate from Rule_Interference: Double_Negative_Trap requires
    explicit evidence of two negations; Rule_Interference is implicit
    sign-removal from negative entries.

  Silent_Sign_Flip
    BEHAVIORAL DEFINITION — do not infer intent:
    A sign-valued output appears in the response AND zero transposition
    working is shown for that specific entry.
    DETECT: Surrounding entries correct and shown with transposition steps.
    This one entry has only a stated value with the wrong sign, no working.
    Do NOT use if any transposition step is shown for that entry.
    MAGNITUDE RULE applies: |wrong| must equal |correct|; if magnitudes
    differ and no working is shown, classify as ARITHMETIC.

───────────────────────────────────────────────────────────────────────────
ARITHMETIC
  The first wrong entry has the correct sign but wrong magnitude. The model
  miscopied the numerical value of an entry (different from INPUT_TRANSCRIPTION
  in that the sign is correct but the digit is wrong after transposition).
  See MAGNITUDE RULE — do not use for sign flips.
  DETECT: Correct sign, wrong numerical value in the transposed position.

───────────────────────────────────────────────────────────────────────────
HALLUCINATION
  The model does not produce a wrong individual entry by computation.
  Instead it stops computing or fabricates the result. Choose one:

  Complete_Collapse
    Model explicitly abandons computation with a meta-statement and fewer
    than 2 rows of Aᵀ have been written. Short response.
    DETECT: Short response + explicit meta-statement + fewer than 2 rows written.

  Teleological_Zeroing
    Model writes a row of Aᵀ as all zeros despite A not having a zero column.
    DETECT: A row of Aᵀ is [0…0] but the corresponding column of A is not.

  Premature_Assertion
    Model transposes some rows then states the full Aᵀ as if complete.
    DETECT: Correct partial rows, then sudden full matrix claim.

  Silent_Omission
    Model produces a full-length response but skips writing out entries for
    one or more rows of Aᵀ.
    NOTE: Do NOT use for truncated responses — use GENERATION_TRUNCATION.

  Ungrounded_Guess
    Model provides no transposition working. States Aᵀ directly with no
    row-by-row reflection shown.
    DETECT: Very short response, no entry-by-entry reflection, Aᵀ presented
    as if obvious.

  Spontaneous_Insertion
    Model correctly transposes most entries then inserts a fabricated entry
    value with no origin in the input matrix A.
    DETECT: All transposition steps up to the insertion are correct. An
    arbitrary value appears in Aᵀ that cannot be traced to any entry in A.

───────────────────────────────────────────────────────────────────────────
INPUT_TRANSCRIPTION
  The model placed the entry at the correct transposed position (j,i) but
  misread the value from A — the source value was wrong because it was
  misread from the problem statement, not a sign flip.
  DETECT: Correct transposed position (j,i) + wrong value (not a sign flip —
  the digit itself is different from A_{ij}).

  SCOPE CHECK (mandatory after finding a miscopied entry):
  Check ALL adjacent entries in the same row/column of the model's written
  submatrix. If the model replaced an entire row or column with fabricated
  data, do NOT use INPUT_TRANSCRIPTION — escalate to HALLUCINATION
  (Silent_Omission or Spontaneous_Insertion).

───────────────────────────────────────────────────────────────────────────
CARRY_DOWN_ERROR
  The model correctly wrote a transposed entry at step N and stated it
  correctly, but then miscopied that same value at step N+1 when carrying
  it forward into another step. A line-to-line copy error.
  DETECT: Entry (Aᵀ)_{ij} stated correctly at step N. Same entry written
  with a changed value at step N+1 with no new transposition applied.
  CRITICAL: The FIRST occurrence must be genuinely correct. If the first
  occurrence is also wrong, classify as ARITHMETIC or SIGN_ERROR instead.
  Boundary with MEMORY_LOSS: CARRY_DOWN_ERROR = N→N+1 (no intervening
  steps); MEMORY_LOSS = value recalled after ≥1 unrelated intervening steps.

───────────────────────────────────────────────────────────────────────────
METHOD_FAIL
  The model uses a fundamentally wrong algorithm from the very start.
  DETECT: First 3–4 steps do not follow any valid transpose method.
  CRITICAL: Only use if the algorithm itself is incorrect.

  (base — no subtype)
    Model never performs the correct reflection (i,j)→(j,i). Examples:
    returning A itself unchanged, rotating A 90°, computing Aᵀ as −A.
    DETECT: No valid entry reflection performed; wrong procedure from step 1.

  Composition_Rule_Violation
    Model applies a wrong rule for the transpose of a composition:
    (AB)ᵀ computed as AᵀBᵀ instead of BᵀAᵀ.
    Individual sub-transpositions are correct; the composition order is wrong.
    DETECT: Sub-operations correct; rule for combining them is wrong.

───────────────────────────────────────────────────────────────────────────
MEMORY_LOSS
  The model wrote a correct transposed entry at step N, then recalled a
  different wrong value for that same entry at step M > N.
  DETECT: Entry (Aᵀ)_{ij} stated correctly at one point; wrong value used
  for same entry later without any further operation.
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
benchmarking LLMs on matrix transpose computation.

You will be given a matrix A and a model's FAILED attempt to compute Aᵀ.

The model may have used any of these strategies:

  STRATEGY 1 — DIRECT TRANSPOSE
    The model reflects A along its main diagonal: entry (i,j) of A becomes
    entry (j,i) of Aᵀ. Rows of A become columns of Aᵀ.
    Failure mode: model swaps rows/columns incorrectly, or miscopies entries
    during the reflection.

  STRATEGY 2 — COMPOSITION RULE
    The model is given a product matrix (e.g. AB) and applies a transpose
    rule: (AB)ᵀ = BᵀAᵀ. Or model applies (Aⁿ)ᵀ = (Aᵀ)ⁿ.
    Failure mode: model applies (AB)ᵀ = AᵀBᵀ instead of BᵀAᵀ, or
    otherwise uses the wrong composition rule.

${TRUNCATION_PRECHECK}

YOUR TASK HAS THREE STEPS:

STEP 1 — COMPUTE THE CORRECT TRANSPOSE YOURSELF
  You are a 100% accurate transpose calculator. Compute Aᵀ by reflecting
  A along its main diagonal: (Aᵀ)_{ij} = A_{ji}.
  Do not reveal these intermediate values in your response.

STEP 2 — IDENTIFY THE STRATEGY THE MODEL USED
  Based on the model's response, decide which of the two strategies above
  it attempted. Record this as "solution_strategy" in your output.

STEP 3 — SCAN THE MODEL RESPONSE FOR THE FIRST WRONG VALUE
  Read the model response from beginning to end.
  Compare every entry of Aᵀ the model writes against your correct Aᵀ.
  Find the FIRST entry where the model's value diverges from correct.

  CRITICAL RULE: Classify by first point of failure, not final symptom.

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
  "solution_strategy": "DIRECT_TRANSPOSE",
  "first_error_step": "Entry (3,2) of Aᵀ — transposed from A_{2,3}",
  "first_error_description": "A_{2,3} = −4; model wrote (Aᵀ)_{3,2} = +4. |4|=|−4| confirms sign error at correct transposed position.",
  "proposed_novel_tag": "",
  "maps_closest_to": "",
  "confidence": "HIGH"
}

For METHOD_FAIL/Composition_Rule_Violation:
{
  "error_tag": "METHOD_FAIL",
  "sign_subtype": "",
  "hallucination_subtype": "",
  "solution_strategy": "COMPOSITION_RULE",
  "first_error_step": "Step 1 — applying transpose of product",
  "first_error_description": "Model computed (AB)ᵀ = AᵀBᵀ instead of BᵀAᵀ. Individual Aᵀ and Bᵀ computed correctly; composition order wrong.",
  "proposed_novel_tag": "Composition_Rule_Violation",
  "maps_closest_to": "",
  "confidence": "HIGH"
}

For all other tags:
{
  "error_tag": "INPUT_TRANSCRIPTION",
  "sign_subtype": "",
  "hallucination_subtype": "",
  "solution_strategy": "DIRECT_TRANSPOSE",
  "first_error_step": "Entry (2,4) of Aᵀ — transposed from A_{4,2}",
  "first_error_description": "A_{4,2} = 3; model wrote (Aᵀ)_{2,4} = 8. Correct transposed position, wrong value misread from A.",
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

Correct Aᵀ: {ground_truth}
Model's extracted Aᵀ answer: {extracted_answer}

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
  2. A model's failed response attempting to compute Aᵀ
  3. A classification made by a junior judge
  4. A targeted verification question for that error type

${ADVERSARIAL_FRAMING}

${ANSWER_CHECK_STEP0}

YOUR PROCESS:
  Step 1: Compute the correct Aᵀ yourself by reflecting A along its main
          diagonal: (Aᵀ)_{ij} = A_{ji}. You are the ground truth oracle.
  Step 2: Read the model response from beginning to end. Compare every entry
          of Aᵀ the model writes against your correct computation.
          Find the FIRST entry where the model diverges.
  Step 3: Answer the verification question with specific evidence quoted
          directly from the response. Confirm or deny the classification.

FIRST-ERROR PRINCIPLE:
  Classify by first point of failure, not final symptom.

${MAGNITUDE_RULE}

STRICT TAXONOMY — primary_tag MUST be exactly one of these ten strings:
  GENERATION_TRUNCATION | FORMATTING_MISMATCH | SIGN_ERROR | ARITHMETIC |
  HALLUCINATION | INPUT_TRANSCRIPTION | CARRY_DOWN_ERROR | METHOD_FAIL |
  MEMORY_LOSS | OTHER_UNMAPPED

SIGN_ERROR subtypes (required when primary_tag = SIGN_ERROR):
  Product_Sign_Error   - correct transposed position, correct magnitude, wrong sign
  Operation_Direction  - entry placed at wrong transposed position
  Rule_Interference    - negative entry triggers additional negation during transpose
  Double_Negative_Trap - negative entry doubly negated during transposition
  Silent_Sign_Flip     - wrong-sign entry with no transposition working shown;
                         MAGNITUDE RULE applies: |wrong| must equal |correct|

HALLUCINATION subtypes (required when primary_tag = HALLUCINATION):
  Complete_Collapse     - explicit abandonment, fewer than 2 rows of Aᵀ written
  Teleological_Zeroing  - row of Aᵀ written as [0…0] despite no zero column in A
  Premature_Assertion   - partial rows then full matrix claimed as complete
  Silent_Omission       - entries silently absent, no meta-statement
  Ungrounded_Guess      - no transposition working shown, Aᵀ stated directly
  Spontaneous_Insertion - fabricated entry inserted after correct transposition chain

METHOD_FAIL subtypes:
  (base)                    - wrong algorithm (returns A, rotates, negates, etc.)
  Composition_Rule_Violation - (AB)ᵀ = AᵀBᵀ instead of BᵀAᵀ

RESPOND IN EXACTLY THIS JSON FORMAT AND NOTHING ELSE:
{
  "verified":              true,
  "primary_tag":           "SIGN_ERROR",
  "sign_subtype":          "Product_Sign_Error",
  "hallucination_subtype": "",
  "proposed_novel_tag":    "",
  "maps_closest_to":       "",
  "forensic_observation":  "Confirmed: A_{2,3} = −4; model wrote (Aᵀ)_{3,2} = +4. Correct position, sign flipped.",
  "error_scope":           "isolated",
  "affected_positions":    ["(Aᵀ)_{3,2}"],
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

error_scope:          "isolated"    = one entry wrong
                      "systematic"  = same error pattern across ≥2 entries
                                      sharing the same tag and subtype
affected_positions:   list the specific (Aᵀ)_{ij} entries where the error
                      manifests (e.g. ["(Aᵀ)_{3,2}", "(Aᵀ)_{4,1}"])
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
ends mid-expression or mid-computation with no final Aᵀ matrix.

Verification question:
  (1) Confirm the response has no complete Aᵀ matrix in the final answer.
  (2) Quote the last ~50 characters of the response. Does it end mid-sentence,
      mid-expression, or mid-transposition step?
  (3) Does the response contain any complete Aᵀ claim anywhere?

CRITICAL: If a complete Aᵀ result exists anywhere in the response, this is
NOT GENERATION_TRUNCATION. Reclassify based on whether that result is correct
or fabricated."""

    # ── FORMATTING_MISMATCH ───────────────────────────────────────────────
    elif tag == "FORMATTING_MISMATCH":
        return """The junior judge classified this as FORMATTING_MISMATCH — the model's
transpose computation is mathematically correct but the final format is wrong.

Verification question:
  (1) Verify every entry of Aᵀ matches your correct computation.
  (2) Identify the specific formatting failure (wrong matrix notation, etc.).
  (3) Confirm every numerical entry value matches the ground truth.

CRITICAL: Do NOT confirm FORMATTING_MISMATCH if any entry value is wrong.
If any entry is wrong, this is a mathematical error — set verified=false
and reclassify."""

    # ── INPUT_TRANSCRIPTION ───────────────────────────────────────────────
    elif tag == "INPUT_TRANSCRIPTION":
        return """The junior judge classified this as INPUT_TRANSCRIPTION — the model
placed the entry at the correct transposed position but wrote the wrong
numerical value (misread from A, not a sign flip).

Verification question:
Identify the EXACT entry that was miscopied. State:
  (1) Position in A: row i, column j, correct value A_{ij},
  (2) Transposed position in Aᵀ: row j, column i,
  (3) Value the model actually wrote at that transposed position.

SCOPE CHECK (mandatory):
Check ALL adjacent entries in the same row/column of the model's written
submatrix. If the model replaced an entire row or column with fabricated
data, reclassify to HALLUCINATION (Silent_Omission or Spontaneous_Insertion).

CRITICAL: If the sign is correct but the digit is wrong, confirm INPUT_TRANSCRIPTION.
If the sign is wrong but the magnitude is correct, reclassify as SIGN_ERROR.
If the entry was placed at the wrong position, reclassify as SIGN_ERROR/Operation_Direction."""

    # ── CARRY_DOWN_ERROR ──────────────────────────────────────────────────
    elif tag == "CARRY_DOWN_ERROR":
        return """The junior judge classified this as CARRY_DOWN_ERROR — the model
correctly wrote a transposed entry at step N but miscopied it at step N+1
when carrying it forward.

Verification question:
  (1) Quote the correctly-stated entry at step N (with step label).
  (2) Quote the changed form at step N+1 (must be adjacent — no intervening
      unrelated steps).
  (3) Confirm the first occurrence matches your correct Aᵀ.
  (4) Confirm there is no new transposition applied between N and N+1.

CRITICAL: If the first occurrence is itself wrong, this is ARITHMETIC or
SIGN_ERROR — not CARRY_DOWN_ERROR. Set verified=false and reclassify.
If separated by ≥1 unrelated intervening steps, reclassify as MEMORY_LOSS."""

    # ── ARITHMETIC ────────────────────────────────────────────────────────
    elif tag == "ARITHMETIC":
        return """The junior judge classified this as ARITHMETIC — correct transposed
position but wrong numerical magnitude (sign is correct).

Verification question:
Find the specific entry where the wrong number first appeared. State:
  (1) Position in A: (i,j) and correct value A_{ij},
  (2) Transposed position in Aᵀ: (j,i),
  (3) Correct value that should appear at (j,i),
  (4) Value the model wrote — confirm sign is correct, only magnitude wrong.

CRITICAL: Apply MAGNITUDE RULE. If |wrong| = |correct| but sign differs,
reclassify as SIGN_ERROR. ARITHMETIC here means the digit was wrong,
not the sign."""

    # ── SIGN_ERROR subtypes ───────────────────────────────────────────────
    elif tag == "SIGN_ERROR" and sign_sub == "Product_Sign_Error":
        return f"""The junior judge classified this as SIGN_ERROR/Product_Sign_Error —
the model reflected the correct entry to the correct position but wrote
the wrong sign on the value.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
Find the exact entry. State:
  (1) A_{{ij}}: position in A and correct value,
  (2) The transposed position (j,i) in Aᵀ,
  (3) Correct value expected at (j,i),
  (4) Value the model wrote — confirm magnitude correct, sign flipped.

CRITICAL: Apply MAGNITUDE RULE. Position must be correct (j,i) and
magnitude must be correct (|wrong| = |correct|).
If position is wrong, reclassify as Operation_Direction.
If magnitude is also wrong, reclassify as INPUT_TRANSCRIPTION or ARITHMETIC."""

    elif tag == "SIGN_ERROR" and sign_sub == "Operation_Direction":
        return f"""The junior judge classified this as SIGN_ERROR/Operation_Direction —
the model placed an entry at the wrong position in Aᵀ (wrong row or column
after the reflection).
Junior judge's first error step: {step or '(not specified)'}

Verification question:
Find the misplaced entry. State:
  (1) The source entry A_{{ij}} and its correct value,
  (2) Correct transposed position (j,i),
  (3) Where the model actually placed this entry (wrong position),
  (4) What the model wrote at the correct position (j,i) instead.

CRITICAL: Confirm the position is wrong, not just the sign.
If position is correct but sign is flipped, reclassify as Product_Sign_Error."""

    elif tag == "SIGN_ERROR" and sign_sub == "Rule_Interference":
        return f"""The junior judge classified this as SIGN_ERROR/Rule_Interference —
a negative entry in A triggered additional negation during transposition,
as if the transpose operation itself negates the entry.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Which entry A_{{ij}} caused the interference? State position and value
      (must be negative).
  (2) What did the model write at (Aᵀ)_{{ji}}?
  (3) Is the model treating transpose as also negating negative entries
      (writing +k where A_{ij} = −k)?

CRITICAL: Entry must be negative. If the model simply misread the sign when
copying without evidence of treating transpose as negation, reclassify as
INPUT_TRANSCRIPTION."""

    elif tag == "SIGN_ERROR" and sign_sub == "Double_Negative_Trap":
        return f"""The junior judge classified this as SIGN_ERROR/Double_Negative_Trap —
a negative entry in A was doubly negated during transposition
(e.g. A_{{ij}} = −k; model writes (Aᵀ)_{{ji}} = +k via −(−k)).
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Which entry A_{{ij}} has the double-negation? State position and value
      (must be negative).
  (2) What did the model write at (Aᵀ)_{{ji}}?
  (3) Is there evidence of two negations (e.g. model writes "−(−k) = k")?
  (4) Apply MAGNITUDE RULE: confirm |result| is correct — only the sign wrong.

CRITICAL: Disambiguate from Rule_Interference: Double_Negative_Trap requires
explicit evidence of two negations. If model just implicitly removes the
negative, use Rule_Interference."""

    elif tag == "SIGN_ERROR" and sign_sub == "Silent_Sign_Flip":
        return f"""The junior judge classified this as SIGN_ERROR/Silent_Sign_Flip —
BEHAVIORAL DEFINITION: a sign-valued output appears in the response AND
zero transposition working is shown for that specific entry.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Identify the last entry correct and shown with transposition steps.
  (2) Identify the first entry with wrong sign — state (j,i), correct value,
      and model's value.
  (3) Confirm NO transposition working is shown for that entry.
  (4) Apply MAGNITUDE RULE: confirm |wrong| = |correct|. If magnitudes differ,
      classify as ARITHMETIC instead.

CRITICAL: If working IS shown for that entry, identify the actual error and
reclassify (Product_Sign_Error, Rule_Interference, or Double_Negative_Trap)."""

    elif tag == "SIGN_ERROR":
        return """The junior judge classified this as SIGN_ERROR but no subtype was given.

Find the first wrong entry. Apply MAGNITUDE RULE:
  — If |wrong| ≠ |correct|: reclassify as ARITHMETIC (not SIGN_ERROR).
  — If |wrong| = |correct|, sign differs: confirm SIGN_ERROR, then identify subtype:

  Product_Sign_Error    — correct transposed position, correct magnitude, wrong sign
  Operation_Direction   — entry placed at wrong transposed position
  Rule_Interference     — negative entry triggers additional negation
  Double_Negative_Trap  — negative entry doubly negated (−(−k) = k)
  Silent_Sign_Flip      — wrong-sign entry with no working shown; |wrong|=|correct|

Populate sign_subtype in your response."""

    # ── HALLUCINATION subtypes ────────────────────────────────────────────
    elif tag == "HALLUCINATION" and halluc_sub == "Teleological_Zeroing":
        return """The junior judge classified this as HALLUCINATION/Teleological_Zeroing —
a row of Aᵀ is written as all zeros despite the corresponding column of A
not being a zero column.

Verification question:
  (1) Which row of Aᵀ was written as [0…0]?
  (2) What is the corresponding column of A? Is it all zeros?
  (3) Does the model show any working for why this row is zero?

CRITICAL: If the corresponding column of A IS all zeros, this is a correct
entry — not Teleological_Zeroing."""

    elif tag == "HALLUCINATION" and halluc_sub == "Complete_Collapse":
        return """The junior judge classified this as HALLUCINATION/Complete_Collapse —
the model explicitly abandoned computation with a meta-statement.

Verification question:
  (1) Quote the exact phrase where the model abandoned computation.
  (2) How many rows of Aᵀ did the model actually write before stopping?
      Must be fewer than 2 for Complete_Collapse.

CRITICAL: If 2 or more rows written, reclassify as Premature_Assertion."""

    elif tag == "HALLUCINATION" and halluc_sub == "Premature_Assertion":
        return """The junior judge classified this as HALLUCINATION/Premature_Assertion —
the model wrote some rows of Aᵀ then stated the full matrix as complete.

Verification question:
  (1) Which rows of Aᵀ did the model actually write out?
  (2) Which rows are absent when the full matrix is claimed?
  (3) Quote the line where the full matrix is asserted prematurely.

CRITICAL: At least one valid row must have been written. If none, use
Ungrounded_Guess. If explicit abandonment phrase, use Complete_Collapse."""

    elif tag == "HALLUCINATION" and halluc_sub == "Silent_Omission":
        return """The junior judge classified this as HALLUCINATION/Silent_Omission —
the model produced a full-length response but silently skipped writing some
rows or entries of Aᵀ.

Verification question:
  (1) Which rows or entries have no transposition shown?
  (2) Is there any meta-statement explaining the skip?
  (3) Does the response jump to the full Aᵀ without showing all steps?
  (4) Does the response end with a complete final answer, or is it truncated?

CRITICAL: If truncated, reclassify as GENERATION_TRUNCATION.
If meta-statement exists, reclassify as Complete_Collapse."""

    elif tag == "HALLUCINATION" and halluc_sub == "Ungrounded_Guess":
        return """The junior judge classified this as HALLUCINATION/Ungrounded_Guess —
essentially no transposition working shown; model states Aᵀ with no derivation.

Verification question:
  (1) Is any row-by-row reflection working shown?
  (2) Are any individual entry transpositions shown?

CRITICAL: If any transposition working exists, use Silent_Omission or
Premature_Assertion instead."""

    elif tag == "HALLUCINATION" and halluc_sub == "Spontaneous_Insertion":
        return """The junior judge classified this as HALLUCINATION/Spontaneous_Insertion —
the model correctly transposed most entries then inserted a fabricated entry
value with no origin in the input matrix A.

Verification question:
  (1) Identify the last entry correctly transposed — state (j,i) and value.
  (2) Identify the inserted entry that has no prior basis — state (j,i)
      and what the model wrote.
  (3) Confirm the inserted value cannot be found at any position in A.

CRITICAL: If the inserted value corresponds to an entry in A (just placed
at the wrong position), reclassify as SIGN_ERROR/Operation_Direction instead.
Spontaneous_Insertion requires a value with NO origin in A."""

    elif tag == "HALLUCINATION":
        return """The junior judge classified this as HALLUCINATION but no subtype was given.

Determine which subtype applies:
  Complete_Collapse     — explicit abandonment + fewer than 2 rows written
  Teleological_Zeroing  — row of Aᵀ written as [0…0] despite non-zero column in A
  Premature_Assertion   — partial rows then full matrix claimed
  Silent_Omission       — entries silently absent, no meta-statement
  Ungrounded_Guess      — no transposition working shown, Aᵀ stated directly
  Spontaneous_Insertion — fabricated entry inserted after correct transposition chain

Populate hallucination_subtype in your response."""

    # ── MEMORY_LOSS ───────────────────────────────────────────────────────
    elif tag == "MEMORY_LOSS":
        return """The junior judge classified this as MEMORY_LOSS — the model wrote a
correct transposed entry at step N then used a wrong value for that same
entry in a later step (recalled after ≥1 unrelated intervening steps).

Verification question:
  (1) Which entry (Aᵀ)_{ij} was stated correctly at the early step?
      Give the step and exact value.
  (2) What wrong value did the model use for that entry later?
  (3) Was the FIRST occurrence genuinely correct per your computation?
  (4) Are the two occurrences separated by ≥1 unrelated intervening steps?

CRITICAL: If the first occurrence is also wrong, root error is ARITHMETIC
or SIGN_ERROR — set verified=false and correct the tag.
If the two occurrences are adjacent (N→N+1), reclassify as CARRY_DOWN_ERROR."""

    # ── METHOD_FAIL ───────────────────────────────────────────────────────
    elif tag == "METHOD_FAIL":
        return """The junior judge classified this as METHOD_FAIL — the model never
performed the correct matrix transpose operation.

Verification question:
Describe what the model did in the first 3–4 steps.

If the model applied (AB)ᵀ = AᵀBᵀ instead of BᵀAᵀ, state this explicitly —
this is Composition_Rule_Violation.

Examples of base METHOD_FAIL:
  - Returning A unchanged (no reflection at all)
  - Rotating A 90° instead of reflecting along diagonal
  - Computing −A and calling it Aᵀ
  - Applying row reduction to A

Examples of Composition_Rule_Violation:
  - (AB)ᵀ assembled as AᵀBᵀ; individual Aᵀ and Bᵀ computed correctly

CRITICAL: If the model correctly reflected some entries but made errors in
individual entries, classify by actual error type (SIGN_ERROR, INPUT_TRANSCRIPTION)
and set verified=false."""

    # ── OTHER_UNMAPPED / UNKNOWN ──────────────────────────────────────────
    else:
        return """The junior judge could not classify this error (UNKNOWN or OTHER_UNMAPPED).

Fresh classification — ignore the previous tag entirely.

Read the model's transpose computation from the beginning.
Find the first step where the model's value diverges from correct Aᵀ.

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
##########################################################################################
# END OF FILE
##########################################################################################

