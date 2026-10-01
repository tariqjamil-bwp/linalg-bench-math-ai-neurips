# ERROR TAXONOMY REFERENCE CARD
# Paste these definitions into every BUILD_SYSTEM and VALIDATE_SYSTEM prompt.
# Replace any existing tag definition with the version below.
# All names and rules are domain-agnostic — valid for det, eig, inv,
# nullity, rank, mul, pow, vec, transpose subcategories.
#
# ─────────────────────────────────────────────────────────────────────────────
# HOW TO USE
# ─────────────────────────────────────────────────────────────────────────────
# 1. In BUILD_SYSTEM: paste TRUNCATION_PRECHECK at the very top (before Step 1),
#    paste MAGNITUDE_RULE just before the tag list, add new tags to the tag list.
#
# 2. In VALIDATE_SYSTEM: paste ADVERSARIAL_FRAMING at the very top,
#    paste ANSWER_CHECK_STEP0 right after, paste MAGNITUDE_RULE before taxonomy,
#    add TRANSCRIPTION_SCOPE_CHECK to the TRANSCRIPTION verification question,
#    add error_scope + affected_positions to the output JSON schema.
# ─────────────────────────────────────────────────────────────────────────────


# ═════════════════════════════════════════════════════════════════════════════
# INSERT AT TOP OF BUILD_SYSTEM — before any classification steps
# ═════════════════════════════════════════════════════════════════════════════

TRUNCATION_PRECHECK = """
MANDATORY FIRST CHECK — BEFORE CLASSIFYING ANYTHING:
Does the response end with a complete final answer (e.g. \\boxed{}) or a
clear numerical conclusion?

IF the response ends mid-expression (e.g. "= "), mid-sentence, or
mid-computation with no final answer:
  → Tag as GENERATION_TRUNCATION. Stop. Do not classify further.

Do NOT tag a truncated response as HALLUCINATION, ARITHMETIC, or
Silent_Omission. Those require a completed response with a wrong value.
A missing value is not a wrong value.
"""


# ═════════════════════════════════════════════════════════════════════════════
# INSERT JUST BEFORE THE TAG LIST — in both BUILD_SYSTEM and VALIDATE_SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

MAGNITUDE_RULE = """
CRITICAL BOUNDARY — SIGN_ERROR vs ARITHMETIC (NON-NEGOTIABLE):
Ask: is the absolute magnitude of the first wrong value correct?

  |wrong| = |correct|  →  MUST be SIGN_ERROR  (sign is flipped)
  |wrong| ≠ |correct|  →  MUST be ARITHMETIC  (magnitude is corrupted)

This applies regardless of HOW the sign was lost — failed multiplication,
dropped negative, double-applied parity, or botched subtraction.
If the magnitude is correct, it is SIGN_ERROR. No exceptions.

Examples:
  correct = -30, model writes +30  →  SIGN_ERROR   (|-30| = |+30|)
  correct = -30, model writes -18  →  ARITHMETIC   (|-18| ≠ |-30|)
  correct =   6, model writes  -6  →  SIGN_ERROR   (|6| = |-6|)
  correct =   6, model writes   8  →  ARITHMETIC   (|8| ≠ |6|)
"""


# ═════════════════════════════════════════════════════════════════════════════
# INSERT AT TOP OF VALIDATE_SYSTEM — before process steps
# ═════════════════════════════════════════════════════════════════════════════

ADVERSARIAL_FRAMING = """
You are an adversarial QA auditor. 
Your job is to PROVE the junior judge wrong wherever they made an error.
Do not set verified=true until you have explicitly checked:

  ✗ Did they confuse a sign flip for an arithmetic error?
    (If |wrong| = |correct|, it MUST be SIGN_ERROR not ARITHMETIC.)
  ✗ Did they miss a truncated response and classify a math error instead?
    (If no final answer exists, it is GENERATION_TRUNCATION.)
  ✗ Did they miss a formatting failure and label it as math?
    (If all intermediates are correct, it is FORMATTING_MISMATCH.)
  ✗ Did they classify by final symptom not first error?
    (Trace back to the earliest divergence from correct.)
  ✗ Did they use a domain-specific subtype name in the wrong context?
    (e.g. "Pivot" language for a cofactor/adjugate method — correct the
    subtype name only, not the primary tag.)

Only after confirming none of the above apply should you set verified=true.
"""

ANSWER_CHECK_STEP0 = """
VALIDATOR STEP 0 — MANDATORY ANSWER CHECK:
1. Compute the correct answer yourself independently.
2. Compare it to the model's final boxed answer.

  IF response is truncated (no final answer):
    → verified=null, tag GENERATION_TRUNCATION. Stop.

  IF model's final answer is CORRECT:
    → Scan for wrong intermediates. If none found, verified=null,
      note "answer correct, no error identifiable."

  IF model's final answer is WRONG:
    → You MUST find a specific wrong intermediate value.
    → verified=null is NOT permitted when the final answer is wrong.
    → A wrong answer with no identified intermediate error = validator failure.
      Keep searching.
"""


# ═════════════════════════════════════════════════════════════════════════════
# TAG DEFINITIONS
# Replace your existing tag definitions with these.
# Keep your domain-specific DETECT examples after each definition.
# ═════════════════════════════════════════════════════════════════════════════

TAGS = """
──────────────────────────────────────────────────────────────────────────────
GENERATION_TRUNCATION  ← NEW TAG
  The response ends mid-expression or mid-computation with no final answer.
  No mathematical error is classifiable — the computation was never finished.
  DETECT: Response ends at "= " or mid-word. No \\boxed{} present.
  Use this before attempting any other classification.

──────────────────────────────────────────────────────────────────────────────
FORMATTING_MISMATCH  ← NEW TAG
  The model's mathematics is 100% correct but the final answer is presented
  incorrectly (missing repeated roots, wrong box format, omitting required
  structure from the output specification).
  DETECT: Every intermediate matches ground truth. Only the final boxed
  presentation fails the formatting requirement.
  CRITICAL: Do NOT use if any intermediate value is wrong.

──────────────────────────────────────────────────────────────────────────────
SIGN_ERROR
  The first wrong value has the correct absolute magnitude but wrong sign.
  See MAGNITUDE RULE above — do not use ARITHMETIC for sign flips.
  Choose exactly one subtype:

  Product_Sign_Error     ← renamed from Row_Multiplier_Sign
    A product of two terms has correct magnitude but wrong sign.
    Applies in ANY algorithm — cofactor expansion, row reduction, dot product,
    scalar multiplication. "Row" is dropped — this is domain-agnostic.
    DETECT: |model value| = |correct value|, sign differs, in a multiply step.

  Operation_Direction
    Model added where it should have subtracted, or vice versa.
    DETECT: Accumulation operator +/− applied in wrong direction.

  Rule_Interference
    A negative matrix entry causes the model to mishandle the sign —
    double-applying or cancelling the negative from that entry.
    DETECT: Entry is negative; model treats it as positive in computation.

  Parity_Sign_Error      ← renamed from Pivot_Sign_Error
    The parity factor (−1)^{i+j} applied with wrong sign, or applied twice.
    Applies to cofactor expansion and adjugate computation.
    "Pivot" is dropped — this tag applies to cofactors, not just pivots.
    DETECT: Minor M_ij correct in magnitude; cofactor sign (±) is flipped.

  Double_Negative_Trap
    Two or more negatives met in a product or chain and the model failed
    to resolve them (e.g. −(−x) written as −x instead of +x).
    DETECT: Expression with two negatives; model produces wrong sign on result.

  Alternating_Drift
    Model correctly tracks a +−+−+ sign pattern for the first few terms
    then loses it midway.
    DETECT: Pattern correct for ≥2 terms, then breaks.

  Cofactor_Neglect
    Model computed a minor correctly but used it directly without applying
    the (−1)^{i+j} parity factor at all.
    DETECT: Minor value used raw in final assembly; no parity sign applied.

  Silent_Sign_Flip       ← replaces Fluency_Masking
    BEHAVIORAL DEFINITION — do not infer intent:
    A sign-valued output appears in the response AND zero intermediate
    computation is shown for that specific step.
    DETECT: Surrounding steps have shown working. This one step has only
    a stated value with the wrong sign, no supporting computation.
    Do NOT use if any computation is shown for that step.

  Parametric_Sign_Flip   ← NULLITY only
    A sign error in a parameterized null-space solution. The free variable
    coefficient has the correct magnitude but wrong sign in the final answer.
    DETECT: Nullity problem; parameterized vector entries have flipped signs
    relative to ground truth. |model value| = |correct value|, sign differs.

──────────────────────────────────────────────────────────────────────────────
ARITHMETIC
  The first wrong value has the correct sign but wrong absolute magnitude.
  Method correct, signs tracked correctly, but a numerical calculation
  produced the wrong number.
  See MAGNITUDE RULE — do not use for sign flips.
  DETECT: Wrong products or sums where the sign of the result is correct
  but the absolute value is wrong.

──────────────────────────────────────────────────────────────────────────────
HALLUCINATION
  The model does not produce a wrong intermediate value by computation.
  Instead it fabricates, abandons, or invents. Choose one subtype:

  Complete_Collapse
    Model explicitly abandons with a meta-statement ("due to complexity",
    "we can simplify", "directly providing the answer"). Short response,
    few or no intermediates computed.

  Teleological_Zeroing
    Full-length response but model forces intermediate values to exactly
    zero without justification. Zeros do not match correct computation.

  Silent_Omission
    Model skips computation blocks silently with no meta-statement.
    Response continues as if skipped steps were completed.
    NOTE: Do NOT use for truncated responses — use GENERATION_TRUNCATION.

  Ungrounded_Guess
    Essentially no working shown. Model outputs a final value with
    minimal or no supporting computation.

  Spontaneous_Insertion  ← NEW SUBTYPE
    Model completes a correct computation chain then inserts a fabricated
    term, constant, or variable with no mathematical origin in prior steps.
    DETECT: All steps up to insertion point are correct. A large or
    arbitrary value appears that cannot be traced to any prior step.
    Example: det(A) = −18 + 30 + 24 + 38480 where 38480 has no origin.

──────────────────────────────────────────────────────────────────────────────
INPUT_TRANSCRIPTION    ← renamed from TRANSCRIPTION
  The model miscopied an entry from the ORIGINAL INPUT matrix.
  Error occurs before any arithmetic — the value was never computed,
  just read from the problem statement incorrectly.
  DETECT: Model's submatrix entry does not match the input matrix.

  SCOPE CHECK (mandatory after finding a miscopied entry):
  Check ALL adjacent entries in the same row and column.
  If the model replaced an entire row or column with fabricated data,
  do NOT use INPUT_TRANSCRIPTION — escalate to HALLUCINATION
  (Silent_Omission or Spontaneous_Insertion).

──────────────────────────────────────────────────────────────────────────────
CARRY_DOWN_ERROR       ← NEW TAG, split from TRANSCRIPTION
  The model correctly computed an intermediate value at step N and stated
  it correctly, but then miscopied that same value at step N+1 when
  carrying it forward. A line-to-line copy error, not an arithmetic error.
  DETECT: Value V correct at step N. Same quantity written as V' ≠ V
  at step N+1 with no new computation.
  CRITICAL: The FIRST occurrence must be genuinely correct. If the first
  occurrence is also wrong, classify as ARITHMETIC or SIGN_ERROR instead.

──────────────────────────────────────────────────────────────────────────────
METHOD_FAIL
  The model uses a fundamentally wrong algorithm from the very start.
  DETECT: First 3-4 steps do not follow any valid method for this task.
  CRITICAL: A valid alternative algorithm is NOT a METHOD_FAIL.
  Only use METHOD_FAIL if the algorithm itself is incorrect AND applied
  incorrectly — not merely different from the expected approach.
  Row reduction used correctly for determinants is NOT METHOD_FAIL.

──────────────────────────────────────────────────────────────────────────────
MEMORY_LOSS
  The model correctly computed and stated a value at step N, then recalled
  a different wrong value for the same quantity at step M > N.
  DETECT: Quantity Q stated as correct value V at step N; stated as wrong
  value W at step M. First occurrence must be correct.
  CRITICAL: If the first occurrence is also wrong, classify as ARITHMETIC
  or SIGN_ERROR, not MEMORY_LOSS.

──────────────────────────────────────────────────────────────────────────────
OTHER_UNMAPPED
  Use ONLY if the error genuinely cannot be mapped to any tag above after
  careful analysis. Populate proposed_novel_tag and maps_closest_to.
  This is not a default — exhaust all other options first.
──────────────────────────────────────────────────────────────────────────────
"""


# ═════════════════════════════════════════════════════════════════════════════
# ADD TO VALIDATE OUTPUT JSON SCHEMA
# Add these two fields to the existing JSON output format in VALIDATE_SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

NEW_JSON_FIELDS = """
  "error_scope":        "isolated",
  "affected_positions": ["C_12"],

  error_scope:          "isolated"    = one position wrong
                        "systematic"  = same error pattern across multiple
                                        positions (e.g. all negative-parity
                                        cofactors have flipped sign)
  affected_positions:   list the specific steps, cells, or cofactors where
                        the error manifests (e.g. ["C_12", "C_21", "C_32"])
"""


# ═════════════════════════════════════════════════════════════════════════════
# QUICK RENAME REFERENCE
# Old name → New name. Update anywhere these strings appear in prompts.
# ═════════════════════════════════════════════════════════════════════════════

RENAMES = {
    "Row_Multiplier_Sign": "Product_Sign_Error",
    "Pivot_Sign_Error":    "Parity_Sign_Error",
    "Fluency_Masking":     "Silent_Sign_Flip",
    "TRANSCRIPTION":       "INPUT_TRANSCRIPTION",
    # CARRY_DOWN_ERROR is new — no old name, add as a new tag
    # GENERATION_TRUNCATION is new — no old name, add as a new tag
    # FORMATTING_MISMATCH is new — no old name, add as a new tag
    # Spontaneous_Insertion is new — add to HALLUCINATION subtypes
}

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY FOR ENGINEER — what to do in each file
# ═════════════════════════════════════════════════════════════════════════════

ENGINEER_INSTRUCTIONS = """
For each of the 9 files (det, eigen, inv, nullity, rank, mul, pow, vec, transpose):

IN BUILD_SYSTEM:
  1. Paste TRUNCATION_PRECHECK at the very top before Step 1.
  2. Paste MAGNITUDE_RULE just before the tag list.
  3. In the tag list:
     a. Add GENERATION_TRUNCATION (new, paste definition above).
     b. Add FORMATTING_MISMATCH (new, paste definition above).
     c. Rename Row_Multiplier_Sign → Product_Sign_Error in text.
     d. Rename Pivot_Sign_Error → Parity_Sign_Error in text.
     e. Replace Fluency_Masking definition with Silent_Sign_Flip definition.
     f. Rename TRANSCRIPTION → INPUT_TRANSCRIPTION, add SCOPE CHECK note.
     g. Add CARRY_DOWN_ERROR as a new tag after INPUT_TRANSCRIPTION.
     h. Add Spontaneous_Insertion to HALLUCINATION subtypes.
  4. Update the example JSON at the bottom to include the new tag names.

IN VALIDATE_SYSTEM:
  1. Paste ADVERSARIAL_FRAMING at the very top.
  2. Paste ANSWER_CHECK_STEP0 immediately after.
  3. Paste MAGNITUDE_RULE before the taxonomy section.
  4. Apply the same tag renames as BUILD_SYSTEM (steps 3a-3h above).
  5. Add error_scope and affected_positions to the output JSON schema.
  6. In the INPUT_TRANSCRIPTION verification question, add SCOPE CHECK.

IN _build_verification_question():
  1. Add a branch for GENERATION_TRUNCATION.
  2. Add a branch for FORMATTING_MISMATCH.
  3. Add a branch for CARRY_DOWN_ERROR.
  4. Update SIGN_ERROR branch: rename subtypes in elif conditions.
  5. Add Spontaneous_Insertion branch under HALLUCINATION.
  6. Update TRANSCRIPTION branch: rename to INPUT_TRANSCRIPTION,
     add scope check text in the question body.

ESTIMATED EFFORT PER FILE: ~20 minutes of find-replace + paste.
TOTAL: ~3 hours for all 9 files.
"""
