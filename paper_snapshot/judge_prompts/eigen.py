"""
judge_prompts/eigen.py — Forensic judge prompts for eigenvalue subcategory.

Revised to align with unified taxonomy (taxonomy_reference_card.py + Final signs.pdf).
14 primary tags: 10 universal + 4 bespoke (ALGEBRAIC_PRECEDENCE, FALSE_VERIFICATION,
VARIABLE_ENTANGLEMENT, GENERATION_LOOP). NUMERICAL_DEGRADATION folded into ARITHMETIC.
"""
from string import Template
from .taxonomy_reference_card import (
    TRUNCATION_PRECHECK,
    MAGNITUDE_RULE,
    ADVERSARIAL_FRAMING,
    ANSWER_CHECK_STEP0,
)

TAG_DEFINITIONS = """\
───────────────────────────────────────────────────────────────────────────
GENERATION_TRUNCATION
  The response ends mid-expression or mid-computation with no final answer.
  No mathematical error is classifiable — the computation was never finished.
  DETECT: Response ends at "= " or mid-word. No \\boxed{} or eigenvalue list present.
  Use this before attempting any other classification.

───────────────────────────────────────────────────────────────────────────
FORMATTING_MISMATCH
  The model's mathematics is 100% correct but the final answer is presented
  incorrectly (missing repeated roots, wrong box format, omitting required
  structure from the output specification).
  DETECT: Every intermediate (polynomial expansion, roots) matches ground truth.
  Only the final presentation fails the formatting requirement.
  CRITICAL: Do NOT use if any intermediate value is wrong.

───────────────────────────────────────────────────────────────────────────
SIGN_ERROR
  The first wrong value is a sign mistake in the characteristic polynomial
  expansion or cofactor assembly. The magnitude may be correct but the sign
  is flipped. Choose exactly one subtype:

  Product_Sign_Error
    A product of two terms has correct magnitude but wrong sign.
    Applies in ANY algorithm — cofactor expansion, row reduction, dot product,
    scalar multiplication. The element a_{1j} multiplied by a minor value,
    or a term in a sub-determinant, has correct |product| but wrong sign.
    DETECT: |product| = |correct|, sign differs in a multiply step.
    Eigen example: a_{12} · M_{12} = (−2) · f(λ) should give −2f(λ) but
    model writes +2f(λ) — magnitude correct, sign flipped.

  Operation_Direction
    Model added where it should have subtracted, or vice versa, in a
    determinant expansion or polynomial accumulation step.
    DETECT: Accumulation operator +/− applied in wrong direction.
    Eigen example: In 2×2 sub-determinant, model writes a·d + b·c instead
    of a·d − b·c, adding where it should subtract.

  Rule_Interference
    At the final polynomial assembly step, all cofactor values are correct but
    the model treats the negative sign of a negative matrix entry as a
    subtraction operator rather than part of the multiplication.
    DETECT: Entry a_{12} = −2; model writes "− 2 * C_12" in the sum instead
    of "(−2) * C_12", double-applying the negative on that polynomial term.

  Parity_Sign_Error
    The model correctly identifies the cofactor position but applies the
    (−1)^{i+j} parity factor with wrong sign on one isolated term, or applies
    it twice. An isolated single parity error (not a pattern drift).
    DETECT: Minor M_{1j} correct in magnitude; cofactor sign (±) is flipped
    on one term without a pattern drift.
    Distinguishable from Alternating_Drift: Parity_Sign_Error is isolated
    (fewer than 2 correct preceding terms in the same expansion).

  Double_Negative_Trap
  Two or more negatives met in a product or chain and the model failed
    to resolve them (e.g. −(−x) written as −x instead of +x).
    DETECT: Model wrote −8 − 3 instead of −8 − (−3) = −5 in a 2×2 block
    of the (A − λI) expansion. Both values must be genuinely negative.

  Alternating_Drift
    The model correctly applied (−1)^{1+j} for the first few cofactors of
    det(A − λI) but then lost the +−+−+ checkerboard pattern. The first wrong
    value is a cofactor with correct magnitude but wrong sign.
    DETECT: Pattern correct for ≥2 terms, then breaks.
    Eigen example: C_11=+M_11, C_12=−M_12 correct, then C_13=−M_13 instead of +M_13.
    Distinguishable from Parity_Sign_Error: Alternating_Drift requires ≥2 correct
    terms before the pattern breaks.

  Cofactor_Neglect
    The model computed a minor polynomial correctly but used it directly in the
    polynomial assembly without applying (−1)^{1+j}.
    DETECT: Model computes M_12 = f(λ) correctly, then writes
    polynomial += a_12 * M_12 instead of polynomial += a_12 * (−1) * M_12.
    Distinguishable from Parity_Sign_Error: Cofactor_Neglect is zero parity
    application; Parity_Sign_Error is wrong parity applied.

  Silent_Sign_Flip
    BEHAVIORAL DEFINITION — do not infer intent:
    A sign-valued output appears in the response AND zero intermediate
    computation is shown for that specific step.
    DETECT: Surrounding steps have shown working. This one step has only
    a stated value with the wrong sign, no supporting computation.
    Do NOT use if any computation is shown for that step.
    MAGNITUDE RULE applies: |wrong| must equal |correct|; if magnitudes
    differ and no working is shown, classify as ARITHMETIC.
    Eigen example: Model writes "C_13 = 644" with no working where correct is −644.

───────────────────────────────────────────────────────────────────────────
ARITHMETIC
  The first wrong value has the correct sign but wrong magnitude during the
  characteristic polynomial expansion. Method is correct, signs are tracked
  correctly at that step, but a numerical calculation produced the wrong number.
  See MAGNITUDE RULE — do not use for sign flips.
  DETECT: Wrong products or sums in 2×2 sub-determinant expansions inside the
  characteristic polynomial. Example: model computes 7·5 − (−3)·(−4) = 47
  instead of 35 − 12 = 23.

  NOTE: Also use ARITHMETIC (not a separate tag) for these precision errors:
  - Premature rounding: converting √17 → 4.123 mid-expansion
  - Compounding drift: small early coefficient error growing into large final error
  - Irrational phobia: forcing irrational eigenvalue (1+√13)/2 ≈ 2.30 to integer λ = 2
  In these cases, explain the nature (rounding/drift/irrational forcing) in
  first_error_description.

───────────────────────────────────────────────────────────────────────────
HALLUCINATION
  The model does not produce a wrong intermediate value by computation.
  Instead it fabricates, abandons, or invents. Choose one subtype:

  Complete_Collapse
    Model explicitly abandons the characteristic polynomial computation with a
    meta-statement. Response is short (under 5000 characters). Writes phrases like
    "due to the complexity", "we can simplify by", "the eigenvalues are approximately"
    and outputs fabricated values.
    DETECT: Short response + explicit meta-statement + missing polynomial expansion.
    Fewer than 3 minors computed before stopping.

  Teleological_Zeroing
    Model produces a full-length response but forces intermediate polynomial
    coefficients or sub-determinants to equal exactly 0 without justification.
    Terms suspiciously cancel perfectly. The model manufactures zeros.
    DETECT: Long response + multiple intermediates = 0 + those zeros do not
    match your correct characteristic polynomial computation.

  Premature_Assertion
    Model begins the characteristic polynomial expansion, computes some (but
    not all) minors, then states eigenvalues as if the computation were complete.
    DETECT: Medium-length response, partial minor computations, then sudden
    eigenvalue claim without finishing the polynomial.

  Silent_Omission
    Model produces a full-length response but silently skips computation blocks —
    minors or cofactors claimed without showing the sub-determinant expansion.
    NOTE: Do NOT use for truncated responses — use GENERATION_TRUNCATION.
    DETECT: Long response + "following the same process..." + sub-determinant
    steps missing for one or more minors.

  Ungrounded_Guess
    Essentially no working shown. Model outputs eigenvalues with minimal or
    no supporting characteristic polynomial computation.
    DETECT: Very short response, no minors or cofactors computed, eigenvalue
    list presented as if obvious.

  Spontaneous_Insertion
    Model completes a correct cofactor expansion chain up to a point, then
    inserts a fabricated polynomial coefficient or minor value with no
    mathematical origin in prior steps.
    DETECT: All sub-determinant computations up to the insertion point are
    correct. A large or arbitrary value appears for a polynomial term that
    cannot be derived from the matrix entries.

───────────────────────────────────────────────────────────────────────────
INPUT_TRANSCRIPTION
  The first wrong value is a matrix entry. The model copied a number incorrectly
  from the input matrix when constructing (A − λI) or a submatrix.
  The error occurs before any arithmetic begins — the value was never computed,
  just read from the problem statement incorrectly.
  DETECT: Compare the model's stated submatrix entries against the original
  matrix A. Find one that does not match.

  SCOPE CHECK (mandatory after finding a miscopied entry):
  Check ALL adjacent entries in the same row and column of the model's
  written-out submatrix. If the model replaced an entire row or column with
  fabricated data, do NOT use INPUT_TRANSCRIPTION — escalate to HALLUCINATION
  (Silent_Omission or Spontaneous_Insertion).

───────────────────────────────────────────────────────────────────────────
CARRY_DOWN_ERROR
  The model correctly computed a polynomial coefficient or minor value at step N
  and stated it correctly, but then miscopied that same value at step N+1 when
  carrying it forward. A line-to-line copy error, not an arithmetic error.
  DETECT: Coefficient or minor stated correctly at step N; written with a changed
  value at step N+1 with no new computation applied.
  CRITICAL: The FIRST occurrence must be genuinely correct. If the first
  occurrence is also wrong, classify as ARITHMETIC or SIGN_ERROR instead.
  Boundary with MEMORY_LOSS: CARRY_DOWN_ERROR = N→N+1 (no intervening
  steps); MEMORY_LOSS = value recalled after ≥1 unrelated intervening steps.

───────────────────────────────────────────────────────────────────────────
METHOD_FAIL
  The model never sets up the characteristic polynomial det(A − λI) = 0 or
  cofactor expansion correctly from the start.
  DETECT: First 3–4 steps do not follow Laplace expansion or any valid
  eigenvalue method at all.
  CRITICAL: A valid alternative algorithm is NOT a METHOD_FAIL. Only use
  METHOD_FAIL if the algorithm itself is incorrect AND applied incorrectly —
  not merely different from the expected approach.
  If the model set up det(A − λI) correctly but made errors within the
  expansion, classify by the actual error type (not METHOD_FAIL).

───────────────────────────────────────────────────────────────────────────
MEMORY_LOSS
  The model correctly computed a polynomial coefficient or minor value at an
  early step and stated it, then used a different wrong value for that same
  quantity at a later step (recalled after ≥1 unrelated intervening steps).
  DETECT: Value V stated correctly at step N, then stated as wrong value W
  at step M where M > N. Example: coefficient of λ^3 = −14 at step 6,
  then used as −41 at step 18.
  CRITICAL: If the first occurrence is also wrong, classify as ARITHMETIC
  or SIGN_ERROR, not MEMORY_LOSS.
  Boundary with CARRY_DOWN_ERROR: MEMORY_LOSS = recalled after ≥1 unrelated
  intervening steps; CARRY_DOWN_ERROR = N→N+1 immediate copy.

───────────────────────────────────────────────────────────────────────────
OTHER_UNMAPPED
  Use ONLY if the error genuinely cannot be mapped to any tag above (including
  the four bespoke eigen tags below) after careful analysis.
  Populate proposed_novel_tag and maps_closest_to.
  This is not a default — exhaust all other options first.

───────────────────────────────────────────────────────────────────────────
— EIGEN-SPECIFIC EXTENDED TAGS (apply only when universal tags do not fit) —
───────────────────────────────────────────────────────────────────────────

ALGEBRAIC_PRECEDENCE
  Structural failure in maintaining mathematical grouping, operator order, or
  distribution across intermediate steps of the characteristic polynomial expansion.
  Try universal tags first; use this only when operator structure — not value —
  is the root cause. Choose one subtype:

  Bracket_Erosion
    The model silently drops parentheses in an intermediate step of the
    polynomial expansion, changing the mathematical expression.
    DETECT: Model writes 2(λ − 3) → 2λ − 3 instead of 2λ − 6 in the
    expansion of a cofactor term.

  PEMDAS_Violation
    The model incorrectly prioritizes operations due to left-to-right reading
    bias during polynomial assembly.
    DETECT: An expression like −3 * (λ^2 − 4) is evaluated as (−3 * λ^2) − 4
    instead of −3λ^2 + 12.

  Exponent_Detachment
    The model fails to distribute an exponent or power across a multiplied term
    in the characteristic polynomial.
    DETECT: (λ − 2)^2 incorrectly expanded as λ^2 − 2 instead of λ^2 − 4λ + 4.

───────────────────────────────────────────────────────────────────────────
FALSE_VERIFICATION
  The model performs a self-check or substitution step to verify a candidate
  eigenvalue but fabricates or misinterprets the result to achieve narrative closure.
  Choose one subtype:

  Circular_Substitution
    The model uses a derived (possibly flawed) intermediate formula to prove
    its derived (possibly flawed) eigenvalue answer is correct. Circular proof.
    DETECT: Model derives incorrect polynomial p(λ), then substitutes its
    guessed eigenvalue into the same p(λ) to "confirm" it.
    Also covers: substituting λ = 3 into the characteristic polynomial,
    getting a non-zero value, but asserting "= 0 ✓" or "verified".

  Tautological_Check
    The model retroactively adds arbitrary constants or fudge factors to force
    an intermediate verification step to equal zero, or performs a trivially
    circular identity as a "check" (e.g., verifying det(A−λI)=0 using the
    same step that produced λ).
    DETECT: Model writes "adjusting for rounding, p(λ) = 0.001 ≈ 0, confirmed."
    with no mathematical justification for the adjustment. Or model restates
    the assumption as proof.

───────────────────────────────────────────────────────────────────────────
VARIABLE_ENTANGLEMENT
  The model mixes up indices, eigenvalue subscripts, or variable scopes during
  the characteristic polynomial expansion or root-labeling steps.
  Choose one subtype:

  Variable_Substitution_Error
    The model substitutes the wrong variable or eigenvalue index into a
    polynomial expression, or mislabels eigenvalue subscripts across steps.
    DETECT: Model evaluates p(λ_2) where it should evaluate p(λ_1), or
    uses a submatrix entry a_{32} where a_{23} is correct.

  Variable_Reassignment
    The model reuses a variable or intermediate value computed in one sub-minor
    context inside a different sub-minor context where it does not apply,
    or treats a polynomial expression f(λ) as a scalar constant in a
    subsequent multiplication step.
    DETECT: Value computed for M_13 is then referenced as M_14; or f(λ) =
    λ^2 − 5λ + 6 is treated as a constant 6 in the next multiplication step.

───────────────────────────────────────────────────────────────────────────
GENERATION_LOOP
  Autoregressive structural failure where the model gets stuck in a repetitive
  cycle during root-finding, rational root search, or polynomial expansion.
  Choose one subtype:

  Repetitive_Generation
    The model generates identical or near-identical algebraic working in a
    loop — either testing the same root candidates repeatedly without progress,
    or repeating the same line of computation 3+ times consecutively.
    DETECT: Model tests λ = 1, 2, 3, −1, −2, then restarts from λ = 1 with
    no new strategy; or the same polynomial substitution p(λ=2) appears 3+
    times consecutively with no new content between repetitions.

───────────────────────────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────────────────────────────────────
# BUILD — SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

BUILD_SYSTEM_TEMPLATE = Template("""You are a forensic mathematics auditor for an AI research paper
benchmarking LLMs on eigenvalue computation.

You will be given a 5×5 or 4×4 matrix and a model's FAILED attempt to compute its eigenvalues.

The model may have attempted any of the following four solution strategies:

  STRATEGY 1 — INVARIANT-BASED REASONING (Null Space / Trace Trick)
    The model bypasses polynomial expansion by exploiting invariants such as
    the matrix trace (sum of eigenvalues), determinant (product of eigenvalues),
    or null-space structure to infer eigenvalues directly.
    Failure mode: model learns the vocabulary of structural shortcuts but cannot
    execute the rigorous logical proof required to finalize the eigenvalues.

  STRATEGY 2 — DIRECT ALGEBRAIC / SYMBOLIC EXPANSION (Brute-Force Polynomial Expansion)
    The model explicitly expands the characteristic polynomial det(A − λI) = 0,
    computes all cofactors, collects terms, and solves the resulting degree-5 polynomial.
    Failure mode: compounding arithmetic degradation over long context windows.

  STRATEGY 3 — RATIONAL ROOT SEARCH & FORMAL HEURISTICS
    The model applies the Rational Root Theorem to discretize the candidate eigenvalue
    space, then tests integer or rational candidates against the characteristic polynomial
    or matrix constraints (e.g., trace, determinant, norm bounds).
    Failure mode: infinite loops when the search space contains no rational roots.

  STRATEGY 4 — BASELINE HEURISTIC GUESSING (Brute-Force Vector / Trace Guessing)
    The model abandons formal methods and guesses integer eigenvalues based on
    surface-level statistical priors (e.g., values that sum to the trace, or
    common small integers).
    Failure mode: pure hallucination with no theorem grounding.

${TRUNCATION_PRECHECK}

YOUR TASK HAS THREE STEPS:

STEP 1 — COMPUTE THE CORRECT ANSWER YOURSELF
  You are a 100% accurate eigenvalue calculator. Compute the eigenvalues of the
  given matrix independently by:
    - Forming the characteristic polynomial det(A − λI) = 0
    - Expanding the 5×5 or 4×4 determinant using cofactor expansion along the first row
    - Computing the five 4×4 minors M_11 through M_15, or four 3×3 minors for a 4×4 matrix
    - Collecting all degree-5 or degree-4 polynomial coefficients correctly
    - Solving for all five or four eigenvalues (real or complex)
  These are your ground truth values. Do not reveal them in your response.

STEP 2 — IDENTIFY THE STRATEGY THE MODEL USED
  Based on the model's response, identify which of the four strategies above it
  attempted. Record this as the "solution_strategy" field in your output.

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

NOTE: Precision loss, premature rounding, and irrational-to-integer forcing
(formerly NUMERICAL_DEGRADATION) are now classified as ARITHMETIC.
Add a note in first_error_description explaining the nature of the arithmetic
error (rounding, compounding drift, or irrational forced to integer).

TIE-BREAKER RULE: If an error seems to fit both a structural/logic tag
(ALGEBRAIC_PRECEDENCE, VARIABLE_ENTANGLEMENT) and a superficial tag
(SIGN_ERROR, ARITHMETIC), prioritize the structural/logic failure.
If a structural failure caused the arithmetic mistake, tag the structural failure.


RESPOND IN EXACTLY THIS JSON FORMAT AND NOTHING ELSE.

For SIGN_ERROR:
{
  "error_tag": "SIGN_ERROR",
  "sign_subtype": "Rule_Interference",
  "hallucination_subtype": "",
  "algebraic_precedence_subtype": "",
  "false_verification_subtype": "",
  "variable_entanglement_subtype": "",
  "generation_loop_subtype": "",
  "solution_strategy": "DIRECT_ALGEBRAIC",
  "first_error_step": "Step 16 — final polynomial assembly",
  "first_error_description": "Entry a_{12}=−2; model wrote − 2*C_12 instead of (−2)*C_12, double-applying the negative on that polynomial term. |term| correct, sign flipped.",
  "proposed_novel_tag": "",
  "maps_closest_to": "",
  "confidence": "HIGH"
}

For HALLUCINATION:
{
  "error_tag": "HALLUCINATION",
  "sign_subtype": "",
  "hallucination_subtype": "Complete_Collapse",
  "algebraic_precedence_subtype": "",
  "false_verification_subtype": "",
  "variable_entanglement_subtype": "",
  "generation_loop_subtype": "",
  "solution_strategy": "BASELINE_HEURISTIC",
  "first_error_step": "Step 8 — after completing only M_11",
  "first_error_description": "Model wrote due to complexity we will simplify and output eigenvalues directly without computing remaining minors.",
  "proposed_novel_tag": "",
  "maps_closest_to": "",
  "confidence": "HIGH"
}

For ALGEBRAIC_PRECEDENCE:
{
  "error_tag": "ALGEBRAIC_PRECEDENCE",
  "sign_subtype": "",
  "hallucination_subtype": "",
  "algebraic_precedence_subtype": "Bracket_Erosion",
  "false_verification_subtype": "",
  "variable_entanglement_subtype": "",
  "generation_loop_subtype": "",
  "solution_strategy": "DIRECT_ALGEBRAIC",
  "first_error_step": "Step 9 — cofactor M_13 expansion",
  "first_error_description": "Model expanded 3(λ − 4) as 3λ − 4 instead of 3λ − 12, silently dropping the distributed multiplication.",
  "proposed_novel_tag": "",
  "maps_closest_to": "",
  "confidence": "HIGH"
}

For all other tags:
{
  "error_tag": "ARITHMETIC",
  "sign_subtype": "",
  "hallucination_subtype": "",
  "algebraic_precedence_subtype": "",
  "false_verification_subtype": "",
  "variable_entanglement_subtype": "",
  "generation_loop_subtype": "",
  "solution_strategy": "DIRECT_ALGEBRAIC",
  "first_error_step": "Step 11 — 2×2 minor inside M_13",
  "first_error_description": "Model computed (−4)*(−1) − (2)*(4) = −12 but correct value is 4 − 8 = −4. Sign correct, magnitude wrong.",
  "proposed_novel_tag": "",
  "maps_closest_to": "",
  "confidence": "MEDIUM"
}

Confidence levels:
  HIGH   — you can point to a specific step and specific wrong value
  MEDIUM — fairly confident but response is ambiguous in places
  LOW    — response is very short or garbled, you are inferring

solution_strategy values:
  INVARIANT_BASED       — Null space / trace trick approach
  DIRECT_ALGEBRAIC      — Brute-force characteristic polynomial expansion
  RATIONAL_ROOT_SEARCH  — Rational Root Theorem with heuristic candidate testing
  BASELINE_HEURISTIC    — Trace/norm guessing without theorem grounding
  MIXED                 — Model switches strategies mid-response
  UNCLEAR               — Cannot determine strategy from response
""").substitute(
    TRUNCATION_PRECHECK=TRUNCATION_PRECHECK,
    MAGNITUDE_RULE=MAGNITUDE_RULE,
    TAG_DEFINITIONS=TAG_DEFINITIONS,
)

BUILD_USER_TEMPLATE = """\
Matrix A:
{matrix_latex}

Correct eigenvalues: {ground_truth}
Model's extracted eigenvalues: {extracted_answer}

Model's full response:
---
{response}
---

Perform the three-step audit described in the system prompt.
Identify the solution strategy used, find the FIRST wrong value, and classify it.
Respond with only the JSON object.
"""


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATE — SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

VALIDATE_SYSTEM_TEMPLATE = Template("""

You will be given:
  1. A square matrix A (typical size: 3×3 to 5×5)
  2. A model's failed response attempting to compute its eigen values
  3. A classification made by a junior judge
  4. A targeted verification question for that error type

${ADVERSARIAL_FRAMING}

${ANSWER_CHECK_STEP0}

YOUR PROCESS:
  Step 1: Compute the correct eigenvalues yourself using cofactor expansion
          along the first row of det(A − λI). Note all correct intermediate values.
          You are the ground truth oracle.
  Step 2: Read the model response step by step. Find the FIRST step where
          the model's value diverges from your correct computation.
  Step 3: Answer the verification question with specific evidence from
          the response. Confirm or deny the classification.

FIRST-ERROR PRINCIPLE:
  Classify by first point of failure, not final symptom.
  A sign error at step 8 cascading to wrong magnitude at step 20 is
  SIGN_ERROR not ARITHMETIC.

TIE-BREAKER: If error fits both structural (ALGEBRAIC_PRECEDENCE,
VARIABLE_ENTANGLEMENT) and superficial (SIGN_ERROR, ARITHMETIC),
prioritize structural failure — it is the root cause.

${MAGNITUDE_RULE}

STRICT TAXONOMY — primary_tag MUST be exactly one of these fourteen strings:
  GENERATION_TRUNCATION | FORMATTING_MISMATCH | SIGN_ERROR | ARITHMETIC |
  HALLUCINATION | INPUT_TRANSCRIPTION | CARRY_DOWN_ERROR | METHOD_FAIL |
  MEMORY_LOSS | OTHER_UNMAPPED |
  ALGEBRAIC_PRECEDENCE | FALSE_VERIFICATION | VARIABLE_ENTANGLEMENT | GENERATION_LOOP

  Do NOT use NUMERICAL_DEGRADATION — it is retired; classify as ARITHMETIC.
  Do NOT invent new primary tags beyond the fourteen above.

SIGN_ERROR subtypes (required when primary_tag = SIGN_ERROR):
  Product_Sign_Error   - product has correct magnitude, wrong sign in a multiply step
  Operation_Direction  - added where subtraction required in accumulation (or vice versa)
  Rule_Interference    - negative element treated as subtraction operator in assembly
  Parity_Sign_Error    - isolated (−1)^{i+j} parity error on one term (not a drift)
  Double_Negative_Trap - two negatives met in 2×2 sub-det, wrong sign on result
  Alternating_Drift    - +−+−+ pattern correct for ≥2 terms then breaks midway
  Cofactor_Neglect     - minor used raw without applying (−1)^{i+j} at all
  Silent_Sign_Flip     - wrong-sign output with zero computation shown for that step;
                         MAGNITUDE RULE applies: |wrong| must equal |correct|

HALLUCINATION subtypes (required when primary_tag = HALLUCINATION):
  Complete_Collapse     - explicit abandonment phrase, fewer than 3 minors computed
  Teleological_Zeroing  - forces polynomial intermediates to 0, long response
  Premature_Assertion   - partial minors computed then eigenvalues claimed as complete
  Silent_Omission       - skips computation blocks silently, no meta-statement
  Ungrounded_Guess      - essentially no working shown, just an eigenvalue list
  Spontaneous_Insertion - fabricated polynomial term inserted after correct chain

ALGEBRAIC_PRECEDENCE subtypes:
  Bracket_Erosion | PEMDAS_Violation | Exponent_Detachment

FALSE_VERIFICATION subtypes:
  Circular_Substitution | Tautological_Check

VARIABLE_ENTANGLEMENT subtypes:
  Variable_Substitution_Error | Variable_Reassignment

GENERATION_LOOP subtypes:
  Repetitive_Generation

RESPOND IN EXACTLY THIS JSON FORMAT AND NOTHING ELSE:
{
  "verified":                        true,
  "primary_tag":                     "INPUT_TRANSCRIPTION",
  "sign_subtype":                    "",
  "hallucination_subtype":           "",
  "algebraic_precedence_subtype":    "",
  "false_verification_subtype":      "",
  "variable_entanglement_subtype":   "",
  "generation_loop_subtype":         "",
  "proposed_novel_tag":              "",
  "maps_closest_to":                 "",
  "forensic_observation":            "Confirmed: model miscopied a(2,3)=−3 as +3 in submatrix for M_12 at step 6",
  "error_scope":                     "isolated",
  "affected_positions":              ["M_12 submatrix, row 2 col 3"],
  "confidence":                      "HIGH"
}

verified=true  - classification confirmed
verified=false - classification wrong, primary_tag is your corrected tag
verified=null  - insufficient evidence (truncated or genuinely ambiguous)

When verified=false, primary_tag must be your corrected tag, not the original.
When verified=false and primary_tag = SIGN_ERROR, populate sign_subtype.
When verified=false and primary_tag = HALLUCINATION, populate hallucination_subtype.
When verified=false and primary_tag is a bespoke tag, populate its subtype field.
forensic_observation always required — at least one sentence of specific evidence.
confidence: HIGH=specific step+value identified, MEDIUM=fairly certain, LOW=inferring.

error_scope:          "isolated"    = one position wrong
                      "systematic"  = same error pattern across ≥2 positions
                                      sharing the same tag and subtype
affected_positions:   list the specific cofactors, polynomial terms, or steps
                      where the error manifests (e.g. ["C_12 at step 6", "C_14 at step 9"])
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
    alg_sub: str,
    fv_sub: str,
    ve_sub: str,
    gl_sub: str,
    strategy: str,
    step: str,
    desc: str,
) -> str:
    """
    Build a targeted verification question for the senior judge based on the
    junior judge's classification. Covers all 14 error categories for eigen.
    """

    # ── GENERATION_TRUNCATION ─────────────────────────────────────────────
    if tag == "GENERATION_TRUNCATION":
        return """The junior judge classified this as GENERATION_TRUNCATION — the response
ends mid-expression or mid-computation with no final eigenvalue answer.

Verification question:
  (1) Confirm the response has no complete eigenvalue list or \\boxed{} in the final answer.
  (2) Quote the last ~50 characters of the response. Does it end mid-sentence,
      mid-expression (e.g. "= "), or mid-computation?
  (3) Does the response contain any complete eigenvalue claim anywhere?

CRITICAL: If a complete eigenvalue claim exists anywhere in the response, this is
NOT GENERATION_TRUNCATION. Reclassify based on whether that claim is correct
or fabricated."""

    # ── FORMATTING_MISMATCH ───────────────────────────────────────────────
    elif tag == "FORMATTING_MISMATCH":
        return """The junior judge classified this as FORMATTING_MISMATCH — the model's
eigenvalue computation is mathematically correct but the final format is wrong.

Verification question:
  (1) Verify every intermediate (polynomial expansion, roots) matches your
      correct computation.
  (2) Identify the specific formatting failure (missing repeated roots,
      wrong box format, etc.).
  (3) Confirm every eigenvalue value itself matches the ground truth.

CRITICAL: Do NOT confirm FORMATTING_MISMATCH if any eigenvalue or intermediate
value is wrong. If any value is wrong, this is a mathematical error —
set verified=false and reclassify."""

    # ── INPUT_TRANSCRIPTION ───────────────────────────────────────────────
    elif tag == "INPUT_TRANSCRIPTION":
        return """The junior judge classified this as INPUT_TRANSCRIPTION — the model
miscopied a matrix entry when constructing (A − λI) or a sub-minor.

Verification question:
Identify the EXACT cell that was miscopied. State:
  (1) Row and column in the original matrix A,
  (2) Correct value at that position,
  (3) Value the model wrote instead when forming (A − λI) or a submatrix.

SCOPE CHECK (mandatory):
Check ALL adjacent entries in the same row and column of the model's
written-out submatrix. If the model replaced an entire row or column with
fabricated data (not just one cell), reclassify to HALLUCINATION
(Silent_Omission or Spontaneous_Insertion).

CRITICAL: You must find both values directly in the response text.
If you cannot point to a specific cell with specific correct and wrong values,
set verified=false. Exact row, column, and both values required."""

    # ── CARRY_DOWN_ERROR ──────────────────────────────────────────────────
    elif tag == "CARRY_DOWN_ERROR":
        return """The junior judge classified this as CARRY_DOWN_ERROR — the model
correctly stated a polynomial coefficient or minor value at step N but
miscopied it at step N+1 when carrying it forward.

Verification question:
  (1) Quote the correctly-stated value at step N (with step label).
  (2) Quote the changed form at step N+1 (must be adjacent — no intervening
      unrelated steps).
  (3) Confirm the first occurrence matches your correct computation.
  (4) Confirm there is no new computation between N and N+1 that would
      legitimately change the value.

CRITICAL: If the first occurrence is itself wrong, this is ARITHMETIC or
SIGN_ERROR — not CARRY_DOWN_ERROR. Set verified=false and reclassify.
If separated by ≥1 unrelated intervening steps, reclassify as MEMORY_LOSS."""

    # ── ARITHMETIC ────────────────────────────────────────────────────────
    elif tag == "ARITHMETIC":
        return """The junior judge classified this as ARITHMETIC — correct method and signs,
but wrong magnitude in a numerical calculation during eigenvalue computation.

Verification question:
Find the specific computation where the wrong number first appeared.
  (1) Were ALL signs in that expression correct going in?
  (2) Apply the MAGNITUDE RULE: is |wrong value| ≠ |correct value|?
  (3) Was the error purely a multiplication or addition mistake with correct signs?
  (4) Was the computation inside a well-formed characteristic polynomial step?

CRITICAL: Apply MAGNITUDE RULE strictly. If |wrong| = |correct|, root error
is SIGN_ERROR — set verified=false and correct the tag. If a bracket was
silently dropped before the arithmetic, root error is ALGEBRAIC_PRECEDENCE.
If a structural failure caused the arithmetic mistake, tag the structural failure.
Also check: if precision loss or irrational-forcing caused the error, confirm
ARITHMETIC and describe the nature in forensic_observation."""

    # ── SIGN_ERROR subtypes ───────────────────────────────────────────────
    elif tag == "SIGN_ERROR" and sign_sub == "Product_Sign_Error":
        return f"""The junior judge classified this as SIGN_ERROR/Product_Sign_Error —
a product of two terms has correct magnitude but wrong sign in the
characteristic polynomial computation.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
Find the exact product where this occurred. State:
  (1) Which terms were multiplied (e.g., a_{{12}} · C_{{12}}),
  (2) Correct values and the sign the product should carry,
  (3) The sign the model actually used,
  (4) The resulting polynomial accumulation error.

CRITICAL: Apply MAGNITUDE RULE. |product| must be correct — only sign flipped.
If |wrong| ≠ |correct|, reclassify as ARITHMETIC.
If the model treated a negative element as a subtraction operator (not a product
sign flip), reclassify as Rule_Interference."""

    elif tag == "SIGN_ERROR" and sign_sub == "Operation_Direction":
        return f"""The junior judge classified this as SIGN_ERROR/Operation_Direction —
the model added where it should have subtracted (or vice versa) in a
determinant expansion or polynomial accumulation step.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
Find the exact accumulation step. State:
  (1) Which computation (e.g., 2×2 minor, polynomial term assembly),
  (2) The correct operation (e.g., a·d − b·c),
  (3) What the model wrote instead,
  (4) The resulting polynomial value vs the correct value.

CRITICAL: Confirm direction of accumulation was wrong — not just the sign
of an individual term. If a term's own sign was flipped, reclassify as
Product_Sign_Error."""

    elif tag == "SIGN_ERROR" and sign_sub == "Rule_Interference":
        return f"""The junior judge classified this as SIGN_ERROR/Rule_Interference —
the model treated a negative matrix entry's sign as a subtraction operator
at the polynomial assembly step.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Which first-row element of A is negative? State position and value.
  (2) Show the exact term as the model wrote it in the polynomial assembly.
  (3) Show how it should have been written: (−k) · C_{{1j}}.

CRITICAL: The element at that position MUST be negative in the original matrix A.
If the element is positive, this classification is impossible — set verified=false."""

    elif tag == "SIGN_ERROR" and sign_sub == "Parity_Sign_Error":
        return f"""The junior judge classified this as SIGN_ERROR/Parity_Sign_Error —
the model correctly identified a cofactor position but applied (−1)^{{i+j}}
with wrong sign on one isolated term (not a drift pattern).
Junior judge's first error step: {step or '(not specified)'}

Verification question:
  (1) Which cofactor has the isolated parity error? State position and the
      correct parity (should be + or −).
  (2) What sign did the model apply?
  (3) Confirm this is an ISOLATED parity error (not a drift pattern).
      How many preceding cofactors in this expansion were correct?

CRITICAL: Distinguish from Alternating_Drift: Parity_Sign_Error is isolated
(fewer than 2 correct preceding terms). Alternating_Drift requires ≥2 correct
terms before the pattern breaks.
If the model simply skipped parity entirely, reclassify as Cofactor_Neglect."""

    elif tag == "SIGN_ERROR" and sign_sub == "Double_Negative_Trap":
        return f"""The junior judge classified this as SIGN_ERROR/Double_Negative_Trap —
inside a 2×2 sub-determinant of the (A − λI) expansion, two negatives met
and the model failed to resolve them correctly.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
Find the exact 2×2 expression where two negatives met and were handled wrong.
  (1) Write out the full expression as the model wrote it.
  (2) Confirm BOTH values involved are genuinely negative in the original matrix A.
  (3) Apply MAGNITUDE RULE: confirm |result| is correct — only the sign wrong.

CRITICAL: If only one value is negative, this cannot be Double_Negative_Trap.
Set verified=false — likely Alternating_Drift or ARITHMETIC."""

    elif tag == "SIGN_ERROR" and sign_sub == "Alternating_Drift":
        return f"""The junior judge classified this as SIGN_ERROR/Alternating_Drift —
the model lost the cofactor checkerboard pattern midway through expanding det(A − λI).
Junior judge's first error step: {step or '(not specified)'}

Verification question:
First, count how many terms appear in the model's first-row cofactor expansion
(this equals the matrix dimension: 4 terms for a 4×4 matrix, 5 for a 5×5).

List the exact sign the model applied to EACH of those terms in order as found
in the response. State at which position the alternating pattern first broke.

REQUIRED: List ALL N signs where N is the number of terms in the expansion.
Do NOT require exactly five signs — a 4×4 expansion has four terms, not five.
If you cannot identify all N signs from the response, set verified=false.
If the pattern never broke, set verified=false.
Drift requires the pattern to be correct for at least two terms before breaking.

Example for 4×4: signs should be +, −, +, − for C_11 through C_14.
Example for 5×5: signs should be +, −, +, −, + for C_11 through C_15."""

    elif tag == "SIGN_ERROR" and sign_sub == "Cofactor_Neglect":
        return f"""The junior judge classified this as SIGN_ERROR/Cofactor_Neglect —
the model computed a minor polynomial correctly but used it directly without
applying (−1)^{{i+j}} at all.
Junior judge's first error step: {step or '(not specified)'}

Verification question:
Find the exact line where a minor polynomial was used as if it were already a cofactor.
State:
  (1) The minor polynomial value (e.g., M_12 = f(λ)),
  (2) The correct cofactor value (with parity sign applied),
  (3) What the model actually used in assembly.

Confirm the ONLY difference is the missing parity sign — polynomial expressions
must otherwise be identical in magnitude.

If magnitudes also differ, this is not Cofactor_Neglect.
If model wrote a sign but got it wrong, reclassify as Alternating_Drift or
Parity_Sign_Error."""

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
  Cofactor_Neglect     — minor used raw without parity factor applied
  Silent_Sign_Flip     — sign error stated with zero computation shown; |wrong|=|correct|

Populate sign_subtype in your response."""

    # ── HALLUCINATION subtypes ────────────────────────────────────────────
    elif tag == "HALLUCINATION" and halluc_sub == "Complete_Collapse":
        return """The junior judge classified this as HALLUCINATION/Complete_Collapse —
explicit abandonment of the eigenvalue computation with a meta-statement.

Verification question:
  (1) Quote the exact phrase where the model abandoned the computation.
  (2) How many of the minors of det(A − λI) did the model actually compute
      before stopping? Must be fewer than 3 for Complete_Collapse.

If the model computed 3 or more minors, consider Teleological_Zeroing,
Silent_Omission, or another tag."""

    elif tag == "HALLUCINATION" and halluc_sub == "Teleological_Zeroing":
        return """The junior judge classified this as HALLUCINATION/Teleological_Zeroing —
the model forced intermediate characteristic polynomial values to zero
without justification.

Verification question:
List THREE intermediate values the model set to zero that should not be zero
per your correct computation of det(A − λI). For each: label, model value (0),
correct value.

If you cannot name three such cases, the classification is likely wrong.
Reclassify to Complete_Collapse or another tag."""

    elif tag == "HALLUCINATION" and halluc_sub == "Premature_Assertion":
        return """The junior judge classified this as HALLUCINATION/Premature_Assertion —
the model began the characteristic polynomial expansion, computed some (but
not all) minors, then stated eigenvalues as if the computation were complete.

Verification question:
  (1) Which minors of det(A − λI) did the model actually expand?
  (2) Which minors are absent when the eigenvalue claim is made?
  (3) Quote the line where eigenvalues are asserted prematurely.

CRITICAL: At least one valid minor expansion must have been performed.
If no minors exist, use Ungrounded_Guess.
If explicit abandonment phrase, use Complete_Collapse."""

    elif tag == "HALLUCINATION" and halluc_sub == "Silent_Omission":
        return """The junior judge classified this as HALLUCINATION/Silent_Omission —
the model skipped characteristic polynomial computation blocks silently
without any meta-statement.

Verification question:
  (1) Which computation steps of the eigenvalue derivation are absent?
  (2) Is there any meta-statement or explanation for the skip?
  (3) Does the response end with a complete final answer, or is it truncated?

CRITICAL: If truncated, reclassify as GENERATION_TRUNCATION.
If there IS a meta-statement, use Complete_Collapse.
If essentially no working exists at all, use Ungrounded_Guess."""

    elif tag == "HALLUCINATION" and halluc_sub == "Ungrounded_Guess":
        return """The junior judge classified this as HALLUCINATION/Ungrounded_Guess —
essentially no working shown, model outputs eigenvalues with no derivation.

Verification question:
Confirm the response contains no meaningful computation — no characteristic
polynomial expansion, no minors, no cofactors, no 2×2 or 3×3 sub-determinants.
  (1) Are any intermediate expansions shown?
  (2) Are any sub-determinants computed?

CRITICAL: If any intermediate computation exists, use Silent_Omission instead."""

    elif tag == "HALLUCINATION" and halluc_sub == "Spontaneous_Insertion":
        return """The junior judge classified this as HALLUCINATION/Spontaneous_Insertion —
the model completed a correct cofactor expansion chain up to a point, then
inserted a fabricated polynomial coefficient or minor value.

Verification question:
  (1) Identify the last minor/cofactor correctly computed — state value.
  (2) Identify the inserted value that has no prior basis — state what the
      model wrote and where.
  (3) Confirm the inserted value cannot be derived from any sub-expansion
      shown in the response.

CRITICAL: If the inserted value can be traced to a preceding computation
(even a wrong one), this is ARITHMETIC or SIGN_ERROR, not Spontaneous_Insertion."""

    elif tag == "HALLUCINATION":
        return """The junior judge classified this as HALLUCINATION but no subtype given.

Determine which subtype applies:
  Complete_Collapse     — explicit abandonment phrase, < 3 minors of det(A−λI) computed
  Teleological_Zeroing  — full response but forces polynomial intermediates to 0
  Premature_Assertion   — partial minors computed then eigenvalues claimed as complete
  Silent_Omission       — skips computation blocks silently, no meta-statement
  Ungrounded_Guess      — essentially no working shown, just an eigenvalue list
  Spontaneous_Insertion — fabricated polynomial term inserted after correct chain

Set the correct hallucination_subtype in your response."""

    # ── MEMORY_LOSS ───────────────────────────────────────────────────────
    elif tag == "MEMORY_LOSS":
        return """The junior judge classified this as MEMORY_LOSS — the model computed
a characteristic polynomial value correctly early then recalled it incorrectly
later (recalled after ≥1 unrelated intervening steps).

Verification question:
  (1) What value was stated correctly at the early step? Give step and value.
  (2) What value did the model use for that same quantity later? Give step and value.
  (3) Was the FIRST occurrence genuinely correct per your computation?
  (4) Are the two occurrences separated by ≥1 unrelated intervening steps?

CRITICAL: If the first occurrence is also wrong, the error is ARITHMETIC or
SIGN_ERROR — set verified=false and correct the tag.
If the two occurrences are adjacent (N→N+1), reclassify as CARRY_DOWN_ERROR."""

    # ── METHOD_FAIL ───────────────────────────────────────────────────────
    elif tag == "METHOD_FAIL":
        return """The junior judge classified this as METHOD_FAIL — the model never set up
the characteristic polynomial det(A − λI) = 0 or cofactor expansion correctly.

Verification question:
Describe what method the model used in the first 3–4 steps.
Confirm it fundamentally differs from forming and expanding det(A − λI).

CRITICAL: Attempting to read eigenvalues from the diagonal (only valid for
diagonal/triangular matrices), applying trace/determinant shortcut without
polynomial expansion, or guessing eigenvalues from norms may be METHOD_FAIL.
Only classify as METHOD_FAIL if the approach is fundamentally wrong from
the outset — not if the model set up det(A − λI) correctly but made errors.
If the model did set up the characteristic polynomial but made errors inside,
classify by actual error type and set verified=false."""

    # ── ALGEBRAIC_PRECEDENCE subtypes ─────────────────────────────────────
    elif tag == "ALGEBRAIC_PRECEDENCE" and alg_sub == "Bracket_Erosion":
        return """The junior judge classified this as ALGEBRAIC_PRECEDENCE/Bracket_Erosion —
the model silently dropped parentheses in an intermediate characteristic
polynomial expansion step.

Verification question:
  (1) Quote the exact line where parentheses were dropped.
  (2) Show the mathematically correct form with parentheses.
  (3) Show the wrong form the model produced without them.

CRITICAL: The wrong value must be traceable entirely to the missing bracket —
not to an arithmetic error after correctly dropping no brackets.
If signs were also wrong, consider SIGN_ERROR as the root cause instead."""

    elif tag == "ALGEBRAIC_PRECEDENCE" and alg_sub == "PEMDAS_Violation":
        return """The junior judge classified this as ALGEBRAIC_PRECEDENCE/PEMDAS_Violation —
the model applied operations in the wrong order due to left-to-right reading
bias during characteristic polynomial assembly.

Verification question:
  (1) Quote the exact expression the model evaluated incorrectly.
  (2) Show the order the model applied operations.
  (3) Show the correct order and result.

CRITICAL: Confirm the error is purely an operator precedence mistake —
not a dropped bracket (Bracket_Erosion) or a sign flip (SIGN_ERROR)."""

    elif tag == "ALGEBRAIC_PRECEDENCE" and alg_sub == "Exponent_Detachment":
        return """The junior judge classified this as ALGEBRAIC_PRECEDENCE/Exponent_Detachment —
the model failed to distribute a power across a multiplied term in the
characteristic polynomial.

Verification question:
  (1) Quote the exact line where the exponent was misapplied.
  (2) Show what the model wrote vs. the correct expansion.
  (3) Confirm the error is the exponent distribution, not an arithmetic slip.

Example: (λ − 2)^2 written as λ^2 − 2 instead of λ^2 − 4λ + 4."""

    elif tag == "ALGEBRAIC_PRECEDENCE":
        return """The junior judge classified this as ALGEBRAIC_PRECEDENCE but no subtype given.

Determine which subtype applies:
  Bracket_Erosion     — silent parenthesis drop changing the expression
  PEMDAS_Violation    — wrong operator priority (left-to-right bias)
  Exponent_Detachment — power not distributed across multiplied term

Set the correct algebraic_precedence_subtype in your response."""

    # ── FALSE_VERIFICATION subtypes ───────────────────────────────────────
    elif tag == "FALSE_VERIFICATION" and fv_sub == "Circular_Substitution":
        return """The junior judge classified this as FALSE_VERIFICATION/Circular_Substitution —
the model used a derived (possibly flawed) characteristic polynomial to prove
its derived (possibly flawed) eigenvalue answer, OR substituted a candidate
eigenvalue, got a non-zero result, but falsely asserted verification passed.

Verification question:
  (1) Show the characteristic polynomial the model derived (possibly wrong).
  (2) Show how the model used substitution to validate its eigenvalues.
  (3) Confirm: does the substitution actually yield zero per your computation?
  (4) If non-zero was produced, quote the exact line where model claimed "= 0 ✓".

If the polynomial is correct AND substitution genuinely yields zero, set
verified=false — the error lies elsewhere."""

    elif tag == "FALSE_VERIFICATION" and fv_sub == "Tautological_Check":
        return """The junior judge classified this as FALSE_VERIFICATION/Tautological_Check —
the model added arbitrary fudge factors to force a verification step to equal
zero, or performed a trivially circular identity as a "check".

Verification question:
  (1) Quote the exact line where an adjustment or circular identity was introduced.
  (2) Confirm the true value of p(λ) at that eigenvalue per your computation.
  (3) Was the adjustment mathematically unjustified?

CRITICAL: Legitimate rounding notes (e.g., "≈ 0 due to floating point") must
be distinguished from manufactured zeros. If the true p(λ) is genuinely very
small, set verified=false and reclassify."""

    elif tag == "FALSE_VERIFICATION":
        return """The junior judge classified this as FALSE_VERIFICATION but no subtype given.

Determine which subtype applies:
  Circular_Substitution — used flawed polynomial to prove flawed eigenvalue, OR
                          substituted λ, got non-zero, but claimed "= 0 ✓"
  Tautological_Check    — added fudge factor to manufacture zero, or circular identity

Set the correct false_verification_subtype in your response."""

    # ── VARIABLE_ENTANGLEMENT subtypes ────────────────────────────────────
    elif tag == "VARIABLE_ENTANGLEMENT" and ve_sub == "Variable_Substitution_Error":
        return """The junior judge classified this as VARIABLE_ENTANGLEMENT/Variable_Substitution_Error —
the model substituted the wrong variable or eigenvalue index into a polynomial
expression, or mislabeled eigenvalue subscripts across steps.

Verification question:
  (1) Identify the exact step where the wrong variable or subscript was used.
  (2) State the correct variable/subscript and the wrong one the model used.
  (3) Confirm the substitution changed the numerical value of a computation.

If the subscripts are only cosmetically mislabeled but the correct values were
used in computation, set verified=false — this may be a notation issue only."""

    elif tag == "VARIABLE_ENTANGLEMENT" and ve_sub == "Variable_Reassignment":
        return """The junior judge classified this as VARIABLE_ENTANGLEMENT/Variable_Reassignment —
the model reused an intermediate value from one sub-minor context in a different
context where it does not apply, or treated a polynomial f(λ) as a scalar constant.

Verification question:
  (1) Which value was computed in which context (state sub-minor label)?
  (2) Where was it incorrectly reused or treated as scalar?
  (3) What is the correct value for that second context?

CRITICAL: If the model legitimately evaluated f(λ) at a specific value, this
is not Variable_Reassignment. Set verified=false and reclassify."""

    elif tag == "VARIABLE_ENTANGLEMENT":
        return """The junior judge classified this as VARIABLE_ENTANGLEMENT but no subtype given.

Determine which subtype applies:
  Variable_Substitution_Error — wrong variable/subscript used, or mislabeled eigenvalues
  Variable_Reassignment       — value reused in wrong minor context, or f(λ) treated as scalar

Set the correct variable_entanglement_subtype in your response."""

    # ── GENERATION_LOOP subtypes ──────────────────────────────────────────
    elif tag == "GENERATION_LOOP" and gl_sub == "Repetitive_Generation":
        return """The junior judge classified this as GENERATION_LOOP/Repetitive_Generation —
the model got stuck in a repetitive cycle, either testing the same root
candidates in a loop or repeating identical algebraic lines consecutively.

Verification question:
  (1) Quote the repeated line, substitution, or candidate set.
  (2) Confirm either:
      (a) Model restarts root candidate testing (λ = 1, 2, ... → restarts from 1), OR
      (b) The same algebraic line appears 3+ times consecutively.
  (3) For (a): Confirm all candidates failed the polynomial check before restart.
  (4) For (b): Confirm there is no new mathematical content between repetitions.

CRITICAL: If the repetitions contain even minor variations (different λ values,
updated coefficients), this is not Repetitive_Generation. Set verified=false
and reclassify to the actual error type (ARITHMETIC, SIGN_ERROR, etc.)."""

    elif tag == "GENERATION_LOOP":
        return """The junior judge classified this as GENERATION_LOOP but no subtype given.

Determine which subtype applies:
  Repetitive_Generation — candidate loop restarts OR identical lines repeated 3+× consecutively

Set the correct generation_loop_subtype in your response."""

    # ── FALLBACK ──────────────────────────────────────────────────────────
    else:
        return """The junior judge could not classify this error (UNKNOWN or OTHER_UNMAPPED).

Fresh classification — ignore the previous tag entirely.

Read the model's eigenvalue computation from the beginning.
Find the first step where the model's value diverges from correct det(A − λI).

Classify using exactly one of the fourteen valid tags:
  GENERATION_TRUNCATION | FORMATTING_MISMATCH | SIGN_ERROR | ARITHMETIC |
  HALLUCINATION | INPUT_TRANSCRIPTION | CARRY_DOWN_ERROR | METHOD_FAIL |
  MEMORY_LOSS | OTHER_UNMAPPED |
  ALGEBRAIC_PRECEDENCE | FALSE_VERIFICATION | VARIABLE_ENTANGLEMENT | GENERATION_LOOP

TIE-BREAKER: If error fits both structural (ALGEBRAIC_PRECEDENCE,
VARIABLE_ENTANGLEMENT) and superficial (SIGN_ERROR, ARITHMETIC), prioritize
the structural failure.

Do NOT use NUMERICAL_DEGRADATION — classify as ARITHMETIC.

For SIGN_ERROR, first apply MAGNITUDE RULE (|wrong|=|correct| required),
then populate sign_subtype:
  Product_Sign_Error | Operation_Direction | Rule_Interference |
  Parity_Sign_Error  | Double_Negative_Trap | Alternating_Drift |
  Cofactor_Neglect   | Silent_Sign_Flip

For HALLUCINATION, populate hallucination_subtype:
  Complete_Collapse | Teleological_Zeroing | Premature_Assertion |
  Silent_Omission   | Ungrounded_Guess     | Spontaneous_Insertion

For bespoke tags, populate the appropriate subtype field:
  ALGEBRAIC_PRECEDENCE: Bracket_Erosion | PEMDAS_Violation | Exponent_Detachment
  FALSE_VERIFICATION: Circular_Substitution | Tautological_Check
  VARIABLE_ENTANGLEMENT: Variable_Substitution_Error | Variable_Reassignment
  GENERATION_LOOP: Repetitive_Generation

Use OTHER_UNMAPPED only if genuinely impossible to map after careful analysis.
Populate proposed_novel_tag and maps_closest_to."""


def build_validate_user_prompt(row: dict, response: str, matrix_latex: str) -> str:
    """
    Build the full user message for the senior validation judge.
    Reads all subtype fields for eigen's extended taxonomy.
    """
    tag     = str(row.get("Error_Tag", "")                       or "UNKNOWN").strip()
    sign_sub = str(row.get("Sign_Subtype", "")                   or "").strip()
    halluc_sub = str(row.get("Hallucination_Subtype", "")        or "").strip()
    alg_sub  = str(row.get("Algebraic_Precedence_Subtype", "")   or "").strip()
    fv_sub   = str(row.get("False_Verification_Subtype", "")     or "").strip()
    ve_sub   = str(row.get("Variable_Entanglement_Subtype", "")  or "").strip()
    gl_sub   = str(row.get("Generation_Loop_Subtype", "")        or "").strip()
    strategy = str(row.get("Solution_Strategy", "")              or "").strip()
    step     = str(row.get("First_Error_Step", "")               or "").strip()
    desc     = str(row.get("First_Error_Description", "")        or "").strip()

    active_subtype = (
        sign_sub or halluc_sub or alg_sub or fv_sub or ve_sub or gl_sub or "(none)"
    )

    vq = _build_verification_question(
        tag, sign_sub, halluc_sub, alg_sub, fv_sub, ve_sub, gl_sub, strategy, step, desc
    )

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
#================================================================================
# END OF FILE
#================================================================================