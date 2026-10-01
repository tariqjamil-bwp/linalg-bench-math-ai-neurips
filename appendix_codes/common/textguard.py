#!/usr/bin/env python3
"""
textguard.py — refuse to send or store a prompt carrying a control character.

SHARED BY BOTH EXPERIMENTS (symlinked into each pipeline's code folder). A
change here lands in two live submissions.

WHAT THIS PREVENTS
──────────────────
In a NON-raw Python string, "\\boxed{}" is not the six characters a LaTeX
reader expects — \\b is an escape, and Python turns it into a single backspace
(chr 8), leaving "\x08oxed{}". The text still looks almost right in a terminal,
because a backspace renders as nothing or as a narrow gap.

This failure mode is silent: a corrupted \\boxed{} renders as nearly the same
text in a terminal, so affected rows look usable and only surface as damage
once something searches for the literal marker or the bytes are diffed.

The same class covers \\a (bell, 7), \\f (form feed, 12), \\v (vertical tab,
11) and \\r — every one of them a plausible first letter after a backslash in
LaTeX or English.

WHY A GUARD RATHER THAN CARE
────────────────────────────
Prompt text reaches the models from three directions — CSV data, Python
literals, and f-string assembly — and only the literals are at risk. That is
exactly the kind of partial hazard that survives review: the safe paths make
the unsafe one look safe by association. A choke-point assertion costs nothing
per call and turns a silent corruption into a build that stops.

Newline and tab are allowed: both are legitimate prompt layout.
"""

from __future__ import annotations

# Everything below 0x20 except \n (10) and \t (9), plus DEL (127).
_FORBIDDEN = {c: f"\\x{c:02x}" for c in list(range(0, 9)) + [11, 12] +
              list(range(14, 32)) + [127]}
_FORBIDDEN[13] = "\\r"      # named separately: \r is the one that looks benign

_NAMES = {7: "\\a bell", 8: "\\b BACKSPACE", 11: "\\v vertical tab",
          12: "\\f form feed", 13: "\\r carriage return", 0: "NUL"}


def find_control_chars(text: str) -> list[tuple[int, int, str]]:
    """(index, codepoint, label) for every forbidden character, in order."""
    return [(i, ord(ch), _NAMES.get(ord(ch), _FORBIDDEN[ord(ch)]))
            for i, ch in enumerate(text) if ord(ch) in _FORBIDDEN]


def assert_clean(text: str, where: str) -> str:
    """Return text unchanged, or raise with enough context to find the source.

    Called at the points where a prompt is finalised, so a corrupted string
    cannot reach a provider or a results file no matter which layer produced it.
    """
    if not isinstance(text, str):
        return text
    hits = find_control_chars(text)
    if not hits:
        return text

    lines = []
    for i, cp, name in hits[:5]:
        lo, hi = max(0, i - 40), min(len(text), i + 40)
        window = repr(text[lo:i]) + "  >>" + name + "<<  " + repr(text[i + 1:hi])
        lines.append(f"    at index {i}: {name} (chr {cp})\n      {window}")

    raise SystemExit(
        f"BLOCKED: control character in {where}.\n"
        + "\n".join(lines)
        + (f"\n    ... and {len(hits) - 5} more" if len(hits) > 5 else "")
        + "\n\n  Almost always an unescaped backslash escape in a Python string:\n"
          '      "\\boxed{}"   ->  backspace + "oxed{}"      WRONG\n'
          '      "\\\\boxed{}"  ->  \\boxed{}                  correct\n'
          '      r"\\boxed{}"  ->  \\boxed{}                  correct\n'
          "\n  Fix the literal. Do not strip the character: a prompt that needed\n"
          "  stripping was already not the prompt the experiment intended to send."
    )
