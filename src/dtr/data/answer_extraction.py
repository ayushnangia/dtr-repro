"""Extract and validate answers from model outputs.

Handles both math benchmarks (\\boxed{} extraction) and GPQA (letter choice).
"""

from __future__ import annotations

import re

MATH_BENCHMARKS = {"aime_2024", "aime_2025", "hmmt_2025"}
GPQA_BENCHMARKS = {"gpqa_diamond"}


# ---------------------------------------------------------------------------
# Boxed answer extraction
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> str | None:
    r"""Extract the answer from the *last* ``\boxed{...}`` in *text*.

    Handles nested braces correctly, e.g. ``\boxed{2^{10}}`` returns
    ``"2^{10}"``.  When multiple ``\boxed{}`` groups exist, the last one
    is returned (matching typical chain-of-thought where the final boxed
    expression is the answer).

    Returns ``None`` if no ``\boxed{}`` is found.
    """
    # Find all \boxed{ positions
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    # Take the last match and extract content with balanced braces
    last_match = matches[-1]
    start = last_match.end()  # position right after \boxed{
    return _extract_balanced_braces(text, start)


def _extract_balanced_braces(text: str, start: int) -> str | None:
    """Extract content between balanced braces starting at *start*.

    *start* should point to the first character after the opening ``{``.
    """
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1

    if depth != 0:
        # Unbalanced braces -- return what we have
        return text[start:pos].strip() or None

    # pos is now one past the closing }, content is [start, pos-1)
    content = text[start : pos - 1]
    return content if content != "" else ""


# ---------------------------------------------------------------------------
# Choice extraction (GPQA)
# ---------------------------------------------------------------------------

_CHOICE_PATTERN = re.compile(
    r"""
    (?:
        \\boxed\{([A-Da-d])\}        # \boxed{A}
        | \(([A-Da-d])\)\s*[\.\,]?$  # (A) at end of line
        | \b([A-Da-d])\)             # A)
        | (?:answer\s+is\s*)[:\s]*\(?([A-Da-d])\)?  # answer is (A) / answer is A
    )
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)


def extract_choice_answer(text: str) -> str | None:
    """Extract a letter choice (A/B/C/D) from model output.

    Tries several patterns in priority order:
    1. ``\\boxed{A}`` -- the standard format we ask for
    2. ``(A)`` at end of a line
    3. ``A)``
    4. ``answer is (A)`` / ``answer is A``

    Returns the uppercase letter, or ``None`` if no choice is found.
    """
    # Priority 1: try \boxed{} with a single letter
    boxed = extract_boxed_answer(text)
    if boxed and re.fullmatch(r"[A-Da-d]", boxed.strip()):
        return boxed.strip().upper()

    # Priority 2: scan with regex patterns (take the last match)
    matches = list(_CHOICE_PATTERN.finditer(text))
    if matches:
        last = matches[-1]
        # One of the groups will be non-None
        for group in last.groups():
            if group is not None:
                return group.upper()

    return None


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

def extract_answer(text: str, benchmark: str) -> str | None:
    """Extract an answer from model output, dispatching by benchmark type.

    Parameters
    ----------
    text:
        Raw model output string.
    benchmark:
        Benchmark name (e.g. ``"aime_2024"``, ``"gpqa_diamond"``).

    Returns
    -------
    str | None
        Extracted answer string, or ``None`` if extraction failed.
    """
    if benchmark in GPQA_BENCHMARKS:
        return extract_choice_answer(text)
    elif benchmark in MATH_BENCHMARKS:
        return extract_boxed_answer(text)
    else:
        # Default: try boxed first, then choice
        result = extract_boxed_answer(text)
        if result is not None:
            return result
        return extract_choice_answer(text)


# ---------------------------------------------------------------------------
# Answer normalization & correctness
# ---------------------------------------------------------------------------

def _normalize_math_answer(answer: str) -> str:
    """Normalize a math answer for comparison.

    Strips whitespace, removes trailing periods, and attempts simple
    numeric normalization (e.g. ``"042"`` -> ``"42"``).
    """
    s = answer.strip().rstrip(".")

    # Remove surrounding dollar signs or LaTeX wrappers
    s = re.sub(r"^\$+|\$+$", "", s).strip()

    # Try to convert to a number for canonical form
    try:
        # Handle fractions like \frac{3}{4}
        frac_match = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
        if frac_match:
            num = int(frac_match.group(1))
            den = int(frac_match.group(2))
            # Return as simplified fraction string
            from math import gcd

            g = gcd(abs(num), abs(den))
            num, den = num // g, den // g
            if den == 1:
                return str(num)
            return f"{num}/{den}"

        # Plain integer or float
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        pass

    return s


def check_correct(
    predicted: str | None,
    gold: str,
    benchmark: str,
) -> bool:
    """Check whether *predicted* matches the *gold* answer.

    Parameters
    ----------
    predicted:
        Extracted model answer (may be ``None`` if extraction failed).
    gold:
        Ground-truth answer string.
    benchmark:
        Benchmark name, used to select the comparison strategy.

    Returns
    -------
    bool
        ``True`` if *predicted* matches *gold* under the appropriate
        normalization for the benchmark.
    """
    if predicted is None:
        return False

    if benchmark in GPQA_BENCHMARKS:
        return predicted.strip().upper() == gold.strip().upper()

    # Math benchmarks: normalize both sides
    return _normalize_math_answer(predicted) == _normalize_math_answer(gold)
