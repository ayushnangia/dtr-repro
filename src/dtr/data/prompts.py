"""Prompt templates for DTR benchmark evaluation.

Based on Tables 4 & 5 of the DTR paper (arXiv 2602.13517).

Math benchmarks (AIME, HMMT) use a simple step-by-step prompt with \\boxed{}.
GPQA uses a multiple-choice prompt with explicit choice listing.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Math prompt (AIME / HMMT) -- Table 4
# ---------------------------------------------------------------------------
MATH_SYSTEM_PROMPT = ""

MATH_USER_TEMPLATE = (
    "{question}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)

# ---------------------------------------------------------------------------
# GPQA prompt -- Table 5
# ---------------------------------------------------------------------------
GPQA_SYSTEM_PROMPT = ""

GPQA_USER_TEMPLATE = (
    "Answer the following multiple choice question. Think step by step before "
    "answering.\n\n"
    "{question}\n\n"
    "Choose from the following:\n"
    "(A) {choice_a}\n"
    "(B) {choice_b}\n"
    "(C) {choice_c}\n"
    "(D) {choice_d}\n\n"
    "The final answer is \\boxed{{X}}."
)

MATH_BENCHMARKS = {"aime_2024", "aime_2025", "hmmt_2025"}
GPQA_BENCHMARKS = {"gpqa_diamond"}


def format_prompt(
    question: str,
    benchmark: str,
    choices: list[str] | None = None,
) -> list[dict]:
    """Build a chat-messages list for the given benchmark question.

    Parameters
    ----------
    question:
        The raw question text.  For GPQA this is the question without choices
        (the function inserts them via *choices*).  If *choices* is ``None``
        and the benchmark is GPQA, the question is used as-is (it may already
        contain embedded choices from the download script).
    benchmark:
        Benchmark name (e.g. ``"aime_2024"``, ``"gpqa_diamond"``).
    choices:
        For GPQA only -- list of four answer-choice strings ``[A, B, C, D]``.
        Ignored for math benchmarks.

    Returns
    -------
    list[dict]
        OpenAI-style chat messages, e.g.
        ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``.
        The system message is omitted when the system prompt is empty.
    """
    if benchmark in MATH_BENCHMARKS:
        return _format_math(question)
    elif benchmark in GPQA_BENCHMARKS:
        return _format_gpqa(question, choices)
    else:
        raise ValueError(
            f"Unknown benchmark {benchmark!r}. "
            f"Expected one of {sorted(MATH_BENCHMARKS | GPQA_BENCHMARKS)}"
        )


def _format_math(question: str) -> list[dict]:
    """Format a math benchmark question into chat messages."""
    messages: list[dict] = []
    if MATH_SYSTEM_PROMPT:
        messages.append({"role": "system", "content": MATH_SYSTEM_PROMPT})
    messages.append({
        "role": "user",
        "content": MATH_USER_TEMPLATE.format(question=question),
    })
    return messages


def _format_gpqa(question: str, choices: list[str] | None) -> list[dict]:
    """Format a GPQA question into chat messages."""
    messages: list[dict] = []
    if GPQA_SYSTEM_PROMPT:
        messages.append({"role": "system", "content": GPQA_SYSTEM_PROMPT})

    if choices and len(choices) == 4:
        content = GPQA_USER_TEMPLATE.format(
            question=question,
            choice_a=choices[0],
            choice_b=choices[1],
            choice_c=choices[2],
            choice_d=choices[3],
        )
    else:
        # Choices are already embedded in the question text (from download script)
        content = (
            f"Answer the following multiple choice question. Think step by step "
            f"before answering.\n\n"
            f"{question}\n\n"
            f"The final answer is \\boxed{{X}}."
        )

    messages.append({"role": "user", "content": content})
    return messages
