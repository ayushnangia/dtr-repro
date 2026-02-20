"""Comprehensive tests for dtr.data.answer_extraction."""

from __future__ import annotations

import pytest

from dtr.data.answer_extraction import (
    check_correct,
    extract_answer,
    extract_boxed_answer,
    extract_choice_answer,
)


# ===================================================================
# extract_boxed_answer
# ===================================================================


class TestExtractBoxedAnswer:
    """Tests for \\boxed{} extraction."""

    def test_simple_integer(self):
        assert extract_boxed_answer(r"The answer is \boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_boxed_answer(r"Thus \boxed{2^{10}}") == "2^{10}"

    def test_deeply_nested_braces(self):
        assert extract_boxed_answer(r"\boxed{\frac{a^{2}}{b}}") == r"\frac{a^{2}}{b}"

    def test_multiple_boxed_takes_last(self):
        text = r"First attempt \boxed{wrong}, correction: \boxed{42}"
        assert extract_boxed_answer(text) == "42"

    def test_no_boxed_returns_none(self):
        assert extract_boxed_answer("the answer is 42") is None

    def test_empty_boxed(self):
        assert extract_boxed_answer(r"\boxed{}") == ""

    def test_fraction(self):
        assert extract_boxed_answer(r"\boxed{\frac{3}{4}}") == r"\frac{3}{4}"

    def test_negative_number(self):
        assert extract_boxed_answer(r"\boxed{-7}") == "-7"

    def test_boxed_with_text_around(self):
        text = "We get $x = 5$, so the answer is $\\boxed{5}$."
        assert extract_boxed_answer(text) == "5"

    def test_boxed_with_spaces(self):
        assert extract_boxed_answer(r"\boxed{ 42 }") == " 42 "

    def test_empty_string(self):
        assert extract_boxed_answer("") is None

    def test_boxed_with_expression(self):
        assert extract_boxed_answer(r"\boxed{x^2 + 3x + 1}") == "x^2 + 3x + 1"

    def test_multiline_text(self):
        text = "Step 1: compute.\nStep 2: simplify.\nTherefore \\boxed{100}."
        assert extract_boxed_answer(text) == "100"

    def test_multiple_nested(self):
        text = r"\boxed{\sqrt{\frac{1}{2}}}"
        assert extract_boxed_answer(text) == r"\sqrt{\frac{1}{2}}"


# ===================================================================
# extract_choice_answer
# ===================================================================


class TestExtractChoiceAnswer:
    """Tests for letter-choice extraction (GPQA)."""

    def test_boxed_letter(self):
        assert extract_choice_answer(r"The final answer is \boxed{A}.") == "A"

    def test_boxed_lowercase_letter(self):
        assert extract_choice_answer(r"\boxed{c}") == "C"

    def test_parenthesized_at_end(self):
        assert extract_choice_answer("I believe the answer is (B)") == "B"

    def test_answer_is_pattern(self):
        assert extract_choice_answer("the answer is A") == "A"

    def test_answer_is_parenthesized(self):
        assert extract_choice_answer("the answer is (D)") == "D"

    def test_no_choice_returns_none(self):
        assert extract_choice_answer("I'm not sure about this one.") is None

    def test_empty_string(self):
        assert extract_choice_answer("") is None

    def test_multiple_choices_takes_last(self):
        text = r"First I thought \boxed{A}, but actually \boxed{B}."
        assert extract_choice_answer(text) == "B"

    def test_letter_d(self):
        assert extract_choice_answer(r"\boxed{D}") == "D"


# ===================================================================
# extract_answer (unified dispatch)
# ===================================================================


class TestExtractAnswer:
    """Tests for benchmark-dispatched extraction."""

    def test_math_benchmark(self):
        assert extract_answer(r"\boxed{42}", "aime_2024") == "42"

    def test_math_benchmark_aime_2025(self):
        assert extract_answer(r"\boxed{7}", "aime_2025") == "7"

    def test_math_benchmark_hmmt(self):
        assert extract_answer(r"\boxed{100}", "hmmt_2025") == "100"

    def test_gpqa_benchmark(self):
        assert extract_answer(r"\boxed{C}", "gpqa_diamond") == "C"

    def test_gpqa_no_answer(self):
        assert extract_answer("I don't know", "gpqa_diamond") is None

    def test_math_no_answer(self):
        assert extract_answer("I don't know", "aime_2024") is None

    def test_unknown_benchmark_tries_boxed(self):
        """Unknown benchmarks should fall back to boxed then choice."""
        assert extract_answer(r"\boxed{42}", "unknown_bench") == "42"


# ===================================================================
# check_correct
# ===================================================================


class TestCheckCorrect:
    """Tests for answer correctness checking."""

    # -- Math benchmarks --

    def test_exact_match(self):
        assert check_correct("42", "42", "aime_2024") is True

    def test_leading_zeros(self):
        assert check_correct("042", "42", "aime_2024") is True

    def test_whitespace_tolerance(self):
        assert check_correct(" 42 ", "42", "aime_2024") is True

    def test_wrong_answer(self):
        assert check_correct("43", "42", "aime_2024") is False

    def test_none_predicted(self):
        assert check_correct(None, "42", "aime_2024") is False

    def test_fraction_normalization(self):
        assert check_correct(r"\frac{3}{4}", r"\frac{3}{4}", "aime_2024") is True

    def test_fraction_simplification(self):
        assert check_correct(r"\frac{6}{8}", r"\frac{3}{4}", "aime_2024") is True

    def test_fraction_to_integer(self):
        assert check_correct(r"\frac{4}{2}", "2", "aime_2024") is True

    def test_negative_number(self):
        assert check_correct("-7", "-7", "hmmt_2025") is True

    def test_float_vs_int(self):
        assert check_correct("3.0", "3", "aime_2025") is True

    # -- GPQA --

    def test_gpqa_correct(self):
        assert check_correct("A", "A", "gpqa_diamond") is True

    def test_gpqa_case_insensitive(self):
        assert check_correct("a", "A", "gpqa_diamond") is True

    def test_gpqa_wrong(self):
        assert check_correct("B", "A", "gpqa_diamond") is False

    def test_gpqa_none(self):
        assert check_correct(None, "A", "gpqa_diamond") is False

    def test_gpqa_whitespace(self):
        assert check_correct(" C ", "C", "gpqa_diamond") is True

    # -- Edge cases --

    def test_trailing_period(self):
        assert check_correct("42.", "42", "aime_2024") is True

    def test_dollar_signs_stripped(self):
        assert check_correct("$42$", "42", "aime_2024") is True

    def test_empty_predicted(self):
        assert check_correct("", "42", "aime_2024") is False

    def test_symbolic_answer(self):
        """Non-numeric answers fall back to string comparison."""
        assert check_correct("x + 1", "x + 1", "hmmt_2025") is True

    def test_symbolic_mismatch(self):
        assert check_correct("x + 1", "x + 2", "hmmt_2025") is False
