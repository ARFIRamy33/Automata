"""Tests of function 57: Arden's lemma (X = AX ∪ B and X = XA ∪ B)."""

import itertools
from dataclasses import FrozenInstanceError

import pytest

from automata_extensions.regex import (
    EMPTY,
    EPSILON,
    ArdenError,
    Regex,
    Symbol,
    arden_star,
    equation_side,
    solve_arden,
)
from test_regex import oracle, words_up_to

LONGEST = 6


def parse(text):
    return Regex.parse(text)


def language(regex, symbols="ab", longest=LONGEST):
    """Words of length <= longest in L(regex), through Python's re (independent oracle)."""
    return {w for w in words_up_to(symbols, longest) if oracle(str(regex), w)}


def same_language(first, second, symbols="ab"):
    """Exact equivalence: both regexes -> complete DFA -> function 25 of the project."""
    one = first.to_nfa(alphabet=symbols).determinize(complete=True)
    two = second.to_nfa(alphabet=symbols).determinize(complete=True)
    return one.has_same_language_as(two)


def fixed_point(a, b, form, start, symbols="ab", longest=LONGEST):
    """Iterate Y -> A·Y ∪ B (or Y·A ∪ B) on words of bounded length.

    Independent of the solver: it only uses membership tests of A and B.
    """
    in_a = {w for w in words_up_to(symbols, longest) if oracle(str(a), w)}
    in_b = {w for w in words_up_to(symbols, longest) if oracle(str(b), w)}
    current = set(start)
    for _ in range(longest + 2):
        nxt = set(in_b)
        for u in in_a:
            for v in list(current):
                w = u + v if form == "AX" else v + u
                if len(w) <= longest:
                    nxt.add(w)
        current = nxt
    return current


A_SAMPLES = ["a", "ab", "a|b", "ba|a", "(ab)(b|a)", "∅", "a(a|b)*", "b*a"]
B_SAMPLES = ["b", "ε", "∅", "ab", "a*", "(a|b)b", "b|ε"]


# ------------------------------------------------------------------- simple equations
def test_57_simple_equation_left_form():
    x = solve_arden(parse("a"), parse("b"))  # X = aX ∪ b
    assert str(x) == "a*b"
    assert language(x) == {"a" * n + "b" for n in range(LONGEST)}


def test_57_simple_equation_right_form():
    x = solve_arden(parse("a"), parse("b"), form="XA")  # X = Xa ∪ b
    assert str(x) == "ba*"
    assert language(x) == {"b" + "a" * n for n in range(LONGEST)}


def test_57_left_and_right_concatenation_are_not_mixed():
    a, b = parse("a"), parse("b")
    left, right = solve_arden(a, b, "AX"), solve_arden(a, b, "XA")
    assert left != right and not same_language(left, right)
    assert oracle(str(left), "aab") and not oracle(str(left), "baa")
    assert oracle(str(right), "baa") and not oracle(str(right), "aab")
    # each solution satisfies its own equation and not the other one
    assert same_language(left, equation_side(a, b, left, "AX"))
    assert same_language(right, equation_side(a, b, right, "XA"))
    assert not same_language(left, equation_side(a, b, left, "XA"))
    assert not same_language(right, equation_side(a, b, right, "AX"))


def test_57_default_form_is_AX():
    assert solve_arden(parse("ab"), parse("c")) == solve_arden(parse("ab"), parse("c"), "AX")


# -------------------------------------------------------------- empty and epsilon cases
def test_57_empty_expression_cases():
    b = parse("ab")
    assert solve_arden(EMPTY, b) == b  # X = ∅X ∪ B = B
    assert solve_arden(EMPTY, b, "XA") == b
    assert solve_arden(parse("a"), EMPTY) == EMPTY  # X = aX ∪ ∅ = ∅
    assert solve_arden(parse("a"), EMPTY, "XA") == EMPTY
    assert solve_arden(EMPTY, EMPTY) == EMPTY


def test_57_epsilon_as_constant_term():
    assert str(solve_arden(parse("a"), EPSILON)) == "a*"
    assert str(solve_arden(parse("ab"), EPSILON, "XA")) == "(ab)*"
    assert solve_arden(EMPTY, EPSILON) == EPSILON


def test_57_epsilon_in_the_coefficient_is_refused():
    for text in ["ε", "a*", "a|ε", "(a|b)*", "(a|ε)(b|ε)"]:
        for form in ("AX", "XA"):
            with pytest.raises(ArdenError, match="empty word belongs to"):
                solve_arden(parse(text), parse("b"), form)


def test_57_refusal_is_justified_by_non_uniqueness():
    a, b = parse("a*"), parse("b")  # ε ∈ L(A)
    smallest = parse("a*b")
    other = parse("(a|b)*")
    for candidate in (smallest, other):
        assert same_language(candidate, equation_side(a, b, candidate, "AX"))
    assert not same_language(smallest, other)  # two different solutions


def test_57_error_class_is_a_value_error_with_explicit_messages():
    assert issubclass(ArdenError, ValueError)
    with pytest.raises(ValueError, match="several solutions"):
        arden_star(EPSILON)
    with pytest.raises(TypeError, match="Regex"):
        solve_arden("a", parse("b"))
    with pytest.raises(TypeError, match="Regex"):
        solve_arden(parse("a"), None)
    with pytest.raises(TypeError, match="Regex"):
        arden_star("a")
    with pytest.raises(ValueError, match="'AX' or 'XA'"):
        solve_arden(parse("a"), parse("b"), form="X=AX")
    with pytest.raises(ValueError, match="'AX' or 'XA'"):
        equation_side(parse("a"), parse("b"), parse("a"), form="?")


# ---------------------------------------------------------- language of the solution
@pytest.mark.parametrize("form", ["AX", "XA"])
@pytest.mark.parametrize("a_text", A_SAMPLES)
@pytest.mark.parametrize("b_text", B_SAMPLES)
def test_57_solution_satisfies_the_equation(a_text, b_text, form):
    a, b = parse(a_text), parse(b_text)
    x = solve_arden(a, b, form)
    assert same_language(x, equation_side(a, b, x, form))  # exact equation X = AX ∪ B


@pytest.mark.parametrize("form", ["AX", "XA"])
@pytest.mark.parametrize("a_text", A_SAMPLES)
@pytest.mark.parametrize("b_text", B_SAMPLES)
def test_57_solution_is_the_unique_fixed_point(a_text, b_text, form):
    a, b = parse(a_text), parse(b_text)
    x = solve_arden(a, b, form)
    everything = set(words_up_to("ab", LONGEST))
    from_nothing = fixed_point(a, b, form, set())
    from_everything = fixed_point(a, b, form, everything)
    assert from_nothing == from_everything  # uniqueness when ε ∉ A
    assert language(x) == from_nothing  # and the solver finds it


def test_57_result_is_simplified_and_inputs_untouched():
    a, b = parse("a"), parse("b")
    before = (str(a), str(b))
    x = solve_arden(a, b)
    assert (str(a), str(b)) == before
    assert isinstance(x, Regex)
    assert str(solve_arden(parse("a|a"), parse("b|b"))) == "a*b"  # a|a = a, b|b = b
    assert solve_arden(parse("a"), parse("ε|∅")) == parse("a*")
    with pytest.raises(FrozenInstanceError):
        x.left = None  # type: ignore[attr-defined]


def test_57_equation_side_is_the_right_hand_side():
    a, b, x = parse("a"), parse("b"), parse("c")
    assert str(equation_side(a, b, x, "AX")) == "ac|b"
    assert str(equation_side(a, b, x, "XA")) == "ca|b"


def test_57_arden_star_returns_the_star_of_a_coefficient_without_epsilon():
    assert arden_star(EMPTY) == EPSILON
    assert str(arden_star(parse("ab"))) == "(ab)*"


def test_57_symbol_objects_and_programmatic_regex():
    x = solve_arden(Symbol("a"), Symbol("b"), "XA")
    assert x == parse("ba*")
    for w in ("b", "ba", "baaa"):
        assert oracle(str(x), w)


# ------------------------------------------------------- least solution when ε ∈ L(A)
@pytest.mark.parametrize("form", ["AX", "XA"])
@pytest.mark.parametrize("a_text", ["a*", "ε", "a|ε", "(a|b)*", "(a|ε)b*"])
@pytest.mark.parametrize("b_text", ["b", "ε", "∅", "ab"])
def test_57_least_solution_with_epsilon_in_a(a_text, b_text, form):
    a, b = parse(a_text), parse(b_text)
    with pytest.raises(ArdenError):
        solve_arden(a, b, form)  # default behaviour is unchanged
    x = solve_arden(a, b, form, least=True)
    assert same_language(x, equation_side(a, b, x, form))  # it is a solution
    assert language(x) == fixed_point(a, b, form, set())  # and the least fixed point


def test_57_least_solution_is_contained_in_every_other_solution():
    a, b = parse("a*"), parse("b")
    least = solve_arden(a, b, least=True)
    assert str(least) == "a*b"
    other = parse("(a|b)*")  # another solution (see the uniqueness test)
    assert same_language(other, equation_side(a, b, other, "AX"))
    assert language(least) <= language(other) and language(least) != language(other)


def test_57_least_flag_does_not_change_the_unique_case():
    for a_text in ["a", "ab", "∅", "a|b"]:
        for form in ("AX", "XA"):
            assert solve_arden(parse(a_text), parse("b"), form, least=True) == solve_arden(
                parse(a_text), parse("b"), form
            )
    assert arden_star(parse("a*"), least=True) == parse("a*")
    with pytest.raises(ArdenError):
        arden_star(parse("a*"))
