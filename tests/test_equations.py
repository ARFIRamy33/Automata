"""Tests of function 58: systems of rational equations."""

import itertools
import random
from dataclasses import FrozenInstanceError

import pytest

from automata_extensions.regex import (
    EMPTY,
    EPSILON,
    ArdenError,
    AutomatonSystem,
    EquationSystem,
    EquationSystemError,
    Regex,
)
from automata_extensions.fa import ExtendedNFA
from test_grammar import make_dfa, make_nfa
from test_regex import dfa_accepts, nfa_accepts, oracle, words_up_to

LONGEST = 5


def parse(text):
    return Regex.parse(text)


def language(regex, symbols="abcd", longest=LONGEST):
    return {w for w in words_up_to(symbols, longest) if oracle(str(regex), w)}


def system(unknowns, coefficients=None, constants=None, form="AX"):
    return EquationSystem.build(
        unknowns,
        {k: parse(v) for k, v in (coefficients or {}).items()},
        {k: parse(v) for k, v in (constants or {}).items()},
        form,
    )


def fixed_point(sys_, start, longest=LONGEST, symbols="abcd"):
    """Independent oracle: iterate the equations on words of bounded length.

    ``start`` is the initial value of every unknown (empty set or every word).
    Only membership of the coefficients in Python's ``re`` is used.
    """
    coefficient = {
        (i, j): language(sys_.coefficients[i][j], symbols, longest)
        for i in range(len(sys_.unknowns))
        for j in range(len(sys_.unknowns))
    }
    constant = [language(c, symbols, longest) for c in sys_.constants]
    current = [set(start) for _ in sys_.unknowns]
    for _ in range(longest + 3):
        nxt = []
        for i in range(len(sys_.unknowns)):
            value = set(constant[i])
            for j in range(len(sys_.unknowns)):
                for u in coefficient[(i, j)]:
                    for v in current[j]:
                        w = u + v if sys_.form == "AX" else v + u
                        if len(w) <= longest:
                            value.add(w)
            nxt.append(value)
        current = nxt
    return current


def check_unique_solution(sys_, symbols="abcd"):
    solution = sys_.solve()
    everything = set(words_up_to(symbols, LONGEST))
    low = fixed_point(sys_, set(), symbols=symbols)
    high = fixed_point(sys_, everything, symbols=symbols)
    assert low == high  # the system has a unique solution (bounded check)
    for i, name in enumerate(sys_.unknowns):
        assert language(solution[name], symbols) == low[i], name
    assert sys_.is_solution(solution)
    return solution


# ------------------------------------------------------------------ one unknown
def test_58_one_unknown():
    s = system(["X"], {("X", "X"): "a"}, {"X": "b"})
    assert str(s) == "X = (a)X ∪ b"
    assert str(s.solve()["X"]) == "a*b"
    check_unique_solution(s)


def test_58_one_unknown_right_form_is_mirrored():
    s = system(["X"], {("X", "X"): "a"}, {"X": "b"}, form="XA")
    assert str(s.solve()["X"]) == "ba*"
    check_unique_solution(s)


def test_58_one_unknown_without_loop_or_without_constant():
    assert str(system(["X"], {}, {"X": "ab"}).solve()["X"]) == "ab"
    assert system(["X"], {("X", "X"): "a"}, {}).solve()["X"] == EMPTY
    assert system(["X"]).solve()["X"] == EMPTY


# ----------------------------------------------------------------- two unknowns
def test_58_two_unknowns_exact_result():
    s = system(["X1", "X2"], {("X1", "X2"): "a", ("X2", "X1"): "b"}, {"X1": "ε"})
    solution = check_unique_solution(s)
    assert solution["X1"].same_language_as(parse("(ab)*"))
    assert solution["X2"].same_language_as(parse("b(ab)*"))
    # with X2 eliminated first the result is the compact form
    compact = s.solve(order=["X2", "X1"])
    assert str(compact["X1"]) == "(ab)*" and str(compact["X2"]) == "b(ab)*"


def test_58_two_unknowns_general_form_of_the_statement():
    s = system(
        ["X1", "X2"],
        {("X1", "X1"): "a", ("X1", "X2"): "b", ("X2", "X1"): "c", ("X2", "X2"): "d"},
        {"X1": "ε", "X2": "a"},
    )
    solution = check_unique_solution(s)
    assert solution["X1"].same_language_as(s.right_hand_side("X1", solution))


def test_58_order_of_concatenation_is_kept():
    left = system(["X1", "X2"], {("X1", "X2"): "a"}, {"X2": "b"})  # X1 = a X2, X2 = b
    right = system(["X1", "X2"], {("X1", "X2"): "a"}, {"X2": "b"}, form="XA")  # X1 = X2 a
    assert str(left.solve()["X1"]) == "ab"
    assert str(right.solve()["X1"]) == "ba"


# ---------------------------------------------------------- several equations
def test_58_chain_of_dependencies():
    s = system(
        ["X1", "X2", "X3", "X4"],
        {("X1", "X2"): "a", ("X2", "X3"): "b", ("X3", "X4"): "c"},
        {"X4": "d", "X1": "ε"},
    )
    solution = check_unique_solution(s)
    assert str(solution["X1"]) == "abcd|ε"
    assert str(solution["X3"]) == "cd"


def test_58_cyclic_dependencies_with_loops():
    s = system(
        ["X1", "X2", "X3"],
        {
            ("X1", "X1"): "a", ("X1", "X2"): "b", ("X2", "X3"): "c",
            ("X3", "X1"): "d", ("X3", "X3"): "a|b",
        },
        {"X1": "ε", "X3": "c"},
    )
    check_unique_solution(s)


def test_58_unknown_that_nobody_uses_and_unused_equations():
    s = system(["X", "Y"], {("X", "X"): "a"}, {"X": "b", "Y": "c"})
    solution = s.solve()
    assert str(solution["X"]) == "a*b" and str(solution["Y"]) == "c"


def test_58_every_elimination_order_gives_the_same_languages():
    s = system(
        ["X1", "X2", "X3"],
        {("X1", "X2"): "a", ("X2", "X3"): "b", ("X3", "X1"): "c", ("X2", "X2"): "d"},
        {"X1": "ε", "X3": "a"},
    )
    reference = s.solve()
    for order in itertools.permutations(s.unknowns):
        other = s.solve(order=order)
        assert s.is_solution(other)
        for name in s.unknowns:
            assert other[name].same_language_as(reference[name]), order


def test_58_random_systems_have_a_unique_correct_solution():
    rng = random.Random(4)
    pool = ["a", "b", "ab", "a|b", "ba", "∅", "c(a|b)"]  # no ε: Arden's condition holds
    for _ in range(40):
        n = rng.randint(1, 3)
        names = [f"X{i}" for i in range(n)]
        coefficients = {(i, j): rng.choice(pool) for i in names for j in names if rng.random() < 0.5}
        constants = {i: rng.choice(["ε", "c", "a", "∅", "a*"]) for i in names if rng.random() < 0.7}
        for form in ("AX", "XA"):
            check_unique_solution(system(names, coefficients, constants, form), symbols="abc")


# ----------------------------------------------------------- epsilon and empty set
def test_58_epsilon_as_constant_and_as_coefficient_between_unknowns():
    s = system(["X1", "X2"], {("X1", "X2"): "ε"}, {"X1": "a", "X2": "b"})  # X1 = X2 ∪ a
    solution = s.solve()
    assert solution["X1"].same_language_as(parse("a|b"))
    check_unique_solution(s)


def test_58_empty_coefficients_and_empty_constants():
    s = system(["X1", "X2"], {("X1", "X2"): "∅", ("X2", "X1"): "a"}, {})
    assert all(v == EMPTY for v in s.solve().values())
    only_constant = system(["X1", "X2"], {("X1", "X2"): "∅"}, {"X1": "ε", "X2": "∅"})
    assert only_constant.solve()["X1"] == EPSILON


def test_58_epsilon_in_a_self_coefficient_is_refused_with_the_unknown_name():
    for text in ["ε", "a*", "a|ε"]:
        s = system(["X", "Y"], {("Y", "Y"): text}, {"X": "a", "Y": "b"})
        with pytest.raises(ArdenError, match="cannot eliminate Y"):
            s.solve()


def test_58_epsilon_cycle_through_several_unknowns_is_refused_then_least_works():
    s = system(["X1", "X2"], {("X1", "X2"): "ε", ("X2", "X1"): "ε"}, {"X2": "a"})
    with pytest.raises(ArdenError, match="empty word"):
        s.solve()
    least = s.solve(least=True)
    assert least["X1"].same_language_as(parse("a")) and least["X2"].same_language_as(parse("a"))
    assert s.is_solution(least)
    # the system really has other solutions (uniqueness fails): that is why it is refused
    other = {"X1": parse("a|b"), "X2": parse("a|b")}
    assert s.is_solution(other)


def test_58_least_option_does_not_change_systems_with_a_unique_solution():
    s = system(["X1", "X2"], {("X1", "X2"): "a", ("X2", "X1"): "b"}, {"X1": "ε"})
    assert dict(s.solve(least=True)) == dict(s.solve())


# ---------------------------------------------------------------- ill-formed systems
def test_58_ill_formed_systems_are_detected():
    with pytest.raises(EquationSystemError, match="at least one"):
        EquationSystem.build([])
    with pytest.raises(EquationSystemError, match="distinct"):
        EquationSystem.build(["X", "X"])
    with pytest.raises(EquationSystemError, match="non-empty"):
        EquationSystem.build([""])
    with pytest.raises(EquationSystemError, match="'Z' is not an unknown"):
        EquationSystem.build(["X"], {("X", "Z"): parse("a")})
    with pytest.raises(EquationSystemError, match="'Z' is not an unknown"):
        EquationSystem.build(["X"], constants={"Z": parse("a")})
    with pytest.raises(EquationSystemError, match="pair of names"):
        EquationSystem.build(["X"], {"X": parse("a")})
    with pytest.raises(EquationSystemError, match="Regex"):
        EquationSystem.build(["X"], {("X", "X"): "a"})
    with pytest.raises(EquationSystemError, match="Regex"):
        EquationSystem.build(["X"], constants={"X": None})
    with pytest.raises(EquationSystemError, match="form"):
        EquationSystem.build(["X"], form="X=AX")
    with pytest.raises(EquationSystemError, match="n×n"):
        EquationSystem(("X", "Y"), ((EMPTY,), (EMPTY,)), (EMPTY, EMPTY))
    with pytest.raises(EquationSystemError, match="one constant"):
        EquationSystem(("X",), ((EMPTY,),), ())


def test_58_bad_elimination_order_and_bad_solution_are_detected():
    s = system(["X", "Y"], {("X", "Y"): "a"}, {"Y": "b"})
    for order in (["X"], ["X", "X"], ["X", "Y", "Y"]):
        with pytest.raises(EquationSystemError, match="exactly once"):
            s.solve(order=order)
    with pytest.raises(EquationSystemError, match="not an unknown"):
        s.solve(order=["X", "Z"])
    with pytest.raises(EquationSystemError, match="missing value"):
        s.is_solution({"X": parse("ab")})


# ------------------------------------------------- solutions satisfy the equations
def test_58_is_solution_accepts_other_forms_and_rejects_wrong_values():
    s = system(["X"], {("X", "X"): "a"}, {"X": "b"})
    assert s.is_solution({"X": parse("a*b")})
    assert s.is_solution({"X": parse("b|aa*b")})  # same language, other expression
    assert not s.is_solution({"X": parse("b")})
    assert not s.is_solution({"X": parse("ba*")})  # the other side's solution


def test_58_system_is_immutable_and_solve_does_not_modify_it():
    s = system(["X1", "X2"], {("X1", "X2"): "a", ("X2", "X1"): "b"}, {"X1": "ε"})
    before = (s.unknowns, s.coefficients, s.constants)
    solution = s.solve()
    assert (s.unknowns, s.coefficients, s.constants) == before
    with pytest.raises(FrozenInstanceError):
        s.form = "XA"
    with pytest.raises(TypeError):
        solution["X1"] = EMPTY  # type: ignore[index]
    assert s.solve() == solution


def test_58_size_of_the_result_stays_small_on_a_chain():
    names = [f"X{i}" for i in range(12)]
    s = EquationSystem.build(
        names, {(names[i], names[i + 1]): parse("a") for i in range(11)}, {names[-1]: EPSILON}
    )
    assert str(s.solve()["X0"]) == "a" * 11
    ring = EquationSystem.build(
        names[:6], {(names[i], names[(i + 1) % 6]): parse("a") for i in range(6)}, {names[0]: EPSILON}
    )
    assert len(str(ring.solve()["X0"])) < 40


# ------------------------------------------------------------------ from automata
def test_58_system_of_a_dfa_both_forms():
    dfa = make_dfa({"q0": {"a": "q1", "b": "q0"}, "q1": {"b": "q1"}}, "q0", {"q1"})
    auto = dfa.to_equation_system()
    assert isinstance(auto, AutomatonSystem) and auto.names[dfa.initial_state] == "X0"
    assert str(auto.system) == "X0 = (b)X0 ∪ (a)X1\nX1 = (b)X1 ∪ ε"
    assert str(auto.solve()) == "b*ab*"
    mirror = dfa.to_equation_system("XA")
    assert mirror.system.form == "XA"
    assert str(mirror.system) == "X0 = X0(b) ∪ ε\nX1 = X0(a) ∪ X1(b)"
    assert mirror.reading == ("X1",)  # the language is the union of the unknowns of final states
    assert mirror.solve().same_language_as(auto.solve())


def test_58_automaton_systems_give_the_language_of_random_automata():
    from test_grammar import random_dfas, random_nfas

    for automaton in itertools.chain(random_dfas(40), random_nfas(40)):
        has_eps = any("" in out and out[""] for out in automaton.transitions.values())
        for form in ("AX", "XA"):
            auto = automaton.to_equation_system(form)
            expression = auto.solve(least=True)
            accepts = nfa_accepts if isinstance(automaton, ExtendedNFA) else dfa_accepts
            for w in words_up_to("ab", 4):
                assert oracle(str(expression), w) == accepts(automaton, w), (form, w)
            if not has_eps:
                assert auto.solve().same_language_as(expression)  # unique: no need for least


def test_58_epsilon_transitions_need_the_least_option():
    nfa = make_nfa({"q0": {"": {"q1"}}, "q1": {"": {"q0"}, "a": {"q2"}}}, "q0", {"q2"})
    auto = nfa.to_equation_system()
    with pytest.raises(ArdenError):
        auto.solve()
    assert auto.solve(least=True).same_language_as(parse("a"))


def test_58_multi_character_symbols_are_refused():
    from automata_extensions.fa import ExtendedDFA

    dfa = ExtendedDFA(
        states={"p", "q"}, input_symbols={"ab"}, transitions={"p": {"ab": "q"}, "q": {}},
        initial_state="p", final_states={"q"}, allow_partial=True,
    )
    with pytest.raises(EquationSystemError, match="several characters"):
        dfa.to_equation_system()
    with pytest.raises(ValueError, match="one-character|several characters"):
        dfa.to_regex_arden()


def test_58_default_order_keeps_expressions_smaller_than_the_worst_order():
    # a hub H linked both ways to four leaves: eliminating the hub first creates 4×4 terms,
    # eliminating the leaves first creates none.
    leaves = [f"L{i}" for i in range(4)]
    coefficients = {("H", leaf): "a" for leaf in leaves}
    coefficients.update({(leaf, "H"): "b" for leaf in leaves})
    constants = {leaf: c for leaf, c in zip(leaves, "cdef")}
    s = system(["H", *leaves], coefficients, constants)
    default = s.solve()
    hub_first = s.solve(order=["H", *leaves])
    size = lambda solution: sum(len(str(v)) for v in solution.values())  # noqa: E731
    assert size(default) < size(hub_first)
    for name in s.unknowns:
        assert default[name].same_language_as(hub_first[name])
    assert s.is_solution(default)
