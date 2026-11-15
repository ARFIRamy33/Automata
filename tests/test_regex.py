"""Tests of the regex module: derivatives (49), Brzozowski DFA (50), Thompson NFA (51)."""

import itertools
import re
from dataclasses import FrozenInstanceError

import pytest

from automata_extensions.fa import ExtendedDFA, ExtendedNFA
from automata_extensions.regex import (
    EMPTY,
    EPSILON,
    Concat,
    Regex,
    RegexSyntaxError,
    Star,
    Symbol,
    Union,
    concat,
    star,
    union,
)

EPS = ""

SAMPLES = [
    "∅", "ε", "()", "a", "a|b", "ab", "a*", "(ab)*", "a(b|c)*", "(a|b)*abb", "a*b*",
    "(a|ε)b", "(ab|a)(c|bc)", "(a*)*", "a|a", "(a*|b)*", "a∅", "∅|a", "ε*", "∅*",
    "(a|b)(a|b)", "a*a", "(ab*)*|c", "\\*a", "(a|b)*a(a|b)", "a(ε|b)*",
]


def words_up_to(symbols, longest):
    for length in range(longest + 1):
        for letters in itertools.product(symbols, repeat=length):
            yield "".join(letters)


def to_python_re(text):
    """Independent oracle: translate the *text* of a formal regex into Python's re."""
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            out.append(re.escape(text[i + 1]))
            i += 2
            continue
        out.append({"ε": "(?:)", "∅": "(?:(?!))"}.get(c, c))
        i += 1
    return re.compile("".join(out))


def oracle(text, word):
    return to_python_re(text).fullmatch(word) is not None


def alphabet_of(text):
    symbols, i = set(), 0
    while i < len(text):
        if text[i] == "\\":
            symbols.add(text[i + 1])
            i += 2
            continue
        if text[i] not in "()|*ε∅":
            symbols.add(text[i])
        i += 1
    return sorted(symbols)


def dfa_accepts(dfa, word):
    state = dfa.initial_state
    for symbol in word:
        state = dfa.transitions.get(state, {}).get(symbol)
        if state is None:
            return False
    return state in dfa.final_states


def nfa_accepts(nfa, word):
    def close(current):
        todo, seen = list(current), set(current)
        while todo:
            for t in nfa.transitions.get(todo.pop(), {}).get(EPS, ()):
                if t not in seen:
                    seen.add(t)
                    todo.append(t)
        return seen

    current = close({nfa.initial_state})
    for symbol in word:
        step = set()
        for s in current:
            step |= set(nfa.transitions.get(s, {}).get(symbol, ()))
        current = close(step)
    return bool(current & set(nfa.final_states))


def parse(text):
    return Regex.parse(text)


# --- syntax and normal form ----------------------------------------------------------


def test_parse_basic_nodes():
    assert parse("a") == Symbol("a")
    assert parse("ε") == EPSILON == parse("()")
    assert parse("∅") == EMPTY
    assert isinstance(parse("ab"), Concat) and isinstance(parse("a|b"), Union)
    assert isinstance(parse("a*"), Star)


def test_priorities_star_then_concatenation_then_union():
    assert parse("ab*") == concat(Symbol("a"), star(Symbol("b")))
    assert parse("a|bc") == union(Symbol("a"), concat(Symbol("b"), Symbol("c")))
    assert parse("(ab)*") == star(concat(Symbol("a"), Symbol("b")))
    assert str(parse("ab*")) == "ab*" and str(parse("(ab)*")) == "(ab)*"
    assert str(parse("(a|b)c")) == "(a|b)c"


def test_escaped_symbols():
    r = parse("\\*\\(\\|")
    assert [p.char for p in r.parts] == ["*", "(", "|"]
    assert str(r) == "\\*\\(\\|"
    assert parse("\\+") == Symbol("+")


def test_print_then_parse_round_trips():
    for text in SAMPLES:
        r = parse(text)
        assert parse(str(r)) == r, text


def test_normal_form_identities():
    a, b = Symbol("a"), Symbol("b")
    assert union(a, EMPTY) == a and union(EMPTY, EMPTY) == EMPTY
    assert union(a, b) == union(b, a) == union(a, union(b, a))
    assert concat(a, EPSILON) == a and concat(a, EMPTY) == EMPTY and concat() == EPSILON
    assert star(EMPTY) == star(EPSILON) == EPSILON
    assert star(star(a)) == star(a)
    assert concat(concat(a, b), a) == concat(a, concat(b, a))


def test_no_incorrect_simplification():
    # r|r* and rr* are different from r*: they must be left alone.
    a = Symbol("a")
    assert union(a, star(a)) != star(a)
    assert concat(a, star(a)) != star(a)
    assert concat(a, b := Symbol("b")) != concat(b, a)


def test_expressions_are_immutable_and_hashable():
    r = parse("a(b|c)*")
    with pytest.raises(FrozenInstanceError):
        r.parts = ()
    assert len({parse("a|b"), parse("b|a"), parse("a|b|a")}) == 1


@pytest.mark.parametrize(
    "bad,fragment",
    [
        ("", "Empty text"),
        ("(", "Missing ')'"),
        ("(a", "Missing ')'"),
        ("a)", "Unmatched ')'"),
        ("a||b", "Empty alternative"),
        ("|a", "Empty alternative"),
        ("a|", "Empty alternative"),
        ("(|a)", "Empty alternative"),
        ("*a", "Nothing to repeat"),
        ("a|*", "Nothing to repeat"),
        ("(*)", "Nothing to repeat"),
        ("a+", "not supported"),
        ("a?", "not supported"),
        ("a\\", "Dangling"),
    ],
)
def test_malformed_expressions_give_explicit_errors(bad, fragment):
    with pytest.raises(RegexSyntaxError, match=re.escape(fragment)):
        parse(bad)


def test_error_messages_give_the_position():
    with pytest.raises(RegexSyntaxError, match="position 2"):
        parse("ab)c")


def test_non_string_input_is_rejected():
    with pytest.raises(TypeError):
        Regex.parse(None)
    with pytest.raises(ValueError):
        Symbol("ab")


# --- function 49: derivatives --------------------------------------------------------


def test_49_nullable():
    for text, expected in [("∅", False), ("ε", True), ("a", False), ("a|ε", True),
                           ("ab", False), ("a*b*", True), ("a*b", False), ("(a|b)*", True)]:
        assert parse(text).nullable() == expected, text


def test_49_derivative_of_the_empty_expression_and_epsilon():
    assert EMPTY.derivative("a") == EMPTY
    assert EPSILON.derivative("a") == EMPTY


def test_49_derivative_of_a_symbol():
    assert parse("a").derivative("a") == EPSILON
    assert parse("a").derivative("b") == EMPTY


def test_49_derivative_of_a_union():
    assert parse("a|b").derivative("a") == EPSILON
    assert parse("ab|ac").derivative("a") == union(Symbol("b"), Symbol("c"))
    assert parse("b|c").derivative("a") == EMPTY


def test_49_derivative_of_a_concatenation():
    assert parse("ab").derivative("a") == Symbol("b")
    assert parse("ab").derivative("b") == EMPTY
    assert parse("abc").derivative("a") == concat(Symbol("b"), Symbol("c"))


def test_49_concatenation_with_a_nullable_first_part():
    # D_a(a*a) = D_a(a*) a | D_a(a) = a*a | ε
    r = parse("a*a")
    assert r.derivative("a") == union(parse("a*a"), EPSILON)
    # D_b(a*b) = ∅·b | D_b(b) = ε   (first part nullable but with a different symbol)
    assert parse("a*b").derivative("b") == EPSILON
    # not nullable first part: no second term
    assert parse("ab").derivative("b") == EMPTY
    # the whole tail may be nullable too
    assert parse("a*b*").derivative("b") == parse("b*")


def test_49_derivative_of_a_star():
    assert parse("a*").derivative("a") == parse("a*")
    assert parse("(ab)*").derivative("a") == parse("b(ab)*")
    assert parse("(ab)*").derivative("b") == EMPTY


def test_49_matches_the_mathematical_definition():
    # D_a(R) = { w | a w ∈ L(R) }, checked word by word with an independent oracle.
    for text in SAMPLES:
        alphabet = alphabet_of(text) or ["a"]
        r = parse(text)
        for a in alphabet + ["z"]:
            derived = str(r.derivative(a))
            for w in words_up_to(alphabet + ["z"], 4):
                assert oracle(derived, w) == oracle(text, a + w), (text, a, w)


def test_49_nullable_matches_the_empty_word():
    for text in SAMPLES:
        assert parse(text).nullable() == oracle(text, ""), text


def test_49_derivative_by_a_word_is_iterated():
    r = parse("(a|b)*abb")
    for word in ("abb", "ab", "bbabb"):
        d = r
        for symbol in word:
            d = d.derivative(symbol)
        assert d.nullable() == oracle("(a|b)*abb", word)


def test_49_does_not_modify_the_expression():
    r = parse("a(b|c)*")
    before = (r, str(r))
    r.derivative("a")
    r.nullable()
    assert (r, str(r)) == before


# --- function 50: Brzozowski automaton -----------------------------------------------


def test_50_single_symbol():
    d = parse("a").brzozowski_dfa()
    assert isinstance(d, ExtendedDFA)
    assert d.states == {parse("a"), EPSILON, EMPTY}
    assert d.initial_state == parse("a")
    assert d.final_states == {EPSILON}
    assert d.transitions[parse("a")] == {"a": EPSILON}
    assert d.transitions[EPSILON] == {"a": EMPTY} and d.transitions[EMPTY] == {"a": EMPTY}


def test_50_union_of_two_symbols():
    d = parse("a|b").brzozowski_dfa()
    assert d.states == {parse("a|b"), EPSILON, EMPTY}
    assert d.transitions[parse("a|b")] == {"a": EPSILON, "b": EPSILON}
    assert d.final_states == {EPSILON}


def test_50_concatenation():
    d = parse("ab").brzozowski_dfa()
    assert d.states == {parse("ab"), parse("b"), EPSILON, EMPTY}
    assert d.transitions[parse("ab")] == {"a": parse("b"), "b": EMPTY}
    assert d.transitions[parse("b")] == {"a": EMPTY, "b": EPSILON}
    assert d.final_states == {EPSILON}


def test_50_kleene_star():
    d = parse("a*").brzozowski_dfa()
    assert d.states == {parse("a*")}  # D_a(a*) = a*: a single state, final
    assert d.final_states == {parse("a*")}
    assert d.transitions[parse("a*")] == {"a": parse("a*")}
    d2 = parse("(ab)*").brzozowski_dfa()
    assert d2.states == {parse("(ab)*"), parse("b(ab)*"), EMPTY}
    assert d2.final_states == {parse("(ab)*")}


def test_50_expression_with_epsilon():
    d = parse("(a|ε)b").brzozowski_dfa()
    assert d.initial_state not in d.final_states
    assert parse("b") in d.states and EPSILON in d.states
    # D_a((a|ε)b) = b ; D_b((a|ε)b) = ε because a|ε is nullable.
    assert d.transitions[d.initial_state] == {"a": parse("b"), "b": EPSILON}


def test_50_empty_expression_and_epsilon_with_an_explicit_alphabet():
    for text, finals in (("∅", set()), ("ε", {EPSILON})):
        d = parse(text).brzozowski_dfa(alphabet="ab")
        assert d.input_symbols == {"a", "b"}
        assert d.final_states == finals
        assert not dfa_accepts(d, "a") and dfa_accepts(d, "") == bool(finals)


def test_50_empty_alphabet_is_handled_explicitly():
    # Either the library accepts an automaton without symbols, or the error is explicit.
    try:
        d = parse("ε").brzozowski_dfa()
    except ValueError as error:
        assert "alphabet" in str(error)
    else:
        assert d.final_states == {EPSILON} and d.input_symbols == frozenset()


def test_50_alphabet_argument_is_checked_and_extra_symbols_go_to_empty():
    with pytest.raises(ValueError, match="missing"):
        parse("ab").brzozowski_dfa(alphabet="a")
    d = parse("ab").brzozowski_dfa(alphabet="abc")
    assert d.transitions[parse("ab")]["c"] == EMPTY


def test_50_final_states_are_the_nullable_ones_and_transitions_are_derivatives():
    for text in SAMPLES:
        d = parse(text).brzozowski_dfa(alphabet=alphabet_of(text) or "a")
        assert d.final_states == {r for r in d.states if r.nullable()}, text
        for state, out in d.transitions.items():
            for symbol, target in out.items():
                assert target == state.derivative(symbol), (text, state, symbol)
        assert d.is_complete(), text


def test_50_language_matches_the_oracle():
    for text in SAMPLES:
        alphabet = alphabet_of(text) or ["a"]
        d = parse(text).brzozowski_dfa(alphabet=alphabet)
        for w in words_up_to(alphabet, 5):
            assert dfa_accepts(d, w) == oracle(text, w), (text, w)


def test_50_only_reachable_and_distinct_states():
    for text in SAMPLES:
        d = parse(text).brzozowski_dfa(alphabet=alphabet_of(text) or "a")
        assert d.accessible_states() == d.states, text
        assert len({str(s) for s in d.states}) == len(d.states), text  # no duplicate expression


def test_50_known_size_and_minimality():
    d = parse("(a|b)*abb").brzozowski_dfa()
    assert len(d.states) == 4  # the classic minimal DFA of (a|b)*abb
    assert d.is_minimal()
    assert len(parse("(a|b)*a(a|b)").brzozowski_dfa().states) == 4


def test_50_agrees_with_a_reference_automaton_built_by_thompson():
    for text in SAMPLES:
        alphabet = alphabet_of(text) or ["a"]
        brz = parse(text).brzozowski_dfa(alphabet=alphabet)
        reference = parse(text).to_nfa(alphabet=alphabet).determinize(complete=True)
        assert brz.has_same_language_as(reference), text


def test_50_runaway_construction_is_stopped():
    with pytest.raises(RuntimeError, match="stopped"):
        parse("(a|b)*abb").brzozowski_dfa(max_states=2)


def test_50_does_not_modify_the_expression():
    r = parse("(a|b)*abb")
    before = (r, str(r))
    r.brzozowski_dfa()
    assert (r, str(r)) == before


def test_50_states_are_displayed_as_regular_expressions():
    d = parse("(ab)*").brzozowski_dfa()
    assert 'label="b(ab)*"' in d.dot()


# --- function 51: Thompson ----------------------------------------------------------


def test_51_single_symbol():
    n = parse("a").to_nfa()
    assert isinstance(n, ExtendedNFA)
    assert n.input_symbols == {"a"} and len(n.final_states) == 1
    assert nfa_accepts(n, "a") and not nfa_accepts(n, "") and not nfa_accepts(n, "aa")
    assert [s for s, out in n.transitions.items() if "a" in out] == [n.initial_state]


def test_51_union_of_two_symbols():
    n = parse("a|b").to_nfa()
    assert [w for w in words_up_to("ab", 2) if nfa_accepts(n, w)] == ["a", "b"]
    assert n.transitions[n.initial_state][EPS] and len(n.transitions[n.initial_state][EPS]) == 2


def test_51_concatenation():
    n = parse("abc").to_nfa()
    assert [w for w in words_up_to("abc", 4) if nfa_accepts(n, w)] == ["abc"]
    assert len(n.states) == 6  # two states per symbol, glued by epsilon transitions


def test_51_star():
    n = parse("a*").to_nfa()
    assert all(nfa_accepts(n, "a" * k) for k in range(5)) and not nfa_accepts(n, "b")
    inner_start = next(iter(n.transitions[n.initial_state][EPS] - set(n.final_states)))
    accept = next(iter(n.final_states))
    assert accept in n.transitions[n.initial_state][EPS]  # ε skips the loop


def test_51_parentheses_and_combined_operators():
    for text in ("(a|b)*abb", "a(b|c)*", "((a|b)c)*|d", "(ab|a)(c|bc)", "a*b*(c|ε)", "((a))"):
        alphabet = alphabet_of(text)
        n = parse(text).to_nfa()
        for w in words_up_to(alphabet, 5):
            assert nfa_accepts(n, w) == oracle(text, w), (text, w)


def test_51_empty_expression_and_epsilon():
    empty = parse("∅").to_nfa(alphabet="a")
    assert not any(nfa_accepts(empty, w) for w in words_up_to("a", 3))
    assert len(empty.states) == 2 and empty.initial_state not in empty.final_states
    eps = parse("ε").to_nfa(alphabet="a")
    assert nfa_accepts(eps, "") and not nfa_accepts(eps, "a")


def test_51_language_matches_the_oracle_on_all_samples():
    for text in SAMPLES:
        alphabet = alphabet_of(text) or ["a"]
        n = parse(text).to_nfa(alphabet=alphabet)
        for w in words_up_to(alphabet, 5):
            assert nfa_accepts(n, w) == oracle(text, w), (text, w)


def test_51_states_are_distinct_integers_without_collision():
    for text in SAMPLES:
        n = parse(text).to_nfa(alphabet=alphabet_of(text) or "a")
        assert n.states == set(range(len(n.states))), text
        assert all(isinstance(s, int) for s in n.states)


def test_51_thompson_shape_every_state_has_at_most_two_outgoing_edges():
    n = parse("(a|b)*abb").to_nfa()
    for out in n.transitions.values():
        assert sum(len(t) for t in out.values()) <= 2
    accept = next(iter(n.final_states))
    assert not any(n.transitions[accept].values())  # nothing leaves the accepting state
    assert not any(n.initial_state in t for out in n.transitions.values() for t in out.values())


def test_51_alphabet_argument_is_checked():
    with pytest.raises(ValueError, match="missing"):
        parse("ab").to_nfa(alphabet="a")
    assert parse("a").to_nfa(alphabet="abc").input_symbols == {"a", "b", "c"}


def test_51_determinization_of_the_result_works():
    d = parse("(a|b)*abb").to_nfa().determinize()
    assert d.is_deterministic()
    for w in words_up_to("ab", 5):
        assert dfa_accepts(d, w) == oracle("(a|b)*abb", w)


def test_51_malformed_expressions_are_rejected_before_any_construction():
    for bad in ("(a", "a||b", "*", "a)"):
        with pytest.raises(RegexSyntaxError):
            parse(bad).to_nfa()


def test_51_does_not_modify_the_expression():
    r = parse("(a|b)*abb")
    before = (r, str(r))
    r.to_nfa()
    assert (r, str(r)) == before


# =====================================================================================
# Functions 53 and 54: DFA -> regular expression
# =====================================================================================

import random  # noqa: E402


def dfa(transitions, initial="q0", finals=(), symbols=None):
    """Build an ExtendedDFA from a possibly partial table."""
    states = set(transitions) | {t for out in transitions.values() for t in out.values()}
    states |= {initial} | set(finals)
    table = {s: dict(transitions.get(s, {})) for s in states}
    used = {a for out in table.values() for a in out}
    return ExtendedDFA(
        states=states,
        input_symbols=set(symbols) if symbols else (used or {"a"}),
        transitions=table,
        initial_state=initial,
        final_states=set(finals),
        allow_partial=True,
    )


def random_dfas(count=120, seed=7):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(1, 4)
        names = [f"s{i}" for i in range(n)]
        table = {
            s: {a: rng.choice(names) for a in "ab" if rng.random() < 0.75} for s in names
        }
        finals = {s for s in names if rng.random() < 0.4}
        yield dfa(table, initial="s0", finals=finals, symbols="ab")


METHODS = ["to_regex_elimination", "to_regex_arden"]


def language_check(automaton, regex, longest=5):
    text = str(regex)
    assert parse(text) == regex  # printable and re-parsable
    for w in words_up_to(sorted(automaton.input_symbols), longest):
        assert oracle(text, w) == dfa_accepts(automaton, w), (text, w)


def exact_check(automaton, regex):
    """Exact equality of languages through Thompson + determinization + function 25."""
    rebuilt = regex.to_nfa(alphabet=automaton.input_symbols).determinize(complete=True)
    assert rebuilt.has_same_language_as(automaton)


@pytest.mark.parametrize("method", METHODS)
def test_53_54_single_state(method):
    only_empty_word = dfa({"q0": {}}, finals={"q0"})
    assert getattr(only_empty_word, method)() == EPSILON
    assert getattr(dfa({"q0": {}}), method)() == EMPTY
    looping = dfa({"q0": {"a": "q0"}}, finals={"q0"})
    assert str(getattr(looping, method)()) == "a*"
    assert getattr(dfa({"q0": {"a": "q0"}}), method)() == EMPTY


@pytest.mark.parametrize("method", METHODS)
def test_53_54_one_transition(method):
    a = dfa({"q0": {"a": "q1"}}, finals={"q1"})
    assert getattr(a, method)() == Symbol("a")


@pytest.mark.parametrize("method", METHODS)
def test_53_54_loop(method):
    a = dfa({"q0": {"a": "q0", "b": "q1"}}, finals={"q1"})
    result = getattr(a, method)()
    language_check(a, result)
    exact_check(a, result)
    assert str(result) == "a*b"


@pytest.mark.parametrize("method", METHODS)
def test_53_54_several_states_and_complete_dfa(method):
    even_a = dfa(
        {"e": {"a": "o", "b": "e"}, "o": {"a": "e", "b": "o"}}, initial="e", finals={"e"}
    )
    result = getattr(even_a, method)()
    language_check(even_a, result)
    exact_check(even_a, result)
    assert result.nullable()


@pytest.mark.parametrize("method", METHODS)
def test_53_54_several_final_states(method):
    a = dfa({"q0": {"a": "q1", "b": "q2"}, "q1": {"a": "q1"}, "q2": {"b": "q2"}}, finals={"q1", "q2"})
    result = getattr(a, method)()
    language_check(a, result)
    exact_check(a, result)
    assert not result.nullable()


@pytest.mark.parametrize("method", METHODS)
def test_53_54_missing_transitions_and_useless_states(method):
    a = dfa(
        {"q0": {"a": "q1", "b": "dead"}, "q1": {"b": "q2"}, "dead": {"a": "dead"}, "u": {"a": "q2"}},
        finals={"q2"},
        symbols="ab",
    )
    result = getattr(a, method)()
    assert str(result) == "ab"  # dead and unreachable states do not appear
    exact_check(a, result)


@pytest.mark.parametrize("method", METHODS)
def test_53_54_empty_language_gives_the_empty_expression(method):
    a = dfa({"q0": {"a": "q1"}, "q1": {"a": "q0"}}, symbols="a")
    assert getattr(a, method)() == EMPTY


@pytest.mark.parametrize("method", METHODS)
def test_53_54_many_random_dfas(method):
    for automaton in random_dfas():
        result = getattr(automaton, method)()
        language_check(automaton, result, longest=5)
        exact_check(automaton, result)


def test_53_and_54_agree_on_the_language():
    for automaton in random_dfas(60, seed=11):
        first = automaton.to_regex_elimination()
        second = automaton.to_regex_arden()
        rebuilt = second.to_nfa(alphabet=automaton.input_symbols).determinize(complete=True)
        assert first.to_nfa(alphabet=automaton.input_symbols).determinize(complete=True).has_same_language_as(rebuilt)


def test_53_and_54_are_really_two_different_methods(monkeypatch):
    automaton = dfa({"q0": {"a": "q0", "b": "q1"}, "q1": {"a": "q0"}}, finals={"q1"})

    def boom(*args, **kwargs):
        raise AssertionError("the other method was called")

    with monkeypatch.context() as m:
        m.setattr(ExtendedDFA, "_eliminate", boom)
        assert automaton.to_regex_arden() is not None  # Arden never uses state elimination
    with monkeypatch.context() as m:
        m.setattr(ExtendedDFA, "to_regex_arden", boom)
        assert automaton.to_regex_elimination() is not None


def test_53_explicit_elimination_order_changes_the_text_not_the_language():
    a = dfa({"q0": {"a": "q1", "b": "q2"}, "q1": {"a": "q2"}, "q2": {"b": "q0"}}, finals={"q2"})
    normalized = a.trim().normalized()
    middle = sorted(normalized.states - {normalized.initial_state, next(iter(normalized.final_states))}, key=repr)
    for order in itertools.permutations(middle):
        result = a.to_regex_elimination(order=order)
        exact_check(a, result)
    with pytest.raises(ValueError, match="exactly once"):
        a.to_regex_elimination(order=middle[:-1])


def test_53_greedy_order_keeps_expressions_small():
    chain = dfa({f"s{i}": {"a": f"s{i + 1}"} for i in range(6)}, initial="s0", finals={"s6"})
    assert str(chain.to_regex_elimination()) == "aaaaaa"
    assert str(chain.to_regex_arden()) == "aaaaaa"


def test_53_54_do_not_modify_the_automaton():
    for automaton in random_dfas(20, seed=3):
        before = (set(automaton.states), dict(automaton.transitions), automaton.final_states)
        automaton.to_regex_elimination()
        automaton.to_regex_arden()
        assert (set(automaton.states), dict(automaton.transitions), automaton.final_states) == before


@pytest.mark.parametrize("method", METHODS)
def test_53_54_multi_character_symbols_are_rejected(method):
    a = ExtendedDFA(
        states={"p", "q"}, input_symbols={"ab"}, transitions={"p": {"ab": "q"}, "q": {}},
        initial_state="p", final_states={"q"}, allow_partial=True,
    )
    with pytest.raises(ValueError, match="one-character|several characters"):
        getattr(a, method)()


def test_54_refuses_automata_with_epsilon_transitions():
    nfa = parse("a*").to_nfa()
    with pytest.raises(ValueError, match="DFA"):
        nfa.to_regex_arden()


def test_54_arden_lemma_on_a_hand_checked_system():
    # X0 = a X0 | b X1 ;  X1 = a X0 | ε   ->   X0 = a* b (a X0 | ε)  ->  (a|ba)* b
    a = dfa({"q0": {"a": "q0", "b": "q1"}, "q1": {"a": "q0"}}, finals={"q1"})
    result = a.to_regex_arden()
    exact_check(a, result)
    assert oracle(str(result), "bab") and oracle(str(result), "aab") and not oracle(str(result), "ba")


def test_53_works_on_an_nfa_with_epsilon_as_well():
    nfa = parse("(a|b)*abb").to_nfa()
    result = nfa.to_regex_elimination()
    for w in words_up_to("ab", 6):
        assert oracle(str(result), w) == oracle("(a|b)*abb", w)
