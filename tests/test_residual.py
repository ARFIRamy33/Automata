"""Tests of function 60: residual automaton."""

import itertools

import pytest

from automata_extensions.fa import ExtendedDFA, ExtendedNFA
from automata_extensions.regex import EMPTY, EPSILON, Regex
from test_grammar import make_dfa, make_nfa, random_dfas, random_nfas
from test_regex import SAMPLES, dfa_accepts, oracle, words_up_to


def parse(text):
    return Regex.parse(text)


def check_language(automaton, text, symbols="ab", longest=6):
    for w in words_up_to(symbols, longest):
        assert dfa_accepts(automaton, w) == oracle(text, w), (text, w)


# ---- simple languages ---------------------------------------------------------------
def test_60_simple_language_states_initial_and_final():
    r = parse("ab")
    dfa = r.residual_automaton()
    assert isinstance(dfa, ExtendedDFA) and dfa.is_complete()
    assert dfa.initial_state == r
    assert set(dfa.states) == {r, parse("b"), EPSILON, EMPTY}  # ab, b, ε and the empty residual
    assert dfa.final_states == {EPSILON}
    assert dfa.transitions[r] == {"a": parse("b"), "b": EMPTY}
    assert dfa.transitions[parse("b")] == {"a": EMPTY, "b": EPSILON}
    assert dfa.transitions[EMPTY] == {"a": EMPTY, "b": EMPTY}
    check_language(dfa, "ab")


def test_60_partial_version_drops_the_empty_residual():
    r = parse("ab")
    dfa = r.residual_automaton(complete=False)
    assert set(dfa.states) == {r, parse("b"), EPSILON} and not dfa.is_complete()
    assert dfa.transitions[r] == {"a": parse("b")} and dfa.transitions[EPSILON] == {}
    check_language(dfa, "ab")


def test_60_language_with_several_prefixes():
    r = parse("(a|b)*abb")
    dfa = r.residual_automaton()
    assert len(dfa.states) == 4
    assert dfa.initial_state == r and len(dfa.final_states) == 1
    assert all(s.nullable() for s in dfa.final_states) and not r.nullable()
    check_language(dfa, "(a|b)*abb")


def test_60_language_containing_epsilon():
    for text in ["a*", "ε|ab", "(ab)*", "a?".replace("?", "|ε")]:
        r = parse(text)
        dfa = r.residual_automaton(alphabet="ab")
        assert dfa.initial_state in dfa.final_states  # the ε-residual contains ε
        check_language(dfa, text)
    only_epsilon = EPSILON.residual_automaton(alphabet="a")
    assert only_epsilon.final_states == {EPSILON} and EMPTY in only_epsilon.states


def test_60_empty_language():
    for complete in (True, False):
        dfa = EMPTY.residual_automaton(alphabet="ab", complete=complete)
        assert set(dfa.states) == {EMPTY} and dfa.final_states == set()
        assert dfa.initial_state == EMPTY
        check_language(dfa, "∅")
    assert EMPTY.residual_automaton(alphabet="a").transitions[EMPTY] == {"a": EMPTY}


def test_60_language_with_many_distinct_residuals():
    r = parse("(a|b)*a(a|b)(a|b)")  # the third symbol from the end is a: 8 residuals
    dfa = r.residual_automaton()
    assert len(dfa.states) == 8
    check_language(dfa, "(a|b)*a(a|b)(a|b)")
    assert len({s for s in dfa.states if s.nullable()}) == 4


# ---- language equality, not syntactic equality ---------------------------------------
def test_60_residuals_equal_as_languages_are_merged():
    r = parse("a*a*")  # D_a(a*a*) = a*a*|a*: another expression, same language as a*a*
    syntactic = r.brzozowski_dfa()
    assert len(syntactic.states) > 1  # function 50 only merges identical normal forms
    dfa = r.residual_automaton()
    assert len(dfa.states) == 1 and dfa.initial_state in dfa.final_states
    assert dfa.transitions[r] == {"a": r}
    check_language(dfa, "a*a*", symbols="a")


def test_60_same_language_as_is_semantic():
    assert parse("a*").same_language_as(parse("ε|aa*"))
    assert parse("(a|b)*").same_language_as(parse("(a*b*)*"))
    assert not parse("a*").same_language_as(parse("a*b"))
    assert parse("a").same_language_as(parse("a"), alphabet="abc")
    assert EMPTY.same_language_as(parse("a∅"))
    assert EPSILON.same_language_as(parse("ε*")) and not EPSILON.same_language_as(EMPTY)
    assert parse("ab").same_language_as(parse("ab")) and parse("a") != parse("ε|a")
    assert not parse("a").same_language_as(parse("b"))


def test_60_number_of_states_equals_the_nerode_index_of_the_language():
    for text in SAMPLES:
        r = parse(text)
        alphabet = sorted(r.alphabet()) or ["a"]
        residual = r.residual_automaton(alphabet=alphabet)
        reference = r.brzozowski_dfa(alphabet)
        assert len(residual.states) == reference.nerode_partition().index, text
        assert residual.is_isomorphic_to(reference.complete().minimize()), text
        assert residual.is_minimal(), text


def test_60_automaton_recognizes_exactly_the_language_of_the_expression():
    for text in SAMPLES:
        r = parse(text)
        alphabet = sorted(r.alphabet()) or ["a"]
        for complete in (True, False):
            dfa = r.residual_automaton(alphabet=alphabet, complete=complete)
            check_language(dfa, text, symbols=alphabet, longest=5)
            assert dfa.has_same_language_as(r.brzozowski_dfa(alphabet))


def test_60_every_state_is_a_residual_of_the_language():
    r = parse("(ab|a)(c|bc)")
    dfa = r.residual_automaton()
    for state in dfa.states:  # a state is the residual by the shortest word leading to it
        word = next(w for w in words_up_to("abc", 5) if _run(dfa, w) == state)
        derived = r
        for symbol in word:
            derived = derived.derivative(symbol)
        assert derived.same_language_as(state)


def _run(dfa, word):
    state = dfa.initial_state
    for symbol in word:
        state = dfa.transitions[state][symbol]
    return state


def test_60_distinct_states_have_distinct_languages_and_no_duplicates():
    for text in ["(a|b)*abb", "a*a*", "(ab)*|c", "a(a|b)*"]:
        dfa = parse(text).residual_automaton()
        states = list(dfa.states)
        for first, second in itertools.combinations(states, 2):
            assert not first.same_language_as(second, alphabet=dfa.input_symbols), (text, first, second)


# ---- options and errors ---------------------------------------------------------------
def test_60_alphabet_extra_symbols_and_errors():
    dfa = parse("a").residual_automaton(alphabet="abc")
    assert dfa.input_symbols == {"a", "b", "c"} and dfa.transitions[parse("a")]["c"] == EMPTY
    with pytest.raises(ValueError, match="missing"):
        parse("ab").residual_automaton(alphabet="a")
    with pytest.raises(ValueError, match="non-empty alphabet"):
        EPSILON.residual_automaton()
    with pytest.raises(RuntimeError, match="derivatives"):
        parse("(a|b)*abb").residual_automaton(max_states=2)


def test_60_expression_is_not_modified_and_result_is_a_valid_new_automaton():
    r = parse("(a|b)*abb")
    before = str(r)
    d1, d2 = r.residual_automaton(), r.residual_automaton()
    assert str(r) == before and d1 is not d2
    assert d1.initial_state in d1.states and d1.final_states <= d1.states
    assert all(set(out) == {"a", "b"} and set(out.values()) <= d1.states for out in d1.transitions.values())


# ---- DFA and NFA versions ---------------------------------------------------------------
def test_60_dfa_version_is_the_quotient_by_the_nerode_classes():
    for dfa in itertools.islice(random_dfas(), 40):
        for complete in (True, False):
            residual = dfa.residual_automaton(complete)
            assert residual.has_same_language_as(dfa)
            assert residual.is_minimal()
            if complete:
                assert residual.is_complete()
                assert len(residual.states) == dfa.nerode_partition().index
                assert residual.is_isomorphic_to(dfa.complete().minimize())


def test_60_dfa_version_merges_equivalent_states():
    dfa = make_dfa(
        {"p": {"a": "p2", "b": "q"}, "p2": {"a": "p", "b": "q2"}, "q": {"a": "p2", "b": "q2"}, "q2": {"a": "p", "b": "q"}},
        "p", {"q", "q2"},
    )
    residual = dfa.residual_automaton()
    assert len(residual.states) == 2
    assert residual.initial_state == frozenset({"p", "p2"}) and residual.final_states == {frozenset({"q", "q2"})}


def test_60_nfa_version_goes_through_determinization():
    for nfa in itertools.islice(random_nfas(), 40):
        residual = nfa.residual_automaton()
        assert isinstance(residual, ExtendedDFA) and residual.is_complete()
        for w in words_up_to("ab", 4):
            from test_regex import nfa_accepts

            assert dfa_accepts(residual, w) == nfa_accepts(nfa, w)
        assert len(residual.states) == nfa.to_dfa(complete=True).nerode_partition().index


def test_60_residual_by_word_is_the_residual_language():
    for dfa in itertools.islice(random_dfas(), 40):
        for u in words_up_to("ab", 3):
            rest = dfa.residual_by_word(u)
            for w in words_up_to("ab", 4):
                assert dfa_accepts(rest, w) == dfa_accepts(dfa, u + w), (u, w)
            assert rest.initial_state in rest.states
    dfa = make_dfa({"q0": {"a": "q1"}, "q1": {}}, "q0", {"q1"})
    dead = dfa.residual_by_word("b")  # falls off the partial automaton
    assert dead.states == {"q0"} and dead.final_states == set()
    assert dfa.residual_by_word("a").final_states == {"q1"} and dfa.residual_by_word("a").initial_state == "q1"
    with pytest.raises(ValueError, match="not a symbol of the alphabet"):
        dfa.residual_by_word("z")


def test_60_residual_by_word_agrees_with_the_derivative_of_the_expression():
    r = parse("(a|b)*abb")
    dfa = r.brzozowski_dfa()
    for u in words_up_to("ab", 4):
        derived = r
        for symbol in u:
            derived = derived.derivative(symbol)
        rest = dfa.residual_by_word(u)
        assert rest.has_same_language_as(derived.brzozowski_dfa(["a", "b"]))


def test_60_residual_by_word_keeps_only_the_states_reachable_from_the_new_initial_state():
    dfa = make_dfa({"q0": {"a": "q1"}, "q1": {"b": "q2"}, "q2": {}, "x": {"a": "q0"}}, "q0", {"q2"})
    rest = dfa.residual_by_word("a")
    assert rest.states == {"q1", "q2"} and rest.initial_state == "q1"
    assert rest.transitions == {"q1": {"b": "q2"}, "q2": {}}
    assert dfa.residual_by_word("").states == {"q0", "q1", "q2"}  # "x" is not reachable
    assert dfa.residual_by_word("ab").states == {"q2"}
