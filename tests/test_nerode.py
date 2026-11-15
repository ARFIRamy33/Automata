"""Tests of function 59: Myhill-Nerode partition (classes of words)."""

import itertools
import random
from dataclasses import FrozenInstanceError

import pytest

from automata_extensions.fa import ExtendedDFA
from automata_extensions.regex import EMPTY, Regex
from test_grammar import make_dfa, random_dfas
from test_regex import dfa_accepts, oracle, words_up_to


def accepts(dfa, word):
    return dfa_accepts(dfa, word)  # a missing transition rejects


def future(dfa, word, symbols, longest):
    """Independent oracle: the set of suffixes (<= longest) that complete ``word`` into L."""
    return frozenset(w for w in words_up_to(symbols, longest) if accepts(dfa, word + w))


def state_future(dfa, state, symbols, longest):
    probe = type(dfa)(
        states=dfa.states, input_symbols=dfa.input_symbols, transitions=dfa.transitions,
        initial_state=state, final_states=dfa.final_states, allow_partial=True,
    )
    return frozenset(w for w in words_up_to(symbols, longest) if dfa_accepts(probe, w))


# ---- sample automata ---------------------------------------------------------------
def chain_all_distinguishable():
    """a^n for n = 0..3 then loop: counts the a's, every state is different."""
    return make_dfa(
        {"s0": {"a": "s1", "b": "s0"}, "s1": {"a": "s2", "b": "s0"},
         "s2": {"a": "s3", "b": "s0"}, "s3": {"a": "s3", "b": "s0"}},
        "s0", {"s3"},
    )


def duplicated_states():
    """Words over {a,b} ending with b, with every state duplicated (6 states, 2 classes)."""
    return make_dfa(
        {"p": {"a": "p2", "b": "q"}, "p2": {"a": "p", "b": "q2"},
         "q": {"a": "p2", "b": "q2"}, "q2": {"a": "p", "b": "q"}},
        "p", {"q", "q2"},
    )


def minimal_even_a():
    return make_dfa({"e": {"a": "o", "b": "e"}, "o": {"a": "e", "b": "o"}}, "e", {"e"})


def incomplete():
    """ab then anything in {a,b}* only after: partial automaton, no sink state."""
    return make_dfa({"q0": {"a": "q1"}, "q1": {"b": "q2"}, "q2": {}}, "q0", {"q2"})


# ---- basic behaviour ---------------------------------------------------------------
def test_59_all_states_distinguishable():
    dfa = chain_all_distinguishable()
    part = dfa.nerode_partition()
    assert len(part) == part.index == 4
    assert part.representatives() == ("", "a", "aa", "aaa")
    assert all(len(c.states) == 1 for c in part)
    assert [c.is_final for c in part] == [False, False, False, True]
    assert part.class_of_word("baaa").index == 3 and part.class_of_word("aaab").index == 0
    assert part.class_of_word("aab").index == 0 and part.class_of_word("aaaa").index == 3


def test_59_several_equivalent_states_are_merged():
    dfa = duplicated_states()
    part = dfa.nerode_partition()
    assert part.index == 2
    assert {c.states for c in part} == {frozenset({"p", "p2"}), frozenset({"q", "q2"})}
    assert part.representatives() == ("", "b")
    assert part.class_of_word("ab") == part.class_of_word("b") == part.class_of_word("bab")
    assert part.are_equivalent("aa", "") and not part.are_equivalent("a", "b")


def test_59_minimal_automaton_has_one_state_per_class():
    dfa = minimal_even_a()
    assert dfa.is_minimal() and dfa.nerode_partition().index == len(dfa.states)
    assert dfa.nerode_partition().are_equivalent("aa", "") and not dfa.nerode_partition().are_equivalent("a", "")


def test_59_non_minimal_automaton_has_fewer_classes_than_states():
    dfa = duplicated_states()
    assert not dfa.is_minimal() and dfa.nerode_partition().index < len(dfa.states)
    assert dfa.nerode_partition().index == len(dfa.minimize().states)


def test_59_reuses_function_30_exactly():
    for dfa in itertools.islice(random_dfas(), 40):
        full = dfa.complete()
        assert {c.states for c in dfa.nerode_partition()} == set(full.myhill_nerode_classes(accessible_only=True))


# ---- incomplete DFA: implicit sink --------------------------------------------------
def test_59_incomplete_automaton_gets_an_implicit_dead_class():
    dfa = incomplete()
    assert not dfa.is_complete()
    part = dfa.nerode_partition()
    assert part.index == 4  # ε, a, ab and the dead class
    dead = [c for c in part if c.is_dead]
    assert len(dead) == 1 and dead[0].representative == "b"  # shortest word that falls off
    assert any(s not in dfa.states for s in dead[0].states)  # it contains the added sink
    assert part.class_of_word("abb") == part.class_of_word("ba") == dead[0]
    assert part.class_of_word("ab").is_final and not part.class_of_word("a").is_final
    assert dfa.is_complete() is False  # the automaton itself is untouched


def test_59_complete_automaton_is_not_extended():
    dfa = minimal_even_a()
    part = dfa.nerode_partition()
    assert all(s in dfa.states for c in part for s in c.states)
    assert not any(c.is_dead for c in part)


def test_59_dead_states_of_a_complete_dfa_form_one_dead_class():
    dfa = make_dfa(
        {"s": {"a": "t", "b": "d1"}, "t": {"a": "d2", "b": "d1"}, "d1": {"a": "d2", "b": "d1"}, "d2": {"a": "d1", "b": "d2"}},
        "s", {"t"},
    )
    part = dfa.nerode_partition()
    dead = [c for c in part if c.is_dead]
    assert len(dead) == 1 and dead[0].states == {"d1", "d2"}


def test_59_empty_and_universal_languages():
    empty = make_dfa({"q": {"a": "q"}}, "q", set())
    p = empty.nerode_partition()
    assert p.index == 1 and p.classes[0].is_dead and not p.classes[0].is_final
    universal = make_dfa({"q": {"a": "q", "b": "q"}}, "q", {"q"})
    p = universal.nerode_partition()
    assert p.index == 1 and p.classes[0].is_final and not p.classes[0].is_dead
    assert p.residual_regex(p.classes[0]).same_language_as(Regex.parse("(a|b)*"))
    assert empty.nerode_partition().residual_regex(empty.nerode_partition().classes[0]) == EMPTY


# ---- the property of the statement: same class <=> same future behaviour -----------------
def test_59_two_words_are_equivalent_iff_their_futures_are_equal():
    for dfa in itertools.islice(random_dfas(), 60):
        part = dfa.nerode_partition()
        bound = len(dfa.states) + 2  # a separating suffix has length < number of classes
        words = list(words_up_to("ab", 3))
        futures = {w: future(dfa, w, "ab", bound) for w in words}
        for u, v in itertools.product(words, repeat=2):
            assert part.are_equivalent(u, v) == (futures[u] == futures[v]), (u, v)


def test_59_states_of_one_class_have_the_same_future_and_other_classes_differ():
    for dfa in itertools.islice(random_dfas(), 60):
        full = dfa.complete()
        part = dfa.nerode_partition()
        bound = len(full.states) + 1
        futures = [{state_future(full, s, "ab", bound) for s in c.states} for c in part]
        assert all(len(f) == 1 for f in futures)  # same behaviour inside a class
        distinct = [next(iter(f)) for f in futures]
        assert len(set(distinct)) == len(distinct)  # and different between classes


def test_59_separating_suffix_really_separates_and_is_shortest():
    for dfa in itertools.islice(random_dfas(), 40):
        part = dfa.nerode_partition()
        words = list(words_up_to("ab", 3))
        for u, v in itertools.product(words, repeat=2):
            suffix = part.separating_suffix(u, v)
            if part.are_equivalent(u, v):
                assert suffix is None
                continue
            assert accepts(dfa, u + suffix) != accepts(dfa, v + suffix)
            shorter = [w for w in words_up_to("ab", len(suffix) - 1) if accepts(dfa, u + w) != accepts(dfa, v + w)]
            assert not shorter


def test_59_representatives_are_the_shortest_words_of_their_class():
    for dfa in itertools.islice(random_dfas(), 40):
        part = dfa.nerode_partition()
        first_seen = {}
        for w in words_up_to("ab", 6):  # shortlex enumeration
            first_seen.setdefault(part.class_of_word(w).index, w)
        for c in part:
            assert c.representative == first_seen[c.index] or len(c.representative) > 6
        assert part.classes[0].representative == ""


def test_59_transition_between_classes_is_well_defined():
    for dfa in itertools.islice(random_dfas(), 40):
        full = dfa.complete()
        part = dfa.nerode_partition()
        for c in part:
            for symbol in "ab":
                targets = {part.class_of_state(full.transitions[s][symbol]) for s in c.states}
                assert len(targets) == 1 and part.transition(c, symbol) in targets
        for w in words_up_to("ab", 3):
            for symbol in "ab":
                assert part.class_of_word(w + symbol) == part.transition(part.class_of_word(w), symbol)


# ---- residuals and quotient --------------------------------------------------------
def test_59_residual_of_each_class_is_the_set_of_good_suffixes():
    for dfa in itertools.islice(random_dfas(), 30):
        part = dfa.nerode_partition()
        for c in part:
            regex = part.residual_regex(c)
            for w in words_up_to("ab", 4):
                assert oracle(str(regex), w) == accepts(dfa, c.representative + w)


def test_59_quotient_automaton_is_the_minimal_dfa():
    for dfa in itertools.islice(random_dfas(), 40):
        part = dfa.nerode_partition()
        quotient = part.quotient_automaton()
        assert quotient.is_complete() and quotient.has_same_language_as(dfa)
        assert len(quotient.states) == part.index and quotient.is_minimal()
        assert quotient.is_isomorphic_to(dfa.complete().minimize())
        partial = part.quotient_automaton(complete=False)
        assert partial.has_same_language_as(dfa)
        # the dead class disappears from a partial DFA (a single initial state stays if L = ∅)
        assert len(partial.states) == max(1, part.index - sum(c.is_dead for c in part))


# ---- validation, immutability, display ---------------------------------------------
def test_59_errors_and_immutability():
    dfa = duplicated_states()
    part = dfa.nerode_partition()
    with pytest.raises(ValueError, match="not a symbol"):
        part.class_of_word("az")
    with pytest.raises(ValueError, match="not a symbol"):
        part.transition(part.classes[0], "z")
    with pytest.raises(ValueError, match="not an accessible state"):
        part.class_of_state("nowhere")
    with pytest.raises(FrozenInstanceError):
        part.classes = ()
    with pytest.raises(FrozenInstanceError):
        part.classes[0].representative = "x"
    assert isinstance(part.classes, tuple) and list(part) == list(part.classes)


def test_59_automaton_is_not_modified_and_inaccessible_states_are_ignored():
    dfa = make_dfa({"q0": {"a": "q0", "b": "q0"}, "lost": {"a": "q0", "b": "lost"}}, "q0", {"q0"})
    before = (set(dfa.states), {s: dict(t) for s, t in dfa.transitions.items()})
    part = dfa.nerode_partition()
    assert part.index == 1 and "lost" not in part.classes[0].states
    assert (set(dfa.states), {s: dict(t) for s, t in dfa.transitions.items()}) == before
    with pytest.raises(ValueError, match="not an accessible state"):
        part.class_of_state("lost")


def test_59_text_display_lists_every_class():
    text = str(duplicated_states().nerode_partition())
    assert text.count("\n") == 1 and "'ε'" in text and "(final)" in text
