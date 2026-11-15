"""Tests of function 62: transition monoid and syntactic monoid."""

import itertools
from dataclasses import FrozenInstanceError

import pytest

from automata_extensions.fa import ExtendedDFA
from automata_extensions.fa.monoid import SyntacticMonoid, TransitionMonoid
from test_grammar import make_dfa, make_nfa, random_dfas
from test_regex import dfa_accepts, words_up_to


def brute_equivalent(dfa, u, v, context=4):
    """Independent oracle: u ~ v iff no context x_y (|x|,|y| <= context) separates them."""
    ctx = list(words_up_to("ab", context))
    return all(
        dfa_accepts(dfa, x + u + y) == dfa_accepts(dfa, x + v + y) for x in ctx for y in ctx
    )


def empty_language():
    return make_dfa({"q0": {"a": "q0", "b": "q0"}}, "q0", set())


def universal_language():
    return make_dfa({"q0": {"a": "q0", "b": "q0"}}, "q0", {"q0"})


def single_word():
    """L = {ab}."""
    return make_dfa({"q0": {"a": "q1"}, "q1": {"b": "q2"}, "q2": {}}, "q0", {"q2"})


def simple_loop():
    """L = a* over {a}."""
    return make_dfa({"q0": {"a": "q0"}}, "q0", {"q0"}, symbols="a")


def even_a():
    return make_dfa({"e": {"a": "o", "b": "e"}, "o": {"a": "e", "b": "o"}}, "e", {"e"})


def duplicated():
    """Words ending with b; states duplicated, so T(A) is bigger than M(L)."""
    return make_dfa(
        {"p": {"a": "p2", "b": "q"}, "p2": {"a": "p", "b": "q2"},
         "q": {"a": "p2", "b": "q2"}, "q2": {"a": "p", "b": "q"}},
        "p", {"q", "q2"},
    )


# ---- required cases ---------------------------------------------------------------
def test_62_empty_language_is_trivial_monoid():
    m = empty_language().syntactic_monoid()
    assert len(m) == 1 and m.element(0).word == ""
    assert m.accepting_elements() == frozenset()
    assert not m.recognizes("") and not m.recognizes("ab")


def test_62_universal_language_is_trivial_monoid():
    m = universal_language().syntactic_monoid()
    assert len(m) == 1
    assert m.accepting_elements() == frozenset({0}) and m.recognizes("abba")


def test_62_single_word_language():
    m = single_word().syntactic_monoid()
    # M({ab}) = {1, a, b, ab, 0}; aa = ba = bb = 0 (zero: no context makes it accepted)
    assert len(m) == 5
    assert [e.word for e in m] == ["", "a", "b", "aa", "ab"]
    zero = m.element_of_word("aa")
    assert m.element_of_word("ba") == m.element_of_word("bb") == zero
    assert m.element_of_word("b") != zero and m.element_of_word("a") != zero
    assert all(m.product(zero, x) == zero == m.product(x, zero) for x in m)
    assert m.recognizes("ab") and not m.recognizes("a") and not m.recognizes("aba")
    assert m.accepting_elements() == frozenset({m.element_of_word("ab").index})


def test_62_simple_loop():
    m = simple_loop().syntactic_monoid()
    assert len(m) == 1 and m.recognizes("") and m.recognizes("aaa")
    assert m.alphabet == ("a",)


def test_62_several_transformations():
    m = even_a().syntactic_monoid()
    assert len(m) == 2  # b acts as the identity: Z/2
    assert m.element_of_word("b") == m.element_of_word("")
    assert m.element_of_word("aa") == m.element_of_word("")
    assert m.element_of_word("a") != m.element_of_word("")
    a = m.element_of_word("a")
    assert m.product(a, a) == m.element_of_word("")
    assert m.multiplication_table() == ((0, 1), (1, 0))


def test_62_transition_monoid_is_not_the_syntactic_monoid():
    dfa = duplicated()
    t, m = dfa.transition_monoid(), dfa.syntactic_monoid()
    assert len(t) == 9 and len(m) == 3
    assert m.transition_monoid == t
    assert len(m.projection) == 9 and set(m.projection) == {0, 1, 2}
    assert m.verify_projection()
    # two different transformations, one syntactic element: a (words a / ba)
    assert t.element_of_word("a") != t.element_of_word("ba")
    assert m.element_of_word("a") == m.element_of_word("ba")


def test_62_syntactic_monoid_equals_transition_monoid_of_minimal_dfa():
    dfa = duplicated()
    minimal = dfa.minimize()
    assert len(dfa.syntactic_monoid()) == len(minimal.transition_monoid())
    # and the monoid of a minimal automaton is already syntactic
    assert len(minimal.syntactic_monoid()) == len(minimal.transition_monoid())


def test_62_neutral_element():
    for dfa in (single_word(), even_a(), duplicated(), empty_language()):
        for m in (dfa.syntactic_monoid(), dfa.transition_monoid()):
            e = m.neutral_element()
            assert e is not None and e.word == "" and e.index == 0
            assert all(m.product(e, x) == x == m.product(x, e) for x in m)


def test_62_closure_and_associativity():
    for dfa in (single_word(), even_a(), duplicated(), empty_language(), simple_loop()):
        for m in (dfa.syntactic_monoid(), dfa.transition_monoid()):
            assert m.is_closed() and m.is_associative() and m.is_monoid()
            n = len(m)
            table = m.multiplication_table()
            assert all(0 <= c < n for row in table for c in row)


def test_62_product_follows_concatenation():
    dfa = single_word()
    m = dfa.syntactic_monoid()
    for u, v in itertools.product(words_up_to("ab", 3), repeat=2):
        assert m.product(m.element_of_word(u), m.element_of_word(v)) == m.element_of_word(u + v)


def test_62_elements_are_the_true_syntactic_classes():
    """Oracle: brute-force contexts, on every language of a random sample."""
    for dfa in list(random_dfas(60, seed=21)) + [single_word(), even_a(), duplicated()]:
        m = dfa.syntactic_monoid()
        probe = list(words_up_to("ab", 3))
        for u, v in itertools.product(probe, repeat=2):
            same = m.element_of_word(u) == m.element_of_word(v)
            assert same == brute_equivalent(dfa, u, v), (u, v)
        # every element is reached by its recorded word
        for e in m:
            assert m.element_of_word(e.word) == e


def test_62_number_of_elements_matches_distinct_classes_of_words():
    for dfa in random_dfas(40, seed=22):
        m = dfa.syntactic_monoid()
        classes = []
        for u in words_up_to("ab", 4):
            if not any(brute_equivalent(dfa, u, w) for w in classes):
                classes.append(u)
        assert len(classes) <= len(m)  # short words reach at most all elements
        reached = {m.element_of_word(u).index for u in words_up_to("ab", 4)}
        assert len(reached) == len(classes)


def test_62_language_recognized_by_the_monoid():
    for dfa in random_dfas(60, seed=23):
        m = dfa.syntactic_monoid()
        for w in words_up_to("ab", 5):
            assert m.recognizes(w) == dfa_accepts(dfa, w)


def test_62_syntactic_is_a_quotient_of_transition_monoid():
    for dfa in random_dfas(60, seed=24):
        m = dfa.syntactic_monoid()
        assert len(m) <= len(m.transition_monoid)
        assert m.verify_projection()
        assert m.is_monoid()


def test_62_same_language_same_syntactic_monoid():
    dfa = duplicated()
    other = dfa.minimize()
    a, b = dfa.syntactic_monoid(), other.syntactic_monoid()
    assert len(a) == len(b)
    assert a.multiplication_table() == b.multiplication_table()
    assert a.accepting_elements() == b.accepting_elements()


def test_62_incomplete_dfa_uses_implicit_sink():
    m = single_word().syntactic_monoid()
    assert m.transition_monoid.acted  # sink included in the acted states
    assert len(single_word().transition_monoid()) == 5


def test_62_no_enumeration_of_sigma_star_large_language():
    """A 12-state counter: the closure is cheap and exact (cyclic group of order 12)."""
    table = {f"c{i}": {"a": f"c{(i + 1) % 12}", "b": f"c{i}"} for i in range(12)}
    m = make_dfa(table, "c0", {"c0"}).syntactic_monoid()
    assert len(m) == 12
    assert m.element_of_word("a" * 12) == m.element_of_word("")
    assert m.element_of_word("a" * 5) != m.element_of_word("")


def test_62_nfa_goes_through_determinization():
    nfa = make_nfa({"q0": {"a": {"q0", "q1"}, "b": {"q0"}}, "q1": {}}, "q0", {"q1"})  # ends with a
    m = nfa.syntactic_monoid()
    assert m.recognizes("ba") and not m.recognizes("ab")
    assert len(m) == len(nfa.to_dfa(complete=True).syntactic_monoid())
    assert len(nfa.transition_monoid()) >= len(m)


def test_62_explanations_mention_the_quotient():
    dfa = duplicated()
    text = dfa.syntactic_monoid().explain()
    assert "quotient" in text and "3" in text and "9" in text
    assert "depends on the automaton" in dfa.transition_monoid().explain()


def test_62_errors_and_limits():
    m = even_a().syntactic_monoid()
    with pytest.raises(ValueError, match="not a letter"):
        m.element_of_word("c")
    with pytest.raises(ValueError, match="too large"):
        m.multiplication_table(max_size=1)
    with pytest.raises(ValueError, match="too large"):
        m.is_associative(max_size=1)
    table = {f"c{i}": {"a": f"c{(i + 1) % 12}", "b": f"c{i}"} for i in range(12)}
    with pytest.raises(RuntimeError, match="more than 5"):
        make_dfa(table, "c0", {"c0"}).syntactic_monoid(max_elements=5)


def test_62_objects_are_immutable_and_automaton_untouched():
    dfa = single_word()
    before = (dfa.states, dfa.transitions, dfa.final_states)
    m = dfa.syntactic_monoid()
    assert (dfa.states, dfa.transitions, dfa.final_states) == before
    with pytest.raises(FrozenInstanceError):
        m.alphabet = ("z",)
    with pytest.raises(FrozenInstanceError):
        m.element(0).word = "x"


def test_62_not_closed_set_is_detected():
    """A subset {1, a, b} of M({ab}) is not closed (a.a = aa is missing): detected, not hidden."""
    m = single_word().syntactic_monoid()
    kept = m.elements[:3]
    broken = type(m)(m.alphabet, m.acted, kept, m._generators, {e.action: e.index for e in kept},
                     m.partition, m.transition_monoid, ())
    assert not broken.is_closed() and not broken.is_monoid()
    with pytest.raises(ValueError, match="not closed"):
        broken.product(1, 1)
    assert broken.product(0, 1) == kept[1]  # identity . a stays inside


def test_62_verify_projection_rejects_bad_projections():
    m = duplicated().syntactic_monoid()
    def tampered(projection):
        return type(m)(m.alphabet, m.acted, m.elements, m._generators, m._index,
                       m.partition, m.transition_monoid, projection)
    assert tampered(m.projection).verify_projection()
    assert not tampered((0,) * 9).verify_projection()  # a morphism, but not surjective
    wrong = list(m.projection)
    other = next(i for i, x in enumerate(wrong) if i and x != wrong[1])
    wrong[1], wrong[other] = wrong[other], wrong[1]
    bad = tampered(tuple(wrong))
    assert wrong != list(m.projection) and not bad.verify_projection()  # surjective, not a morphism


def test_62_associativity_check_detects_a_non_associative_law():
    """Rock-paper-scissors table on 3 elements has no associativity: must be reported."""
    m = even_a().syntactic_monoid()
    class Fake(type(m)):
        def multiplication_table(self, max_size=200):
            return ((0, 1, 2), (1, 2, 0), (2, 0, 0))  # (1*1)*1 = 0 ... 1*(1*1) = 1*2 = 0
        def __len__(self):
            return 3
    fake = Fake(m.alphabet, m.acted, m.elements, m._generators, m._index)
    n = 3
    t = fake.multiplication_table()
    assert any(t[t[i][j]][k] != t[i][t[j][k]] for i in range(n) for j in range(n) for k in range(n))
    assert not fake.is_associative()
    assert m.is_associative()
