"""Tests of function 61: morphisms of automata."""

from dataclasses import FrozenInstanceError

import pytest

from automata_extensions.fa import ExtendedDFA
from automata_extensions.fa.morphism import (
    AutomatonMorphism, MorphismError, Violation, morphism_violations,
)
from test_grammar import make_dfa, make_nfa, random_dfas
from test_regex import dfa_accepts, words_up_to


def ends_with_b():
    return make_dfa({"p": {"a": "p", "b": "q"}, "q": {"a": "p", "b": "q"}}, "p", {"q"})


def duplicated():
    return make_dfa(
        {"p": {"a": "p2", "b": "q"}, "p2": {"a": "p", "b": "q2"},
         "q": {"a": "p2", "b": "q2"}, "q2": {"a": "p", "b": "q"}},
        "p", {"q", "q2"},
    )


DUP_TO_MIN = {"p": "p", "p2": "p", "q": "q", "q2": "q"}


def kinds(violations):
    return sorted(v.kind for v in violations)


def same_language(a, b, n=6):
    return all(dfa_accepts(a, w) == dfa_accepts(b, w) for w in words_up_to("ab", n))


def included(a, b, n=6):
    return all(dfa_accepts(b, w) for w in words_up_to("ab", n) if dfa_accepts(a, w))


# ---- valid morphisms ------------------------------------------------------------------
def test_61_valid_non_injective_morphism():
    a, b = duplicated(), ends_with_b()
    assert a.morphism_violations(b, DUP_TO_MIN) == ()
    assert a.is_morphism_to(b, DUP_TO_MIN)
    h = a.morphism_to(b, DUP_TO_MIN)
    assert isinstance(h, AutomatonMorphism)
    assert not h.is_injective and h.is_surjective and not h.is_bijective
    assert not h.is_isomorphism
    assert h("p2") == "p" and h.image() == {"p", "q"}
    assert h.kernel() == {frozenset({"p", "p2"}), frozenset({"q", "q2"})}
    assert same_language(a, b)


def test_61_identity_is_an_isomorphism():
    a = duplicated()
    h = a.morphism_to(a, {q: q for q in a.states}, strict=True)
    assert h.is_bijective and h.is_isomorphism and h.strict


def test_61_renaming_is_isomorphism_and_agrees_with_is_isomorphic_to():
    a = ends_with_b()
    b = make_dfa({"x": {"a": "x", "b": "y"}, "y": {"a": "x", "b": "y"}}, "x", {"y"})
    h = a.morphism_to(b, {"p": "x", "q": "y"})
    assert h.is_isomorphism and a.is_isomorphic_to(b)


def test_61_morphism_vs_isomorphism_vs_strict():
    """Bijective morphism that is not an isomorphism: final state not reflected."""
    a = make_dfa({"s": {"a": "t"}, "t": {"a": "t"}}, "s", {"t"}, symbols="a")
    b = make_dfa({"s": {"a": "t"}, "t": {"a": "t"}}, "s", {"s", "t"}, symbols="a")
    m = {"s": "s", "t": "t"}
    h = a.morphism_to(b, m)
    assert h.is_bijective and not h.is_isomorphism  # morphism, not isomorphism
    assert kinds(a.morphism_violations(b, m, strict=True)) == ["strict-final"]
    assert not a.is_isomorphic_to(b)
    assert included(a, b) and not same_language(a, b)


# ---- invalid maps, explicit messages ---------------------------------------------------
def test_61_transitions_not_preserved():
    a, b = duplicated(), ends_with_b()
    bad = {**DUP_TO_MIN, "p2": "q"}  # p -a-> p2 gives q but b: p -a-> p
    v = a.morphism_violations(b, bad)
    assert "transition" in kinds(v)
    text = " ".join(str(x) for x in v)
    assert "'p'" in text and "'q'" in text and "--'a'-->" in text
    with pytest.raises(MorphismError, match="transition") as err:
        a.morphism_to(b, bad)
    assert all(isinstance(x, Violation) for x in err.value.violations)
    assert isinstance(err.value, ValueError)


def test_61_initial_not_preserved():
    a, b = ends_with_b(), ends_with_b()
    v = a.morphism_violations(b, {"p": "q", "q": "q"})
    assert "initial" in kinds(v)
    assert any("initial state of the target is 'p'" in x.message for x in v if x.kind == "initial")


def test_61_finals_not_preserved():
    a, b = ends_with_b(), ends_with_b()
    v = a.morphism_violations(b, {"p": "p", "q": "p"})
    assert "final" in kinds(v)
    assert "'q' is final but h('q') = 'p' is not final" in " ".join(str(x) for x in v)


def test_61_incompatible_alphabets():
    a = ends_with_b()
    b = make_dfa({"p": {"a": "p", "c": "p"}}, "p", {"p"}, symbols="ac")
    v = a.morphism_violations(b, {"p": "p", "q": "p"})
    assert kinds(v) == ["alphabet"]
    assert "['b']" in v[0].message and "['c']" in v[0].message
    assert not a.is_morphism_to(b, {"p": "p", "q": "p"})


def test_61_domain_and_codomain_errors():
    a, b = ends_with_b(), ends_with_b()
    assert kinds(a.morphism_violations(b, {"p": "p"})) == ["domain"]
    assert kinds(a.morphism_violations(b, {"p": "p", "q": "q", "z": "p"})) == ["domain"]
    v = a.morphism_violations(b, {"p": "p", "q": "nowhere"})
    assert kinds(v) == ["codomain"] and "'nowhere'" in v[0].message
    with pytest.raises(TypeError, match="mapping"):
        a.morphism_violations(b, [("p", "p")])


def test_61_all_violations_are_reported_together():
    a, b = ends_with_b(), ends_with_b()
    v = a.morphism_violations(b, {"p": "q", "q": "p"})
    assert {"initial", "final", "transition"} <= set(kinds(v))


# ---- partial DFAs ---------------------------------------------------------------------
def test_61_partial_source_unconstrained_where_undefined():
    """Nothing is required where delta_A is undefined: B may have more transitions."""
    a = make_dfa({"s": {"a": "t"}, "t": {}}, "s", {"t"})
    b = make_dfa({"s": {"a": "t", "b": "s"}, "t": {"a": "t", "b": "t"}}, "s", {"t"})
    m = {"s": "s", "t": "t"}
    assert a.morphism_violations(b, m) == ()
    assert included(a, b)
    # strict forbids the extra transitions of B
    v = a.morphism_violations(b, m, strict=True)
    assert set(kinds(v)) == {"strict-transition"}
    assert any("has no 'b'-transition" in x.message for x in v)


def test_61_missing_transitions_in_target():
    a = make_dfa({"s": {"a": "t", "b": "t"}, "t": {}}, "s", {"t"})
    b = make_dfa({"s": {"a": "t"}, "t": {}}, "s", {"t"})
    v = a.morphism_violations(b, {"s": "s", "t": "t"})
    assert kinds(v) == ["missing-transition"]
    assert "no transition from h('s') = 's' on 'b'" in v[0].message
    with pytest.raises(MorphismError, match="missing-transition"):
        a.morphism_to(b, {"s": "s", "t": "t"})


def test_61_partial_both_sides_strict_isomorphism():
    a = make_dfa({"s": {"a": "t"}, "t": {}}, "s", {"t"})
    b = make_dfa({"u": {"a": "v"}, "v": {}}, "u", {"v"})
    h = a.morphism_to(b, {"s": "u", "t": "v"}, strict=True)
    assert h.is_isomorphism and a.is_isomorphic_to(b)


def test_61_partial_target_lacking_transition_is_not_completed_silently():
    """The sink of complete() is not invented: the map must target real states."""
    a = make_dfa({"s": {"a": "s"}}, "s", {"s"}, symbols="a")
    b = make_dfa({"s": {}}, "s", {"s"}, symbols="a")
    assert kinds(a.morphism_violations(b, {"s": "s"})) == ["missing-transition"]
    assert a.morphism_violations(a.complete(), {"s": "s"}) == ()


# ---- strict morphism => equal languages --------------------------------------------------
def test_61_strict_morphism_preserves_language_random():
    count = 0
    for a in random_dfas(120, seed=31):
        full = a.complete()
        min_h = full.minimization_morphism() if hasattr(full, "minimization_morphism") else None
        assert min_h is not None and min_h.strict
        # the map is defined on the accessible part only; check on that part
        reach = full.induced_subautomaton(full.accessible_states())
        assert reach.morphism_violations(min_h.target, dict(min_h.mapping), strict=True) == ()
        assert same_language(a, min_h.target)
        count += 1
    assert count == 120


def test_61_non_strict_morphism_only_gives_inclusion():
    for a in random_dfas(60, seed=32):
        full = a.complete()
        b = make_dfa({"z": {"a": "z", "b": "z"}}, "z", {"z"})
        h = full.morphism_to(b, {q: "z" for q in full.states})
        assert included(full, b)
        assert h.kernel() == {frozenset(full.states)}


# ---- quotient / minimization morphisms -------------------------------------------------
def test_61_minimization_morphism():
    a = duplicated()
    h = a.minimization_morphism()
    assert h.strict and h.is_surjective and not h.is_injective
    assert len(h.target.states) == 2
    assert h.target.has_same_language_as(a)
    assert h.kernel() == {frozenset({"p", "p2"}), frozenset({"q", "q2"})}
    assert h.target.is_isomorphic_to(a.minimize())


def test_61_minimal_dfa_gives_isomorphism():
    a = ends_with_b()
    h = a.minimization_morphism()
    assert h.is_isomorphism


def test_61_quotient_by_valid_and_invalid_partitions():
    a = duplicated()
    h = a.quotient_by([{"p", "p2"}, {"q", "q2"}])
    assert h.strict and len(h.target.states) == 2
    with pytest.raises(ValueError, match="final"):
        a.quotient_by([{"p", "q"}, {"p2", "q2"}])
    with pytest.raises(ValueError, match="empty"):
        a.quotient_by([{"p", "p2"}, {"q", "q2"}, set()])
    with pytest.raises(ValueError):
        a.quotient_by([{"p", "p2"}, {"p", "q", "q2"}])
    with pytest.raises(ValueError):
        a.quotient_by([{"p", "p2"}])
    with pytest.raises(ValueError):
        a.quotient_by([{"p", "p2", "zz"}, {"q", "q2"}])
    # not a congruence: successors of y and z fall in different classes
    cyc = make_dfa({"x": {"a": "y"}, "y": {"a": "z"}, "z": {"a": "x"}}, "x", {"x"}, symbols="a")
    with pytest.raises(ValueError):
        cyc.quotient_by([{"x"}, {"y", "z"}])  # y->z, z->x: successors in different classes


# ---- composition, NFA, immutability ------------------------------------------------------
def test_61_composition():
    a, b = duplicated(), ends_with_b()
    h1 = a.morphism_to(b, DUP_TO_MIN, strict=True)
    ren = make_dfa({"x": {"a": "x", "b": "y"}, "y": {"a": "x", "b": "y"}}, "x", {"y"})
    h2 = b.morphism_to(ren, {"p": "x", "q": "y"}, strict=True)
    h = h1.compose(h2)
    assert h("p2") == "x" and h("q2") == "y" and h.strict
    assert a.morphism_violations(ren, dict(h.mapping), strict=True) == ()
    with pytest.raises(ValueError, match="source of the second"):
        h1.compose(h1)  # target of h1 has states {p, q}, source of h1 has four


def test_61_nfa_morphism_with_epsilon():
    a = make_nfa({"s": {"": {"t"}, "a": {"t"}}, "t": {}}, "s", {"t"})
    b = make_nfa({"u": {"": {"v"}, "a": {"v", "u"}}, "v": {}}, "u", {"v"})
    assert a.morphism_violations(b, {"s": "u", "t": "v"}) == ()
    c = make_nfa({"u": {"a": {"v"}}, "v": {}}, "u", {"v"})
    v = a.morphism_violations(c, {"s": "u", "t": "v"})
    assert kinds(v) == ["missing-transition"] and "ε" in v[0].message
    s = a.morphism_violations(b, {"s": "u", "t": "v"}, strict=True)
    assert kinds(s) == ["strict-transition"]  # b has the extra u -a-> u


def test_61_nfa_target_choice_must_contain_image():
    a = make_nfa({"s": {"a": {"t"}}, "t": {}}, "s", {"t"})
    b = make_nfa({"s": {"a": {"t", "w"}}, "t": {}, "w": {}}, "s", {"t"})
    assert a.morphism_violations(b, {"s": "s", "t": "t"}) == ()
    v = a.morphism_violations(b, {"s": "s", "t": "w"})
    assert kinds(v) == ["final"]  # w not final; the transition itself is fine


def test_61_immutability_and_originals_untouched():
    a, b = duplicated(), ends_with_b()
    snap = (a.states, a.transitions, b.states, b.transitions)
    h = a.morphism_to(b, DUP_TO_MIN)
    mapping = dict(DUP_TO_MIN)
    mapping["p"] = "zzz"  # caller mutating its dict must not affect h
    assert h("p") == "p"
    with pytest.raises(TypeError):
        h.mapping["p"] = "q"
    with pytest.raises(FrozenInstanceError):
        h.strict = True
    with pytest.raises(ValueError, match="not a state"):
        h("nope")
    assert snap == (a.states, a.transitions, b.states, b.transitions)
    a.minimization_morphism()
    assert snap[:2] == (a.states, a.transitions)


def test_61_module_function_works_on_plain_automata_too():
    a, b = ends_with_b(), ends_with_b()
    assert morphism_violations(a, b, {"p": "p", "q": "q"}) == ()
    assert str(Violation("x", "msg")) == "[x] msg"


def test_61_isomorphism_requires_surjectivity():
    """Injective strict morphism into a bigger automaton (extra state) is not an isomorphism."""
    a = ends_with_b()
    b = make_dfa(
        {"p": {"a": "p", "b": "q"}, "q": {"a": "p", "b": "q"}, "extra": {"a": "extra"}},
        "p", {"q"},
    )
    m = {"p": "p", "q": "q"}
    assert a.morphism_violations(b, m, strict=True) == ()
    h = a.morphism_to(b, m, strict=True)
    assert h.is_injective and not h.is_surjective
    assert not h.is_isomorphism and not a.is_isomorphic_to(b)
