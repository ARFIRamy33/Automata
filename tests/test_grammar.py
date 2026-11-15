"""Tests of functions 55 and 56: regular grammars <-> automata, derivation trees."""

import itertools
import random
from dataclasses import FrozenInstanceError

import pytest

from automata_extensions.fa import ExtendedDFA, ExtendedNFA
from automata_extensions.grammar import (
    GrammarError,
    LeftRegularGrammar,
    NotDerivableError,
    ParseTree,
    Production,
    RegularGrammar,
)

EPS = ""


# ---------------------------------------------------------------- independent oracles
def words(symbols, longest):
    for n in range(longest + 1):
        yield from itertools.product(symbols, repeat=n)


def grammar_language(grammar, longest):
    """Oracle: rewrite sentential forms until only terminals remain (no grammar method)."""
    found, todo, seen = set(), [(grammar.start,)], set()
    while todo:
        form = todo.pop()
        if form in seen:
            continue
        seen.add(form)
        position = next((i for i, s in enumerate(form) if s in grammar.nonterminals), None)
        if position is None:
            found.add(form)
            continue
        if sum(s in grammar.terminals for s in form) > longest:
            continue
        for p in grammar.productions:
            if p.head == form[position]:
                todo.append(form[:position] + p.body + form[position + 1 :])
    return {w for w in found if len(w) <= longest}


def targets(automaton, value):
    """Destinations of one table cell: a set for an NFA, one state for a DFA."""
    if value is None:
        return set()
    return set(value) if isinstance(automaton, ExtendedNFA) else {value}


def automaton_accepts(automaton, word):
    def close(current):
        todo, seen = list(current), set(current)
        while todo:
            for t in targets(automaton, automaton.transitions.get(todo.pop(), {}).get(EPS)):
                if t not in seen:
                    seen.add(t)
                    todo.append(t)
        return seen

    current = close({automaton.initial_state})
    for symbol in word:
        step = set()
        for state in current:
            step |= targets(automaton, automaton.transitions.get(state, {}).get(symbol))
        current = close(step)
    return bool(current & set(automaton.final_states))


def automaton_language(automaton, longest):
    alphabet = sorted(s for s in automaton.input_symbols if s != EPS)
    return {w for w in words(alphabet, longest) if automaton_accepts(automaton, w)}


def make_dfa(table, initial="q0", finals=(), symbols="ab"):
    states = set(table) | {t for out in table.values() for t in out.values()} | {initial} | set(finals)
    return ExtendedDFA(
        states=states,
        input_symbols=set(symbols),
        transitions={s: dict(table.get(s, {})) for s in states},
        initial_state=initial,
        final_states=set(finals),
        allow_partial=True,
    )


def make_nfa(table, initial="q0", finals=(), symbols="ab"):
    states = set(table) | {t for out in table.values() for v in out.values() for t in v} | {initial} | set(finals)
    return ExtendedNFA(
        states=states,
        input_symbols=set(symbols),
        transitions={s: {a: frozenset(v) for a, v in table.get(s, {}).items()} for s in states},
        initial_state=initial,
        final_states=set(finals),
    )


def random_dfas(count=80, seed=5):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(1, 4)
        names = [f"s{i}" for i in range(n)]
        table = {s: {a: rng.choice(names) for a in "ab" if rng.random() < 0.75} for s in names}
        yield make_dfa(table, "s0", {s for s in names if rng.random() < 0.4})


def random_nfas(count=80, seed=9):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(1, 4)
        names = [f"s{i}" for i in range(n)]
        table = {
            s: {
                a: {t for t in names if rng.random() < 0.35}
                for a in ("a", "b", EPS)
                if rng.random() < 0.7
            }
            for s in names
        }
        table = {s: {a: v for a, v in out.items() if v} for s, out in table.items()}
        yield make_nfa(table, "s0", {s for s in names if rng.random() < 0.4}, symbols="ab")


def random_grammars(count=100, seed=3):
    rng = random.Random(seed)
    for _ in range(count):
        heads = ["S", "A", "B"][: rng.randint(1, 3)]
        rules = {}
        for h in heads:
            bodies = set()
            for _ in range(rng.randint(0, 3)):
                kind = rng.choice(["eps", "t", "tn", "tn"])
                a = rng.choice("ab")
                bodies.add({"eps": "", "t": a, "tn": a + rng.choice(heads)}[kind])
            rules[h] = sorted(bodies)
        yield RegularGrammar.from_rules("S", rules)


# ------------------------------------------------------------ function 55: automaton -> G
def test_55_dfa_to_grammar_exact_productions():
    dfa = make_dfa({"e": {"a": "o", "b": "e"}, "o": {"a": "e", "b": "o"}}, "e", {"e"})
    g = dfa.to_right_grammar()
    assert g.start == "S"
    assert g.nonterminals == {"S", "A0"}
    assert g.terminals == {"a", "b"}
    assert g.productions == {
        Production("S", ("a", "A0")),
        Production("S", ("b", "S")),
        Production("S", ()),
        Production("A0", ("a", "S")),
        Production("A0", ("b", "A0")),
    }


def test_55_one_nonterminal_per_state_and_initial_is_start():
    dfa = make_dfa({"x": {"a": "y"}, "y": {"b": "z"}, "z": {}}, "x", {"z"})
    g = dfa.to_right_grammar()
    assert len(g.nonterminals) == len(dfa.states) == 3
    assert g.start == "S"
    assert [p for p in g.productions if p.head == "S"] == [Production("S", ("a", "A0"))]


def test_55_every_final_state_gets_an_epsilon_production():
    dfa = make_dfa({"q0": {"a": "q1", "b": "q2"}}, "q0", {"q1", "q2"})
    g = dfa.to_right_grammar()
    assert sum(1 for p in g.productions if p.body == ()) == 2


def test_55_nondeterministic_automaton():
    nfa = make_nfa({"q0": {"a": {"q0", "q1"}}, "q1": {"b": {"q2"}}}, "q0", {"q2"})
    g = nfa.to_right_grammar()
    assert sum(1 for p in g.productions if p.head == "S" and p.body[0] == "a") == 2
    assert grammar_language(g, 5) == automaton_language(nfa, 5)


def test_55_epsilon_transitions_are_removed_first():
    nfa = make_nfa({"q0": {EPS: {"q1"}}, "q1": {"a": {"q2"}, EPS: {"q0"}}}, "q0", {"q2"}, symbols="a")
    g = nfa.to_right_grammar()
    assert all(not (len(p.body) == 1 and p.body[0] in g.nonterminals) for p in g.productions)
    assert grammar_language(g, 4) == {("a",)}


def test_55_names_never_collide_with_terminals():
    dfa = make_dfa({"q0": {"S": "q1", "A0": "q2"}, "q1": {}, "q2": {}}, "q0", {"q1", "q2"}, symbols=["S", "A0"])
    g = dfa.to_right_grammar()
    assert g.nonterminals.isdisjoint(g.terminals)
    assert len(g.nonterminals) == 3
    assert grammar_language(g, 2) == {("S",), ("A0",)}


def test_55_arbitrary_state_types():
    dfa = make_dfa({(0, 1): {"a": frozenset({1})}, frozenset({1}): {"b": (0, 1)}}, (0, 1), {frozenset({1})})
    g = dfa.to_right_grammar()
    assert grammar_language(g, 4) == automaton_language(dfa, 4)


def test_55_language_is_preserved_on_random_automata():
    for automaton in itertools.chain(random_dfas(), random_nfas()):
        g = automaton.to_right_grammar()
        assert grammar_language(g, 5) == automaton_language(automaton, 5)


def test_55_automaton_and_grammar_are_not_modified():
    for automaton in random_nfas(15):
        before = (set(automaton.states), dict(automaton.transitions), set(automaton.final_states))
        g = automaton.to_right_grammar()
        assert (set(automaton.states), dict(automaton.transitions), set(automaton.final_states)) == before
        with pytest.raises(FrozenInstanceError):
            g.start = "X"


# ------------------------------------------------------------ function 55: grammar -> NFA
def test_55_grammar_to_nfa_basic_forms():
    g = RegularGrammar.from_rules("S", {"S": ["aA", "b"], "A": ["bB", ""], "B": ["a"]})
    nfa = g.to_nfa()
    assert isinstance(nfa, ExtendedNFA)
    assert nfa.initial_state == "S"
    assert nfa.transitions["S"]["a"] == {"A"}
    assert automaton_language(nfa, 5) == grammar_language(g, 5) == {("b",), ("a",), ("a", "b", "a")}


def test_55_epsilon_productions():
    g = RegularGrammar.from_rules("S", {"S": ["", "aS"]})
    nfa = g.to_nfa()
    assert "S" in nfa.final_states
    assert automaton_language(nfa, 4) == {("a",) * n for n in range(5)}
    only_eps = RegularGrammar.from_rules("S", {"S": [""]})
    assert automaton_language(only_eps.to_nfa(), 3) == {()}


def test_55_empty_language_grammar():
    g = RegularGrammar.from_rules("S", {"S": ["aS"]})
    assert automaton_language(g.to_nfa(), 4) == set()


def test_55_no_epsilon_transition_in_result():
    for g in random_grammars(30):
        assert all(EPS not in out for out in g.to_nfa().transitions.values())


def test_55_final_state_name_avoids_collisions():
    g = RegularGrammar.from_rules("F", {"F": ["aG", "b"], "G": ["a"]})
    nfa = g.to_nfa()
    assert len(nfa.states) == 3  # F, G and a fresh final state
    assert automaton_language(nfa, 3) == {("b",), ("a", "a")}
    g2 = RegularGrammar.from_rules("F", {"F": ["a"], "F'": ["b"]})
    assert len(g2.to_nfa().states) == 3 and automaton_language(g2.to_nfa(), 2) == {("a",)}


def test_55_to_dfa_is_deterministic_and_equivalent():
    g = RegularGrammar.from_rules("S", {"S": ["aS", "aA", "b"], "A": ["b", ""]})
    dfa = g.to_dfa()
    assert dfa.is_deterministic()
    assert automaton_language(dfa, 5) == grammar_language(g, 5)
    assert dfa.is_complete() is False and g.to_dfa(complete=True).is_complete()


def test_55_language_is_preserved_on_random_grammars():
    for g in random_grammars():
        assert automaton_language(g.to_nfa(), 5) == grammar_language(g, 5)


def test_55_round_trips():
    for automaton in itertools.islice(random_dfas(), 40):
        g = automaton.to_right_grammar()
        back = g.to_nfa()
        assert automaton_language(back, 5) == automaton_language(automaton, 5)
        assert grammar_language(back.to_right_grammar(), 5) == grammar_language(g, 5)
    for g in itertools.islice(random_grammars(), 40):
        assert grammar_language(g.to_nfa().to_right_grammar(), 5) == grammar_language(g, 5)


# ----------------------------------------------------------------- invalid grammars
@pytest.mark.parametrize(
    "rules, message",
    [
        ({"S": ["A"], "A": ["a"]}, "unit productions"),
        ({"S": ["Aa"], "A": ["b"]}, "not right-linear"),
        ({"S": ["abS"]}, "unsupported shape"),
        ({"S": ["aSS"]}, "unsupported shape"),
        ({"S": ["aAb"], "A": ["a"]}, "unsupported shape"),
    ],
)
def test_55_non_regular_grammars_are_refused(rules, message):
    with pytest.raises(GrammarError, match=message):
        RegularGrammar.from_rules("S", rules)


def test_55_other_invalid_grammars():
    with pytest.raises(GrammarError, match="start symbol"):
        RegularGrammar(frozenset({"S"}), frozenset({"a"}), frozenset(), "T")
    with pytest.raises(GrammarError, match="both terminal and nonterminal"):
        RegularGrammar(frozenset({"S", "a"}), frozenset({"a"}), frozenset(), "S")
    with pytest.raises(GrammarError, match="unknown symbol"):
        RegularGrammar(frozenset({"S"}), frozenset({"a"}), frozenset({Production("S", ("z",))}), "S")
    with pytest.raises(GrammarError, match="not a nonterminal"):
        RegularGrammar(frozenset({"S"}), frozenset({"a"}), frozenset({Production("X", ("a",))}), "S")
    with pytest.raises(GrammarError, match="non-empty"):
        RegularGrammar(frozenset({"S"}), frozenset({""}), frozenset(), "S")


def test_55_grammar_is_immutable_and_hashable():
    g = RegularGrammar.from_rules("S", {"S": ["aS", ""]})
    with pytest.raises(FrozenInstanceError):
        g.start = "T"
    assert isinstance(g.productions, frozenset)
    assert g == RegularGrammar.from_rules("S", {"S": ["", "aS"]}) and hash(g)
    assert str(g) == "S → ε | aS"


# ------------------------------------------------------------ function 56: derivation trees
EXAMPLE = {"S": ["aA"], "A": ["bB"], "B": [""]}


def test_56_example_of_the_statement():
    g = RegularGrammar.from_rules("S", EXAMPLE)
    tree = g.parse("ab")
    assert str(tree) == "S(a A(b B(ε)))"
    assert tree.symbol == "S" and tree.kind == "nonterminal"
    assert tree.production == Production("S", ("a", "A"))
    a, a_sub = tree.children
    assert (a.kind, a.symbol, a.children) == ("terminal", "a", ())
    b, b_sub = a_sub.children
    assert b.symbol == "b" and b_sub.symbol == "B"
    assert b_sub.production == Production("B", ())
    assert b_sub.children == (ParseTree.epsilon(),)
    assert tree.word() == ("a", "b")
    assert tree.productions() == (
        Production("S", ("a", "A")), Production("A", ("b", "B")), Production("B", ()),
    )
    assert tree.derivation_steps() == (("S",), ("a", "A"), ("a", "b", "B"), ("a", "b"))
    assert (tree.height(), tree.size()) == (3, 6)


def test_56_word_not_derivable():
    g = RegularGrammar.from_rules("S", EXAMPLE)
    for bad in ("", "a", "abb", "b", "ba"):
        assert not g.accepts(bad)
        assert g.parse_all(bad) == ()
        with pytest.raises(NotDerivableError, match="cannot be derived"):
            g.parse(bad)
    with pytest.raises(NotDerivableError, match="not a terminal"):
        g.parse("az")
    assert not g.accepts("az") and g.count_trees("az") == 0


def test_56_empty_word():
    g = RegularGrammar.from_rules("S", {"S": ["", "aS"]})
    tree = g.parse("")
    assert str(tree) == "S(ε)" and tree.word() == () and tree.derivation_steps() == (("S",), ())
    assert g.accepts("") and g.count_trees("") == 1
    with pytest.raises(NotDerivableError):
        RegularGrammar.from_rules("S", {"S": ["a"]}).parse("")


def test_56_epsilon_production_in_the_middle_and_end():
    g = RegularGrammar.from_rules("S", {"S": ["aA"], "A": ["", "bA"]})
    assert str(g.parse("a")) == "S(a A(ε))"
    assert str(g.parse("abb")) == "S(a A(b A(b A(ε))))"


def test_56_terminal_production():
    g = RegularGrammar.from_rules("S", {"S": ["aA"], "A": ["b"]})
    assert str(g.parse("ab")) == "S(a A(b))"


def test_56_recursive_productions_terminate():
    g = RegularGrammar.from_rules("S", {"S": ["aS", "b"]})
    assert g.parse("a" * 300 + "b").height() == 301
    assert not g.accepts("a" * 300)
    loop = RegularGrammar.from_rules("S", {"S": ["aS"]})  # never terminates as a derivation
    assert loop.parse_all("aaaa") == () and not loop.accepts("aaaa")
    mutual = RegularGrammar.from_rules("S", {"S": ["aA", ""], "A": ["bS", "aA"]})
    assert str(mutual.parse("abab")) == "S(a A(b S(a A(b S(ε)))))"
    assert {(), ("a", "b"), ("a", "a", "b"), ("a", "a", "a", "b")} <= mutual.generate(4)
    assert mutual.generate(4) == grammar_language(mutual, 4)


def test_56_ambiguity_policy():
    g = RegularGrammar.from_rules("S", {"S": ["aA", "aB"], "A": ["b"], "B": ["b"]})
    assert g.count_trees("ab") == 2 and g.is_ambiguous_for("ab")
    trees = g.parse_all("ab")
    assert len(trees) == 2 and trees[0] != trees[1]
    assert {str(t) for t in trees} == {"S(a A(b))", "S(a B(b))"}
    assert g.parse("ab") == trees[0] == g.parse("ab")  # deterministic: the first one
    assert g.parse_all("ab", limit=1) == (trees[0],)
    assert g.parse_all("ab", limit=0) == ()
    unambiguous = RegularGrammar.from_rules("S", EXAMPLE)
    assert not unambiguous.is_ambiguous_for("ab")
    with pytest.raises(ValueError, match="non-negative"):
        g.parse_all("ab", limit=-1)


def test_56_highly_ambiguous_grammar_is_counted_without_enumeration():
    g = RegularGrammar.from_rules("S", {"S": ["aS", "aT", ""], "T": ["aS", "aT", ""]})
    assert g.count_trees("a" * 40) > 10**9
    assert len(g.parse_all("a" * 40, limit=5)) == 5


def test_56_sequence_of_multi_character_terminals():
    g = RegularGrammar(
        frozenset({"S"}), frozenset({"if", "x"}),
        frozenset({Production("S", ("if", "S")), Production("S", ("x",))}), "S",
    )
    assert g.accepts(["if", "if", "x"]) and not g.accepts(["if"])
    assert g.parse(["if", "x"]).word() == ("if", "x")


def test_56_trees_agree_with_language_oracle_and_replay():
    for g in random_grammars():
        language = grammar_language(g, 4)
        for w in words(sorted(g.terminals), 4):
            assert g.accepts(w) == (w in language)
            trees = g.parse_all(w)
            assert bool(trees) == (w in language)
            assert g.count_trees(w) == len(trees)
            assert len(set(trees)) == len(trees)
            for tree in trees:
                assert tree.word() == w
                assert replay(g, tree) == w


def replay(grammar, tree):
    """Independent check: apply the listed productions to the start symbol."""
    form = [grammar.start]
    for production in tree.productions():
        assert production in grammar.productions
        index = next(i for i, s in enumerate(form) if s in grammar.nonterminals)
        assert form[index] == production.head
        form[index : index + 1] = list(production.body)
    assert all(s in grammar.terminals for s in form)
    return tuple(form)


def test_56_derivation_steps_are_valid_rewrites():
    for g in itertools.islice(random_grammars(), 40):
        for w in words(sorted(g.terminals), 3):
            for tree in g.parse_all(w, limit=3):
                steps = tree.derivation_steps()
                assert steps[0] == (g.start,) and steps[-1] == w
                for before, after in zip(steps, steps[1:]):
                    i = next(k for k, s in enumerate(before) if s in g.nonterminals)
                    assert any(
                        p.head == before[i] and after == before[:i] + p.body + before[i + 1 :]
                        for p in g.productions
                    )


def test_56_tree_structure_and_immutability():
    g = RegularGrammar.from_rules("S", EXAMPLE)
    tree = g.parse("ab")
    with pytest.raises(FrozenInstanceError):
        tree.symbol = "X"
    assert isinstance(tree.children, tuple) and tree.is_leaf is False
    assert tree.children[0].is_leaf and ParseTree.epsilon().is_leaf
    assert ParseTree.terminal("a") == ParseTree.terminal("a") and hash(tree)
    assert tree.pretty().splitlines()[0] == "S    [S → a A]"
    dot = tree.to_dot()
    assert dot.startswith("digraph") and dot.count("->") == tree.size() - 1
    assert "shape=box" in dot and str(Production("A", ())) == "A → ε"


# ======================================================================================
# Limits lifted: left-linear grammars, liberal rules, text format, SVG
# ======================================================================================
import xml.etree.ElementTree as ET  # noqa: E402


def random_left_grammars(count=100, seed=13):
    rng = random.Random(seed)
    for _ in range(count):
        heads = ["S", "A", "B"][: rng.randint(1, 3)]
        rules = {}
        for h in heads:
            bodies = set()
            for _ in range(rng.randint(0, 3)):
                kind = rng.choice(["eps", "t", "nt", "nt"])
                a = rng.choice("ab")
                bodies.add({"eps": "", "t": a, "nt": rng.choice(heads) + a}[kind])
            rules[h] = sorted(bodies)
        yield LeftRegularGrammar.from_rules("S", rules)


def rules_language(start, rules, longest):
    """Oracle for liberal rules: rewrite the (unique) nonterminal of a sentential form."""
    heads = set(rules)
    parsed = {h: [() if b in ("", "ε") else tuple(b) for b in bodies] for h, bodies in rules.items()}
    found, todo, seen = set(), [(start,)], set()
    while todo:
        form = todo.pop()
        if form in seen:
            continue
        seen.add(form)
        position = next((i for i, x in enumerate(form) if x in heads), None)
        if position is None:
            if len(form) <= longest:
                found.add(form)
            continue
        if sum(x not in heads for x in form) > longest:
            continue
        for body in parsed[form[position]]:
            todo.append(form[:position] + body + form[position + 1 :])
    return found


# ---------------------------------------------------- to_left_grammar (automaton -> left)
def test_left_dfa_to_left_grammar_exact_productions():
    dfa = make_dfa({"e": {"a": "o", "b": "e"}, "o": {"a": "e", "b": "o"}}, "e", {"e"})
    g = dfa.to_left_grammar()
    assert isinstance(g, LeftRegularGrammar) and g.start == "S"
    assert g.nonterminals == {"S", "A0", "A1"}  # one per state plus the fresh start symbol
    # A0 = initial (even), A1 = odd
    assert g.productions == {
        Production("A0", ()),
        Production("A0", ("A1", "a")), Production("A0", ("A0", "b")),
        Production("A1", ("A0", "a")), Production("A1", ("A1", "b")),
        Production("S", ()),
        Production("S", ("A1", "a")), Production("S", ("A0", "b")),
    }


def test_left_language_preserved_on_random_automata():
    for automaton in itertools.chain(random_dfas(), random_nfas()):
        g = automaton.to_left_grammar()
        assert grammar_language(g, 5) == automaton_language(automaton, 5)


def test_left_names_never_collide_and_automaton_is_untouched():
    dfa = make_dfa({"q0": {"S": "q1", "A0": "q1"}, "q1": {}}, "q0", {"q1"}, symbols=["S", "A0"])
    before = (set(dfa.states), dict(dfa.transitions))
    g = dfa.to_left_grammar()
    assert g.nonterminals.isdisjoint(g.terminals) and (set(dfa.states), dict(dfa.transitions)) == before
    assert grammar_language(g, 2) == {("S",), ("A0",)}


def test_left_epsilon_transitions_and_empty_language():
    nfa = make_nfa({"q0": {EPS: {"q1"}}, "q1": {"a": {"q2"}}}, "q0", {"q2"})
    g = nfa.to_left_grammar()
    assert all(len(p.body) != 1 or p.body[0] in g.terminals for p in g.productions)
    assert grammar_language(g, 3) == {("a",)}
    nothing = make_dfa({"q0": {"a": "q1"}}, "q0", set())
    assert grammar_language(nothing.to_left_grammar(), 3) == set()
    only_eps = make_dfa({"q0": {}}, "q0", {"q0"})
    assert grammar_language(only_eps.to_left_grammar(), 3) == {()}


# ---------------------------------------------------------- LeftRegularGrammar itself
def test_left_grammar_to_nfa_and_membership():
    g = LeftRegularGrammar.from_rules("S", {"S": ["Sa", "Ab"], "A": ["a", ""]})
    assert grammar_language(g, 5) == {("b",), ("a", "b")} | {("b",) + ("a",) * k for k in range(1, 5)} | {
        ("a", "b") + ("a",) * k for k in range(0, 4)
    }
    nfa = g.to_nfa()
    assert automaton_language(nfa, 5) == grammar_language(g, 5)
    for w in words("ab", 5):
        assert g.accepts(w) == (w in grammar_language(g, 5))
    assert not g.accepts("az")


def test_left_random_grammars_nfa_dfa_and_counts():
    for g in random_left_grammars():
        language = grammar_language(g, 4)
        assert automaton_language(g.to_nfa(), 4) == language
        assert automaton_language(g.to_dfa(), 4) == language
        assert {tuple(w) for w in g.generate(4)} == language
        for w in words("ab", 4):
            trees = g.parse_all(w)
            assert g.accepts(w) == bool(trees) == (w in language)
            assert g.count_trees(w) == len(trees)
            for tree in trees:
                assert tree.word() == w and replay(g, tree) == w


def test_left_tree_structure_follows_the_bodies():
    g = LeftRegularGrammar.from_rules("S", {"S": ["Sa", "b"]})
    tree = g.parse("baa")
    assert str(tree) == "S(S(S(b) a) a)"
    assert tree.production == Production("S", ("S", "a"))
    assert tree.word() == ("b", "a", "a")
    assert tree.derivation_steps() == (("S",), ("S", "a"), ("S", "a", "a"), ("b", "a", "a"))
    assert tree.height() == 3
    with pytest.raises(NotDerivableError, match="cannot be derived"):
        g.parse("ab")
    with pytest.raises(NotDerivableError, match="not a terminal"):
        g.parse("bz")


def test_left_epsilon_empty_word_and_ambiguity():
    g = LeftRegularGrammar.from_rules("S", {"S": ["Aa", "Ba"], "A": ["", "b"], "B": ["", "b"]})
    assert g.count_trees("a") == 2 and g.is_ambiguous_for("a")
    assert {str(t) for t in g.parse_all("a")} == {"S(A(ε) a)", "S(B(ε) a)"}
    assert g.count_trees("ba") == 2 and g.parse_all("ba", limit=1)[0].word() == ("b", "a")
    h = LeftRegularGrammar.from_rules("S", {"S": ["Sa", ""]})
    assert str(h.parse("")) == "S(ε)" and h.accepts("aaa")


def test_left_recursion_terminates():
    g = LeftRegularGrammar.from_rules("S", {"S": ["Sa", "b"]})
    assert g.parse("b" + "a" * 300).height() == 301
    loop = LeftRegularGrammar.from_rules("S", {"S": ["Sa"]})
    assert not loop.accepts("aaaa") and loop.parse_all("aaaa") == ()


def test_left_reversed_and_mirror_language():
    g = LeftRegularGrammar.from_rules("S", {"S": ["Sa", "Ab"], "A": ["a", ""]})
    mirror = g.reversed()
    assert isinstance(mirror, RegularGrammar)
    assert {w[::-1] for w in grammar_language(mirror, 5)} == grammar_language(g, 5)


def test_left_invalid_shapes_have_explicit_messages():
    with pytest.raises(GrammarError, match="not left-linear"):
        LeftRegularGrammar.from_rules("S", {"S": ["aS"]})
    with pytest.raises(GrammarError, match="unit productions"):
        LeftRegularGrammar.from_rules("S", {"S": ["A"], "A": ["a"]})
    with pytest.raises(GrammarError, match="unsupported shape"):
        LeftRegularGrammar.from_rules("S", {"S": ["SSa"]})
    with pytest.raises(GrammarError, match="not right-linear"):
        RegularGrammar.from_rules("S", {"S": ["Sa"]})


def test_left_and_right_conversions_keep_the_language():
    for g in itertools.islice(random_grammars(), 40):
        left = g.to_left_grammar()
        assert isinstance(left, LeftRegularGrammar)
        assert grammar_language(left, 4) == grammar_language(g, 4)
        assert grammar_language(left.to_right_grammar(), 4) == grammar_language(g, 4)
    for g in itertools.islice(random_left_grammars(), 40):
        assert grammar_language(g.to_right_grammar(), 4) == grammar_language(g, 4)


def test_left_and_right_classes_are_different_and_immutable():
    r = RegularGrammar.from_rules("S", {"S": [""]})
    l = LeftRegularGrammar.from_rules("S", {"S": [""]})
    assert r != l and hash(l)
    with pytest.raises(FrozenInstanceError):
        l.start = "X"


# ------------------------------------------------------------------ from_linear (liberal)
def test_linear_long_bodies_are_cut_with_fresh_nonterminals():
    g = RegularGrammar.from_linear("S", {"S": ["abcT", "aa"], "T": ["b"]})
    assert all(p.body == () or len(p.body) <= 2 for p in g.productions)
    assert len(g.nonterminals) == 2 + 3  # S, T, plus two for abcT and one for aa
    assert grammar_language(g, 6) == {("a", "b", "c", "b"), ("a", "a")}


def test_linear_unit_productions_are_removed():
    g = RegularGrammar.from_linear("S", {"S": ["A", "bS"], "A": ["B", "a"], "B": ["", "cB"]})
    assert all(not (len(p.body) == 1 and p.body[0] in g.nonterminals) for p in g.productions)
    assert grammar_language(g, 5) == rules_language("S", {"S": ["A", "bS"], "A": ["B", "a"], "B": ["", "cB"]}, 5)
    cycle = RegularGrammar.from_linear("S", {"S": ["A", "a"], "A": ["S", "b"]})
    assert grammar_language(cycle, 3) == {("a",), ("b",)}
    selfloop = RegularGrammar.from_linear("S", {"S": ["S", "a"]})
    assert grammar_language(selfloop, 3) == {("a",)}


def test_linear_fresh_names_avoid_existing_symbols():
    # bodies given as tuples because "S_1" has several characters
    g = RegularGrammar.from_linear("S", {"S": [("a", "b", "S_1"), ("a", "b")], "S_1": [("c",)]})
    assert {"S", "S_1"} < g.nonterminals and "S_2" in g.nonterminals  # S_1 was taken
    assert g.nonterminals == {"S", "S_1", "S_2", "S_3"}  # one fresh name per long body
    assert grammar_language(g, 6) == {("a", "b", "c"), ("a", "b")}


def test_linear_nonterminal_in_the_middle_is_refused():
    with pytest.raises(GrammarError, match="before the end"):
        RegularGrammar.from_linear("S", {"S": ["aSb"]})
    with pytest.raises(GrammarError, match="left-linear"):
        LeftRegularGrammar.from_linear("S", {"S": ["aSb"]})


def test_linear_random_rules_keep_the_language():
    rng = random.Random(21)
    for _ in range(150):
        heads = ["S", "A", "B"][: rng.randint(1, 3)]
        rules = {}
        for h in heads:
            bodies = set()
            for _ in range(rng.randint(0, 3)):
                letters = "".join(rng.choice("ab") for _ in range(rng.randint(0, 3)))
                tail = rng.choice(heads + [""] * 2)
                bodies.add(letters + tail)
            rules[h] = sorted(bodies)
        g = RegularGrammar.from_linear("S", rules)
        assert grammar_language(g, 5) == rules_language("S", rules, 5), rules


def test_linear_left_version_keeps_the_language():
    rng = random.Random(22)
    for _ in range(100):
        heads = ["S", "A"][: rng.randint(1, 2)]
        rules = {}
        for h in heads:
            bodies = set()
            for _ in range(rng.randint(0, 3)):
                head_part = rng.choice(heads + [""] * 2)
                letters = "".join(rng.choice("ab") for _ in range(rng.randint(0, 2)))
                bodies.add(head_part + letters)
            rules[h] = sorted(bodies)
        g = LeftRegularGrammar.from_linear("S", rules)
        mirrored = {h: ["".join(reversed(b)) for b in bodies] for h, bodies in rules.items()}
        expected = {w[::-1] for w in rules_language("S", mirrored, 4)}
        assert grammar_language(g, 4) == expected, rules


# ----------------------------------------------------------------------- from_text
def test_text_basic_and_start_symbol():
    g = RegularGrammar.from_text("S -> a A | b\nA -> b B | ε\nB -> ε")
    assert g.start == "S" and g.nonterminals == {"S", "A", "B"} and g.terminals == {"a", "b"}
    assert grammar_language(g, 3) == {("b",), ("a",), ("a", "b")}
    assert RegularGrammar.from_text("A -> a\nS -> a A", start="S").start == "S"


def test_text_multi_character_symbols_comments_and_arrows():
    text = """
    # a comment
    Expr → if Cond | go Expr | ε
    Cond -> yes
    Expr -> done      # same head on a later line
    """.replace("      # same head on a later line", "")
    g = RegularGrammar.from_text(text)
    assert g.terminals == {"if", "go", "yes", "done"} and g.start == "Expr"
    assert g.accepts(["go", "go", "if", "yes"]) and not g.accepts(["if"])
    assert g.accepts(["go", "done"]) and g.accepts([])


def test_text_empty_alternative_means_epsilon():
    g = RegularGrammar.from_text("S -> a S |")
    assert g.accepts("aaa") and g.accepts("")


def test_text_errors_are_explicit():
    with pytest.raises(GrammarError, match="missing '->'"):
        RegularGrammar.from_text("S = a")
    with pytest.raises(GrammarError, match="no production"):
        RegularGrammar.from_text("\n# nothing\n")
    with pytest.raises(GrammarError, match="one symbol"):
        RegularGrammar.from_text("S T -> a")
    with pytest.raises(GrammarError, match="one symbol"):
        RegularGrammar.from_text(" -> a")
    with pytest.raises(GrammarError, match="unit productions"):
        RegularGrammar.from_text("S -> A\nA -> a")
    with pytest.raises(GrammarError, match="unsupported shape"):
        RegularGrammar.from_text("S -> a b")  # two terminals: refused unless normalized


def test_text_normalize_option_and_left_grammars():
    g = RegularGrammar.from_text("S -> a b S | A\nA -> c", normalize=True)
    assert grammar_language(g, 5) == {("a", "b", "c"), ("c",), ("a", "b", "a", "b", "c")}
    left = LeftRegularGrammar.from_text("S -> S a | b")
    assert left.parse("baa").word() == ("b", "a", "a")
    left_norm = LeftRegularGrammar.from_text("S -> S a b | c", normalize=True)
    assert left_norm.accepts("cabab") and not left_norm.accepts("caba")


# ------------------------------------------------------------------------------- SVG
def svg_texts(svg):
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    return root, [t.text for t in root.iter(ns + "text")]


def test_svg_is_well_formed_and_complete():
    g = RegularGrammar.from_rules("S", EXAMPLE)
    tree = g.parse("ab")
    root, texts = svg_texts(tree.to_svg())
    assert root.tag.endswith("svg") and root.get("viewBox")
    assert texts == ["S", "a", "A", "b", "B", "ε"] or sorted(texts) == sorted(["S", "a", "A", "b", "B", "ε"])
    assert len(list(root.iter("{http://www.w3.org/2000/svg}line"))) == tree.size() - 1
    assert 'class="bg"' in tree.to_svg() and "prefers-color-scheme" in tree.to_svg()


def test_svg_escapes_special_characters_and_handles_small_trees():
    g = RegularGrammar(
        frozenset({"S"}), frozenset({"<", "&"}),
        frozenset({Production("S", ("<", "S")), Production("S", ("&",))}), "S",
    )
    root, texts = svg_texts(g.parse(["<", "&"]).to_svg())
    assert "<" in texts and "&" in texts
    ET.fromstring(ParseTree.epsilon().to_svg())
    ET.fromstring(ParseTree.terminal("x").to_svg())


def test_svg_is_deterministic_and_works_for_left_trees_and_deep_trees():
    left = LeftRegularGrammar.from_rules("S", {"S": ["Sa", "b"]})
    svg = left.parse("baaa").to_svg()
    assert svg == left.parse("baaa").to_svg()
    root, texts = svg_texts(svg)
    assert [t for t in texts if t in ("a", "b")] == ["b", "a", "a", "a"]  # frontier order
    deep = RegularGrammar.from_rules("S", {"S": ["aS", "b"]}).parse("a" * 40 + "b")
    ET.fromstring(deep.to_svg())


def test_svg_geometry_inner_nodes_are_centred_and_rows_follow_depth():
    tree = RegularGrammar.from_rules("S", {"S": ["aA"], "A": ["b"]}).parse("ab")  # S(a A(b))
    ns = "{http://www.w3.org/2000/svg}"
    root = ET.fromstring(tree.to_svg())
    pos = {t.text: (float(t.get("x")), float(t.get("y"))) for t in root.iter(ns + "text")}
    assert pos["S"][0] == pytest.approx((pos["a"][0] + pos["A"][0]) / 2)  # parent between children
    assert pos["A"][0] == pytest.approx(pos["b"][0])  # single child directly below
    assert pos["a"][0] < pos["b"][0]  # leaves keep the word order
    assert pos["S"][1] < pos["A"][1] < pos["b"][1]  # one row per depth level
