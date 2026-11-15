"""Audit (prompt 22): differential tests against independent oracles, with fresh seeds.

Each oracle is written without using the code under test: bounded word enumeration,
Python's ``re``, a private NFA simulator, mutual reachability by Floyd-Warshall,
pumping-lemma bounds, and counting distinct bounded futures for Myhill-Nerode.
"""

import itertools
import random

import pytest

from automata_extensions.fa import ExtendedDFA, ExtendedNFA
from automata_extensions.regex import Regex
from test_grammar import automaton_accepts, make_dfa, make_nfa, random_dfas, random_nfas
from test_regex import oracle, words_up_to


def lang(automaton, longest=6):
    return {w for w in words_up_to("ab", longest) if automaton_accepts(automaton, w)}


def dfa_edges(dfa):
    return {(q, t) for q, out in dfa.transitions.items() for t in out.values() if t is not None}


def closure(states, edges):
    """Reflexive-transitive closure by Floyd-Warshall (independent of the DFS under test)."""
    reach = {(p, q) for p in states for q in states if p == q} | set(edges)
    for k in states:
        for i in states:
            for j in states:
                if (i, k) in reach and (k, j) in reach:
                    reach.add((i, j))
    return reach


def bigger_dfas(count, seed):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(1, 6)
        names = [f"s{i}" for i in range(n)]
        table = {s: {a: rng.choice(names) for a in "ab" if rng.random() < 0.8} for s in names}
        yield make_dfa(table, "s0", {s for s in names if rng.random() < 0.35})


# ---- graph algorithms -------------------------------------------------------------------
def test_audit_reachability_family_against_floyd_warshall():
    for dfa in bigger_dfas(150, 101):
        states, edges = dfa.states, dfa_edges(dfa)
        reach = closure(states, edges)
        acc = {q for q in states if (dfa.initial_state, q) in reach}
        coacc = {q for q in states if any((q, f) in reach for f in dfa.final_states)}
        assert dfa.accessible_states() == acc
        assert dfa.coaccessible_states() == coacc
        assert dfa.useful_states() == acc & coacc
        assert set(dfa.dfs_iterative()) == acc == set(dfa.bfs())
        assert len(dfa.bfs()) == len(set(dfa.bfs())) and dfa.bfs()[0] == dfa.initial_state
        kept = (acc & coacc) or {dfa.initial_state}  # documented convention for the empty language
        assert dfa.is_trim() == (kept == set(states))
        trimmed = dfa.trim()
        assert lang(trimmed) == lang(dfa)
        assert set(trimmed.states) == kept
        for q in states:
            assert dfa.is_accessible(q) == (q in acc)
            assert dfa.reachable_from(q) == {r for r in states if (q, r) in reach}


def test_audit_scc_and_cycles_against_mutual_reachability():
    for dfa in bigger_dfas(150, 102):
        states, edges = dfa.states, dfa_edges(dfa)
        reach = closure(states, edges)
        expected = {
            frozenset(r for r in states if (q, r) in reach and (r, q) in reach) for q in states
        }
        assert set(dfa.strongly_connected_components()) == expected
        # a state lies on a cycle iff some successor reaches it back
        def cyclic(q):
            return any((q, r) in edges and (r, q) in reach for r in states)
        assert dfa.has_cycle() == any(cyclic(q) for q in states)
        for q in states:
            assert dfa.has_cycle_from(q) == any((q, r) in reach and cyclic(r) for r in states)


def test_audit_empty_and_infinite_against_bounded_words():
    for dfa in bigger_dfas(150, 103):
        n = len(dfa.states) + 1  # + implicit sink
        words = list(words_up_to("ab", 2 * n - 1))
        accepted = [w for w in words if automaton_accepts(dfa, w)]
        assert dfa.is_empty() == (not [w for w in accepted if len(w) < n])
        assert dfa.is_empty() == (not accepted)
        assert dfa.is_infinite() == any(n <= len(w) < 2 * n for w in accepted)  # pumping lemma
    for nfa in random_nfas(80, 104):
        n = len(nfa.states) + 1
        accepted = [w for w in words_up_to("ab", 2 * n) if automaton_accepts(nfa, w)]
        assert nfa.is_empty() == (not accepted)


# ---- language operations ---------------------------------------------------------------------
def test_audit_boolean_operations_and_relations():
    pool = list(bigger_dfas(40, 105))
    for a, b in itertools.product(pool[:14], pool[14:28]):
        la, lb = lang(a), lang(b)
        assert lang(a.complemented()) == {w for w in words_up_to("ab", 6)} - la
        assert lang(a.language_union(b)) == la | lb
        assert lang(a.language_intersection(b)) == la & lb
        assert lang(a.language_difference(b)) == la - lb
        assert lang(a.language_concatenation(b)) == {
            u + v for u in la for v in lb if len(u + v) <= 6
        } & set(words_up_to("ab", 6))
    for a in pool:
        assert lang(a.language_star()) >= {""}
        assert a.has_same_language_as(a.minimize()) and a.has_same_language_as(a.complete())
        assert not a.has_same_language_as(a.complemented())


def test_audit_inclusion_and_equality_exact_with_pumping_bound():
    """Words up to 2n are enough to decide inclusion/equality for DFAs with <= n states."""
    pool = list(bigger_dfas(30, 106))
    for a, b in itertools.product(pool, pool[:10]):
        n = len(a.states) + len(b.states) + 2
        words = list(words_up_to("ab", min(n, 9)))
        included = all(automaton_accepts(b, w) for w in words if automaton_accepts(a, w))
        if n <= 9:
            assert a.is_included_in(b) == included
            equal = included and all(automaton_accepts(a, w) for w in words if automaton_accepts(b, w))
            assert a.has_same_language_as(b) == equal
        elif not included:
            assert not a.is_included_in(b)


def small_dfas(count, seed):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(1, 3)
        names = [f"s{i}" for i in range(n)]
        table = {s: {a: rng.choice(names) for a in "ab" if rng.random() < 0.85} for s in names}
        yield make_dfa(table, "s0", {s for s in names if rng.random() < 0.45})


def test_audit_quotients_against_definition():
    """Witnesses are searched up to 9 letters: enough for 3x3 = 9 product states."""
    pool = list(small_dfas(24, 107))
    for a, b in itertools.product(pool[:6], pool[6:12]):
        short = [u for u in words_up_to("ab", 9) if automaton_accepts(a, u)]
        left = a.language_left_quotient(b)
        right = a.language_right_quotient(b)
        for w in words_up_to("ab", 3):
            # L(a)^-1 L(b) = { w | exists u in L(a): u w in L(b) }
            assert automaton_accepts(left, w) == any(automaton_accepts(b, u + w) for u in short)
            # L(a) / L(b) = { w | exists u in L(b): w u in L(a) }
            ub = [u for u in words_up_to("ab", 9) if automaton_accepts(b, u)]
            assert automaton_accepts(right, w) == any(automaton_accepts(a, w + u) for u in ub)


# ---- Myhill-Nerode and minimization -----------------------------------------------------------
def test_audit_nerode_classes_minimization_against_bounded_futures():
    for dfa in bigger_dfas(120, 108):
        n = len(dfa.states) + 1
        suffixes = list(words_up_to("ab", min(n, 7)))
        full = dfa.complete()
        probe = lambda q: frozenset(
            s for s in suffixes
            if automaton_accepts(
                type(full)(states=full.states, input_symbols=full.input_symbols,
                           transitions=full.transitions, initial_state=q,
                           final_states=full.final_states), s)
        )
        acc = full.accessible_states()
        futures = {q: probe(q) for q in acc}
        groups = {frozenset(q for q in acc if futures[q] == f) for f in set(futures.values())}
        assert set(dfa.complete().myhill_nerode_classes(accessible_only=True)) == groups
        mini = dfa.minimize()
        # documented convention: a partial DFA stays partial, its dead class is not materialised
        # (the initial state is always kept)
        initial_group = next(g for g in groups if dfa.initial_state in g)
        kept = [g for g in groups if g == initial_group or any(futures[q] for q in g)]
        assert len(mini.states) == (len(groups) if dfa.is_complete() else len(kept))
        assert dfa.has_same_language_as(mini)
        assert mini.is_minimal() and mini.minimize().is_isomorphic_to(mini)
        if dfa.is_complete():
            assert dfa.is_minimal() == (len(groups) == len(dfa.states) and acc == dfa.states)


# ---- epsilon handling, determinization --------------------------------------------------------
def test_audit_epsilon_closure_determinization_and_epsilon_removal():
    for nfa in random_nfas(150, 109):
        for q in nfa.states:
            seen, todo = {q}, [q]
            while todo:
                for t in nfa.transitions.get(todo.pop(), {}).get("", ()):
                    if t not in seen:
                        seen.add(t)
                        todo.append(t)
            assert nfa.epsilon_closure(q) == seen
        base = lang(nfa)
        assert lang(nfa.determinize()) == base == lang(nfa.to_dfa()) == lang(nfa.to_dfa(complete=True))
        free = nfa.remove_epsilon_transitions()
        assert lang(free) == base
        assert all("" not in out or not out[""] for out in free.transitions.values())
        assert lang(nfa.reversed_automaton()) == {w[::-1] for w in base}


# ---- regular expressions --------------------------------------------------------------------------
def random_regex(rng, depth=3):
    if depth == 0 or rng.random() < 0.25:
        return rng.choice(["a", "b", "ε", "a", "b"])
    kind = rng.choice(["cat", "cat", "or", "star"])
    if kind == "star":
        return "(" + random_regex(rng, depth - 1) + ")*"
    left, right = random_regex(rng, depth - 1), random_regex(rng, depth - 1)
    return "(" + left + ("|" if kind == "or" else "") + right + ")"


def test_audit_thompson_derivatives_brzozowski_against_python_re():
    rng = random.Random(110)
    for _ in range(120):
        text = random_regex(rng)
        regex = Regex.parse(text)
        nfa, dfa = regex.to_nfa(alphabet="ab"), regex.brzozowski_dfa("ab")
        for w in words_up_to("ab", 6):
            expected = oracle(text, w)
            assert automaton_accepts(nfa, w) == expected, (text, w)
            assert automaton_accepts(dfa, w) == expected, (text, w)
            derived = regex
            for symbol in w:
                derived = derived.derivative(symbol)
            assert derived.nullable() == expected, (text, w)  # D_w(r) contains ε iff w in L(r)
        for symbol in "ab":
            d = regex.derivative(symbol)
            for w in words_up_to("ab", 4):
                assert oracle(str(d), w) == oracle(text, symbol + w), (text, symbol, w)


def test_audit_dfa_to_regex_three_ways():
    for dfa in bigger_dfas(60, 111):
        for method in (dfa.to_regex_elimination, dfa.to_regex_arden):
            text = str(method())
            for w in words_up_to("ab", 6):
                assert oracle(text, w) == automaton_accepts(dfa, w), (method.__name__, text, w)
    for nfa in random_nfas(40, 112):
        text = str(nfa.to_regex_elimination())
        for w in words_up_to("ab", 5):
            assert oracle(text, w) == automaton_accepts(nfa, w), (text, w)


def test_audit_eliminate_states_gives_the_language_for_every_order():
    rng = random.Random(116)
    for nfa in random_nfas(80, 113):
        norm = nfa.normalized()
        assert lang(norm, 5) == lang(nfa, 5)
        middle = [q for q in norm.states if q not in (norm.initial_state, *norm.final_states)]
        for order in (None, rng.sample(middle, len(middle))):
            text = norm.eliminate_states(order)
            for w in words_up_to("ab", 5):
                expected = automaton_accepts(nfa, w)
                assert (text is not None and oracle(text, w)) == expected, (text, w)


def test_audit_residual_automaton_and_systems():
    for dfa in bigger_dfas(60, 114):
        res = dfa.residual_automaton()
        assert lang(res) == lang(dfa)
        assert len(res.states) == len(dfa.complete().minimize().states)  # complete residual automaton
    rng = random.Random(115)
    for _ in range(40):
        text = random_regex(rng)
        res = Regex.parse(text).residual_automaton(alphabet="ab")
        for w in words_up_to("ab", 5):
            assert automaton_accepts(res, w) == oracle(text, w), (text, w)


# ---- remaining functions: simulation, traces, completeness, prefix-closure, distinguishability ------------
from test_grammar import grammar_language  # noqa: E402


def step(dfa, state, symbol):
    return dfa.transitions.get(state, {}).get(symbol)


def test_audit_accepts_and_execution_trace_against_private_simulators():
    for dfa in bigger_dfas(100, 120):
        for w in words_up_to("ab", 5):
            assert dfa.accepts(w) == automaton_accepts(dfa, w)
            trace = dfa.execution_trace(w)
            state, expected = dfa.initial_state, [dfa.initial_state]
            blocked = None
            for i, symbol in enumerate(w):
                state = step(dfa, state, symbol)
                if state is None:
                    blocked = i
                    break
                expected.append(state)
            assert list(trace.configurations) == expected and trace.blocked_at == blocked
            assert trace.accepted == (blocked is None and expected[-1] in dfa.final_states)
    for nfa in random_nfas(100, 121):
        for w in words_up_to("ab", 4):
            # configurations are sets of states: the final one must be what the private simulator reaches
            assert nfa.execution_trace(w).accepted == automaton_accepts(nfa, w)


@pytest.mark.skipif(not hasattr(ExtendedNFA, "accepts_input"), reason="needs the real automata-lib NFA")
def test_audit_nfa_accepts_native_wrapper():
    for nfa in random_nfas(100, 121):
        for w in words_up_to("ab", 4):
            assert nfa.accepts(w) == automaton_accepts(nfa, w)


def test_audit_predecessors_induced_subautomaton_completion():
    for dfa in bigger_dfas(100, 122):
        states = dfa.states
        reach = closure(states, dfa_edges(dfa))
        edges = dfa_edges(dfa)
        for q in states:
            # a path of length >= 1: some edge (p, r) with p reaching... r reaches q or r == q
            assert dfa.predecessor_states(q) == {
                p for p in states if any((p, r) in edges and (r, q) in reach for r in states)
            }
        full = dfa.complete()
        assert full.is_complete() and lang(full) == lang(dfa)
        assert dfa.is_complete() == all(
            step(dfa, q, a) is not None for q in states for a in "ab"
        )
        if dfa.is_complete():
            assert full.states == dfa.states  # nothing is added when already complete
        else:
            assert len(full.states) == len(dfa.states) + 1
        keep = {q for q in states if q != dfa.initial_state} | {dfa.initial_state}
        sub = dfa.induced_subautomaton(keep)
        assert sub.states == keep and lang(sub) == lang(dfa)


def test_audit_prefix_closure():
    for dfa in bigger_dfas(150, 123):
        L = lang(dfa, 6)
        largest = dfa.largest_prefix_closed_sublanguage()
        expected = {w for w in L if all(w[:i] in L for i in range(len(w) + 1))}
        assert lang(largest, 6) == expected
        assert largest.is_prefix_closed()


def test_audit_prefix_closure_exact_on_small_automata():
    for dfa in small_dfas(150, 124):
        n = len(dfa.states) + 1
        L = {w for w in words_up_to("ab", 2 * n) if automaton_accepts(dfa, w)}
        assert dfa.is_prefix_closed() == all(w[:i] in L for w in L for i in range(len(w)))


def test_audit_distinguishing_words_universality_determinism():
    for dfa in bigger_dfas(80, 125):
        for p, q in itertools.combinations(sorted(dfa.states), 2):
            suffixes = list(words_up_to("ab", len(dfa.states) + 1))

            def reads(start, w):
                s = start
                for c in w:
                    s = step(dfa, s, c)
                    if s is None:
                        return False
                return s in dfa.final_states

            separating = [w for w in suffixes if reads(p, w) != reads(q, w)]
            word = dfa.distinguishing_word(p, q)
            assert dfa.are_distinguishable(p, q) == bool(separating)
            if word is not None:
                assert reads(p, word) != reads(q, word)
                assert len(word) == min(map(len, separating))  # shortest
        n = len(dfa.states) + 1
        assert dfa.is_universal() == all(automaton_accepts(dfa, w) for w in words_up_to("ab", n))
        assert dfa.is_deterministic()
    for nfa in random_nfas(80, 126):
        has_choice = any(
            len(v) > 1 or (a == "" and v) for out in nfa.transitions.values() for a, v in out.items()
        )
        assert nfa.is_deterministic() == (not has_choice)


def test_audit_grammars_from_automata():
    for dfa in bigger_dfas(40, 127):
        assert {"".join(w) for w in grammar_language(dfa.to_right_grammar(), 5)} == lang(dfa, 5)
        assert {"".join(w) for w in grammar_language(dfa.to_left_grammar(), 5)} == lang(dfa, 5)
    for nfa in random_nfas(40, 128):
        assert {"".join(w) for w in grammar_language(nfa.to_right_grammar(), 5)} == lang(nfa, 5)
    for dfa in bigger_dfas(20, 129):
        g = dfa.to_right_grammar()
        back = g.to_nfa()
        assert lang(back, 5) == lang(dfa, 5)
        for w in sorted(lang(dfa, 4))[:5]:
            trees = g.parse_all(tuple(w))
            assert trees and all(t.word() == tuple(w) for t in trees)


def test_audit_nfa_execution_trace_regression():
    """Regression (found by the audit): ExtendedNFA.execution_trace raised NotImplementedError.

    Function 16 asks for 'the states **or sets of states** met': an NFA with an epsilon move
    and a choice must give closed sets.
    """
    nfa = make_nfa(
        {"p": {"": {"q"}, "a": {"p"}}, "q": {"a": {"r", "q"}}, "r": {"b": {"f"}}, "f": {}},
        "p", {"f"},
    )
    trace = nfa.execution_trace("ab")
    assert trace.configurations == (
        frozenset({"p", "q"}),
        frozenset({"p", "q", "r"}),
        frozenset({"f"}),
    )
    assert trace.accepted and trace.blocked_at is None
    blocked = nfa.execution_trace("abb")
    assert blocked.blocked_at == 2 and not blocked.accepted and len(blocked.configurations) == 3
    assert nfa.execution_trace("c").blocked_at == 0  # symbol outside the alphabet
    assert nfa.execution_trace("").configurations == (frozenset({"p", "q"}),)
    assert not nfa.execution_trace("").accepted


def test_audit_double_reversal_known_limit_of_fn50_wording():
    """Cahier des charges line 50 says 'double reverse + determinization'.

    The project implements fn 50 as the derivative automaton (as the prompt required). Composing
    ``reversed_automaton().determinize()`` twice keeps the right language, but is NOT always
    minimal: automata-lib allows a single initial state, so the reversal of an automaton with
    several final states adds an artificial initial state whose subset stays a separate state.
    Documented here so the limit cannot be forgotten; the excess observed is at most one state.
    """
    excess = set()
    for dfa in bigger_dfas(200, 130):
        twice = dfa.reversed_automaton().determinize().reversed_automaton().determinize()
        assert dfa.has_same_language_as(twice)
        excess.add(len(twice.states) - len(dfa.minimize().states))
    assert max(excess) <= 1  # (-1: minimize() of a complete DFA keeps its dead state, the partial double reversal does not)
    assert 1 in excess  # the limit is real: the composition is not a minimization


def test_audit_cartesian_product_with_every_boolean_rule():
    pool = list(bigger_dfas(24, 131))
    rules = {
        "and": lambda x, y: x and y, "or": lambda x, y: x or y,
        "minus": lambda x, y: x and not y, "xor": lambda x, y: x != y,
        "nand": lambda x, y: not (x and y),
    }
    for a, b in itertools.product(pool[:8], pool[8:16]):
        for name, rule in rules.items():
            prod = a.cartesian_product(b, rule)
            for w in words_up_to("ab", 5):
                assert automaton_accepts(prod, w) == rule(
                    automaton_accepts(a, w), automaton_accepts(b, w)
                ), (name, w)
            # only reachable pairs are built (lazy construction)
            assert len(prod.states) <= (len(a.states) + 1) * (len(b.states) + 1)
