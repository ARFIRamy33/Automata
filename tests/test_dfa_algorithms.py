"""Unit tests for DFS, BFS and accessible_states on ExtendedDFA."""

import pytest
from automata.fa.dfa import DFA

from automata_extensions.fa import ExtendedDFA


def make_dfa(transitions, initial="q0", finals=()):
    """Build a (possibly partial) ExtendedDFA from a transitions dict.

    The states are the keys of ``transitions``; the alphabet is the set of
    symbols used.  ``allow_partial=True`` lets some states lack transitions.
    """
    symbols = {a for outgoing in transitions.values() for a in outgoing}
    return ExtendedDFA(
        states=set(transitions),
        input_symbols=symbols or {"a"},
        transitions=transitions,
        initial_state=initial,
        final_states=set(finals),
        allow_partial=True,
    )


@pytest.fixture
def simple():
    """q0 --a--> q1 --b--> q2."""
    return make_dfa(
        {"q0": {"a": "q1"}, "q1": {"b": "q2"}, "q2": {}},
        finals={"q2"},
    )


@pytest.fixture
def diamond():
    """Two paths q0 -> q3: through q1 (symbol a) and through q2 (symbol b)."""
    return make_dfa(
        {
            "q0": {"a": "q1", "b": "q2"},
            "q1": {"a": "q3"},
            "q2": {"a": "q3"},
            "q3": {},
        },
        finals={"q3"},
    )


@pytest.fixture
def with_unreachable():
    """q2 points to q1 but nothing leads to q2; q3 is isolated."""
    return make_dfa({"q0": {"a": "q1"}, "q1": {}, "q2": {"a": "q1"}, "q3": {}})


@pytest.fixture
def with_loop():
    """Self-loop on q1 and a cycle q0 -> q1 -> q0."""
    return make_dfa({"q0": {"a": "q1"}, "q1": {"a": "q1", "b": "q0"}})


@pytest.fixture
def initial_without_transition():
    """q0 has no outgoing transition, although q1 leads to q0."""
    return make_dfa({"q0": {}, "q1": {"a": "q0"}})


ALL_AUTOMATA = [
    "simple",
    "diamond",
    "with_unreachable",
    "with_loop",
    "initial_without_transition",
]


# --- 1. simple automaton with three states -------------------------------


def test_simple_dfs_iterative(simple):
    assert simple.dfs_iterative() == ["q0", "q1", "q2"]


def test_simple_dfs_recursive(simple):
    assert simple.dfs_recursive() == ["q0", "q1", "q2"]


def test_simple_bfs(simple):
    assert simple.bfs() == ["q0", "q1", "q2"]


def test_simple_accessible_states(simple):
    assert simple.accessible_states() == {"q0", "q1", "q2"}


# --- 2. several paths ----------------------------------------------------


def test_diamond_dfs_goes_deep_first(diamond):
    # q3 is reached through q1 before q2 is visited.
    assert diamond.dfs_iterative() == ["q0", "q1", "q3", "q2"]
    assert diamond.dfs_recursive() == ["q0", "q1", "q3", "q2"]


def test_diamond_bfs_goes_level_by_level(diamond):
    # Level 0: q0 / level 1: q1, q2 / level 2: q3.
    assert diamond.bfs() == ["q0", "q1", "q2", "q3"]


def test_diamond_each_state_visited_once(diamond):
    for order in (diamond.dfs_iterative(), diamond.dfs_recursive(), diamond.bfs()):
        assert len(order) == len(set(order)) == 4


def test_diamond_accessible_states(diamond):
    assert diamond.accessible_states() == {"q0", "q1", "q2", "q3"}


# --- 3. unreachable state ------------------------------------------------


def test_unreachable_states_are_not_visited(with_unreachable):
    assert with_unreachable.dfs_iterative() == ["q0", "q1"]
    assert with_unreachable.dfs_recursive() == ["q0", "q1"]
    assert with_unreachable.bfs() == ["q0", "q1"]


def test_unreachable_accessible_states(with_unreachable):
    assert with_unreachable.accessible_states() == {"q0", "q1"}


# --- 4. loop -------------------------------------------------------------


def test_loop_terminates_and_visits_once(with_loop):
    assert with_loop.dfs_iterative() == ["q0", "q1"]
    assert with_loop.dfs_recursive() == ["q0", "q1"]
    assert with_loop.bfs() == ["q0", "q1"]


def test_loop_accessible_states(with_loop):
    assert with_loop.accessible_states() == {"q0", "q1"}


# --- 5. initial state without any transition -----------------------------


def test_initial_without_transition(initial_without_transition):
    a = initial_without_transition
    assert a.dfs_iterative() == ["q0"]
    assert a.dfs_recursive() == ["q0"]
    assert a.bfs() == ["q0"]
    assert a.accessible_states() == {"q0"}


# --- 6. start from a state other than the initial one --------------------


def test_start_from_another_state(diamond):
    assert diamond.dfs_iterative("q2") == ["q2", "q3"]
    assert diamond.dfs_recursive("q2") == ["q2", "q3"]
    assert diamond.bfs("q2") == ["q2", "q3"]


def test_start_from_unreachable_state(with_unreachable):
    # q2 is not accessible from q0, but it can still start a traversal.
    assert with_unreachable.bfs("q2") == ["q2", "q1"]


def test_invalid_start_state_raises(simple):
    for method in (simple.dfs_iterative, simple.dfs_recursive, simple.bfs):
        with pytest.raises(ValueError):
            method("does_not_exist")


# --- 7. DFS and BFS visit the same states --------------------------------


@pytest.mark.parametrize("name", ALL_AUTOMATA)
def test_dfs_and_bfs_visit_same_set(name, request):
    a = request.getfixturevalue(name)
    assert set(a.dfs_iterative()) == set(a.dfs_recursive()) == set(a.bfs())


@pytest.mark.parametrize("name", ALL_AUTOMATA)
def test_dfs_iterative_matches_recursive_order(name, request):
    a = request.getfixturevalue(name)
    assert a.dfs_iterative() == a.dfs_recursive()


@pytest.mark.parametrize("name", ALL_AUTOMATA)
def test_accessible_states_equals_bfs_set(name, request):
    a = request.getfixturevalue(name)
    assert a.accessible_states() == set(a.bfs())


# --- properties of the design --------------------------------------------


def test_accessible_states_is_immutable_and_contains_initial(with_unreachable):
    result = with_unreachable.accessible_states()
    assert isinstance(result, frozenset)
    assert with_unreachable.initial_state in result


def test_automaton_is_not_modified(diamond):
    transitions_before = {q: dict(t) for q, t in diamond.transitions.items()}
    diamond.dfs_iterative()
    diamond.dfs_recursive()
    diamond.bfs()
    diamond.accessible_states()
    assert {q: dict(t) for q, t in diamond.transitions.items()} == transitions_before


def test_original_dfa_class_is_untouched():
    for name in ("dfs_iterative", "dfs_recursive", "bfs", "accessible_states"):
        assert not hasattr(DFA, name)


# === is_accessible, coaccessible_states, is_coaccessible ==================


@pytest.fixture
def dead_end():
    """q1 is final; q2 is a sink (loops on itself, cannot reach a final state)."""
    return make_dfa(
        {"q0": {"a": "q1", "b": "q2"}, "q1": {}, "q2": {"a": "q2"}},
        finals={"q1"},
    )


@pytest.fixture
def mixed():
    """q0 -> q1 (final); q2 -> q1 but q2 is unreachable; q3 is isolated."""
    return make_dfa(
        {"q0": {"a": "q1"}, "q1": {}, "q2": {"a": "q1"}, "q3": {}},
        finals={"q1"},
    )


@pytest.fixture
def loop_with_final():
    """Cycle q0 <-> q1 with a self-loop on q1; q0 is final."""
    return make_dfa(
        {"q0": {"a": "q1"}, "q1": {"a": "q1", "b": "q0"}},
        finals={"q0"},
    )


# --- is_accessible --------------------------------------------------------


def test_is_accessible_true_and_false(mixed):
    assert mixed.is_accessible("q0")
    assert mixed.is_accessible("q1")
    assert not mixed.is_accessible("q2")
    assert not mixed.is_accessible("q3")


def test_is_accessible_initial_state_without_transition(initial_without_transition):
    assert initial_without_transition.is_accessible("q0")
    assert not initial_without_transition.is_accessible("q1")


def test_is_accessible_unknown_state_raises(simple):
    with pytest.raises(ValueError):
        simple.is_accessible("does_not_exist")


# --- coaccessible_states --------------------------------------------------


def test_coaccessible_simple(simple):
    assert simple.coaccessible_states() == {"q0", "q1", "q2"}


def test_coaccessible_several_paths(diamond):
    assert diamond.coaccessible_states() == {"q0", "q1", "q2", "q3"}


def test_coaccessible_excludes_sink(dead_end):
    assert dead_end.coaccessible_states() == {"q0", "q1"}


def test_coaccessible_with_loop(loop_with_final):
    assert loop_with_final.coaccessible_states() == {"q0", "q1"}


def test_coaccessible_without_final_state(with_unreachable):
    assert with_unreachable.coaccessible_states() == frozenset()


def test_coaccessible_includes_unreachable_states(mixed):
    # q2 cannot be reached from q0, but it leads to the final state q1.
    assert mixed.coaccessible_states() == {"q0", "q1", "q2"}
    assert mixed.accessible_states() == {"q0", "q1"}


def test_coaccessible_isolated_state_is_excluded(mixed):
    assert "q3" not in mixed.coaccessible_states()


def test_coaccessible_final_state_without_transition(initial_without_transition):
    a = make_dfa({"q0": {}, "q1": {"a": "q0"}}, finals={"q0"})
    assert a.coaccessible_states() == {"q0", "q1"}


def test_coaccessible_is_immutable_and_contains_final_states(dead_end):
    result = dead_end.coaccessible_states()
    assert isinstance(result, frozenset)
    assert dead_end.final_states <= result


# --- is_coaccessible ------------------------------------------------------


def test_is_coaccessible_true_and_false(dead_end):
    assert dead_end.is_coaccessible("q0")
    assert dead_end.is_coaccessible("q1")
    assert not dead_end.is_coaccessible("q2")


def test_is_coaccessible_unknown_state_raises(simple):
    with pytest.raises(ValueError):
        simple.is_coaccessible("does_not_exist")


@pytest.mark.parametrize(
    "name", ["simple", "diamond", "dead_end", "mixed", "loop_with_final"]
)
def test_is_methods_agree_with_sets(name, request):
    a = request.getfixturevalue(name)
    for q in a.states:
        assert a.is_accessible(q) == (q in a.accessible_states())
        assert a.is_coaccessible(q) == (q in a.coaccessible_states())


def test_new_methods_do_not_modify_automaton(mixed):
    transitions_before = {q: dict(t) for q, t in mixed.transitions.items()}
    finals_before = set(mixed.final_states)
    mixed.is_accessible("q0")
    mixed.coaccessible_states()
    mixed.is_coaccessible("q1")
    assert {q: dict(t) for q, t in mixed.transitions.items()} == transitions_before
    assert set(mixed.final_states) == finals_before


# === strongly_connected_components, has_cycle, has_cycle_from =============


@pytest.fixture
def one_scc():
    """Cycle q0 -> q1 -> q2 -> q0: a single strongly connected component."""
    return make_dfa({"q0": {"a": "q1"}, "q1": {"a": "q2"}, "q2": {"a": "q0"}})


@pytest.fixture
def two_cycles_chained():
    """{q0, q1} and {q2, q3} are cycles; q1 -> q2 (one way only); q4 is isolated."""
    return make_dfa(
        {
            "q0": {"a": "q1"},
            "q1": {"a": "q0", "b": "q2"},
            "q2": {"a": "q3"},
            "q3": {"a": "q2"},
            "q4": {},
        }
    )


@pytest.fixture
def dag():
    """No cycle at all: q0 -> q1 -> q2 and q0 -> q2."""
    return make_dfa({"q0": {"a": "q1", "b": "q2"}, "q1": {"a": "q2"}, "q2": {}})


@pytest.fixture
def two_cycles_sharing_state():
    """Cycles q0 <-> q1 and q1 <-> q2 share q1: they form one single SCC."""
    return make_dfa(
        {"q0": {"a": "q1"}, "q1": {"a": "q0", "b": "q2"}, "q2": {"a": "q1"}}
    )


@pytest.fixture
def self_loop():
    """q0 -> q1 and a self-loop on q1."""
    return make_dfa({"q0": {"a": "q1"}, "q1": {"a": "q1"}})


@pytest.fixture
def cycle_after_chain():
    """q0 -> q1 -> q2 <-> q3, then q3 -> q4 (dead end); q5 -> q5 is apart."""
    return make_dfa(
        {
            "q0": {"a": "q1"},
            "q1": {"a": "q2"},
            "q2": {"a": "q3"},
            "q3": {"a": "q2", "b": "q4"},
            "q4": {},
            "q5": {"a": "q5"},
        }
    )


@pytest.fixture
def cycle_only_unreachable():
    """q0 -> q1 has no cycle; the cycle q2 <-> q3 is not accessible from q0."""
    return make_dfa(
        {"q0": {"a": "q1"}, "q1": {}, "q2": {"a": "q3"}, "q3": {"a": "q2"}}
    )


@pytest.fixture
def two_cycles_reachable():
    """q0 leads to the cycle q1 <-> q2 and to the cycle q3 <-> q4."""
    return make_dfa(
        {
            "q0": {"a": "q1", "b": "q3"},
            "q1": {"a": "q2"},
            "q2": {"a": "q1"},
            "q3": {"a": "q4"},
            "q4": {"a": "q3"},
        }
    )


SCC_AUTOMATA = [
    "simple",
    "diamond",
    "with_unreachable",
    "with_loop",
    "initial_without_transition",
    "one_scc",
    "two_cycles_chained",
    "dag",
    "two_cycles_sharing_state",
    "self_loop",
    "cycle_after_chain",
]


# --- strongly_connected_components ----------------------------------------


def test_scc_single_component(one_scc):
    assert one_scc.strongly_connected_components() == {
        frozenset({"q0", "q1", "q2"})
    }


def test_scc_several_components(two_cycles_chained):
    # q1 -> q2 exists but q2 cannot come back: two different components.
    assert two_cycles_chained.strongly_connected_components() == {
        frozenset({"q0", "q1"}),
        frozenset({"q2", "q3"}),
        frozenset({"q4"}),
    }


def test_scc_isolated_state_is_its_own_component(two_cycles_chained):
    assert frozenset({"q4"}) in two_cycles_chained.strongly_connected_components()


def test_scc_no_cycle_gives_one_component_per_state(dag):
    assert dag.strongly_connected_components() == {
        frozenset({"q0"}),
        frozenset({"q1"}),
        frozenset({"q2"}),
    }


def test_scc_several_cycles_sharing_a_state_merge(two_cycles_sharing_state):
    assert two_cycles_sharing_state.strongly_connected_components() == {
        frozenset({"q0", "q1", "q2"})
    }


def test_scc_respects_direction_of_transitions(simple):
    # q0 -> q1 -> q2 is connected if the arrows are ignored, but it is not
    # strongly connected: connectivity is not strong connectivity.
    assert simple.strongly_connected_components() == {
        frozenset({"q0"}),
        frozenset({"q1"}),
        frozenset({"q2"}),
    }


def test_scc_self_loop_does_not_merge_states(self_loop):
    assert self_loop.strongly_connected_components() == {
        frozenset({"q0"}),
        frozenset({"q1"}),
    }


def test_scc_includes_states_not_accessible_from_initial(with_unreachable):
    assert with_unreachable.strongly_connected_components() == {
        frozenset({"q0"}),
        frozenset({"q1"}),
        frozenset({"q2"}),
        frozenset({"q3"}),
    }


@pytest.mark.parametrize("name", SCC_AUTOMATA)
def test_scc_is_a_partition_of_the_states(name, request):
    a = request.getfixturevalue(name)
    components = a.strongly_connected_components()
    all_states = [q for component in components for q in component]
    assert sorted(all_states) == sorted(a.states)  # each state exactly once
    assert all(component for component in components)  # none is empty


def test_scc_on_long_path_does_not_hit_recursion_limit():
    n = 3000
    transitions = {f"q{i}": {"a": f"q{i + 1}"} for i in range(n - 1)}
    transitions[f"q{n - 1}"] = {}
    long_path = make_dfa(transitions)
    assert len(long_path.strongly_connected_components()) == n


# --- has_cycle -------------------------------------------------------------


def test_has_cycle_false_without_cycle(dag):
    assert not dag.has_cycle()


def test_has_cycle_false_on_chain(simple):
    assert not simple.has_cycle()


def test_has_cycle_simple_cycle(one_scc):
    assert one_scc.has_cycle()


def test_has_cycle_several_cycles(two_cycles_chained):
    assert two_cycles_chained.has_cycle()


def test_has_cycle_self_loop(self_loop):
    assert self_loop.has_cycle()


def test_has_cycle_joining_paths_are_not_a_cycle(diamond):
    # q3 is reached twice (through q1 and q2), which is not a cycle.
    assert not diamond.has_cycle()


def test_has_cycle_sees_cycle_in_unreachable_part(cycle_only_unreachable):
    assert cycle_only_unreachable.has_cycle()


def test_has_cycle_unreachable_states_without_cycle(with_unreachable):
    assert not with_unreachable.has_cycle()


def test_has_cycle_on_long_path_does_not_hit_recursion_limit():
    n = 3000
    transitions = {f"q{i}": {"a": f"q{i + 1}"} for i in range(n - 1)}
    transitions[f"q{n - 1}"] = {}
    assert not make_dfa(transitions).has_cycle()
    transitions[f"q{n - 1}"] = {"a": "q0"}
    assert make_dfa(transitions).has_cycle()


# --- has_cycle_from --------------------------------------------------------


def test_cycle_from_state_without_reachable_cycle(cycle_after_chain):
    assert not cycle_after_chain.has_cycle_from("q4")


def test_cycle_from_state_that_reaches_a_cycle(cycle_after_chain):
    assert cycle_after_chain.has_cycle_from("q0")


def test_cycle_from_state_inside_a_cycle(cycle_after_chain):
    assert cycle_after_chain.has_cycle_from("q2")
    assert cycle_after_chain.has_cycle_from("q3")


def test_cycle_from_state_leading_to_a_cycle(cycle_after_chain):
    # q1 is not in the cycle but leads to it.
    assert cycle_after_chain.has_cycle_from("q1")


def test_cycle_from_ignores_cycles_that_are_not_accessible(cycle_after_chain):
    # q5 has a self-loop, but q4 cannot reach it.
    assert cycle_after_chain.has_cycle()
    assert not cycle_after_chain.has_cycle_from("q4")


def test_cycle_from_self_loop_state(cycle_after_chain):
    assert cycle_after_chain.has_cycle_from("q5")


def test_cycle_from_unknown_state_raises(simple):
    with pytest.raises(ValueError):
        simple.has_cycle_from("does_not_exist")


def test_cycle_from_with_several_accessible_cycles(two_cycles_reachable):
    assert two_cycles_reachable.has_cycle_from("q0")


def test_cycle_from_differs_from_global_detection(cycle_only_unreachable):
    assert cycle_only_unreachable.has_cycle()
    assert not cycle_only_unreachable.has_cycle_from("q0")
    assert cycle_only_unreachable.has_cycle_from("q2")


def test_new_methods_do_not_modify_the_automaton(two_cycles_chained):
    before = {q: dict(t) for q, t in two_cycles_chained.transitions.items()}
    two_cycles_chained.strongly_connected_components()
    two_cycles_chained.has_cycle()
    two_cycles_chained.has_cycle_from("q0")
    after = {q: dict(t) for q, t in two_cycles_chained.transitions.items()}
    assert after == before


# === is_empty, is_infinite, accepts =======================================

# Native isempty()/isfinite() exist on the real DFA of automata-lib 9.2.0.
# They are used as an independent cross-check when available.
needs_native = pytest.mark.skipif(
    not (hasattr(DFA, "isempty") and hasattr(DFA, "isfinite")),
    reason="native isempty/isfinite not available",
)


@pytest.fixture
def no_final_accessible():
    """q0 -> q1, no final state at all."""
    return make_dfa({"q0": {"a": "q1"}, "q1": {}})


@pytest.fixture
def no_transition_no_final():
    """A single state without transition and not final."""
    return make_dfa({"q0": {}})


@pytest.fixture
def initial_is_final():
    """A single state without transition, which is final."""
    return make_dfa({"q0": {}}, finals={"q0"})


@pytest.fixture
def final_unreachable():
    """q2 is final but nothing leads to it from q0."""
    return make_dfa(
        {"q0": {"a": "q1"}, "q1": {}, "q2": {"a": "q1"}}, finals={"q2"}
    )


@pytest.fixture
def useless_cycle_unreachable():
    """q0 -> q1 (final).  The cycle q2 <-> q3 leads to q1 (coaccessible) but
    nothing leads to it from q0 (not accessible)."""
    return make_dfa(
        {
            "q0": {"a": "q1"},
            "q1": {},
            "q2": {"a": "q3"},
            "q3": {"a": "q2", "b": "q1"},
        },
        finals={"q1"},
    )


@pytest.fixture
def one_useful_cycle_among_two():
    """Useful cycle q1 <-> q2 (q1 -> q4 final); useless self-loop on q3."""
    return make_dfa(
        {
            "q0": {"a": "q1", "b": "q3"},
            "q1": {"a": "q2", "b": "q4"},
            "q2": {"a": "q1"},
            "q3": {"a": "q3"},
            "q4": {},
        },
        finals={"q4"},
    )


@pytest.fixture
def cycle_with_final_self_loop():
    """q0 -> q1, q1 is final and loops on itself: a*-like infinite language."""
    return make_dfa({"q0": {"a": "q1"}, "q1": {"a": "q1"}}, finals={"q1"})


@pytest.fixture
def even_a():
    """Complete DFA over {a, b}: words with an even number of 'a'."""
    return ExtendedDFA(
        states={"even", "odd"},
        input_symbols={"a", "b"},
        transitions={
            "even": {"a": "odd", "b": "even"},
            "odd": {"a": "even", "b": "odd"},
        },
        initial_state="even",
        final_states={"even"},
    )


@pytest.fixture
def only_ab():
    """Partial DFA recognising exactly the word 'ab'."""
    return make_dfa(
        {"q0": {"a": "q1"}, "q1": {"b": "q2"}, "q2": {}}, finals={"q2"}
    )


@pytest.fixture
def empty_word_loop():
    """Initial state is final and loops on 'a' (language a*)."""
    return make_dfa({"q0": {"a": "q0"}}, finals={"q0"})


# --- is_empty --------------------------------------------------------------


def test_is_empty_no_accessible_final_state(no_final_accessible):
    assert no_final_accessible.is_empty()


def test_is_empty_false_with_accessible_final_state(simple):
    assert not simple.is_empty()


def test_is_empty_automaton_without_transition(no_transition_no_final):
    assert no_transition_no_final.is_empty()


def test_is_empty_false_when_initial_state_is_final(initial_is_final):
    assert not initial_is_final.is_empty()


def test_is_empty_unreachable_final_state_does_not_count(final_unreachable):
    assert final_unreachable.is_empty()


def test_is_empty_with_cycle_but_no_final_state(with_loop):
    assert with_loop.is_empty()


# --- is_infinite -----------------------------------------------------------


def test_is_infinite_accessible_and_coaccessible_cycle(loop_with_final):
    assert loop_with_final.is_infinite()


def test_is_infinite_final_self_loop(cycle_with_final_self_loop):
    assert cycle_with_final_self_loop.is_infinite()


def test_is_infinite_false_for_unreachable_cycle(useless_cycle_unreachable):
    # The cycle exists and is coaccessible, but q0 can never enter it.
    assert useless_cycle_unreachable.has_cycle()
    assert not useless_cycle_unreachable.is_infinite()


def test_is_infinite_false_for_cycle_that_cannot_reach_a_final_state(dead_end):
    # q2 is accessible and loops, but no final state can be reached from it.
    assert dead_end.has_cycle_from("q0")
    assert not dead_end.is_infinite()


def test_is_infinite_false_without_cycle(diamond):
    assert not diamond.is_infinite()


def test_is_infinite_one_useful_cycle_among_two(one_useful_cycle_among_two):
    assert one_useful_cycle_among_two.is_infinite()


def test_is_infinite_false_if_only_the_useless_cycle_remains():
    # Same shape as above, but the useful cycle is removed.
    a = make_dfa(
        {
            "q0": {"a": "q1", "b": "q3"},
            "q1": {"b": "q4"},
            "q3": {"a": "q3"},
            "q4": {},
        },
        finals={"q4"},
    )
    assert not a.is_infinite()


def test_is_infinite_false_for_finite_language(only_ab):
    assert not only_ab.is_infinite()


def test_is_infinite_false_for_empty_language(with_loop):
    # Empty language (no final state) with a cycle: finite, not infinite.
    assert with_loop.is_empty()
    assert not with_loop.is_infinite()


# --- cross-check with the native methods ----------------------------------

NATIVE_CHECK_AUTOMATA = [
    "simple",
    "diamond",
    "with_unreachable",
    "with_loop",
    "dead_end",
    "mixed",
    "loop_with_final",
    "no_final_accessible",
    "initial_is_final",
    "final_unreachable",
    "useless_cycle_unreachable",
    "one_useful_cycle_among_two",
    "cycle_with_final_self_loop",
    "only_ab",
    "empty_word_loop",
]


@needs_native
@pytest.mark.parametrize("name", NATIVE_CHECK_AUTOMATA)
def test_is_empty_agrees_with_native_isempty(name, request):
    a = request.getfixturevalue(name)
    assert a.is_empty() == a.isempty()


@needs_native
@pytest.mark.parametrize("name", NATIVE_CHECK_AUTOMATA)
def test_is_infinite_agrees_with_native_isfinite(name, request):
    a = request.getfixturevalue(name)
    assert a.is_infinite() == (not a.isfinite())


# --- accepts ---------------------------------------------------------------


def test_accepts_accepted_word(only_ab):
    assert only_ab.accepts("ab")


def test_accepts_rejected_word(only_ab):
    assert not only_ab.accepts("a")  # stops in a non-final state


def test_accepts_empty_word_when_initial_state_is_final(empty_word_loop):
    assert empty_word_loop.accepts("")


def test_accepts_empty_word_rejected_when_initial_state_is_not_final(only_ab):
    assert not only_ab.accepts("")


def test_accepts_word_with_several_symbols(even_a):
    assert even_a.accepts("aab")
    assert even_a.accepts("abba")
    assert not even_a.accepts("ab")
    assert not even_a.accepts("bab")


def test_accepts_loops_over_long_words(empty_word_loop):
    assert empty_word_loop.accepts("a" * 5000)


def test_accepts_symbol_outside_the_alphabet_is_false(only_ab):
    assert not only_ab.accepts("ac")
    assert not only_ab.accepts("c")


def test_accepts_symbol_outside_the_alphabet_even_if_state_is_final(
    empty_word_loop,
):
    # Without the check, "ab" could be wrongly handled; the alphabet is {a}.
    assert not empty_word_loop.accepts("ab")


def test_accepts_interrupted_by_missing_transition(only_ab):
    # 'b' is in the alphabet but q0 has no transition on 'b'.
    assert not only_ab.accepts("b")
    # After "ab" we are in q2, which has no transition at all.
    assert not only_ab.accepts("aba")
    assert not only_ab.accepts("abab")


@pytest.mark.parametrize("word", ["", "a", "b", "ab", "ba", "aab", "bab", "abba"])
def test_accepts_agrees_with_native_accepts_input(even_a, word):
    assert even_a.accepts(word) == even_a.accepts_input(word)


def test_accepts_does_not_modify_the_automaton(only_ab):
    before = {q: dict(t) for q, t in only_ab.transitions.items()}
    only_ab.accepts("ab")
    only_ab.accepts("zz")
    assert {q: dict(t) for q, t in only_ab.transitions.items()} == before


def test_new_names_are_not_native_to_dfa():
    for name in ("is_empty", "is_infinite", "accepts"):
        assert not hasattr(DFA, name)


# === execution_trace, reachable_from, predecessor_states ==================

from dataclasses import FrozenInstanceError  # noqa: E402

from automata_extensions.fa.mixins.analyse import (  # noqa: E402
    AnalyseMixin,
    ExecutionTrace,
)


class ToyNFA(AnalyseMixin):
    """Tiny NFA used ONLY to test the generic trace logic with sets of states.

    It is not ExtendedNFA (which does not exist yet) and has no epsilon
    transitions.  It implements the three hooks of the mixin.
    """

    def __init__(self, transitions, initial, finals):
        self.transitions = transitions
        self.initial_state = initial
        self.final_states = frozenset(finals)

    def _initial_configuration(self):
        return frozenset({self.initial_state})

    def _next_configuration(self, configuration, symbol):
        following = frozenset(
            target
            for state in configuration
            for target in self.transitions.get(state, {}).get(symbol, ())
        )
        return following or None

    def _is_accepting_configuration(self, configuration):
        return not configuration.isdisjoint(self.final_states)


@pytest.fixture
def ends_with_ab():
    """NFA for words ending with 'ab': p -a-> {p, q}, p -b-> p, q -b-> r."""
    return ToyNFA(
        {"p": {"a": {"p", "q"}, "b": {"p"}}, "q": {"b": {"r"}}},
        initial="p",
        finals={"r"},
    )


@pytest.fixture
def fan_in():
    """q4 -> q0, and q0, q1, q2 all lead to q3 (several direct predecessors)."""
    return make_dfa(
        {
            "q0": {"a": "q3"},
            "q1": {"a": "q3"},
            "q2": {"a": "q3"},
            "q3": {},
            "q4": {"a": "q0"},
        }
    )


# --- execution_trace: DFA ---------------------------------------------------


def test_trace_accepted_word(only_ab):
    trace = only_ab.execution_trace("ab")
    assert trace.configurations == ("q0", "q1", "q2")
    assert trace.accepted is True
    assert trace.blocked_at is None


def test_trace_rejected_word_keeps_every_step(even_a):
    trace = even_a.execution_trace("ab")
    assert trace.configurations == ("even", "odd", "odd")
    assert trace.accepted is False
    assert trace.blocked_at is None  # fully read, but not in a final state


def test_trace_several_symbols_checks_each_step(even_a):
    trace = even_a.execution_trace("aabab")
    # even -a-> odd -a-> even -b-> even -a-> odd -b-> odd
    assert trace.configurations == ("even", "odd", "even", "even", "odd", "odd")
    assert trace.accepted is False


def test_trace_empty_word_accepted(empty_word_loop):
    trace = empty_word_loop.execution_trace("")
    assert trace.configurations == ("q0",)
    assert trace.accepted is True
    assert trace.blocked_at is None


def test_trace_empty_word_rejected(only_ab):
    trace = only_ab.execution_trace("")
    assert trace.configurations == ("q0",)
    assert trace.accepted is False


def test_trace_missing_transition_stops_the_trace(only_ab):
    trace = only_ab.execution_trace("aba")
    # "ab" is read, then q2 has no transition on 'a'.
    assert trace.configurations == ("q0", "q1", "q2")
    assert trace.blocked_at == 2
    assert trace.accepted is False  # even though q2 is final


def test_trace_blocked_on_first_symbol(only_ab):
    trace = only_ab.execution_trace("b")
    assert trace.configurations == ("q0",)
    assert trace.blocked_at == 0
    assert trace.accepted is False


def test_trace_symbol_outside_the_alphabet(only_ab):
    trace = only_ab.execution_trace("ac")
    assert trace.configurations == ("q0", "q1")
    assert trace.blocked_at == 1
    assert trace.accepted is False


def test_trace_has_one_more_configuration_than_symbols_when_fully_read(even_a):
    for word in ["", "a", "ab", "abba"]:
        assert len(even_a.execution_trace(word).configurations) == len(word) + 1


@pytest.mark.parametrize("word", ["", "a", "b", "ab", "aa", "aab", "abc", "ba", "z"])
def test_trace_accepted_agrees_with_accepts(only_ab, even_a, word):
    for automaton in (only_ab, even_a):
        assert automaton.execution_trace(word).accepted == automaton.accepts(word)


def test_trace_is_immutable_and_keeps_the_word(only_ab):
    trace = only_ab.execution_trace("ab")
    assert isinstance(trace, ExecutionTrace)
    assert trace.word == "ab"
    with pytest.raises(FrozenInstanceError):
        trace.accepted = False


def test_trace_does_not_modify_the_automaton(only_ab):
    before = {q: dict(t) for q, t in only_ab.transitions.items()}
    only_ab.execution_trace("ab")
    only_ab.execution_trace("abz")
    assert {q: dict(t) for q, t in only_ab.transitions.items()} == before


@pytest.mark.skipif(
    not hasattr(DFA, "read_input_stepwise"), reason="native stepwise reading missing"
)
@pytest.mark.parametrize("word", ["", "a", "ab", "aab", "abba", "bbb"])
def test_trace_agrees_with_native_read_input_stepwise(even_a, word):
    # Only the configurations after each symbol are compared, so the result
    # does not depend on whether the native generator also yields the
    # initial configuration.
    native = list(even_a.read_input_stepwise(word, ignore_rejection=True))
    expected = native[len(native) - len(word):]
    assert list(even_a.execution_trace(word).configurations[1:]) == expected


# --- execution_trace: NFA (toy class, sets of states) ---------------------------


def test_trace_nfa_follows_all_the_paths(ends_with_ab):
    trace = ends_with_ab.execution_trace("aab")
    assert trace.configurations == (
        frozenset({"p"}),
        frozenset({"p", "q"}),
        frozenset({"p", "q"}),
        frozenset({"p", "r"}),
    )
    assert trace.accepted is True


def test_trace_nfa_rejected_word(ends_with_ab):
    trace = ends_with_ab.execution_trace("abb")
    assert trace.configurations == (
        frozenset({"p"}),
        frozenset({"p", "q"}),
        frozenset({"p", "r"}),
        frozenset({"p"}),
    )
    assert trace.accepted is False


def test_trace_nfa_empty_word(ends_with_ab):
    trace = ends_with_ab.execution_trace("")
    assert trace.configurations == (frozenset({"p"}),)
    assert trace.accepted is False


def test_trace_nfa_blocked_when_the_set_becomes_empty(ends_with_ab):
    trace = ends_with_ab.execution_trace("ac")
    assert trace.configurations == (frozenset({"p"}), frozenset({"p", "q"}))
    assert trace.blocked_at == 1
    assert trace.accepted is False


# --- reachable_from ---------------------------------------------------------


def test_reachable_from_state_with_several_successors(diamond):
    assert diamond.reachable_from("q0") == {"q0", "q1", "q2", "q3"}


def test_reachable_from_state_without_successor(diamond):
    assert diamond.reachable_from("q3") == {"q3"}


def test_reachable_from_graph_with_several_paths(diamond):
    assert diamond.reachable_from("q1") == {"q1", "q3"}


def test_reachable_from_handles_cycles(with_loop):
    assert with_loop.reachable_from("q1") == {"q0", "q1"}
    assert with_loop.reachable_from("q0") == {"q0", "q1"}


def test_reachable_from_state_other_than_initial(diamond):
    assert diamond.initial_state == "q0"
    assert diamond.reachable_from("q2") == {"q2", "q3"}


def test_reachable_from_inaccessible_state(with_unreachable):
    # q2 cannot be reached from q0, but q1 can be reached from q2.
    assert "q2" not in with_unreachable.accessible_states()
    assert with_unreachable.reachable_from("q2") == {"q2", "q1"}


def test_reachable_from_includes_the_start_state(initial_without_transition):
    assert initial_without_transition.reachable_from("q0") == {"q0"}


def test_reachable_from_unknown_state_raises(simple):
    with pytest.raises(ValueError):
        simple.reachable_from("does_not_exist")
    with pytest.raises(ValueError):
        simple.reachable_from(None)


def test_reachable_from_initial_state_equals_accessible_states(with_unreachable):
    initial = with_unreachable.initial_state
    assert with_unreachable.reachable_from(initial) == with_unreachable.accessible_states()


def test_reachable_from_returns_an_immutable_set(diamond):
    assert isinstance(diamond.reachable_from("q0"), frozenset)


# --- predecessor_states -------------------------------------------------------


def test_predecessors_single_direct_predecessor(simple):
    assert simple.predecessor_states("q1") == {"q0"}


def test_predecessors_several_direct_predecessors(fan_in):
    # q0, q1, q2 are direct; q4 is indirect (q4 -> q0 -> q3).
    assert fan_in.predecessor_states("q3") == {"q0", "q1", "q2", "q4"}


def test_predecessors_several_levels(simple):
    assert simple.predecessor_states("q2") == {"q0", "q1"}


def test_predecessors_indirect_ones_are_included(diamond):
    # q1 and q2 are direct predecessors of q3, q0 is indirect.
    assert diamond.predecessor_states("q3") == {"q0", "q1", "q2"}


def test_predecessors_state_without_predecessor(simple):
    assert simple.predecessor_states("q0") == frozenset()


def test_predecessors_isolated_state(with_unreachable):
    assert with_unreachable.predecessor_states("q3") == frozenset()


def test_predecessors_target_is_not_included_without_cycle(simple):
    assert "q2" not in simple.predecessor_states("q2")


def test_predecessors_cycle_includes_the_target(with_loop):
    # q0 -> q1 -> q0: each state can come back to itself.
    assert with_loop.predecessor_states("q0") == {"q0", "q1"}
    assert with_loop.predecessor_states("q1") == {"q0", "q1"}


def test_predecessors_self_loop_on_the_target(self_loop):
    assert self_loop.predecessor_states("q1") == {"q0", "q1"}


def test_predecessors_self_loop_elsewhere_does_not_include_target(self_loop):
    # q0 has a successor (q1) but nobody leads to q0, even with a loop on q1.
    assert self_loop.predecessor_states("q0") == frozenset()


def test_predecessors_unknown_state_raises(simple):
    with pytest.raises(ValueError):
        simple.predecessor_states("does_not_exist")


def test_predecessors_returns_an_immutable_set(diamond):
    assert isinstance(diamond.predecessor_states("q3"), frozenset)


@pytest.mark.parametrize("name", SCC_AUTOMATA)
def test_predecessors_match_paths_of_positive_length(name, request):
    # q is a predecessor of t  <=>  t is reachable from some successor of q.
    a = request.getfixturevalue(name)
    for target in a.states:
        for state in a.states:
            reaches_target = any(
                target in a.reachable_from(successor)
                for successor in a._successors(state)
            )
            assert (state in a.predecessor_states(target)) == reaches_target


@pytest.mark.parametrize("name", ["simple", "dead_end", "mixed", "loop_with_final"])
def test_coaccessible_states_are_final_states_and_their_predecessors(name, request):
    a = request.getfixturevalue(name)
    expected = set(a.final_states)
    for final in a.final_states:
        expected |= a.predecessor_states(final)
    assert a.coaccessible_states() == expected


# --- names and automaton integrity ----------------------------------------------


def test_new_names_are_not_native_to_dfa():
    for name in ("execution_trace", "reachable_from", "predecessor_states"):
        assert not hasattr(DFA, name)


def test_native_predecessors_and_successors_are_not_shadowed():
    # automata-lib already has DFA.predecessors / DFA.successors (on words).
    for name in ("predecessors", "successors"):
        assert getattr(ExtendedDFA, name, None) is getattr(DFA, name, None)


def test_reachability_methods_do_not_modify_the_automaton(diamond):
    before = {q: dict(t) for q, t in diamond.transitions.items()}
    diamond.reachable_from("q0")
    diamond.predecessor_states("q3")
    assert {q: dict(t) for q, t in diamond.transitions.items()} == before


# === induced_subautomaton, is_complete, complete ==========================

from itertools import product  # noqa: E402


def snapshot(automaton):
    """Everything that defines an automaton, as plain comparable data."""
    return (
        frozenset(automaton.states),
        frozenset(automaton.input_symbols),
        {q: dict(t) for q, t in automaton.transitions.items()},
        automaton.initial_state,
        frozenset(automaton.final_states),
    )


def words_up_to(symbols, max_length):
    """All words over ``symbols`` of length 0..max_length."""
    for length in range(max_length + 1):
        for letters in product(symbols, repeat=length):
            yield "".join(letters)


@pytest.fixture
def chain4():
    """q0 -a-> q1 -a-> q2 -a-> q3 and q1 -b-> q3; finals q2 and q3."""
    return make_dfa(
        {
            "q0": {"a": "q1"},
            "q1": {"a": "q2", "b": "q3"},
            "q2": {"a": "q3"},
            "q3": {},
        },
        finals={"q2", "q3"},
    )


@pytest.fixture
def one_missing():
    """Alphabet {a, b}; only q1 has no transition on 'b'."""
    return make_dfa({"q0": {"a": "q0", "b": "q1"}, "q1": {"a": "q0"}}, finals={"q1"})


@pytest.fixture
def complete_but_flag_partial():
    """Every transition is defined, although allow_partial=True is used."""
    return make_dfa({"q0": {"a": "q0", "b": "q1"}, "q1": {"a": "q0", "b": "q1"}})


# --- induced_subautomaton -------------------------------------------------------


def test_induced_with_several_states(chain4):
    sub = chain4.induced_subautomaton({"q0", "q1", "q3"})
    assert sub.states == {"q0", "q1", "q3"}
    assert sub.transitions == {"q0": {"a": "q1"}, "q1": {"b": "q3"}, "q3": {}}
    assert sub.initial_state == "q0"
    assert sub.final_states == {"q3"}


def test_induced_keeps_only_transitions_with_both_ends_kept(chain4):
    sub = chain4.induced_subautomaton({"q0", "q1", "q3"})
    # q1 -a-> q2 is removed because q2 is not kept; q1 -b-> q3 stays.
    assert "a" not in sub.transitions["q1"]
    assert sub.transitions["q1"] == {"b": "q3"}


def test_induced_with_all_the_states_gives_the_same_automaton(chain4):
    sub = chain4.induced_subautomaton(chain4.states)
    assert snapshot(sub) == snapshot(chain4)
    assert sub is not chain4


def test_induced_with_a_single_state(chain4):
    sub = chain4.induced_subautomaton({"q0"})
    assert sub.states == {"q0"}
    assert sub.transitions == {"q0": {}}
    assert sub.final_states == frozenset()


def test_induced_single_state_that_is_initial_and_final(initial_is_final):
    sub = initial_is_final.induced_subautomaton({"q0"})
    assert sub.final_states == {"q0"}
    assert sub.accepts("")


def test_induced_keeps_self_loops_of_kept_states(self_loop):
    sub = self_loop.induced_subautomaton({"q0", "q1"})
    assert sub.transitions["q1"] == {"a": "q1"}


def test_induced_empty_set_raises(chain4):
    with pytest.raises(ValueError):
        chain4.induced_subautomaton(set())


def test_induced_without_the_initial_state_raises(chain4):
    with pytest.raises(ValueError):
        chain4.induced_subautomaton({"q1", "q2", "q3"})


def test_induced_with_unknown_state_raises(chain4):
    with pytest.raises(ValueError):
        chain4.induced_subautomaton({"q0", "does_not_exist"})


def test_induced_drops_final_states_that_are_not_kept(chain4):
    sub = chain4.induced_subautomaton({"q0", "q1", "q2"})  # q3 (final) removed
    assert sub.final_states == {"q2"}


def test_induced_all_final_states_removed_gives_empty_language(chain4):
    sub = chain4.induced_subautomaton({"q0", "q1"})
    assert sub.final_states == frozenset()
    assert sub.is_empty()


def test_induced_keeps_the_alphabet(chain4):
    # Symbol 'b' is no longer used by any transition but stays in the alphabet.
    sub = chain4.induced_subautomaton({"q0", "q2"})
    assert sub.input_symbols == chain4.input_symbols == {"a", "b"}
    assert sub.transitions == {"q0": {}, "q2": {}}


def test_induced_accepts_any_iterable_and_duplicates(chain4):
    expected = snapshot(chain4.induced_subautomaton({"q0", "q1"}))
    assert snapshot(chain4.induced_subautomaton(["q0", "q1", "q1"])) == expected
    assert snapshot(chain4.induced_subautomaton(q for q in ("q0", "q1"))) == expected


def test_induced_returns_an_extended_dfa(chain4):
    assert isinstance(chain4.induced_subautomaton({"q0", "q1"}), ExtendedDFA)


def test_induced_does_not_modify_the_original(chain4):
    before = snapshot(chain4)
    chain4.induced_subautomaton({"q0", "q1", "q3"})
    chain4.induced_subautomaton({"q0"})
    assert snapshot(chain4) == before


def test_induced_language_is_included_in_the_original_language(chain4):
    sub = chain4.induced_subautomaton({"q0", "q1", "q3"})
    for word in words_up_to("ab", 5):
        if sub.accepts(word):
            assert chain4.accepts(word)


def test_induced_result_works_with_the_other_algorithms(chain4):
    sub = chain4.induced_subautomaton({"q0", "q1", "q3"})
    assert sub.accessible_states() == {"q0", "q1", "q3"}
    assert sub.accepts("ab")
    assert not sub.accepts("aa")


# --- is_complete ------------------------------------------------------------------


def test_is_complete_on_a_complete_dfa(even_a):
    assert even_a.is_complete()


def test_is_complete_false_with_one_missing_transition(one_missing):
    assert not one_missing.is_complete()


def test_is_complete_false_with_several_missing_transitions(only_ab):
    # q0 lacks 'b', q1 lacks 'a', q2 lacks both.
    assert not only_ab.is_complete()


def test_is_complete_false_for_state_without_any_transition(
    initial_without_transition,
):
    assert not initial_without_transition.is_complete()


def test_is_complete_single_state_with_loops():
    assert make_dfa({"q0": {"a": "q0", "b": "q0"}}).is_complete()


def test_is_complete_does_not_depend_on_the_allow_partial_flag(
    complete_but_flag_partial, even_a
):
    # even_a was built with allow_partial=False (library checked it), the other
    # one with allow_partial=True: both tables are complete.
    assert complete_but_flag_partial.is_complete()
    assert even_a.is_complete()


def test_completeness_is_not_determinism(one_missing):
    # A DFA is deterministic by construction: at most one target per (state,
    # symbol).  It is still not complete.
    assert isinstance(one_missing, DFA)
    assert not one_missing.is_complete()
    # A transition table is a dict: a (state, symbol) pair has 0 or 1 target.
    assert all(
        isinstance(target, str)
        for outgoing in one_missing.transitions.values()
        for target in outgoing.values()
    )


def test_is_complete_does_not_modify_the_automaton(one_missing):
    before = snapshot(one_missing)
    one_missing.is_complete()
    assert snapshot(one_missing) == before


# --- complete ---------------------------------------------------------------------


def test_complete_adds_one_sink_state(only_ab):
    completed = only_ab.complete()
    new_states = completed.states - only_ab.states
    assert len(new_states) == 1
    assert completed.is_complete()


def test_complete_sends_missing_transitions_to_the_sink(only_ab):
    completed = only_ab.complete()
    (sink,) = completed.states - only_ab.states
    for state in only_ab.states:
        for symbol in only_ab.input_symbols:
            expected = only_ab.transitions[state].get(symbol, sink)
            assert completed.transitions[state][symbol] == expected


def test_complete_sink_loops_on_every_symbol(only_ab):
    completed = only_ab.complete()
    (sink,) = completed.states - only_ab.states
    assert completed.transitions[sink] == {a: sink for a in only_ab.input_symbols}


def test_complete_sink_is_not_final(only_ab):
    completed = only_ab.complete()
    (sink,) = completed.states - only_ab.states
    assert sink not in completed.final_states
    assert completed.final_states == only_ab.final_states


def test_complete_keeps_initial_state_and_alphabet(only_ab):
    completed = only_ab.complete()
    assert completed.initial_state == only_ab.initial_state
    assert completed.input_symbols == only_ab.input_symbols


def test_complete_on_an_already_complete_dfa_adds_no_state(even_a):
    completed = even_a.complete()
    assert snapshot(completed) == snapshot(even_a)
    assert completed is not even_a


def test_complete_on_a_complete_table_built_with_partial_flag(
    complete_but_flag_partial,
):
    completed = complete_but_flag_partial.complete()
    assert completed.states == complete_but_flag_partial.states


def test_complete_with_chosen_sink_name(only_ab):
    completed = only_ab.complete(sink_state="sink")
    assert completed.states == only_ab.states | {"sink"}
    assert completed.transitions["sink"] == {"a": "sink", "b": "sink"}
    assert completed.transitions["q0"]["b"] == "sink"


def test_complete_does_not_modify_the_original(only_ab):
    before = snapshot(only_ab)
    only_ab.complete()
    only_ab.complete(sink_state="sink")
    assert snapshot(only_ab) == before


def test_complete_returns_an_extended_dfa(only_ab):
    assert isinstance(only_ab.complete(), ExtendedDFA)


def test_complete_is_idempotent(only_ab):
    once = only_ab.complete()
    twice = once.complete()
    assert snapshot(twice) == snapshot(once)


COMPLETE_CHECK_AUTOMATA = [
    "simple",
    "diamond",
    "with_unreachable",
    "with_loop",
    "initial_without_transition",
    "dead_end",
    "chain4",
    "one_missing",
    "only_ab",
    "empty_word_loop",
    "initial_is_final",
]


@pytest.mark.parametrize("name", COMPLETE_CHECK_AUTOMATA)
def test_complete_preserves_the_language(name, request):
    a = request.getfixturevalue(name)
    completed = a.complete()
    assert completed.is_complete()
    # Words over the alphabet plus one foreign symbol 'z', up to length 5.
    for word in words_up_to(sorted(a.input_symbols) + ["z"], 5):
        assert completed.accepts(word) == a.accepts(word)


@pytest.mark.parametrize("name", COMPLETE_CHECK_AUTOMATA)
def test_complete_keeps_every_original_transition(name, request):
    a = request.getfixturevalue(name)
    completed = a.complete()
    for state, outgoing in a.transitions.items():
        for symbol, target in outgoing.items():
            assert completed.transitions[state][symbol] == target


# --- names and native methods ------------------------------------------------------


def test_new_names_do_not_replace_native_methods():
    for name in ("induced_subautomaton", "is_complete", "complete"):
        assert not hasattr(DFA, name)
    # The native method wrapped by complete() is still the native one.
    assert getattr(ExtendedDFA, "to_complete") is getattr(DFA, "to_complete")


# === complemented, cartesian_product, is_included_in ======================

import operator  # noqa: E402


@pytest.fixture
def a_even():
    """Over {a}: an even number of 'a' (complete)."""
    return make_dfa({"e": {"a": "o"}, "o": {"a": "e"}}, initial="e", finals={"e"})


@pytest.fixture
def empty_ab():
    """Complete DFA over {a, b} with no final state: the empty language."""
    return make_dfa({"q0": {"a": "q0", "b": "q0"}})


@pytest.fixture
def universal_ab():
    """Complete DFA over {a, b} accepting every word."""
    return make_dfa({"q0": {"a": "q0", "b": "q0"}}, finals={"q0"})


@pytest.fixture
def odd_a():
    """Complete DFA over {a, b}: an odd number of 'a'."""
    return ExtendedDFA(
        states={"even", "odd"},
        input_symbols={"a", "b"},
        transitions={
            "even": {"a": "odd", "b": "even"},
            "odd": {"a": "even", "b": "odd"},
        },
        initial_state="even",
        final_states={"odd"},
    )


@pytest.fixture
def ab_or_b():
    """Partial DFA recognising exactly {ab, b}."""
    return make_dfa(
        {"q0": {"a": "q1", "b": "q2"}, "q1": {"b": "q2"}, "q2": {}}, finals={"q2"}
    )


@pytest.fixture
def only_c():
    """Alphabet {c}: recognises exactly the word 'c'."""
    return make_dfa({"r0": {"c": "r1"}, "r1": {}}, initial="r0", finals={"r1"})


@pytest.fixture
def b_star():
    """Alphabet {b}: recognises b*."""
    return make_dfa({"y0": {"b": "y0"}}, initial="y0", finals={"y0"})


@pytest.fixture
def period2():
    """x0 <-> x1 over {a}; final x0 (an even number of 'a')."""
    return make_dfa({"x0": {"a": "x1"}, "x1": {"a": "x0"}}, initial="x0", finals={"x0"})


@pytest.fixture
def sticky():
    """y0 -a-> y1, y1 loops; final y1 (at least one 'a')."""
    return make_dfa({"y0": {"a": "y1"}, "y1": {"a": "y1"}}, initial="y0", finals={"y1"})


PAIR_AUTOMATA = [
    "simple",
    "diamond",
    "with_loop",
    "dead_end",
    "chain4",
    "one_missing",
    "only_ab",
    "even_a",
    "odd_a",
    "empty_word_loop",
    "a_even",
    "only_c",
    "ab_or_b",
    "empty_ab",
    "universal_ab",
    "initial_without_transition",
    "b_star",
]

PRODUCT_RULES = {
    "and": lambda a, b: a and b,
    "or": lambda a, b: a or b,
    "difference": lambda a, b: a and not b,
    "xor": lambda a, b: a != b,
}


def all_automata(request):
    return {name: request.getfixturevalue(name) for name in PAIR_AUTOMATA}


# --- complemented ---------------------------------------------------------------


def test_complement_of_a_complete_dfa_swaps_final_states(even_a):
    c = even_a.complemented()
    assert c.states == even_a.states
    assert c.transitions == even_a.transitions
    assert c.initial_state == even_a.initial_state
    assert c.final_states == {"odd"}


def test_complement_of_a_complete_dfa_recognises_the_opposite(even_a):
    c = even_a.complemented()
    for word in words_up_to("ab", 5):
        assert c.accepts(word) == (not even_a.accepts(word))


def test_complement_of_an_incomplete_dfa_adds_a_sink(only_ab):
    assert not only_ab.is_complete()
    c = only_ab.complemented()
    assert c.is_complete()
    assert len(c.states) == len(only_ab.states) + 1


def test_complement_of_an_incomplete_dfa_accepts_blocked_words(only_ab):
    c = only_ab.complemented()
    assert not c.accepts("ab")  # the only word of the original language
    assert c.accepts("")
    assert c.accepts("b")  # blocked in the original (no transition): now accepted
    assert c.accepts("aba")
    for word in words_up_to("ab", 5):
        assert c.accepts(word) == (not only_ab.accepts(word))


def test_complement_of_the_empty_language_is_universal(empty_ab):
    c = empty_ab.complemented()
    assert c.final_states == c.states
    assert not c.is_empty()
    assert all(c.accepts(word) for word in words_up_to("ab", 4))


def test_complement_of_the_universal_language_is_empty(universal_ab):
    c = universal_ab.complemented()
    assert c.is_empty()
    assert not any(c.accepts(word) for word in words_up_to("ab", 4))


def test_complement_keeps_the_alphabet(only_ab, even_a, a_even):
    for automaton in (only_ab, even_a, a_even):
        assert automaton.complemented().input_symbols == automaton.input_symbols


def test_complement_rejects_words_with_foreign_symbols(even_a):
    # Σ* is taken over the alphabet: such a word is in neither language.
    assert not even_a.accepts("z")
    assert not even_a.complemented().accepts("z")


def test_complement_does_not_modify_the_original(only_ab, even_a):
    for automaton in (only_ab, even_a):
        before = snapshot(automaton)
        automaton.complemented()
        assert snapshot(automaton) == before


def test_complement_returns_a_new_complete_extended_dfa(even_a):
    c = even_a.complemented()
    assert isinstance(c, ExtendedDFA)
    assert c is not even_a
    assert c.is_complete()


def test_complement_every_word_is_in_exactly_one_language(request):
    for name, a in all_automata(request).items():
        c = a.complemented()
        for word in words_up_to(sorted(a.input_symbols), 4):
            assert a.accepts(word) != c.accepts(word), (name, word)


def test_complement_twice_gives_the_same_language(request):
    for name, a in all_automata(request).items():
        twice = a.complemented().complemented()
        for word in words_up_to(sorted(a.input_symbols), 4):
            assert twice.accepts(word) == a.accepts(word), (name, word)


@pytest.mark.skipif(not hasattr(DFA, "complement"), reason="native complement missing")
@pytest.mark.parametrize("name", ["even_a", "odd_a", "only_ab", "ab_or_b", "universal_ab"])
def test_complement_agrees_with_native_complement(name, request):
    a = request.getfixturevalue(name)
    mine, native = a.complemented(), a.complement()
    for word in words_up_to(sorted(a.input_symbols), 5):
        assert mine.accepts(word) == native.accepts_input(word)


# --- cartesian_product ------------------------------------------------------------


def test_product_states_are_reachable_pairs_only(period2, sticky):
    p = period2.cartesian_product(sticky, operator.and_)
    assert p.states == {("x0", "y0"), ("x1", "y1"), ("x0", "y1")}
    # The 4th possible pair ("x1", "y0") cannot be reached: it is not built.
    assert ("x1", "y0") not in p.states
    assert p.accessible_states() == p.states


def test_product_initial_state_is_the_pair_of_initial_states(period2, sticky):
    p = period2.cartesian_product(sticky, operator.and_)
    assert p.initial_state == ("x0", "y0")


def test_product_transitions_move_both_components(period2, sticky):
    p = period2.cartesian_product(sticky, operator.and_)
    assert p.transitions == {
        ("x0", "y0"): {"a": ("x1", "y1")},
        ("x1", "y1"): {"a": ("x0", "y1")},
        ("x0", "y1"): {"a": ("x1", "y1")},
    }


def test_product_final_states_follow_the_rule(period2, sticky):
    expected = {
        "and": {("x0", "y1")},
        "or": {("x0", "y0"), ("x1", "y1"), ("x0", "y1")},
        "difference": {("x0", "y0")},
        "xor": {("x0", "y0"), ("x1", "y1")},
    }
    for rule_name, finals in expected.items():
        p = period2.cartesian_product(sticky, PRODUCT_RULES[rule_name])
        assert p.final_states == finals, rule_name


def test_product_is_not_always_an_intersection(period2, sticky):
    inter = period2.cartesian_product(sticky, PRODUCT_RULES["and"])
    union = period2.cartesian_product(sticky, PRODUCT_RULES["or"])
    assert inter.transitions == union.transitions  # same product graph
    assert inter.final_states != union.final_states  # different final states


def test_product_with_identical_alphabets_has_no_blocked_component(period2, sticky):
    p = period2.cartesian_product(sticky, operator.and_)
    assert p.input_symbols == {"a"}
    assert all(None not in pair for pair in p.states)
    assert p.is_complete()


def test_product_with_different_alphabets(empty_word_loop, b_star):
    # a* over {a} and b* over {b}: the product reads {a, b}.
    p = empty_word_loop.cartesian_product(b_star, operator.and_)
    assert p.input_symbols == {"a", "b"}
    assert p.initial_state == ("q0", "y0")
    assert p.states == {("q0", "y0"), ("q0", None), (None, "y0"), (None, None)}
    assert p.transitions[("q0", "y0")] == {"a": ("q0", None), "b": (None, "y0")}
    assert p.transitions[("q0", None)] == {"a": ("q0", None), "b": (None, None)}
    assert p.transitions[(None, None)] == {"a": (None, None), "b": (None, None)}
    assert p.is_complete()


def test_product_different_alphabets_intersection_language(empty_word_loop, b_star):
    p = empty_word_loop.cartesian_product(b_star, operator.and_)
    assert p.final_states == {("q0", "y0")}
    assert p.accepts("")  # the only common word
    assert not p.accepts("a")
    assert not p.accepts("b")
    assert not p.accepts("ab")


def test_product_different_alphabets_union_language(empty_word_loop, b_star):
    # The rule "or" needs the blocked components: a word of a* is still
    # accepted although b* cannot read 'a'.
    p = empty_word_loop.cartesian_product(b_star, operator.or_)
    assert p.accepts("")
    assert p.accepts("aaa")
    assert p.accepts("bb")
    assert not p.accepts("ab")
    assert not p.accepts("ba")


def test_product_with_several_states_each(chain4, even_a):
    p = chain4.cartesian_product(even_a, operator.and_)
    assert p.initial_state == ("q0", "even")
    assert p.accessible_states() == p.states
    assert len(p.states) <= len(chain4.states) * len(even_a.states)
    assert p.accepts("aa")  # q0 -a-> q1 -a-> q2 (final); two 'a' (even)
    assert not p.accepts("a")


def test_product_accepts_a_native_dfa_as_second_operand(period2):
    plain = DFA(
        states={"y0", "y1"},
        input_symbols={"a"},
        transitions={"y0": {"a": "y1"}, "y1": {"a": "y1"}},
        initial_state="y0",
        final_states={"y1"},
    )
    p = period2.cartesian_product(plain, operator.and_)
    assert isinstance(p, ExtendedDFA)
    assert p.final_states == {("x0", "y1")}


@pytest.mark.parametrize("rule_name", sorted(PRODUCT_RULES))
def test_product_recognises_the_combination_of_the_two_languages(rule_name, request):
    rule = PRODUCT_RULES[rule_name]
    automata = all_automata(request)
    for name_a, a in automata.items():
        for name_b, b in automata.items():
            p = a.cartesian_product(b, rule)
            assert p.is_complete()
            alphabet = sorted(set(a.input_symbols) | set(b.input_symbols)) + ["z"]
            for word in words_up_to(alphabet, 3):
                expected = rule(a.accepts(word), b.accepts(word))
                assert p.accepts(word) == expected, (rule_name, name_a, name_b, word)


def test_product_does_not_modify_the_operands(period2, sticky):
    before_a, before_b = snapshot(period2), snapshot(sticky)
    period2.cartesian_product(sticky, operator.and_)
    assert snapshot(period2) == before_a
    assert snapshot(sticky) == before_b


# --- is_included_in -----------------------------------------------------------------


def test_inclusion_true(a_even, empty_word_loop):
    # (aa)* is included in a*.
    assert a_even.is_included_in(empty_word_loop)


def test_inclusion_false(a_even, empty_word_loop):
    assert not empty_word_loop.is_included_in(a_even)  # "a" is in a* only


def test_inclusion_is_not_equality(only_ab, ab_or_b):
    assert only_ab.is_included_in(ab_or_b)
    assert not ab_or_b.is_included_in(only_ab)  # "b" is only in ab_or_b


def test_inclusion_of_two_equal_languages(empty_word_loop):
    # Same language a*, recognised by two different DFAs.
    same = make_dfa(
        {"s0": {"a": "s1"}, "s1": {"a": "s1"}}, initial="s0", finals={"s0", "s1"}
    )
    assert empty_word_loop.is_included_in(same)
    assert same.is_included_in(empty_word_loop)


def test_every_language_is_included_in_itself(request):
    for name, a in all_automata(request).items():
        assert a.is_included_in(a), name


def test_empty_language_is_included_in_any_language(empty_ab, only_ab, even_a):
    assert empty_ab.is_included_in(only_ab)
    assert empty_ab.is_included_in(even_a)
    assert empty_ab.is_included_in(empty_ab)


def test_empty_language_is_included_even_with_another_alphabet(only_ab):
    empty_c = make_dfa({"r": {"c": "r"}}, initial="r")
    assert empty_c.is_included_in(only_ab)


def test_language_without_reachable_final_state_is_included(
    initial_without_transition, only_ab
):
    assert initial_without_transition.is_included_in(only_ab)


def test_any_language_is_included_in_the_universal_language(
    only_ab, even_a, empty_ab, universal_ab
):
    for automaton in (only_ab, even_a, empty_ab, universal_ab):
        assert automaton.is_included_in(universal_ab)


def test_universal_language_is_not_included_in_a_smaller_one(universal_ab, only_ab):
    assert not universal_ab.is_included_in(only_ab)


def test_inclusion_with_incomplete_automata(only_ab, ab_or_b):
    assert not only_ab.is_complete()
    assert not ab_or_b.is_complete()
    assert only_ab.is_included_in(ab_or_b)
    assert not ab_or_b.is_included_in(only_ab)


def test_inclusion_with_different_alphabets(only_c, only_ab, a_even, even_a):
    # "c" is accepted by only_c but cannot even be read by only_ab.
    assert not only_c.is_included_in(only_ab)
    # (aa)* over {a} is included in "even number of a" over {a, b}...
    assert a_even.is_included_in(even_a)
    # ... but not the other way: "b" is accepted by even_a, unreadable by a_even.
    assert not even_a.is_included_in(a_even)


def test_inclusion_over_a_smaller_alphabet_in_the_universal_language(
    empty_word_loop, universal_ab
):
    assert empty_word_loop.is_included_in(universal_ab)  # a* inside {a, b}*
    assert not universal_ab.is_included_in(empty_word_loop)


def test_inclusion_agrees_with_intersection_with_the_complement(request):
    # L(A) ⊆ L(B)  <=>  L(A) ∩ complement(L(B)) is empty, over all pairs.
    automata = all_automata(request)
    for name_a, a in automata.items():
        for name_b, b in automata.items():
            witness = a.cartesian_product(b, PRODUCT_RULES["difference"])
            assert a.is_included_in(b) == witness.is_empty(), (name_a, name_b)


def test_inclusion_false_means_a_counterexample_word_may_be_found(request):
    # If a short word is accepted by A and rejected by B, A is not included.
    automata = all_automata(request)
    for name_a, a in automata.items():
        for name_b, b in automata.items():
            alphabet = sorted(set(a.input_symbols) | set(b.input_symbols))
            counterexample = any(
                a.accepts(w) and not b.accepts(w) for w in words_up_to(alphabet, 4)
            )
            if counterexample:
                assert not a.is_included_in(b), (name_a, name_b)


def test_inclusion_does_not_modify_the_operands(only_ab, ab_or_b):
    before_a, before_b = snapshot(only_ab), snapshot(ab_or_b)
    only_ab.is_included_in(ab_or_b)
    ab_or_b.is_included_in(only_ab)
    assert snapshot(only_ab) == before_a
    assert snapshot(ab_or_b) == before_b


@pytest.mark.skipif(not hasattr(DFA, "issubset"), reason="native issubset missing")
@pytest.mark.parametrize("name_a", ["even_a", "odd_a", "empty_ab", "universal_ab"])
@pytest.mark.parametrize("name_b", ["even_a", "odd_a", "empty_ab", "universal_ab"])
def test_inclusion_agrees_with_native_issubset(name_a, name_b, request):
    a = request.getfixturevalue(name_a)
    b = request.getfixturevalue(name_b)
    assert a.is_included_in(b) == a.issubset(b)


# --- names and native methods ---------------------------------------------------------


def test_new_names_do_not_replace_native_methods():
    for name in ("complemented", "cartesian_product", "is_included_in"):
        assert not hasattr(DFA, name)
    for name in ("complement", "issubset", "intersection"):
        assert getattr(ExtendedDFA, name, None) is getattr(DFA, name, None)


# === has_same_language_as, is_isomorphic_to, is_prefix_closed =============


def renamed(automaton, naming):
    """Copy of ``automaton`` with its states renamed through ``naming``."""
    return ExtendedDFA(
        states={naming[q] for q in automaton.states},
        input_symbols=automaton.input_symbols,
        transitions={
            naming[q]: {a: naming[t] for a, t in out.items()}
            for q, out in automaton.transitions.items()
        },
        initial_state=naming[automaton.initial_state],
        final_states={naming[q] for q in automaton.final_states},
        allow_partial=True,
    )


def shuffled_names(automaton):
    """A renaming that is not the identity: new names, reverse order."""
    ordered = sorted(automaton.states, key=repr, reverse=True)
    return {state: ("n", index) for index, state in enumerate(ordered)}


@pytest.fixture
def two_state_a_star():
    """a* again, with two final states s0 -a-> s1 -a-> s1."""
    return make_dfa(
        {"s0": {"a": "s1"}, "s1": {"a": "s1"}}, initial="s0", finals={"s0", "s1"}
    )


@pytest.fixture
def two_cycle_a_star():
    """a* again, with a 2-cycle t0 <-> t1 (both final)."""
    return make_dfa(
        {"t0": {"a": "t1"}, "t1": {"a": "t0"}}, initial="t0", finals={"t0", "t1"}
    )


# --- has_same_language_as ---------------------------------------------------------


def test_same_language_identical_automata(only_ab):
    assert only_ab.has_same_language_as(only_ab)
    assert only_ab.has_same_language_as(renamed(only_ab, {q: q for q in only_ab.states}))


def test_same_language_with_different_states(empty_word_loop, two_state_a_star):
    # Both recognise a*, with 1 and 2 states.
    assert empty_word_loop.has_same_language_as(two_state_a_star)
    assert two_state_a_star.has_same_language_as(empty_word_loop)


def test_same_language_false_for_different_languages(only_ab, ab_or_b):
    assert not only_ab.has_same_language_as(ab_or_b)
    assert not ab_or_b.has_same_language_as(only_ab)


def test_same_language_with_incomplete_automata(only_ab, ab_or_b):
    assert not only_ab.is_complete() and not ab_or_b.is_complete()
    assert only_ab.has_same_language_as(only_ab.complete())
    assert ab_or_b.has_same_language_as(ab_or_b.complete())
    assert not only_ab.has_same_language_as(ab_or_b)


def test_same_language_with_different_alphabets(empty_word_loop):
    # a* read by a DFA over {a} and by a partial DFA over {a, b} with no 'b'.
    wider = ExtendedDFA(
        states={"q"},
        input_symbols={"a", "b"},
        transitions={"q": {"a": "q"}},
        initial_state="q",
        final_states={"q"},
        allow_partial=True,
    )
    assert empty_word_loop.has_same_language_as(wider)
    assert wider.has_same_language_as(empty_word_loop)


def test_same_language_false_when_the_wider_alphabet_is_really_used(
    a_even, even_a
):
    # even_a accepts "b", which a_even cannot read.
    assert not a_even.has_same_language_as(even_a)


def test_same_language_both_empty(empty_ab, initial_without_transition):
    empty_c = make_dfa({"r": {"c": "r"}}, initial="r")
    assert empty_ab.has_same_language_as(initial_without_transition)
    assert empty_ab.has_same_language_as(empty_c)  # even with another alphabet
    assert empty_c.has_same_language_as(empty_ab)


def test_same_language_empty_versus_non_empty(empty_ab, only_ab):
    assert not empty_ab.has_same_language_as(only_ab)
    assert not only_ab.has_same_language_as(empty_ab)


def test_same_language_when_both_inclusions_hold(a_even, empty_word_loop):
    # Here only one inclusion holds: (aa)* is in a*, not the reverse.
    assert a_even.is_included_in(empty_word_loop)
    assert not empty_word_loop.is_included_in(a_even)
    assert not a_even.has_same_language_as(empty_word_loop)
    # For equal languages both inclusions hold.
    same = make_dfa(
        {"s0": {"a": "s1"}, "s1": {"a": "s2"}, "s2": {"a": "s1"}},
        initial="s0",
        finals={"s0", "s2"},
    )  # (aa)* again: s0, then s1 <-> s2 alternating
    assert a_even.is_included_in(same) and same.is_included_in(a_even)
    assert a_even.has_same_language_as(same)


def test_same_language_is_not_structural_equality(empty_word_loop, two_state_a_star):
    assert empty_word_loop.has_same_language_as(two_state_a_star)
    assert snapshot(empty_word_loop) != snapshot(two_state_a_star)


def test_same_language_accepts_a_native_dfa(only_ab):
    plain = DFA(
        states={"x", "y", "z"},
        input_symbols={"a", "b"},
        transitions={"x": {"a": "y"}, "y": {"b": "z"}, "z": {}},
        initial_state="x",
        final_states={"z"},
        allow_partial=True,
    )
    assert only_ab.has_same_language_as(plain)


def test_same_language_does_not_modify_the_operands(only_ab, ab_or_b):
    before_a, before_b = snapshot(only_ab), snapshot(ab_or_b)
    only_ab.has_same_language_as(ab_or_b)
    assert snapshot(only_ab) == before_a
    assert snapshot(ab_or_b) == before_b


def test_same_language_relation_properties(request):
    automata = all_automata(request)
    for name_a, a in automata.items():
        assert a.has_same_language_as(a), name_a
        for name_b, b in automata.items():
            equal = a.has_same_language_as(b)
            assert equal == b.has_same_language_as(a), (name_a, name_b)
            assert equal == (a.is_included_in(b) and b.is_included_in(a))
            symmetric_difference = a.cartesian_product(b, PRODUCT_RULES["xor"])
            assert equal == symmetric_difference.is_empty(), (name_a, name_b)


def test_same_language_false_when_a_short_word_differs(request):
    automata = all_automata(request)
    for name_a, a in automata.items():
        for name_b, b in automata.items():
            alphabet = sorted(set(a.input_symbols) | set(b.input_symbols))
            if any(a.accepts(w) != b.accepts(w) for w in words_up_to(alphabet, 4)):
                assert not a.has_same_language_as(b), (name_a, name_b)


def test_equality_operator_is_not_redefined():
    assert ExtendedDFA.__eq__ is DFA.__eq__
    assert ExtendedDFA.__hash__ is DFA.__hash__


# --- is_isomorphic_to ---------------------------------------------------------------


def test_isomorphic_same_automaton_with_other_state_names(only_ab):
    other = renamed(only_ab, {"q0": "start", "q1": "middle", "q2": "end"})
    assert only_ab.is_isomorphic_to(other)
    assert other.is_isomorphic_to(only_ab)


def test_isomorphic_with_a_complex_renaming(request):
    for name, a in all_automata(request).items():
        b = renamed(a, shuffled_names(a))
        assert a.is_isomorphic_to(b), name
        assert b.is_isomorphic_to(a), name
        assert a.is_isomorphic_to(a), name


def test_isomorphic_renaming_with_swapped_roles():
    # Same shape; the names of the two non-initial states are swapped.
    a = make_dfa(
        {"q0": {"a": "q1", "b": "q2"}, "q1": {"a": "q1"}, "q2": {}}, finals={"q1"}
    )
    b = make_dfa(
        {"q0": {"a": "q2", "b": "q1"}, "q2": {"a": "q2"}, "q1": {}}, finals={"q2"}
    )
    assert a.is_isomorphic_to(b)


def test_same_language_but_not_isomorphic_different_number_of_states(
    empty_word_loop, two_state_a_star
):
    assert empty_word_loop.has_same_language_as(two_state_a_star)
    assert not empty_word_loop.is_isomorphic_to(two_state_a_star)


def test_same_language_and_same_size_but_not_isomorphic(
    two_state_a_star, two_cycle_a_star
):
    # a* with a loop versus a* with a 2-cycle, both with 2 final states.
    assert two_state_a_star.has_same_language_as(two_cycle_a_star)
    assert len(two_state_a_star.states) == len(two_cycle_a_star.states)
    assert not two_state_a_star.is_isomorphic_to(two_cycle_a_star)


def test_not_isomorphic_with_a_different_number_of_states(only_ab, chain4):
    assert not only_ab.is_isomorphic_to(chain4)


def test_not_isomorphic_with_different_transitions():
    a = make_dfa({"p0": {"a": "p1"}, "p1": {"a": "p1"}}, initial="p0", finals={"p1"})
    b = make_dfa({"q0": {"a": "q1"}, "q1": {"a": "q0"}}, initial="q0", finals={"q1"})
    assert not a.is_isomorphic_to(b)


def test_not_isomorphic_when_a_symbol_differs():
    # Same shape, but the transition is labelled 'a' in one and 'b' in the other.
    a = make_dfa({"p0": {"a": "p1"}, "p1": {}}, initial="p0", finals={"p1"})
    b = make_dfa({"q0": {"b": "q1"}, "q1": {}}, initial="q0", finals={"q1"})
    assert not a.is_isomorphic_to(b)


def test_not_isomorphic_when_a_transition_goes_elsewhere():
    a = make_dfa({"p0": {"a": "p1", "b": "p0"}, "p1": {}}, initial="p0", finals={"p1"})
    b = make_dfa({"q0": {"a": "q1", "b": "q1"}, "q1": {}}, initial="q0", finals={"q1"})
    assert not a.is_isomorphic_to(b)


def test_not_isomorphic_with_different_final_states():
    a = make_dfa({"p0": {"a": "p1"}, "p1": {}}, initial="p0", finals={"p1"})
    b = make_dfa({"q0": {"a": "q1"}, "q1": {}}, initial="q0", finals={"q0"})
    c = make_dfa({"r0": {"a": "r1"}, "r1": {}}, initial="r0", finals={"r0", "r1"})
    assert not a.is_isomorphic_to(b)  # a different state is final
    assert not a.is_isomorphic_to(c)  # a different number of final states


def test_not_isomorphic_with_different_alphabets():
    narrow = make_dfa({"q": {"a": "q"}}, initial="q", finals={"q"})
    wide = ExtendedDFA(
        states={"q"},
        input_symbols={"a", "b"},
        transitions={"q": {"a": "q"}},
        initial_state="q",
        final_states={"q"},
        allow_partial=True,
    )
    assert narrow.has_same_language_as(wide)  # same language...
    assert not narrow.is_isomorphic_to(wide)  # ...but the alphabet differs


def test_not_isomorphic_when_the_initial_state_is_not_matched():
    # Same states, same transitions, but the initial state is not the same one.
    a = make_dfa({"x": {"a": "y"}, "y": {"a": "y"}}, initial="x", finals={"y"})
    b = make_dfa({"x": {"a": "y"}, "y": {"a": "y"}}, initial="y", finals={"y"})
    assert not a.is_isomorphic_to(b)


def test_isomorphic_with_unaccessible_states():
    a = make_dfa(
        {"q0": {"a": "q0"}, "u1": {"a": "q0"}, "u2": {}}, initial="q0", finals={"q0"}
    )
    b = make_dfa(
        {"r0": {"a": "r0"}, "v2": {}, "v1": {"a": "r0"}}, initial="r0", finals={"r0"}
    )
    assert a.is_isomorphic_to(b)


def test_not_isomorphic_when_unaccessible_parts_differ():
    a = make_dfa({"q0": {"a": "q0"}, "u": {}}, initial="q0", finals={"q0"})
    b = make_dfa({"r0": {"a": "r0"}, "v": {}}, initial="r0", finals={"r0", "v"})
    c = make_dfa({"s0": {"a": "s0"}, "w": {"a": "w"}}, initial="s0", finals={"s0"})
    assert not a.is_isomorphic_to(b)  # the free state is final in b only
    assert not a.is_isomorphic_to(c)  # the free state loops in c only


def test_isomorphism_search_can_backtrack():
    # Free states u1 -a-> u2 (u2 final) and u3 -a-> u4 (u4 not final) on one
    # side; the other side lists the same two chains in the opposite order.
    # The first guess for u1 (a chain ending in a non-final state) passes the
    # local checks and only fails one step later, so the search must undo it.
    a = make_dfa(
        {
            "q0": {},
            "u1": {"a": "u2"},
            "u2": {},
            "u3": {"a": "u4"},
            "u4": {},
        },
        initial="q0",
        finals={"q0", "u2"},
    )
    b = make_dfa(
        {
            "r0": {},
            "v1": {"a": "v2"},
            "v2": {},
            "v3": {"a": "v4"},
            "v4": {},
        },
        initial="r0",
        finals={"r0", "v4"},
    )
    assert a.is_isomorphic_to(b)
    assert b.is_isomorphic_to(a)


def test_not_isomorphic_when_the_labels_differ_with_the_same_alphabet():
    # Same alphabet {a, b} and same shape, but the transitions use other symbols.
    a = make_dfa({"p0": {"a": "p1"}, "p1": {"b": "p1"}}, initial="p0", finals={"p1"})
    b = make_dfa({"q0": {"b": "q1"}, "q1": {"a": "q1"}}, initial="q0", finals={"q1"})
    assert a.input_symbols == b.input_symbols == {"a", "b"}
    assert not a.is_isomorphic_to(b)


def test_isomorphism_search_must_undo_a_wrong_guess():
    # Integer state names, so the order in which the states are tried is the
    # same on every run (ascending).  In A, 2 -a-> 1 and 3 is a lone state.
    # In B, 3 -a-> 2 and 1 is a lone state.  The first guess 1 -> 1 passes the
    # local checks but makes 2 impossible to map: it has to be undone, then
    # 1 -> 2 is the right choice.
    a = make_dfa({0: {}, 1: {}, 2: {"a": 1}, 3: {}}, initial=0)
    b = make_dfa({0: {}, 1: {}, 2: {}, 3: {"a": 2}}, initial=0)
    assert a.is_isomorphic_to(b)
    assert b.is_isomorphic_to(a)


def test_isomorphism_implies_same_language_and_is_symmetric(request):
    automata = all_automata(request)
    for name_a, a in automata.items():
        for name_b, b in automata.items():
            iso = a.is_isomorphic_to(b)
            assert iso == b.is_isomorphic_to(a), (name_a, name_b)
            if iso:
                assert a.has_same_language_as(b), (name_a, name_b)
                assert len(a.states) == len(b.states)


def test_isomorphism_accepts_a_native_dfa(only_ab):
    plain = DFA(
        states={"x", "y", "z"},
        input_symbols={"a", "b"},
        transitions={"x": {"a": "y"}, "y": {"b": "z"}, "z": {}},
        initial_state="x",
        final_states={"z"},
        allow_partial=True,
    )
    assert only_ab.is_isomorphic_to(plain)


def test_isomorphism_does_not_modify_the_operands(only_ab, ab_or_b):
    before_a, before_b = snapshot(only_ab), snapshot(ab_or_b)
    only_ab.is_isomorphic_to(renamed(only_ab, shuffled_names(only_ab)))
    only_ab.is_isomorphic_to(ab_or_b)
    assert snapshot(only_ab) == before_a
    assert snapshot(ab_or_b) == before_b


# --- is_prefix_closed -----------------------------------------------------------------


@pytest.fixture
def prefix_closed_language():
    """{ε, a, ab}: every state is final."""
    return make_dfa(
        {"q0": {"a": "q1"}, "q1": {"b": "q2"}, "q2": {}}, finals={"q0", "q1", "q2"}
    )


def test_prefix_closed_language(prefix_closed_language):
    assert prefix_closed_language.is_prefix_closed()


def test_language_that_is_not_prefix_closed(only_ab, a_even):
    assert not only_ab.is_prefix_closed()  # "ab" is accepted, "" and "a" are not
    assert not a_even.is_prefix_closed()  # "aa" accepted but "a" is not


def test_empty_language_is_prefix_closed(empty_ab, initial_without_transition):
    assert empty_ab.is_prefix_closed()
    assert initial_without_transition.is_prefix_closed()


def test_language_with_only_the_empty_word(initial_is_final):
    assert initial_is_final.is_prefix_closed()
    # Same language {ε}, with a dead non-final state after the first symbol.
    with_dead_state = make_dfa(
        {"q0": {"a": "q1"}, "q1": {"a": "q1"}}, initial="q0", finals={"q0"}
    )
    assert with_dead_state.is_prefix_closed()


def test_language_with_several_words_sharing_prefixes():
    # {ε, a, aa, ab}
    language = make_dfa(
        {
            "q0": {"a": "q1"},
            "q1": {"a": "q2", "b": "q3"},
            "q2": {},
            "q3": {},
        },
        finals={"q0", "q1", "q2", "q3"},
    )
    assert language.is_prefix_closed()
    # {a, aa, ab} without the empty word: not prefix-closed.
    without_epsilon = make_dfa(
        {
            "q0": {"a": "q1"},
            "q1": {"a": "q2", "b": "q3"},
            "q2": {},
            "q3": {},
        },
        finals={"q1", "q2", "q3"},
    )
    assert not without_epsilon.is_prefix_closed()


def test_word_with_a_prefix_that_is_not_accepted():
    # {ε, ab}: the prefix "a" of "ab" is not accepted.
    language = make_dfa(
        {"q0": {"a": "q1"}, "q1": {"b": "q2"}, "q2": {}}, finals={"q0", "q2"}
    )
    assert not language.is_prefix_closed()


def test_prefix_closed_with_unaccessible_states():
    # The unaccessible states u0, u1 are not final, but they play no role.
    language = make_dfa(
        {"q0": {"a": "q1"}, "q1": {}, "u0": {"a": "u1"}, "u1": {"a": "q1"}},
        initial="q0",
        finals={"q0", "q1"},
    )
    assert language.is_prefix_closed()
    # An unaccessible final state does not help either.
    unaccessible_final = make_dfa(
        {"q0": {"a": "q1"}, "q1": {}, "u": {}},
        initial="q0",
        finals={"q1", "u"},
    )
    assert not unaccessible_final.is_prefix_closed()  # "a" accepted, "" not


def test_prefix_closed_ignores_a_non_coaccessible_sink(prefix_closed_language):
    # The completed DFA has a non-final sink; the language is the same.
    completed = prefix_closed_language.complete()
    assert completed.is_prefix_closed()
    assert completed.final_states < completed.states


def test_prefix_closed_matches_the_definition_on_all_the_automata(request):
    # Brute force: every prefix of every accepted word (up to a length that
    # is enough for these small automata) must be accepted.
    for name, a in all_automata(request).items():
        alphabet = sorted(a.input_symbols)
        longest = 2 * len(a.states)
        expected = all(
            a.accepts(word[:cut])
            for word in words_up_to(alphabet, longest)
            if a.accepts(word)
            for cut in range(len(word) + 1)
        )
        assert a.is_prefix_closed() == expected, name


def test_prefix_closed_does_not_modify_the_automaton(only_ab):
    before = snapshot(only_ab)
    only_ab.is_prefix_closed()
    assert snapshot(only_ab) == before


# --- names -------------------------------------------------------------------------------


def test_new_names_are_not_native_to_dfa():
    for name in ("has_same_language_as", "is_isomorphic_to", "is_prefix_closed"):
        assert not hasattr(DFA, name)


# =====================================================================================
# Functions 28, 29, 30: largest prefix-closed sublanguage, distinguishable states,
# Myhill-Nerode classes
# =====================================================================================


def brute_prefix_closed_part(automaton, longest):
    """Words of L all of whose prefixes are in L (brute force)."""
    return {
        word
        for word in words_up_to(sorted(automaton.input_symbols), longest)
        if all(automaton.accepts(word[:cut]) for cut in range(len(word) + 1))
    }


def language_up_to(automaton, longest):
    return {
        word
        for word in words_up_to(sorted(automaton.input_symbols), longest)
        if automaton.accepts(word)
    }


@pytest.fixture
def repairable():
    """L = {ε, a, ab, abb, b...}: q0, q1 final; q2 not final; q3 final behind q2."""
    return make_dfa(
        {
            "q0": {"a": "q1", "b": "q2"},
            "q1": {"a": "q1", "b": "q3"},
            "q2": {"a": "q3"},
            "q3": {},
        },
        finals={"q0", "q1", "q3"},
    )


# --- function 28 ---------------------------------------------------------------------


def test_28_prefix_closed_language_is_unchanged(prefix_closed_language):
    result = prefix_closed_language.largest_prefix_closed_sublanguage()
    assert language_up_to(result, 4) == language_up_to(prefix_closed_language, 4)
    assert result.is_prefix_closed()


def test_28_not_prefix_closed_language_is_repaired(repairable):
    result = repairable.largest_prefix_closed_sublanguage()
    # "ba" is accepted by the original but "b" is not: it must disappear.
    assert repairable.accepts("ba")
    assert not result.accepts("ba")
    assert result.accepts("") and result.accepts("a") and result.accepts("aab")
    assert result.is_prefix_closed()
    assert not repairable.is_prefix_closed()


def test_28_matches_the_definition_by_brute_force(request):
    for name, a in all_automata(request).items():
        result = a.largest_prefix_closed_sublanguage()
        longest = 2 * len(a.states)
        assert language_up_to(result, longest) == brute_prefix_closed_part(
            a, longest
        ), name


def test_28_result_is_included_in_the_original_and_prefix_closed(request):
    for name, a in all_automata(request).items():
        result = a.largest_prefix_closed_sublanguage()
        assert language_up_to(result, 5) <= language_up_to(a, 5), name
        assert result.is_prefix_closed(), name


def test_28_empty_language(simple):
    # q0 is not final: even though "ab" is accepted, K is empty.
    result = simple.largest_prefix_closed_sublanguage()
    assert result.isempty() if hasattr(result, "isempty") else result.is_empty()
    assert language_up_to(result, 4) == set()


def test_28_empty_word_depends_on_the_initial_state(only_ab, prefix_closed_language):
    assert not only_ab.largest_prefix_closed_sublanguage().accepts("")
    assert prefix_closed_language.largest_prefix_closed_sublanguage().accepts("")


def test_28_several_final_states(repairable):
    result = repairable.largest_prefix_closed_sublanguage()
    # q3 is final and reached through final states only (q0, q1): kept.
    assert result.final_states == {"q0", "q1", "q3"}
    assert result.accepts("ab") and not result.accepts("ba")


def test_28_inaccessible_states_are_dropped():
    a = make_dfa(
        {"q0": {"a": "q1"}, "q1": {}, "q9": {"a": "q1"}},
        finals={"q0", "q1", "q9"},
    )
    result = a.largest_prefix_closed_sublanguage()
    assert result.states == {"q0", "q1"}


def test_28_does_not_modify_the_automaton(repairable):
    before = snapshot(repairable)
    result = repairable.largest_prefix_closed_sublanguage()
    assert snapshot(repairable) == before
    assert result is not repairable
    assert isinstance(result, ExtendedDFA)


# --- function 29 ---------------------------------------------------------------------


def test_29_one_final_one_not_final(simple):
    assert simple.distinguishing_word("q1", "q2") == ""
    assert simple.are_distinguishable("q1", "q2")


def test_29_shortest_distinguishing_word(simple):
    assert simple.distinguishing_word("q0", "q1") == "b"
    assert simple.accepts("ab") and not simple.accepts("b")


def test_29_identical_states_are_not_distinguishable(simple):
    assert simple.distinguishing_word("q1", "q1") is None
    assert not simple.are_distinguishable("q1", "q1")


def test_29_different_transitions(even_a):
    word = even_a.distinguishing_word("even", "odd")
    assert word == ""


def test_29_equivalent_states_with_different_names():
    a = make_dfa(
        {"p": {"a": "q"}, "q": {"a": "p"}, "r": {"a": "s"}, "s": {"a": "r"}},
        initial="p",
        finals={"p", "r"},
    )
    assert a.distinguishing_word("p", "r") is None
    assert not a.are_distinguishable("q", "s")


def test_29_unknown_states(simple):
    with pytest.raises(ValueError):
        simple.distinguishing_word("q0", "nope")
    with pytest.raises(ValueError):
        simple.are_distinguishable("nope", "q0")


def test_29_word_really_distinguishes_and_none_means_equivalent(request):
    for name, a in all_automata(request).items():
        alphabet = sorted(a.input_symbols)
        for p in a.states:
            for q in a.states:
                word = a.distinguishing_word(p, q)
                brute = [
                    w
                    for w in words_up_to(alphabet, 2 * len(a.states))
                    if run_accepts(a, p, w) != run_accepts(a, q, w)
                ]
                if word is None:
                    assert not brute, (name, p, q)
                else:
                    assert run_accepts(a, p, word) != run_accepts(a, q, word)
                    assert len(word) == len(brute[0]), (name, p, q)


def run_accepts(automaton, state, word):
    """Acceptance of ``word`` when starting from ``state`` (None = blocked)."""
    for symbol in word:
        state = automaton.transitions.get(state, {}).get(symbol)
        if state is None:
            return False
    return state in automaton.final_states


def test_29_is_symmetric_and_does_not_modify(request):
    for name, a in all_automata(request).items():
        before = snapshot(a)
        for p in a.states:
            for q in a.states:
                assert a.are_distinguishable(p, q) == a.are_distinguishable(q, p)
        assert snapshot(a) == before


def test_29_missing_transition_is_a_dead_state():
    # x has no transition, y loops without final: both reject everything.
    a = make_dfa({"s": {"a": "x", "b": "y"}, "x": {}, "y": {"a": "y"}}, initial="s", finals=set())
    assert not a.are_distinguishable("x", "y")


# --- function 30 ---------------------------------------------------------------------


def test_30_all_states_distinguishable(simple):
    classes = simple.myhill_nerode_classes()
    assert classes == frozenset({frozenset({"q0"}), frozenset({"q1"}), frozenset({"q2"})})


def test_30_several_equivalent_states():
    a = make_dfa(
        {"p": {"a": "q"}, "q": {"a": "p"}, "r": {"a": "s"}, "s": {"a": "r"}},
        initial="p",
        finals={"p", "r"},
    )
    assert a.myhill_nerode_classes() == frozenset(
        {frozenset({"p", "r"}), frozenset({"q", "s"})}
    )


def test_30_minimal_dfa_has_singletons(even_a):
    assert even_a.myhill_nerode_classes() == frozenset(
        {frozenset({"even"}), frozenset({"odd"})}
    )


def test_30_classes_are_not_just_final_versus_non_final():
    # 3 non-final states: s0 -a-> s1 -a-> s2(final); s3 -a-> s3 never accepts.
    a = make_dfa(
        {"s0": {"a": "s1"}, "s1": {"a": "s2"}, "s2": {}, "s3": {"a": "s3"}},
        initial="s0",
        finals={"s2"},
    )
    classes = a.myhill_nerode_classes()
    # s0, s1, s3 are all non-final, yet pairwise distinguishable.
    assert classes == frozenset(frozenset({q}) for q in ("s0", "s1", "s2", "s3"))


def test_30_inaccessible_states_convention(with_unreachable):
    everything = with_unreachable.myhill_nerode_classes()
    assert frozenset().union(*everything) == with_unreachable.states
    only = with_unreachable.myhill_nerode_classes(accessible_only=True)
    assert frozenset().union(*only) == with_unreachable.accessible_states()


def test_30_inaccessible_equivalent_state_is_merged_by_default():
    a = make_dfa(
        {"q0": {"a": "q1"}, "q1": {}, "z": {"a": "q1"}}, finals={"q1"}
    )
    # z is inaccessible and behaves like q0.
    assert frozenset({"q0", "z"}) in a.myhill_nerode_classes()
    assert frozenset({"q0"}) in a.myhill_nerode_classes(accessible_only=True)


def test_30_is_a_partition_consistent_with_function_29(request):
    for name, a in all_automata(request).items():
        for flag in (False, True):
            classes = a.myhill_nerode_classes(accessible_only=flag)
            considered = a.accessible_states() if flag else a.states
            assert all(classes_group for classes_group in classes)
            assert sum(len(c) for c in classes) == len(considered), name
            assert frozenset().union(*classes) == considered, name
            for cls in classes:
                for other in classes:
                    for p in cls:
                        for q in other:
                            same = cls is other or cls == other
                            assert a.are_distinguishable(p, q) != same, (name, p, q)


def test_30_does_not_modify_the_automaton(even_a):
    before = snapshot(even_a)
    even_a.myhill_nerode_classes()
    assert snapshot(even_a) == before


def test_30_dead_states_are_grouped_without_inventing_a_state(dead_end):
    classes = dead_end.myhill_nerode_classes()
    assert frozenset().union(*classes) == dead_end.states
    assert None not in frozenset().union(*classes)


def test_new_names_28_29_30_are_not_native_to_dfa():
    for name in (
        "largest_prefix_closed_sublanguage",
        "distinguishing_word",
        "are_distinguishable",
        "myhill_nerode_classes",
    ):
        assert not hasattr(DFA, name)


# =====================================================================================
# Functions 31, 32, 33: minimality, minimization, determinism
# =====================================================================================

from automata_extensions.fa.mixins.analyse import AnalyseMixin  # noqa: E402


@pytest.fixture
def redundant():
    """Complete DFA over {a}: p0 -> p1 -> p2 -> p1; p1 and p2 are equivalent."""
    return make_dfa(
        {"p0": {"a": "p1"}, "p1": {"a": "p2"}, "p2": {"a": "p1"}},
        initial="p0",
        finals={"p1", "p2"},
    )


@pytest.fixture
def two_unreachable():
    """Minimal core q0 -a-> q1(final) plus two inaccessible states."""
    return make_dfa(
        {
            "q0": {"a": "q1"},
            "q1": {},
            "u1": {"a": "q1"},
            "u2": {"a": "u1"},
        },
        finals={"q1"},
    )


def language_from(automaton, word_source, longest):
    return {
        w
        for w in words_up_to(sorted(word_source.input_symbols), longest)
        if automaton.accepts(w)
    }


# --- function 31 ---------------------------------------------------------------------


def test_31_minimal_complete_dfa(even_a):
    assert even_a.is_minimal()


def test_31_two_equivalent_states(redundant):
    assert not redundant.is_minimal()


def test_31_one_unreachable_state(with_unreachable):
    assert not with_unreachable.is_minimal()


def test_31_several_unreachable_states(two_unreachable):
    # u1, u2 and q1 are pairwise distinguishable except nothing is equivalent
    # to nothing: only accessibility makes it non-minimal.
    assert not two_unreachable.is_minimal()
    assert two_unreachable.accessible_states() != two_unreachable.states


def test_31_distinguishable_but_inaccessible_is_not_minimal():
    a = make_dfa({"q0": {}, "z": {"a": "q0"}}, initial="q0", finals={"q0"})
    assert a.myhill_nerode_classes() == frozenset({frozenset({"q0"}), frozenset({"z"})})
    assert not a.is_minimal()  # absence of equivalent states is not enough


def test_31_single_state():
    assert make_dfa({"q": {}}, initial="q").is_minimal()
    assert make_dfa({"q": {"a": "q"}}, initial="q", finals={"q"}).is_minimal()


def test_31_incomplete_minimal(simple):
    assert not simple.is_complete()
    assert simple.is_minimal()


def test_31_incomplete_with_a_useless_dead_state():
    a = make_dfa(
        {"q0": {"a": "q1", "b": "d"}, "q1": {}, "d": {"a": "d"}}, finals={"q1"}
    )
    assert not a.is_minimal()  # d does nothing a missing transition would not


def test_31_complete_dfa_keeps_its_sink():
    a = make_dfa(
        {"q0": {"a": "q1", "b": "d"}, "q1": {"a": "d", "b": "d"}, "d": {"a": "d", "b": "d"}},
        finals={"q1"},
    )
    assert a.is_complete() and a.is_minimal()


def test_31_agrees_with_the_classes_and_with_minimize(request):
    for name, a in all_automata(request).items():
        by_classes = a.accessible_states() == a.states and all(
            len(c) == 1 for c in a.myhill_nerode_classes()
        )
        if a.is_minimal():
            assert by_classes, name
        assert a.is_minimal() == (len(a.minimize().states) == len(a.states)), name


# --- function 32 ---------------------------------------------------------------------


def test_32_already_minimal_keeps_the_size(even_a):
    m = even_a.minimize()
    assert len(m.states) == 2 and m.is_minimal()
    assert m.is_isomorphic_to(even_a.minimize())


def test_32_merges_equivalent_states(redundant):
    m = redundant.minimize()
    assert len(m.states) == 2
    assert frozenset({"p1", "p2"}) in m.states
    assert m.initial_state == frozenset({"p0"})
    assert m.final_states == {frozenset({"p1", "p2"})}
    assert m.is_minimal()


def test_32_removes_unreachable_states(two_unreachable):
    m = two_unreachable.minimize()
    assert len(m.states) == 2
    assert language_from(m, two_unreachable, 4) == {"a"}


def test_32_several_classes_and_language_preserved(request):
    for name, a in all_automata(request).items():
        m = a.minimize()
        assert m.is_minimal(), name
        longest = 2 * len(a.states) + 1
        assert language_from(m, a, longest) == language_from(a, a, longest), name
        assert len(m.states) <= len(a.states), name


def test_32_incomplete_stays_incomplete(simple):
    m = simple.minimize()
    assert not m.is_complete()
    assert language_from(m, simple, 4) == {"ab"}


def test_32_incomplete_drops_dead_states():
    a = make_dfa(
        {"q0": {"a": "q1", "b": "d"}, "q1": {}, "d": {"a": "d"}}, finals={"q1"}
    )
    m = a.minimize()
    assert len(m.states) == 2 and m.is_minimal()
    assert language_from(m, a, 4) == {"a"}


def test_32_empty_language_partial_keeps_the_initial_state(with_unreachable):
    m = with_unreachable.minimize()
    assert m.states == {frozenset({"q0", "q1"})}
    assert language_from(m, with_unreachable, 3) == set()


def test_32_complete_stays_complete():
    a = make_dfa(
        {"q0": {"a": "q1", "b": "d"}, "q1": {"a": "d", "b": "d"}, "d": {"a": "d", "b": "d"}},
        finals={"q1"},
    )
    m = a.minimize()
    assert m.is_complete() and len(m.states) == 3


def test_32_names_never_collide(request):
    for name, a in all_automata(request).items():
        m = a.minimize()
        flat = [s for cls in m.states for s in cls]
        assert len(flat) == len(set(flat)), name


def test_32_does_not_modify_the_automaton(redundant):
    before = snapshot(redundant)
    m = redundant.minimize()
    assert snapshot(redundant) == before
    assert m is not redundant and isinstance(m, ExtendedDFA)


def test_32_minimization_is_idempotent(redundant):
    once = redundant.minimize()
    assert len(once.minimize().states) == len(once.states)
    assert once.minimize().is_isomorphic_to(once.minimize())


# --- function 33 ---------------------------------------------------------------------


class FakeFA(AnalyseMixin):
    """Stand-in with an automata-lib NFA layout (no NFA class in the project)."""

    def __init__(self, transitions, initial_state="q0", **extra):
        self.transitions = transitions
        self.initial_state = initial_state
        self.states = set(transitions)
        for key, value in extra.items():
            setattr(self, key, value)


def test_33_complete_dfa(even_a):
    assert even_a.is_complete() and even_a.is_deterministic()


def test_33_incomplete_dfa_is_still_deterministic(simple):
    assert not simple.is_complete()
    assert simple.is_deterministic()


def test_33_dfa_with_collection_valued_states_is_still_deterministic(redundant):
    # minimize() names states with frozensets: a cell is still one state.
    assert redundant.minimize().is_deterministic()


def test_33_nfa_with_several_targets():
    nfa = FakeFA({"q0": {"a": {"q0", "q1"}}, "q1": {}})
    assert not nfa.is_deterministic()


def test_33_nfa_with_single_targets_is_deterministic():
    nfa = FakeFA({"q0": {"a": {"q1"}}, "q1": {"b": {"q0"}}})
    assert nfa.is_deterministic()


def test_33_epsilon_transition():
    assert not FakeFA({"q0": {"": {"q1"}}, "q1": {}}).is_deterministic()
    assert FakeFA({"q0": {"": set()}, "q1": {}}).is_deterministic()


def test_33_several_initial_states():
    nfa = FakeFA({"q0": {}, "q1": {}}, initial_states={"q0", "q1"})
    assert not nfa.is_deterministic()
    assert FakeFA({"q0": {}}, initial_states={"q0"}).is_deterministic()


def test_33_does_not_modify_the_automaton(even_a):
    before = snapshot(even_a)
    even_a.is_deterministic()
    assert snapshot(even_a) == before


def test_new_names_31_32_33_are_not_native_to_dfa():
    for name in ("is_minimal", "minimize", "is_deterministic"):
        assert not hasattr(DFA, name)


# =====================================================================================
# Functions 34, 35, 36: epsilon closure, determinization, reversal
# =====================================================================================

from automata_extensions.fa import ExtendedNFA  # noqa: E402

EPS = ""


def make_nfa(transitions, initial="q0", finals=(), symbols=None):
    """Build an ExtendedNFA; ``transitions`` = {state: {symbol: iterable}}.

    Every state gets a key (even without transition) and epsilon is ``""``.
    """
    table = {s: {a: frozenset(t) for a, t in out.items()} for s, out in transitions.items()}
    states = set(table) | {t for out in table.values() for ts in out.values() for t in ts}
    states |= {initial} | set(finals)
    for s in states:
        table.setdefault(s, {})
    used = {a for out in table.values() for a in out if a != EPS}
    return ExtendedNFA(
        states=states,
        input_symbols=set(symbols) if symbols is not None else (used or {"a"}),
        transitions=table,
        initial_state=initial,
        final_states=set(finals),
    )


def nfa_accepts(automaton, word):
    """Independent oracle: subset simulation written from scratch for the tests."""

    def close(current):
        todo, seen = list(current), set(current)
        while todo:
            s = todo.pop()
            for t in automaton.transitions.get(s, {}).get(EPS, ()):
                if t not in seen:
                    seen.add(t)
                    todo.append(t)
        return seen

    current = close({automaton.initial_state})
    for symbol in word:
        step = set()
        for s in current:
            step |= set(automaton.transitions.get(s, {}).get(symbol, ()))
        current = close(step)
    return bool(current & set(automaton.final_states))


def nfa_language(automaton, longest):
    alphabet = sorted(automaton.input_symbols - {EPS})
    return {w for w in words_up_to(alphabet, longest) if nfa_accepts(automaton, w)}


def dfa_language(automaton, alphabet, longest):
    return {
        w
        for w in words_up_to(sorted(alphabet), longest)
        if run_accepts(automaton, automaton.initial_state, w)
    }


def nfa_snapshot(a):
    return (
        frozenset(a.states),
        frozenset(a.input_symbols),
        {q: {s: frozenset(t) for s, t in o.items()} for q, o in a.transitions.items()},
        a.initial_state,
        frozenset(a.final_states),
    )


@pytest.fixture
def eps_chain():
    """q0 -ε-> q1 -ε-> q2 -a-> q3 (final)."""
    return make_nfa(
        {"q0": {EPS: {"q1"}}, "q1": {EPS: {"q2"}}, "q2": {"a": {"q3"}}},
        finals={"q3"},
    )


@pytest.fixture
def eps_cycle():
    """q0 -ε-> q1 -ε-> q2 -ε-> q0 (cycle), q2 -a-> q3 (final)."""
    return make_nfa(
        {"q0": {EPS: {"q1"}}, "q1": {EPS: {"q2"}}, "q2": {EPS: {"q0"}, "a": {"q3"}}},
        finals={"q3"},
    )


@pytest.fixture
def nondeterministic():
    """Words over {a,b} ending with 'ab': q0 -a-> {q0,q1}, q1 -b-> q2."""
    return make_nfa(
        {"q0": {"a": {"q0", "q1"}, "b": {"q0"}}, "q1": {"b": {"q2"}}},
        finals={"q2"},
    )


NFA_LIST = [
    "eps_chain",
    "eps_cycle",
    "nondeterministic",
    "nfa_diamond",
    "nfa_empty",
    "nfa_one_state",
    "nfa_empty_word",
    "nfa_eps_mix",
]


@pytest.fixture
def nfa_diamond():
    """Two paths for 'a' then 'b' and one final per branch."""
    return make_nfa(
        {"q0": {"a": {"q1", "q2"}}, "q1": {"b": {"q3"}}, "q2": {"b": {"q4"}}},
        finals={"q3", "q4"},
    )


@pytest.fixture
def nfa_empty():
    """The final state cannot be reached."""
    return make_nfa({"q0": {"a": {"q0"}}, "q1": {}}, finals={"q1"})


@pytest.fixture
def nfa_one_state():
    """One state, final, loops on a and b: L = {a,b}*."""
    return make_nfa({"q0": {"a": {"q0"}, "b": {"q0"}}}, finals={"q0"})


@pytest.fixture
def nfa_empty_word():
    """L = {ε, a}: initial is final."""
    return make_nfa({"q0": {"a": {"q1"}}}, finals={"q0", "q1"})


@pytest.fixture
def nfa_eps_mix():
    """ε-moves in the middle and a non-trivial final set; L over {a,b}."""
    return make_nfa(
        {
            "q0": {"a": {"q1"}, EPS: {"q2"}},
            "q1": {"b": {"q1", "q3"}},
            "q2": {"b": {"q4"}, EPS: {"q1"}},
            "q4": {EPS: {"q2"}},
        },
        finals={"q3", "q4"},
    )


def all_nfas(request):
    return {name: request.getfixturevalue(name) for name in NFA_LIST}


# --- function 34 ---------------------------------------------------------------------


def test_34_state_without_epsilon_transition(eps_chain):
    assert eps_chain.epsilon_closure("q3") == frozenset({"q3"})
    assert eps_chain.epsilon_closure("q2") == frozenset({"q2"})  # only an 'a' move


def test_34_one_epsilon_transition():
    nfa = make_nfa({"q0": {EPS: {"q1"}}}, finals={"q1"})
    assert nfa.epsilon_closure("q0") == frozenset({"q0", "q1"})
    assert nfa.epsilon_closure("q1") == frozenset({"q1"})  # not symmetric


def test_34_successive_epsilon_transitions(eps_chain):
    assert eps_chain.epsilon_closure("q0") == frozenset({"q0", "q1", "q2"})
    assert eps_chain.epsilon_closure("q1") == frozenset({"q1", "q2"})


def test_34_several_epsilon_paths():
    nfa = make_nfa(
        {"s": {EPS: {"x", "y"}}, "x": {EPS: {"t"}}, "y": {EPS: {"t"}}, "t": {}}
    )
    assert nfa.epsilon_closure("s") == frozenset({"s", "x", "y", "t"})


def test_34_epsilon_cycle_terminates(eps_cycle):
    expected = frozenset({"q0", "q1", "q2"})
    for state in ("q0", "q1", "q2"):
        assert eps_cycle.epsilon_closure(state) == expected


def test_34_epsilon_self_loop():
    nfa = make_nfa({"q0": {EPS: {"q0"}}})
    assert nfa.epsilon_closure("q0") == frozenset({"q0"})


def test_34_set_of_states(eps_chain):
    assert eps_chain.epsilon_closure_of({"q1", "q3"}) == frozenset({"q1", "q2", "q3"})
    assert eps_chain.epsilon_closure_of(["q0", "q1"]) == frozenset({"q0", "q1", "q2"})


def test_34_empty_set(eps_chain):
    assert eps_chain.epsilon_closure_of([]) == frozenset()


def test_34_already_visited_state_and_duplicates(eps_chain):
    # q1 is in the closure of q0; asking for both gives the same union.
    assert eps_chain.epsilon_closure_of(["q0", "q1", "q0"]) == eps_chain.epsilon_closure("q0")


def test_34_closure_contains_the_start_and_is_idempotent(request):
    for name, a in all_nfas(request).items():
        for state in a.states:
            closure = a.epsilon_closure(state)
            assert state in closure, name
            assert a.epsilon_closure_of(closure) == closure, name


def test_34_unknown_state(eps_chain):
    with pytest.raises(ValueError):
        eps_chain.epsilon_closure("zzz")
    with pytest.raises(ValueError):
        eps_chain.epsilon_closure_of(["q0", "zzz"])


def test_34_tuple_state_names_are_not_confused_with_sets():
    nfa = make_nfa({("a", 1): {EPS: {("b", 2)}}, ("b", 2): {}}, initial=("a", 1))
    assert nfa.epsilon_closure(("a", 1)) == frozenset({("a", 1), ("b", 2)})


def test_34_dfa_closures_are_singletons(even_a):
    assert even_a.epsilon_closure("even") == frozenset({"even"})
    assert even_a.epsilon_closure_of(["even", "odd"]) == frozenset({"even", "odd"})


def test_34_does_not_modify_the_automaton(eps_cycle):
    before = nfa_snapshot(eps_cycle)
    eps_cycle.epsilon_closure("q0")
    assert nfa_snapshot(eps_cycle) == before


# --- function 35 ---------------------------------------------------------------------


def test_35_simple_nfa_without_epsilon(nondeterministic):
    dfa = nondeterministic.determinize()
    assert isinstance(dfa, ExtendedDFA)
    assert dfa.initial_state == frozenset({"q0"})
    assert dfa.transitions[frozenset({"q0"})]["a"] == frozenset({"q0", "q1"})
    assert dfa_language(dfa, {"a", "b"}, 5) == nfa_language(nondeterministic, 5)
    assert "aab" in dfa_language(dfa, {"a", "b"}, 3)


def test_35_several_transitions_for_one_symbol(nfa_diamond):
    dfa = nfa_diamond.determinize()
    assert dfa.transitions[frozenset({"q0"})]["a"] == frozenset({"q1", "q2"})
    assert frozenset({"q3", "q4"}) in dfa.final_states


def test_35_epsilon_transitions(eps_chain):
    dfa = eps_chain.determinize()
    assert dfa.initial_state == frozenset({"q0", "q1", "q2"})
    assert dfa_language(dfa, {"a"}, 3) == {"a"}


def test_35_epsilon_cycle(eps_cycle):
    dfa = eps_cycle.determinize()
    assert dfa.initial_state == frozenset({"q0", "q1", "q2"})
    assert dfa_language(dfa, {"a"}, 3) == {"a"}


def test_35_initial_state_is_final_through_epsilon():
    nfa = make_nfa({"q0": {EPS: {"q1"}}}, finals={"q1"})
    dfa = nfa.determinize()
    assert dfa.initial_state in dfa.final_states  # the empty word is accepted


def test_35_empty_language(nfa_empty):
    dfa = nfa_empty.determinize()
    assert dfa.final_states == frozenset()
    assert dfa_language(dfa, {"a"}, 4) == set()


def test_35_result_with_a_single_state(nfa_one_state):
    dfa = nfa_one_state.determinize()
    assert dfa.states == {frozenset({"q0"})}
    assert dfa.final_states == dfa.states
    assert dfa_language(dfa, {"a", "b"}, 3) == nfa_language(nfa_one_state, 3)


def test_35_language_is_preserved_and_result_is_deterministic(request):
    for name, a in all_nfas(request).items():
        for complete in (False, True):
            dfa = a.determinize(complete=complete)
            assert dfa.is_deterministic(), name
            if complete:
                assert dfa.is_complete(), name
            alphabet = a.input_symbols - {EPS}
            for longest in (5,):
                assert dfa_language(dfa, alphabet, longest) == nfa_language(a, longest), name


def test_35_complete_option_adds_the_empty_subset(nfa_diamond):
    partial = nfa_diamond.determinize()
    full = nfa_diamond.determinize(complete=True)
    assert not partial.is_complete()
    assert full.is_complete()
    assert frozenset() in full.states and frozenset() not in partial.states
    assert frozenset() not in full.final_states


def test_35_each_subset_appears_once_and_all_are_reachable(request):
    for name, a in all_nfas(request).items():
        dfa = a.determinize()
        assert dfa.accessible_states() == dfa.states, name
        assert all(len(s) >= 1 for s in dfa.states), name


def test_35_does_not_modify_the_automaton(request):
    for name, a in all_nfas(request).items():
        before = nfa_snapshot(a)
        a.determinize()
        a.determinize(complete=True)
        assert nfa_snapshot(a) == before, name


def test_35_on_a_dfa_gives_an_equivalent_dfa(even_a):
    dfa = even_a.determinize()
    assert dfa_language(dfa, {"a", "b"}, 4) == dfa_language(even_a, {"a", "b"}, 4)
    assert len(dfa.states) == 2


@pytest.mark.skipif(not hasattr(DFA, "from_nfa"), reason="native from_nfa missing")
def test_35_agrees_with_native_from_nfa(request):
    for name, a in all_nfas(request).items():
        native = DFA.from_nfa(a)
        ours = a.determinize()
        for word in words_up_to(sorted(a.input_symbols - {EPS}), 5):
            assert native.accepts_input(word) == run_accepts(
                ours, ours.initial_state, word
            ), (name, word)


# --- function 36 ---------------------------------------------------------------------


def reverse_words(words):
    return {w[::-1] for w in words}


def test_36_single_word(simple):
    rev = simple.reversed_automaton()
    assert isinstance(rev, ExtendedNFA)
    assert nfa_language(rev, 4) == {"ba"}
    assert rev.initial_state == "q2" and rev.final_states == {"q0"}  # no fresh state


def test_36_several_words(request):
    a = request.getfixturevalue("ab_or_b")
    assert nfa_language(a.reversed_automaton(), 4) == reverse_words(
        dfa_language(a, a.input_symbols, 4)
    )


def test_36_several_final_states_use_a_fresh_initial_state(nfa_diamond):
    rev = nfa_diamond.reversed_automaton()
    assert rev.initial_state == -1
    assert rev.transitions[-1][EPS] == {"q3", "q4"}
    assert rev.final_states == {"q0"}
    assert nfa_language(rev, 4) == {"ba"}


def test_36_fresh_state_name_never_collides():
    nfa = make_nfa({-1: {"a": {-2}}, -2: {}}, initial=-1, finals={-1, -2})
    rev = nfa.reversed_automaton()
    assert rev.initial_state == -3
    assert len(rev.states) == 3


def test_36_several_paths(nondeterministic):
    rev = nondeterministic.reversed_automaton()
    assert nfa_language(rev, 5) == reverse_words(nfa_language(nondeterministic, 5))


def test_36_epsilon_transitions_are_kept_reversed(eps_chain):
    rev = eps_chain.reversed_automaton()
    assert rev.transitions["q1"][EPS] == {"q0"}
    assert rev.transitions["q2"][EPS] == {"q1"}
    assert nfa_language(rev, 3) == {"a"}


def test_36_reversed_language_on_all_the_nfas(request):
    for name, a in all_nfas(request).items():
        rev = a.reversed_automaton()
        assert nfa_language(rev, 5) == reverse_words(nfa_language(a, 5)), name


def test_36_reversed_language_on_all_the_dfas(request):
    for name, a in all_automata(request).items():
        rev = a.reversed_automaton()
        longest = 2 * len(a.states)
        assert nfa_language(rev, longest) == reverse_words(
            dfa_language(a, a.input_symbols, longest)
        ), name


def test_36_empty_language_stays_empty(nfa_empty, with_unreachable):
    assert nfa_language(nfa_empty.reversed_automaton(), 4) == set()
    assert nfa_language(with_unreachable.reversed_automaton(), 4) == set()


def test_36_empty_word(nfa_empty_word, even_a):
    assert "" in nfa_language(nfa_empty_word.reversed_automaton(), 2)
    assert "" in nfa_language(even_a.reversed_automaton(), 2)
    not_empty_word = make_nfa({"q0": {"a": {"q1"}}}, finals={"q1"})
    assert "" not in nfa_language(not_empty_word.reversed_automaton(), 2)


def test_36_reversing_twice_gives_the_same_language(request):
    for name, a in all_nfas(request).items():
        twice = a.reversed_automaton().reversed_automaton()
        assert nfa_language(twice, 5) == nfa_language(a, 5), name


def test_36_reverse_of_a_dfa_is_not_forced_to_be_deterministic(redundant):
    rev = redundant.reversed_automaton()
    assert not rev.is_deterministic()  # two final states: epsilon edges from the fresh initial state


def test_36_does_not_modify_the_automaton(request, even_a):
    for name, a in all_nfas(request).items():
        before = nfa_snapshot(a)
        a.reversed_automaton()
        assert nfa_snapshot(a) == before, name
    before = snapshot(even_a)
    even_a.reversed_automaton()
    assert snapshot(even_a) == before


def test_36_native_reverse_is_not_hidden():
    assert "reverse" not in vars(ExtendedNFA)
    assert "reverse" not in vars(AnalyseMixin)


def test_new_names_34_35_36_are_not_native():
    for name in ("epsilon_closure", "epsilon_closure_of", "determinize", "reversed_automaton"):
        assert not hasattr(DFA, name)


def test_33_on_a_real_extended_nfa(nondeterministic, eps_chain, nfa_one_state):
    assert not nondeterministic.is_deterministic()
    assert not eps_chain.is_deterministic()
    assert nfa_one_state.is_deterministic()


# =====================================================================================
# Functions 37, 38: epsilon removal, universal language  (39: see the report)
# =====================================================================================


def has_epsilon(automaton):
    return any(EPS in out for out in automaton.transitions.values())


# --- function 37 ---------------------------------------------------------------------


def test_37_without_epsilon_keeps_the_language(nondeterministic):
    result = nondeterministic.remove_epsilon_transitions()
    assert isinstance(result, ExtendedNFA)
    assert result.transitions == nondeterministic.transitions
    assert result.final_states == nondeterministic.final_states


def test_37_one_epsilon_transition_and_final_states():
    nfa = make_nfa({"q0": {EPS: {"q1"}}}, finals={"q1"})
    result = nfa.remove_epsilon_transitions()
    assert result.final_states == {"q0", "q1"}  # E(q0) contains the final q1
    assert not has_epsilon(result)
    assert "" in nfa_language(result, 2)


def test_37_successive_epsilon_transitions(eps_chain):
    result = eps_chain.remove_epsilon_transitions()
    for state in ("q0", "q1", "q2"):
        assert result.transitions[state] == {"a": frozenset({"q3"})}
    assert result.initial_state == "q0"
    assert nfa_language(result, 3) == {"a"}


def test_37_epsilon_cycle(eps_cycle):
    result = eps_cycle.remove_epsilon_transitions()
    assert not has_epsilon(result)
    assert nfa_language(result, 3) == {"a"}


def test_37_several_paths():
    nfa = make_nfa(
        {"s": {EPS: {"x", "y"}}, "x": {"a": {"t"}}, "y": {"a": {"u"}, EPS: {"t"}}},
        finals={"t"},
    )
    result = nfa.remove_epsilon_transitions()
    assert result.transitions["s"]["a"] == {"t", "u"}
    assert nfa_language(result, 3) == nfa_language(nfa, 3)


def test_37_empty_language_and_empty_word(nfa_empty, nfa_empty_word):
    assert nfa_language(nfa_empty.remove_epsilon_transitions(), 4) == set()
    assert nfa_language(nfa_empty_word.remove_epsilon_transitions(), 3) == {"", "a"}


def test_37_alphabet_is_preserved_and_no_epsilon_remains(request):
    for name, a in all_nfas(request).items():
        result = a.remove_epsilon_transitions()
        assert result.input_symbols == a.input_symbols - {EPS}, name
        assert result.states == a.states and result.initial_state == a.initial_state, name
        assert not has_epsilon(result), name
        assert result.is_deterministic() == (
            all(len(t) <= 1 for o in result.transitions.values() for t in o.values())
        ), name


def test_37_language_equivalence_on_all_the_automata(request):
    for name, a in all_nfas(request).items():
        result = a.remove_epsilon_transitions()
        for longest in (5,):
            assert nfa_language(result, longest) == nfa_language(a, longest), name


def test_37_on_a_dfa(even_a):
    result = even_a.remove_epsilon_transitions()
    assert isinstance(result, ExtendedNFA)
    alphabet = sorted(even_a.input_symbols)
    assert nfa_language(result, 4) == dfa_language(even_a, alphabet, 4)


def test_37_does_not_modify_the_automaton(request):
    for name, a in all_nfas(request).items():
        before = nfa_snapshot(a)
        a.remove_epsilon_transitions()
        assert nfa_snapshot(a) == before, name


@pytest.mark.skipif(not hasattr(ExtendedNFA, "eliminate_lambda"), reason="native missing")
def test_37_agrees_with_native_eliminate_lambda(request):
    for name, a in all_nfas(request).items():
        native = a.eliminate_lambda()
        ours = a.remove_epsilon_transitions()
        assert nfa_language(native, 5) == nfa_language(ours, 5), name


# --- function 38 ---------------------------------------------------------------------


def total_dfa(final_states, symbols=("a", "b")):
    """One-state complete DFA over ``symbols``."""
    return ExtendedDFA(
        states={"s"},
        input_symbols=set(symbols),
        transitions={"s": {a: "s" for a in symbols}},
        initial_state="s",
        final_states=set(final_states),
    )


def test_38_universal_dfa():
    assert total_dfa({"s"}).is_universal()


def test_38_empty_language():
    assert not total_dfa(set()).is_universal()


def test_38_rejects_only_the_empty_word():
    a = make_dfa(
        {"i": {"a": "t", "b": "t"}, "t": {"a": "t", "b": "t"}}, initial="i", finals={"t"}
    )
    assert a.is_complete() and not a.is_universal()
    assert not run_accepts(a, "i", "") and run_accepts(a, "i", "ab")


def test_38_rejects_only_one_word():
    # Accepts everything except "b".
    a = make_dfa(
        {
            "i": {"a": "t", "b": "u"},
            "t": {"a": "t", "b": "t"},
            "u": {"a": "t", "b": "t"},
        },
        initial="i",
        finals={"i", "t"},
    )
    assert not a.is_universal()


def test_38_incomplete_is_not_universal_even_if_all_states_are_final(
    prefix_closed_language,
):
    assert not prefix_closed_language.is_complete()
    assert not prefix_closed_language.is_universal()


def test_38_complete_is_not_universal_in_general(even_a):
    assert even_a.is_complete() and not even_a.is_universal()


def test_38_universal_language_needs_no_completeness_when_alphabet_is_one_symbol():
    # Complete table but not universal vs. universal: both exist.
    assert make_dfa({"q": {"a": "q"}}, initial="q", finals={"q"}).is_universal()


def test_38_nfa_recognising_sigma_star(nondeterministic):
    # {a,b}* written as an NFA with two states and an epsilon move.
    universal = make_nfa(
        {"q0": {"a": {"q0"}, "b": {"q0"}, EPS: {"q1"}}, "q1": {}}, finals={"q1"}
    )
    assert universal.is_universal()
    assert not nondeterministic.is_universal()  # only words ending with ab


def test_38_nfa_union_covering_sigma_star():
    # L1 = words ending with a, L2 = words not ending with a (incl. ε): union = Σ*.
    nfa = make_nfa(
        {
            "s": {EPS: {"x", "y"}},
            "x": {"a": {"x", "xf"}, "b": {"x"}},
            "y": {"a": {"yn"}, "b": {"y"}},
            "yn": {"a": {"yn"}, "b": {"y"}},
        },
        initial="s",
        finals={"xf", "y"},
    )
    assert nfa.is_universal()


def test_38_nfa_missing_one_word():
    # Only the words without 'b' are accepted (a 'b' leads to the dead state q1).
    nfa = make_nfa({"q0": {"a": {"q0"}, "b": {"q1"}}, "q1": {}}, finals={"q0"})
    assert not nfa.is_universal()
    assert not nfa_accepts(nfa, "b") and nfa_accepts(nfa, "")


def test_38_several_symbols_matches_brute_force(request):
    for name, a in all_nfas(request).items():
        alphabet = sorted(a.input_symbols - {EPS})
        brute = all(nfa_accepts(a, w) for w in words_up_to(alphabet, 6))
        assert a.is_universal() == brute, name
    for name, a in all_automata(request).items():
        alphabet = sorted(a.input_symbols)
        brute = all(run_accepts(a, a.initial_state, w) for w in words_up_to(alphabet, 6))
        assert a.is_universal() == brute, name


def test_38_empty_alphabet():
    try:
        accepting = ExtendedDFA(
            states={"s"}, input_symbols=set(), transitions={"s": {}},
            initial_state="s", final_states={"s"},
        )
        rejecting = ExtendedDFA(
            states={"s"}, input_symbols=set(), transitions={"s": {}},
            initial_state="s", final_states=set(),
        )
    except Exception as error:  # the library may forbid an empty alphabet
        pytest.skip(f"empty alphabet not allowed: {error}")
    assert accepting.is_universal()  # Σ* = {ε}
    assert not rejecting.is_universal()


def test_38_does_not_modify_the_automaton(request, even_a):
    for name, a in all_nfas(request).items():
        before = nfa_snapshot(a)
        a.is_universal()
        assert nfa_snapshot(a) == before, name
    before = snapshot(even_a)
    even_a.is_universal()
    assert snapshot(even_a) == before


def test_new_names_37_38_are_not_native():
    for name in ("remove_epsilon_transitions", "is_universal"):
        assert not hasattr(DFA, name)


# =====================================================================================
# Functions 40, 41, 42: DOT export, SVG/PDF/TikZ export, union of languages
# =====================================================================================

import shutil  # noqa: E402
import subprocess  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

from automata_extensions.fa.mixins import visualisation  # noqa: E402
from automata_extensions.fa.mixins.visualisation import GraphvizNotFoundError  # noqa: E402

needs_dot = pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz 'dot' not installed")
needs_latex = pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")


@pytest.fixture
def special_names():
    """State names full of characters that DOT and LaTeX treat specially."""
    return make_nfa(
        {
            'q"0': {"a": {"b\\x", "{s}"}, EPS: {"{s}"}},
            "b\\x": {"b": {'q"0'}},
            "{s}": {"a": {"{s}"}},
            "50%_#&$^~": {},
        },
        initial='q"0',
        finals={"{s}"},
    )


# --- function 40 ---------------------------------------------------------------------


def test_40_simple_dfa(simple):
    text = simple.dot()
    assert text.startswith('digraph "automaton" {') and text.rstrip().endswith("}")
    assert "rankdir=LR;" in text
    assert 'n0 [shape=circle, label="q0"];' in text  # initial state first
    assert 'n1 [shape=circle, label="q1"];' in text
    assert 'n2 [shape=doublecircle, label="q2"];' in text
    assert "__start__ -> n0;" in text
    assert 'n0 -> n1 [label="a"];' in text and 'n1 -> n2 [label="b"];' in text


def test_40_nfa_with_several_transitions(nondeterministic):
    text = nondeterministic.dot()
    assert text.count("->") == 1 + 3  # start arrow + q0->q0 (a,b), q0->q1, q1->q2
    assert 'label="a, b"' in text  # two symbols between the same states: one edge
    assert text.count("__start__") == 2  # declared and used once


def test_40_several_final_states(nfa_diamond):
    text = nfa_diamond.dot()
    assert text.count("doublecircle") == 2
    assert text.count("shape=circle") == 1 + 3  # default node line + 3 non-final states


def test_40_initial_and_final_state_is_both(nfa_empty_word):
    text = nfa_empty_word.dot()
    assert 'n0 [shape=doublecircle, label="q0"];' in text
    assert "__start__ -> n0;" in text


def test_40_epsilon_transitions(eps_chain):
    text = eps_chain.dot()
    assert 'n0 -> n1 [label="ε"];' in text
    assert "label=\"\"" in text  # only the invisible start point has an empty label


def test_40_epsilon_is_listed_after_the_symbols():
    nfa = make_nfa({"q0": {"a": {"q1"}, EPS: {"q1"}}}, finals={"q1"})
    assert 'label="a, ε"' in nfa.dot()


def test_40_special_characters_are_escaped(special_names):
    text = special_names.dot()
    assert 'label="q\\"0"' in text
    assert 'label="b\\\\x"' in text
    assert 'label="{s}"' in text  # braces are harmless inside quotes
    assert 'label="50%_#&$^~"' in text
    assert "\n\n" not in text.strip()


def test_40_new_lines_in_names_are_escaped():
    a = make_nfa({"two\nlines": {}}, initial="two\nlines")
    assert 'label="two\\nlines"' in a.dot()
    assert "\ntwo" not in a.dot()


def test_40_set_and_tuple_state_names(redundant):
    text = redundant.minimize().dot()
    assert 'label="{p1, p2}"' in text
    empty = make_nfa({"q": {"a": {"r"}}, "r": {}}, finals={"r"}).determinize(complete=True)
    assert 'label="∅"' in empty.dot()
    t = make_dfa({("a", 1): {}}, initial=("a", 1))
    assert 'label="(a, 1)"' in t.dot()


def test_40_identical_names_of_different_types_stay_distinct():
    a = make_dfa({1: {"a": "1"}, "1": {}}, initial=1)
    text = a.dot()
    assert "n0 -> n1" in text and text.count('label="1"') == 2


@needs_dot
def test_40_output_is_accepted_by_graphviz(special_names, simple, nondeterministic):
    for automaton in (special_names, simple, nondeterministic):
        result = subprocess.run(
            ["dot", "-Tsvg"], input=automaton.dot().encode(), capture_output=True
        )
        assert result.returncode == 0, result.stderr.decode()


def test_40_file_is_written(tmp_path, special_names):
    target = tmp_path / "out.dot"
    text = special_names.dot(target)
    assert target.read_text(encoding="utf-8") == text
    assert "ε" in target.read_text(encoding="utf-8")


def test_40_invalid_path_writes_nothing(tmp_path, simple):
    with pytest.raises(FileNotFoundError):
        simple.dot(tmp_path / "missing_dir" / "out.dot")
    with pytest.raises(IsADirectoryError):
        simple.dot(tmp_path)


def test_40_does_not_modify_the_automaton(special_names, simple):
    before = nfa_snapshot(special_names)
    special_names.dot()
    assert nfa_snapshot(special_names) == before
    before = snapshot(simple)
    simple.dot()
    assert snapshot(simple) == before


def test_40_is_reproducible(special_names):
    assert special_names.dot() == special_names.dot()


# --- function 41 ---------------------------------------------------------------------


@needs_dot
def test_41_svg_of_a_dfa(tmp_path, simple):
    target = tmp_path / "dfa.svg"
    text = simple.svg(target)
    assert target.read_text(encoding="utf-8") == text
    root = ET.fromstring(text.encode())
    assert root.tag.endswith("svg")
    assert {"q0", "q1", "q2"} <= {t.text for t in root.iter() if t.tag.endswith("text")}
    assert text != simple.dot()  # a real drawing, not the DOT text


@needs_dot
def test_41_svg_returns_the_text_without_a_path(simple):
    assert "<svg" in simple.svg()


@needs_dot
def test_41_pdf_of_a_dfa(tmp_path, simple):
    target = tmp_path / "dfa.pdf"
    assert simple.pdf(target) == target
    data = target.read_bytes()
    assert data.startswith(b"%PDF-") and b"%%EOF" in data[-1024:]
    assert data != simple.dot().encode()


@needs_dot
def test_41_pdf_content_is_drawn(tmp_path, simple):
    if shutil.which("pdftotext") is None:
        pytest.skip("pdftotext not installed")
    target = simple.pdf(tmp_path / "d.pdf")
    out = subprocess.run(["pdftotext", str(target), "-"], capture_output=True, text=True).stdout
    assert "q0" in out and "q2" in out


@needs_dot
def test_41_nfa_and_epsilon_exports(tmp_path, special_names, eps_chain):
    for automaton in (special_names, eps_chain):
        svg = automaton.svg(tmp_path / "n.svg")
        assert "<svg" in svg
        assert automaton.pdf(tmp_path / "n.pdf").read_bytes().startswith(b"%PDF-")
    assert "ε" in eps_chain.svg()


def test_41_missing_graphviz_is_reported_and_nothing_is_written(
    tmp_path, simple, monkeypatch
):
    monkeypatch.setattr(visualisation.shutil, "which", lambda name: None)
    with pytest.raises(GraphvizNotFoundError, match="Graphviz"):
        simple.svg(tmp_path / "a.svg")
    with pytest.raises(GraphvizNotFoundError):
        simple.pdf(tmp_path / "a.pdf")
    assert list(tmp_path.iterdir()) == []
    # The exports that need no Graphviz still work.
    assert simple.dot(tmp_path / "a.dot") and simple.latex_tikz(tmp_path / "a.tex")


def test_41_graphviz_failure_is_reported(tmp_path, simple, monkeypatch):
    monkeypatch.setattr(visualisation.shutil, "which", lambda name: "/bin/false")
    with pytest.raises(RuntimeError, match="Graphviz failed"):
        simple.svg(tmp_path / "a.svg")
    assert not (tmp_path / "a.svg").exists()


def test_41_invalid_output_path(tmp_path, simple):
    bad = tmp_path / "no_such_dir" / "x"
    for call in (simple.svg, simple.pdf, simple.latex_tikz):
        with pytest.raises(FileNotFoundError):
            call(bad)
    assert not (tmp_path / "no_such_dir").exists()  # nothing created implicitly


def test_41_tikz_content(simple, tmp_path):
    text = simple.latex_tikz()
    assert text.startswith("\\begin{tikzpicture}") and text.rstrip().endswith(
        "\\end{tikzpicture}"
    )
    assert "\\node[state, initial] (s0)" in text
    assert "\\node[state, accepting] (s2)" in text
    assert "\\path (s0) edge node {a} (s1);" in text
    assert "\\documentclass" not in text
    full = simple.latex_tikz(tmp_path / "a.tex", standalone=True)
    assert full.startswith("\\documentclass") and "\\usetikzlibrary{automata" in full
    assert (tmp_path / "a.tex").read_text(encoding="utf-8") == full


def test_41_tikz_loops_and_opposite_edges(nfa_one_state):
    assert "edge[loop above]" in nfa_one_state.latex_tikz()
    a = make_dfa({"p": {"a": "q"}, "q": {"b": "p"}}, initial="p")
    text = a.latex_tikz()
    assert text.count("edge[bend left]") == 2


def test_41_tikz_epsilon_and_escaping(special_names):
    text = special_names.latex_tikz()
    assert "$\\varepsilon$" in text
    assert "b\\textbackslash{}x" in text and "\\{s\\}" in text
    assert "50\\%\\_\\#\\&\\$\\^{}\\~{}" in text
    assert 'q\\texttt{"}0' in text


def test_41_tikz_inaccessible_states_are_drawn(with_unreachable):
    text = with_unreachable.latex_tikz()
    assert text.count("\\node[") == 4


@needs_latex
def test_41_tikz_really_compiles(tmp_path, special_names, simple, eps_cycle, nfa_empty_word):
    for index, automaton in enumerate((special_names, simple, eps_cycle, nfa_empty_word)):
        source = tmp_path / f"t{index}.tex"
        automaton.latex_tikz(source, standalone=True)
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", source.name],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout[-800:]
        assert (tmp_path / f"t{index}.pdf").read_bytes().startswith(b"%PDF-")


def test_41_does_not_modify_the_automaton(special_names):
    before = nfa_snapshot(special_names)
    special_names.latex_tikz()
    assert nfa_snapshot(special_names) == before


def test_new_names_40_41_are_not_native():
    for name in ("dot", "svg", "pdf", "latex_tikz"):
        assert not hasattr(DFA, name)
    for name in ("show_diagram",):
        assert name not in vars(visualisation.VisualisationMixin)


# --- function 42 ---------------------------------------------------------------------


def language_of(automaton, alphabet, longest):
    """Words over ``alphabet`` accepted by a DFA or an NFA (independent oracle)."""
    if isinstance(automaton, ExtendedNFA):
        return {w for w in words_up_to(sorted(alphabet), longest) if nfa_accepts(automaton, w)}
    return {
        w
        for w in words_up_to(sorted(alphabet), longest)
        if run_accepts(automaton, automaton.initial_state, w)
    }


def check_union(first, second, alphabet=None, longest=5):
    result = first.language_union(second)
    alphabet = alphabet or (
        (set(first.input_symbols) | set(second.input_symbols)) - {EPS}
    )
    expected = language_of(first, alphabet, longest) | language_of(second, alphabet, longest)
    assert language_of(result, alphabet, longest) == expected
    return result


def test_42_two_simple_languages(only_ab, a_even):
    result = check_union(only_ab, a_even)
    assert isinstance(result, ExtendedDFA)


def test_42_disjoint_languages():
    left = make_dfa({"s": {"a": "t"}, "t": {}}, initial="s", finals={"t"})
    right = make_dfa({"s": {"b": "t"}, "t": {}}, initial="s", finals={"t"})
    result = check_union(left, right)
    assert language_of(result, {"a", "b"}, 3) == {"a", "b"}


def test_42_identical_languages(even_a):
    result = check_union(even_a, even_a)
    assert language_of(result, {"a", "b"}, 4) == language_of(even_a, {"a", "b"}, 4)


def test_42_with_the_empty_language(even_a, empty_ab):
    result = check_union(even_a, empty_ab)
    assert language_of(result, {"a", "b"}, 4) == language_of(even_a, {"a", "b"}, 4)
    assert language_of(empty_ab.language_union(empty_ab), {"a", "b"}, 4) == set()


def test_42_with_the_universal_language(even_a, universal_ab):
    result = check_union(even_a, universal_ab)
    assert result.is_universal() and universal_ab.is_universal()


def test_42_different_alphabets():
    only_a = make_dfa({"s": {"a": "s"}}, initial="s", finals={"s"})  # a*
    only_c = make_dfa({"s": {"c": "t"}, "t": {}}, initial="s", finals={"t"})  # {c}
    result = check_union(only_a, only_c)
    assert result.input_symbols == {"a", "c"}
    assert language_of(result, {"a", "c"}, 3) == {"", "a", "aa", "aaa", "c"}


def test_42_incomplete_automata(simple, one_missing):
    assert not simple.is_complete()
    check_union(simple, one_missing)
    check_union(one_missing, simple)


def test_42_dfas_on_all_the_pair_fixtures(request):
    items = list(all_automata(request).items())
    for (n1, a), (n2, b) in zip(items, items[1:] + items[:1]):
        check_union(a, b, longest=4)


def test_42_with_nfas_uses_a_fresh_initial_state(nondeterministic, eps_chain):
    result = nondeterministic.language_union(eps_chain)
    assert isinstance(result, ExtendedNFA)
    assert result.initial_state == ("start",)
    assert result.transitions[("start",)][EPS] == {(0, "q0"), (1, "q0")}
    check_union(nondeterministic, eps_chain)


def test_42_mixed_dfa_and_nfa(even_a, nondeterministic):
    check_union(even_a, nondeterministic)
    check_union(nondeterministic, even_a)


def test_42_nfa_with_different_alphabets(eps_chain):
    other = make_nfa({"s": {"b": {"t"}}}, initial="s", finals={"t"})
    result = check_union(eps_chain, other)
    assert result.input_symbols == {"a", "b"}


def test_42_names_never_collide():
    first = make_nfa({"q0": {"a": {"q1"}}}, finals={"q1"})
    second = make_nfa({"q0": {"b": {"q1"}}}, finals={"q1"})  # same names!
    result = first.language_union(second)
    assert len(result.states) == 5
    assert language_of(result, {"a", "b"}, 2) == {"a", "b"}


def test_42_dfa_with_frozenset_states_is_not_read_as_an_nfa(redundant, nondeterministic):
    minimized = redundant.minimize()  # state names are frozensets
    check_union(minimized, nondeterministic, longest=4)


def test_42_does_not_modify_the_operands(even_a, nondeterministic, simple):
    for first, second in ((even_a, simple), (nondeterministic, even_a), (simple, nondeterministic)):
        snap = (
            nfa_snapshot(first) if isinstance(first, ExtendedNFA) else snapshot(first),
            nfa_snapshot(second) if isinstance(second, ExtendedNFA) else snapshot(second),
        )
        first.language_union(second)
        after = (
            nfa_snapshot(first) if isinstance(first, ExtendedNFA) else snapshot(first),
            nfa_snapshot(second) if isinstance(second, ExtendedNFA) else snapshot(second),
        )
        assert snap == after


def test_42_union_is_symmetric_on_the_language(only_ab, even_a):
    alphabet = {"a", "b"}
    assert language_of(only_ab.language_union(even_a), alphabet, 5) == language_of(
        even_a.language_union(only_ab), alphabet, 5
    )


@pytest.mark.skipif(not hasattr(DFA, "union"), reason="native DFA.union missing")
def test_42_agrees_with_native_union(even_a, odd_a):
    native = DFA.union(even_a, odd_a)
    ours = even_a.language_union(odd_a)
    for word in words_up_to(["a", "b"], 5):
        assert native.accepts_input(word) == run_accepts(ours, ours.initial_state, word)


def test_new_name_42_is_not_native():
    assert not hasattr(DFA, "language_union")


# =====================================================================================
# Functions 7, 8, 9 (useful states, trim, is_trim) and 39 (state elimination)
# =====================================================================================

import re  # noqa: E402


@pytest.fixture
def messy():
    """q0 -a-> q1 -b-> q2(final); q0 -c-> d (dead end, with a loop); u unreachable -a-> q2."""
    return make_dfa(
        {
            "q0": {"a": "q1", "c": "d"},
            "q1": {"b": "q2"},
            "q2": {},
            "d": {"a": "d"},
            "u": {"a": "q2"},
        },
        finals={"q2"},
    )


# --- function 7: useful states ---------------------------------------------------------


def test_7_useful_states(messy):
    assert messy.useful_states() == {"q0", "q1", "q2"}  # d is dead, u is inaccessible


def test_7_is_accessible_and_coaccessible(request):
    for name, a in all_automata(request).items():
        assert a.useful_states() == a.accessible_states() & a.coaccessible_states(), name


def test_7_empty_language_has_no_useful_state(with_unreachable, nfa_empty):
    assert with_unreachable.useful_states() == frozenset()
    assert nfa_empty.useful_states() == frozenset()


def test_7_all_states_useful(simple):
    assert simple.useful_states() == simple.states


def test_7_on_an_nfa_with_epsilon(eps_chain):
    assert eps_chain.useful_states() == {"q0", "q1", "q2", "q3"}


def test_7_matches_the_definition_by_words(request):
    for name, a in all_automata(request).items():
        alphabet = sorted(a.input_symbols)
        expected = set()
        for word in words_up_to(alphabet, len(a.states)):
            state = a.initial_state
            for symbol in word:
                state = a.transitions.get(state, {}).get(symbol)
                if state is None:
                    break
            if state is None:
                continue
            for suffix in words_up_to(alphabet, len(a.states)):
                if run_accepts(a, state, suffix):
                    expected.add(state)
                    break
        assert a.useful_states() == expected, name


# --- function 8: trim -----------------------------------------------------------------


def test_8_removes_dead_and_inaccessible_states(messy):
    result = messy.trim()
    assert isinstance(result, ExtendedDFA)
    assert result.states == {"q0", "q1", "q2"}
    assert result.transitions == {"q0": {"a": "q1"}, "q1": {"b": "q2"}, "q2": {}}
    assert result.is_trim()


def test_8_keeps_the_language(request):
    for name, a in all_automata(request).items():
        t = a.trim()
        alphabet = a.input_symbols
        longest = 2 * len(a.states)
        assert dfa_language(t, alphabet, longest) == dfa_language(a, alphabet, longest), name
        assert t.is_trim(), name
        assert t.initial_state == a.initial_state, name


def test_8_empty_language_keeps_only_the_initial_state(with_unreachable):
    t = with_unreachable.trim()
    assert t.states == {"q0"} and t.final_states == frozenset() and t.transitions == {"q0": {}}
    assert t.is_trim()


def test_8_removes_the_sink_of_a_complete_dfa():
    a = make_dfa(
        {"q0": {"a": "q1", "b": "d"}, "q1": {"a": "d", "b": "d"}, "d": {"a": "d", "b": "d"}},
        finals={"q1"},
    )
    t = a.trim()
    assert t.states == {"q0", "q1"} and not t.is_complete()


def test_8_trim_is_idempotent(messy):
    once = messy.trim()
    assert snapshot(once.trim()) == snapshot(once)


def test_8_nfa_with_epsilon():
    nfa = make_nfa(
        {"q0": {EPS: {"q1", "dead"}, "a": {"dead"}}, "q1": {"b": {"q2"}}, "dead": {}, "u": {"a": {"q2"}}},
        finals={"q2"},
    )
    t = nfa.trim()
    assert isinstance(t, ExtendedNFA)
    assert t.states == {"q0", "q1", "q2"}
    assert t.transitions["q0"] == {EPS: {"q1"}}  # the 'a' edge to a removed state is gone
    assert nfa_language(t, 3) == nfa_language(nfa, 3) == {"b"}


def test_8_nfa_language_on_all_nfas(request):
    for name, a in all_nfas(request).items():
        t = a.trim()
        assert nfa_language(t, 5) == nfa_language(a, 5), name
        assert t.is_trim(), name


def test_8_does_not_modify_the_automaton(messy, eps_cycle):
    before = snapshot(messy)
    messy.trim()
    assert snapshot(messy) == before
    before = nfa_snapshot(eps_cycle)
    eps_cycle.trim()
    assert nfa_snapshot(eps_cycle) == before


# --- function 9: is_trim ---------------------------------------------------------------


def test_9_trim_and_not_trim(simple, messy, with_unreachable):
    assert simple.is_trim()
    assert not messy.is_trim()
    assert not with_unreachable.is_trim()  # empty language but 4 states


def test_9_one_state_automata():
    assert make_dfa({"q": {}}, initial="q").is_trim()  # empty language convention
    assert make_dfa({"q": {"a": "q"}}, initial="q", finals={"q"}).is_trim()


def test_9_is_equivalent_to_trim_keeping_everything(request):
    for name, a in all_automata(request).items():
        assert a.is_trim() == (len(a.trim().states) == len(a.states)), name
    for name, a in all_nfas(request).items():
        assert a.is_trim() == (len(a.trim().states) == len(a.states)), name


def test_9_a_trim_dfa_may_be_incomplete_and_a_complete_one_is_rarely_trim(even_a, simple):
    assert simple.is_trim() and not simple.is_complete()
    assert even_a.is_trim()  # complete and every state useful


# --- function 39: state elimination ----------------------------------------------------


def regex_language(regex, alphabet, longest):
    """Words matched by ``regex`` (text of the project's syntax, translated for ``re``)."""
    if regex is None:
        return set()
    pattern = re.compile(
        regex.replace("()", "(?:)").replace("ε", "(?:)").replace("∅", "(?:(?!))")
    )
    return {w for w in words_up_to(sorted(alphabet), longest) if pattern.fullmatch(w)}


@pytest.fixture
def normalized_nfa():
    """start -a-> m1 -ε-> m2 -b-> end, plus m1 -b-> m1 and an a-b parallel edge."""
    return make_nfa(
        {
            "i": {"a": {"m1", "m2"}},
            "m1": {"b": {"m1"}, EPS: {"m2"}},
            "m2": {"b": {"f"}, "a": {"f"}},
            "f": {},
        },
        initial="i",
        finals={"f"},
    )


def test_39_valid_normalized_automaton(normalized_nfa):
    assert normalized_nfa.is_normalized()
    regex = normalized_nfa.eliminate_states()
    assert regex_language(regex, {"a", "b"}, 6) == nfa_language(normalized_nfa, 6)


def test_39_simple_chain_gives_a_concatenation():
    nfa = make_nfa({"i": {"a": {"x"}}, "x": {"b": {"f"}}, "f": {}}, initial="i", finals={"f"})
    assert nfa.eliminate_states() == "ab"


def test_39_loop_gives_a_star_and_epsilon_is_the_empty_word():
    nfa = make_nfa({"i": {EPS: {"x"}}, "x": {"a": {"x"}, EPS: {"f"}}, "f": {}}, initial="i", finals={"f"})
    assert nfa.eliminate_states() == "a*"


def test_39_parallel_edges_give_a_union():
    nfa = make_nfa({"i": {"a": {"f"}, "b": {"f"}}, "f": {}}, initial="i", finals={"f"})
    assert nfa.eliminate_states() == "a|b"


def test_39_precedence_is_rendered_with_parentheses():
    nfa = make_nfa(
        {"i": {"a": {"x", "y"}}, "x": {"b": {"f"}}, "y": {"c": {"y", "f"}}, "f": {}},
        initial="i",
        finals={"f"},
    )
    regex = nfa.eliminate_states()
    assert regex_language(regex, {"a", "b", "c"}, 5) == nfa_language(nfa, 5) == {
        "ab", "ac", "acc", "accc", "acccc"
    }


def test_39_several_intermediate_states_and_epsilon_cycle():
    nfa = make_nfa(
        {
            "i": {EPS: {"x"}},
            "x": {EPS: {"y"}, "a": {"f"}},
            "y": {EPS: {"x"}, "b": {"y"}},
            "f": {},
        },
        initial="i",
        finals={"f"},
    )
    assert regex_language(nfa.eliminate_states(), {"a", "b"}, 5) == nfa_language(nfa, 5)


def test_39_empty_language_returns_none():
    nfa = make_nfa({"i": {"a": {"x"}}, "x": {}, "f": {}}, initial="i", finals={"f"})
    assert nfa.is_normalized() and nfa.eliminate_states() is None


def test_39_initial_and_final_directly_linked():
    nfa = make_nfa({"i": {"a": {"f"}, EPS: {"f"}}, "f": {}}, initial="i", finals={"f"})
    assert regex_language(nfa.eliminate_states(), {"a"}, 3) == {"", "a"}


def test_39_every_order_gives_the_same_language(normalized_nfa):
    import itertools

    expected = nfa_language(normalized_nfa, 6)
    for order in itertools.permutations(["m1", "m2"]):
        regex = normalized_nfa.eliminate_states(order=order)
        assert regex_language(regex, {"a", "b"}, 6) == expected, order


def test_39_bad_orders_are_rejected(normalized_nfa):
    for bad in (["m1"], ["m1", "m1"], ["m1", "m2", "i"], ["m1", "zzz"]):
        with pytest.raises(ValueError):
            normalized_nfa.eliminate_states(order=bad)


def test_39_preconditions(simple, nondeterministic):
    # initial state with an incoming transition
    loop = make_nfa({"i": {"a": {"i", "f"}}, "f": {}}, initial="i", finals={"f"})
    assert not loop.is_normalized()
    # two final states / a final state with an outgoing transition / initial = final
    two_finals = make_nfa({"i": {"a": {"f1", "f2"}}, "f1": {}, "f2": {}}, initial="i", finals={"f1", "f2"})
    final_out = make_nfa({"i": {"a": {"f"}}, "f": {"a": {"i2"}}, "i2": {}}, initial="i", finals={"f"})
    same = make_nfa({"i": {}}, initial="i", finals={"i"})
    no_final = make_nfa({"i": {}}, initial="i")
    assert simple.is_normalized()  # a chain DFA with one final state is normalised
    for bad in (loop, two_finals, final_out, same, no_final, nondeterministic):
        assert not bad.is_normalized()
        with pytest.raises(ValueError, match="normalised"):
            bad.eliminate_states()


def test_39_special_characters_are_escaped_and_long_symbols_rejected():
    nfa = make_nfa({"i": {"*": {"f"}}, "f": {}}, initial="i", finals={"f"})
    assert nfa.eliminate_states() == "\\*"
    long = make_nfa({"i": {"ab": {"f"}}, "f": {}}, initial="i", finals={"f"})
    with pytest.raises(ValueError, match="one character"):
        long.eliminate_states()


def test_39_normalized_builds_an_equivalent_normalized_automaton(request):
    for name, a in all_nfas(request).items():
        n = a.normalized()
        assert n.is_normalized(), name
        assert nfa_language(n, 5) == nfa_language(a, 5), name
        assert regex_language(n.eliminate_states(), a.input_symbols - {EPS}, 5) == nfa_language(a, 5), name
    for name, a in all_automata(request).items():
        n = a.normalized()
        assert n.is_normalized(), name
        longest = 4
        assert regex_language(n.eliminate_states(), a.input_symbols, longest) == dfa_language(
            a, a.input_symbols, longest
        ), name


def test_39_normalized_names_do_not_collide():
    a = make_nfa({("start",): {"a": {("accept",)}}, ("accept",): {}}, initial=("start",), finals={("accept",)})
    n = a.normalized()
    assert len(n.states) == 4 and n.is_normalized()
    assert nfa_language(n, 2) == {"a"}


def test_39_known_result_even_number_of_a(even_a):
    regex = even_a.normalized().eliminate_states()
    assert regex_language(regex, {"a", "b"}, 6) == dfa_language(even_a, {"a", "b"}, 6)


def test_39_does_not_modify_the_automaton(normalized_nfa, even_a):
    before = nfa_snapshot(normalized_nfa)
    normalized_nfa.eliminate_states()
    normalized_nfa.is_normalized()
    assert nfa_snapshot(normalized_nfa) == before
    before = snapshot(even_a)
    even_a.normalized()
    assert snapshot(even_a) == before


def test_new_names_7_8_9_39_are_not_native():
    for name in ("useful_states", "trim", "is_trim", "is_normalized", "normalized", "eliminate_states"):
        assert not hasattr(DFA, name)


# =====================================================================================
# Functions 43, 44, 45: intersection, difference, concatenation
# =====================================================================================


def accepts_any(automaton, word):
    """Acceptance by a DFA or an NFA (independent oracle)."""
    if isinstance(automaton, ExtendedNFA):
        return nfa_accepts(automaton, word)
    return run_accepts(automaton, automaton.initial_state, word)


def union_alphabet(first, second):
    return (set(first.input_symbols) | set(second.input_symbols)) - {EPS}


def check_operation(first, second, operation, rule, longest=5):
    result = getattr(first, operation)(second)
    for word in words_up_to(sorted(union_alphabet(first, second)), longest):
        assert accepts_any(result, word) == rule(first, second, word), (operation, word)
    return result


def inter_rule(a, b, w):
    return accepts_any(a, w) and accepts_any(b, w)


def diff_rule(a, b, w):
    return accepts_any(a, w) and not accepts_any(b, w)


def concat_rule(a, b, w):
    return any(accepts_any(a, w[:i]) and accepts_any(b, w[i:]) for i in range(len(w) + 1))


def operand_snapshot(automaton):
    return nfa_snapshot(automaton) if isinstance(automaton, ExtendedNFA) else snapshot(automaton)


@pytest.fixture
def dfa_pairs(request):
    """Several pairs of DFAs with common words, disjoint, incomplete, other alphabets."""
    names = [
        "even_a", "odd_a", "only_ab", "a_even", "empty_ab", "universal_ab",
        "ab_or_b", "only_c", "b_star", "simple", "one_missing", "dead_end",
    ]
    automata = [request.getfixturevalue(n) for n in names]
    return [(a, b) for a in automata for b in automata]


# --- function 43 ---------------------------------------------------------------------


def test_43_identical_languages(even_a):
    result = check_operation(even_a, even_a, "language_intersection", inter_rule)
    assert isinstance(result, ExtendedDFA)
    assert dfa_language(result, {"a", "b"}, 5) == dfa_language(even_a, {"a", "b"}, 5)


def test_43_disjoint_languages(even_a, odd_a):
    result = check_operation(even_a, odd_a, "language_intersection", inter_rule)
    assert dfa_language(result, {"a", "b"}, 5) == set()


def test_43_common_words(even_a, only_ab):
    result = check_operation(even_a, only_ab, "language_intersection", inter_rule)
    # only_ab = {ab}: one 'a' (odd), so nothing in common with even_a ...
    assert dfa_language(result, {"a", "b"}, 4) == set()
    common = even_a.language_intersection(make_dfa(
        {"s": {"a": "t", "b": "s"}, "t": {"a": "u", "b": "t"}, "u": {"a": "u", "b": "u"}},
        initial="s", finals={"u"}))  # at least two 'a'
    assert dfa_language(common, {"a", "b"}, 4) == {
        w for w in words_up_to(["a", "b"], 4) if w.count("a") >= 2 and w.count("a") % 2 == 0
    }


def test_43_with_empty_and_universal(even_a, empty_ab, universal_ab):
    assert dfa_language(even_a.language_intersection(empty_ab), {"a", "b"}, 4) == set()
    r = even_a.language_intersection(universal_ab)
    assert dfa_language(r, {"a", "b"}, 4) == dfa_language(even_a, {"a", "b"}, 4)


def test_43_different_alphabets(even_a, only_c):
    r = check_operation(even_a, only_c, "language_intersection", inter_rule)
    assert r.input_symbols == {"a", "b", "c"}
    only_a = make_dfa({"s": {"a": "s"}}, initial="s", finals={"s"})
    r2 = check_operation(only_a, even_a, "language_intersection", inter_rule)
    assert dfa_language(r2, {"a"}, 4) == {"", "aa", "aaaa"}


def test_43_final_pairs_need_both_components_final(only_ab, ab_or_b):
    r = only_ab.language_intersection(ab_or_b)
    for (p, q) in r.final_states:
        assert p in only_ab.final_states and q in ab_or_b.final_states
    for (p, q) in r.states - r.final_states:
        assert not (p in only_ab.final_states and q in ab_or_b.final_states)


def test_43_only_accessible_pairs_are_built(simple, one_missing):
    r = simple.language_intersection(one_missing)
    assert r.accessible_states() == r.states
    assert len(r.states) <= len(simple.states) * (len(one_missing.states) + 1)


def test_43_all_dfa_pairs(dfa_pairs):
    for a, b in dfa_pairs:
        check_operation(a, b, "language_intersection", inter_rule, longest=4)


def test_43_with_nfas_and_epsilon(nondeterministic, eps_chain, even_a, nfa_eps_mix, nfa_diamond):
    for a, b in (
        (nondeterministic, even_a), (even_a, nondeterministic), (eps_chain, nfa_eps_mix),
        (nfa_eps_mix, nfa_diamond), (nondeterministic, nondeterministic),
    ):
        r = check_operation(a, b, "language_intersection", inter_rule)
        assert isinstance(r, ExtendedNFA)
        assert not any(EPS in out for out in r.transitions.values())


def test_43_does_not_modify_the_operands(even_a, nondeterministic, simple):
    for first, second in ((even_a, simple), (nondeterministic, even_a), (simple, nondeterministic)):
        before = (operand_snapshot(first), operand_snapshot(second))
        first.language_intersection(second)
        assert (operand_snapshot(first), operand_snapshot(second)) == before


# --- function 44 ---------------------------------------------------------------------


def test_44_identical_languages_give_the_empty_language(even_a):
    r = check_operation(even_a, even_a, "language_difference", diff_rule)
    assert dfa_language(r, {"a", "b"}, 5) == set()
    assert r.is_empty()


def test_44_disjoint_languages(even_a, odd_a):
    r = check_operation(even_a, odd_a, "language_difference", diff_rule)
    assert dfa_language(r, {"a", "b"}, 4) == dfa_language(even_a, {"a", "b"}, 4)


def test_44_with_empty_and_universal(even_a, empty_ab, universal_ab):
    r = even_a.language_difference(empty_ab)
    assert dfa_language(r, {"a", "b"}, 4) == dfa_language(even_a, {"a", "b"}, 4)
    assert empty_ab.language_difference(even_a).is_empty()
    assert even_a.language_difference(universal_ab).is_empty()
    assert universal_ab.language_difference(even_a).is_universal() is False
    assert dfa_language(universal_ab.language_difference(even_a), {"a", "b"}, 3) == dfa_language(
        odd_a_language(), {"a", "b"}, 3
    )


def odd_a_language():
    return make_dfa(
        {"e": {"a": "o", "b": "e"}, "o": {"a": "e", "b": "o"}}, initial="e", finals={"o"}
    )


def test_44_inclusions(even_a):
    # L(A) included in L(B): empty difference; L(B) included in L(A): the rest.
    small = make_dfa({"s": {"b": "s"}}, initial="s", finals={"s"})  # b*
    # b* is included in even_a (zero 'a').
    assert small.language_difference(even_a).is_empty()
    r = even_a.language_difference(small)
    assert not r.is_empty()
    assert run_accepts(r, r.initial_state, "aa") and not run_accepts(r, r.initial_state, "bb")


def test_44_different_alphabets_and_incomplete(only_c, even_a, simple, one_missing):
    # Words with 'c' are never rejected by even_a (it cannot read them): kept from A.
    r = check_operation(only_c, even_a, "language_difference", diff_rule)
    assert run_accepts(r, r.initial_state, "c")
    check_operation(even_a, only_c, "language_difference", diff_rule)
    check_operation(simple, one_missing, "language_difference", diff_rule)
    check_operation(one_missing, simple, "language_difference", diff_rule)


def test_44_all_dfa_pairs(dfa_pairs):
    for a, b in dfa_pairs:
        check_operation(a, b, "language_difference", diff_rule, longest=4)


def test_44_with_nfas(nondeterministic, eps_chain, even_a, nfa_eps_mix):
    for a, b in ((nondeterministic, even_a), (even_a, nondeterministic), (eps_chain, nfa_eps_mix),
                 (nfa_eps_mix, eps_chain), (nondeterministic, nondeterministic)):
        r = check_operation(a, b, "language_difference", diff_rule)
        assert isinstance(r, ExtendedDFA)


def test_44_equals_intersection_with_the_complement(even_a, only_ab):
    comp = only_ab.complemented()
    left = even_a.language_difference(only_ab)
    right = even_a.language_intersection(comp)
    for w in words_up_to(["a", "b"], 5):
        assert run_accepts(left, left.initial_state, w) == run_accepts(right, right.initial_state, w)


def test_44_does_not_modify_the_operands(even_a, nondeterministic, simple):
    for first, second in ((even_a, simple), (nondeterministic, even_a), (simple, nondeterministic)):
        before = (operand_snapshot(first), operand_snapshot(second))
        first.language_difference(second)
        assert (operand_snapshot(first), operand_snapshot(second)) == before


# --- function 45 ---------------------------------------------------------------------


def test_45_two_simple_languages():
    first = make_dfa({"s": {"a": "t"}, "t": {}}, initial="s", finals={"t"})  # {a}
    second = make_dfa({"s": {"b": "t"}, "t": {}}, initial="s", finals={"t"})  # {b}
    r = check_operation(first, second, "language_concatenation", concat_rule)
    assert isinstance(r, ExtendedNFA)
    assert nfa_language(r, 3) == {"ab"}


def test_45_epsilon_transitions_link_finals_to_the_second_initial(simple):
    other = make_dfa({"s": {"b": "t"}, "t": {}}, initial="s", finals={"t"})
    r = simple.language_concatenation(other)
    assert r.initial_state == (0, "q0")
    assert r.final_states == {(1, "t")}
    assert r.transitions[(0, "q2")][EPS] == {(1, "s")}
    assert EPS not in r.transitions[(0, "q1")]
    assert r.transitions[(0, "q0")] == {"a": {(0, "q1")}}  # operand transitions are kept


def test_45_with_the_empty_language(even_a, empty_ab):
    for a, b in ((even_a, empty_ab), (empty_ab, even_a), (empty_ab, empty_ab)):
        r = check_operation(a, b, "language_concatenation", concat_rule)
        assert nfa_language(r, 4) == set()


def test_45_with_the_empty_word_language(even_a):
    only_empty_word = make_dfa({"q": {}}, initial="q", finals={"q"})
    r1 = check_operation(even_a, only_empty_word, "language_concatenation", concat_rule)
    r2 = check_operation(only_empty_word, even_a, "language_concatenation", concat_rule)
    expected = dfa_language(even_a, {"a", "b"}, 5)
    assert nfa_language(r1, 5) == nfa_language(r2, 5) == expected
    both = check_operation(only_empty_word, only_empty_word, "language_concatenation", concat_rule)
    assert nfa_language(both, 2) == {""}


def test_45_two_languages_containing_the_empty_word(even_a, b_star):
    r = check_operation(even_a, b_star, "language_concatenation", concat_rule)
    assert "" in nfa_language(r, 1)


def test_45_several_ways_to_split(nondeterministic):
    # 'a'* . 'a'* : "aaa" can be cut in four places.
    a_star = make_dfa({"s": {"a": "s"}}, initial="s", finals={"s"})
    r = check_operation(a_star, a_star, "language_concatenation", concat_rule)
    assert nfa_language(r, 4) == {"", "a", "aa", "aaa", "aaaa"}
    check_operation(nondeterministic, nondeterministic, "language_concatenation", concat_rule)


def test_45_different_alphabets(only_c, even_a):
    r = check_operation(only_c, even_a, "language_concatenation", concat_rule)
    assert r.input_symbols == {"a", "b", "c"}
    assert "caa" in nfa_language(r, 3) and "ac" not in nfa_language(r, 3)


def test_45_same_state_names_do_not_collide():
    first = make_nfa({"q0": {"a": {"q1"}}}, finals={"q1"})
    second = make_nfa({"q0": {"b": {"q1"}}}, finals={"q1"})
    r = first.language_concatenation(second)
    assert len(r.states) == 4
    assert nfa_language(r, 3) == {"ab"}


def test_45_keeps_the_epsilon_transitions_of_the_operands(eps_chain, eps_cycle):
    r = eps_chain.language_concatenation(eps_cycle)
    assert r.transitions[(0, "q0")][EPS] == {(0, "q1")}
    assert r.transitions[(1, "q2")][EPS] == {(1, "q0")}
    check_operation(eps_chain, eps_cycle, "language_concatenation", concat_rule)


def test_45_all_dfa_pairs(dfa_pairs):
    for a, b in dfa_pairs:
        check_operation(a, b, "language_concatenation", concat_rule, longest=4)


def test_45_with_nfas(nondeterministic, eps_chain, even_a, nfa_eps_mix, nfa_empty_word):
    for a, b in ((nondeterministic, even_a), (even_a, nfa_eps_mix), (nfa_empty_word, eps_chain),
                 (eps_chain, nfa_empty_word), (nfa_eps_mix, nfa_eps_mix)):
        check_operation(a, b, "language_concatenation", concat_rule)


def test_45_final_state_of_the_first_that_is_also_final_in_the_result_is_not_kept():
    first = make_dfa({"s": {}}, initial="s", finals={"s"})  # {ε}
    second = make_dfa({"s": {"a": "t"}, "t": {}}, initial="s", finals={"t"})  # {a}
    r = first.language_concatenation(second)
    assert (0, "s") not in r.final_states
    assert nfa_language(r, 2) == {"a"}


def test_45_does_not_modify_the_operands(even_a, nondeterministic, simple):
    for first, second in ((even_a, simple), (nondeterministic, even_a), (simple, nondeterministic)):
        before = (operand_snapshot(first), operand_snapshot(second))
        first.language_concatenation(second)
        assert (operand_snapshot(first), operand_snapshot(second)) == before


def test_new_names_43_44_45_are_not_native():
    for name in ("language_intersection", "language_difference", "language_concatenation"):
        assert not hasattr(DFA, name)


@pytest.mark.skipif(not hasattr(DFA, "intersection"), reason="native DFA.intersection missing")
def test_43_agrees_with_native_intersection(even_a, odd_a, universal_ab):
    for other in (odd_a, universal_ab, even_a):
        native = DFA.intersection(even_a, other)
        ours = even_a.language_intersection(other)
        for word in words_up_to(["a", "b"], 5):
            assert native.accepts_input(word) == run_accepts(ours, ours.initial_state, word)


@pytest.mark.skipif(not hasattr(DFA, "difference"), reason="native DFA.difference missing")
def test_44_agrees_with_native_difference(even_a, odd_a, universal_ab):
    for other in (odd_a, universal_ab, even_a):
        native = DFA.difference(even_a, other)
        ours = even_a.language_difference(other)
        for word in words_up_to(["a", "b"], 5):
            assert native.accepts_input(word) == run_accepts(ours, ours.initial_state, word)


# =====================================================================================
# Functions 46, 47, 48: Kleene star, left quotient, right quotient
# =====================================================================================

QUOTIENT_DFAS = [
    "even_a", "odd_a", "only_ab", "a_even", "empty_ab", "universal_ab",
    "ab_or_b", "only_c", "b_star", "simple", "one_missing", "dead_end",
]
QUOTIENT_NFAS = [
    "nondeterministic", "eps_chain", "nfa_diamond", "nfa_empty",
    "nfa_one_state", "nfa_empty_word", "nfa_eps_mix", "eps_cycle",
]
PREFIX_LENGTH = 6  # longest witness word u tried by the brute-force oracles


def words_of(automaton, alphabet, longest):
    return [w for w in words_up_to(sorted(alphabet), longest) if accepts_any(automaton, w)]


def star_rule(automaton, word):
    """w in L* iff w cuts into pieces of L (dynamic programming, oracle)."""
    reachable = [True] + [False] * len(word)
    for end in range(1, len(word) + 1):
        reachable[end] = any(
            reachable[start] and accepts_any(automaton, word[start:end])
            for start in range(end)
        )
    return reachable[len(word)]


def quotient_pairs(request):
    fixtures = {n: request.getfixturevalue(n) for n in QUOTIENT_DFAS + QUOTIENT_NFAS}
    dfas = [fixtures[n] for n in QUOTIENT_DFAS]
    nfas = [fixtures[n] for n in QUOTIENT_NFAS]
    pairs = [(a, b) for a in dfas for b in dfas]
    pairs += [(a, b) for a in nfas for b in nfas[:4]]
    pairs += [(a, b) for a in dfas[:6] for b in nfas[:6]]
    pairs += [(a, b) for a in nfas[:6] for b in dfas[:6]]
    return pairs


def left_oracle(first, second, word, alphabet):
    return any(
        accepts_any(second, u + word) for u in words_of(first, alphabet, PREFIX_LENGTH)
    )


def right_oracle(first, second, word, alphabet):
    return any(
        accepts_any(first, word + u) for u in words_of(second, alphabet, PREFIX_LENGTH)
    )


# --- function 46 ---------------------------------------------------------------------


def test_46_star_of_the_empty_language(empty_ab):
    r = empty_ab.language_star()
    assert isinstance(r, ExtendedNFA)
    assert nfa_language(r, 3) == {""}


def test_46_star_of_the_empty_word_language():
    only_empty = make_dfa({"q": {}}, initial="q", finals={"q"})
    assert nfa_language(only_empty.language_star(), 3) == {""}


def test_46_star_of_a_single_word(simple):
    r = simple.language_star()
    assert nfa_language(r, 6) == {"", "ab", "abab", "ababab"}
    assert nfa_accepts(r, "") and not nfa_accepts(r, "aba") and not nfa_accepts(r, "abb")


def test_46_star_of_several_words(ab_or_b):
    r = ab_or_b.language_star()
    base = words_of(ab_or_b, {"a", "b"}, 4)
    assert "ab" in base and "b" in base
    assert nfa_accepts(r, "abbab") and nfa_accepts(r, "bbb") and not nfa_accepts(r, "aa")


def test_46_new_initial_state_and_epsilon_transitions(simple):
    r = simple.language_star()
    assert r.initial_state == ("start",)
    assert ("start",) in r.final_states
    assert r.transitions[("start",)] == {EPS: {(0, "q0")}}
    assert r.transitions[(0, "q2")][EPS] == {(0, "q0")}  # old final -> old initial
    assert r.final_states == {("start",), (0, "q2")}
    assert r.input_symbols == simple.input_symbols
    assert len(r.states) == len(simple.states) + 1


def test_46_old_initial_state_with_incoming_edges_needs_the_new_initial_state():
    # L = words ending with 'b' with the initial state on a loop: marking it final
    # would wrongly accept "a".
    looping = make_dfa({"s": {"a": "s", "b": "f"}, "f": {}}, initial="s", finals={"f"})
    r = looping.language_star()
    assert not nfa_accepts(r, "a")
    assert nfa_accepts(r, "aab") and nfa_accepts(r, "abb") and nfa_accepts(r, "")


def test_46_matches_the_definition_on_all_automata(request):
    for name in QUOTIENT_DFAS + QUOTIENT_NFAS:
        a = request.getfixturevalue(name)
        r = a.language_star()
        alphabet = union_alphabet(a, a)
        for word in words_up_to(sorted(alphabet), 5):
            assert nfa_accepts(r, word) == star_rule(a, word), (name, word)


def test_46_is_idempotent_on_the_language(simple):
    once = simple.language_star()
    twice = once.language_star()
    assert nfa_language(once, 6) == nfa_language(twice, 6)


def test_46_does_not_modify_the_automaton(simple, nondeterministic, eps_chain):
    for a in (simple, nondeterministic, eps_chain):
        before = operand_snapshot(a)
        a.language_star()
        assert operand_snapshot(a) == before


def test_46_names_never_collide():
    a = make_nfa({("start",): {"a": {("start",)}}}, initial=("start",), finals={("start",)})
    r = a.language_star()
    assert len(r.states) == 2
    assert nfa_language(r, 3) == {"", "a", "aa", "aaa"}


# --- function 47 ---------------------------------------------------------------------


def test_47_simple_language():
    first = make_dfa({"s": {"a": "t"}, "t": {}}, initial="s", finals={"t"})  # {a}
    second = make_dfa(
        {"s": {"a": "t"}, "t": {"b": "u", "a": "v"}, "u": {}, "v": {}}, initial="s", finals={"u", "v"}
    )  # {ab, aa}
    r = first.language_left_quotient(second)
    assert isinstance(r, ExtendedNFA)
    assert nfa_language(r, 3) == {"b", "a"}  # a^{-1} {ab, aa}


def test_47_argument_order_is_prefix_language_first():
    prefixes = make_dfa({"s": {"a": "t"}, "t": {}}, initial="s", finals={"t"})  # {a}
    target = make_dfa(
        {"s": {"a": "t"}, "t": {"b": "u"}, "u": {}}, initial="s", finals={"u"}
    )  # {ab}
    assert nfa_language(prefixes.language_left_quotient(target), 3) == {"b"}
    assert nfa_language(target.language_left_quotient(prefixes), 3) == set()  # not symmetric


def test_47_first_language_empty(empty_ab, even_a):
    r = empty_ab.language_left_quotient(even_a)
    assert nfa_language(r, 4) == set()
    assert r.transitions[("start",)][EPS] == frozenset()


def test_47_second_language_empty(even_a, empty_ab):
    assert nfa_language(even_a.language_left_quotient(empty_ab), 4) == set()


def test_47_first_language_contains_the_empty_word(even_a, only_ab):
    # even_a contains ε: the quotient contains L2 itself (u = ε).
    r = even_a.language_left_quotient(only_ab)
    assert "ab" in nfa_language(r, 3)
    assert nfa_language(r, 3) == {w for w in words_up_to(["a", "b"], 3) if left_oracle(even_a, only_ab, w, {"a", "b"})}


def test_47_several_prefixes():
    prefixes = make_dfa(
        {"s": {"a": "t", "b": "t"}, "t": {}}, initial="s", finals={"t"}
    )  # {a, b}
    target = make_dfa(
        {"s": {"a": "x", "b": "y"}, "x": {"c": "f"}, "y": {"d": "f"}, "f": {}},
        initial="s",
        finals={"f"},
    )  # {ac, bd}
    r = prefixes.language_left_quotient(target)
    assert nfa_language(r, 2) == {"c", "d"}


def test_47_several_initial_states_through_epsilon_transitions():
    prefixes = make_dfa({"s": {"a": "t", "b": "t"}, "t": {}}, initial="s", finals={"t"})
    target = make_dfa(
        {"s": {"a": "x", "b": "y"}, "x": {"c": "f"}, "y": {"d": "f"}, "f": {}},
        initial="s",
        finals={"f"},
    )
    r = prefixes.language_left_quotient(target)
    assert r.initial_state == ("start",)
    assert r.transitions[("start",)][EPS] == {(0, "x"), (0, "y")}  # two possible starts


def test_47_different_alphabets(only_c, even_a):
    for first, second in ((only_c, even_a), (even_a, only_c), (only_c, only_c)):
        r = first.language_left_quotient(second)
        assert r.input_symbols == second.input_symbols
        for word in words_up_to(sorted(union_alphabet(first, second)), 4):
            assert nfa_accepts(r, word) == left_oracle(first, second, word, union_alphabet(first, second))


def test_47_matches_the_definition_on_many_pairs(request):
    for first, second in quotient_pairs(request):
        alphabet = union_alphabet(first, second)
        r = first.language_left_quotient(second)
        for word in words_up_to(sorted(alphabet), 3):
            assert nfa_accepts(r, word) == left_oracle(first, second, word, alphabet), word


def test_47_does_not_modify_the_operands(even_a, nondeterministic, simple):
    for first, second in ((even_a, simple), (nondeterministic, even_a), (simple, nondeterministic)):
        before = (operand_snapshot(first), operand_snapshot(second))
        first.language_left_quotient(second)
        assert (operand_snapshot(first), operand_snapshot(second)) == before


# --- function 48 ---------------------------------------------------------------------


def test_48_simple_languages():
    dividend = make_dfa(
        {"s": {"a": "t"}, "t": {"b": "u"}, "u": {}}, initial="s", finals={"u"}
    )  # {ab}
    divisor = make_dfa({"s": {"b": "t"}, "t": {}}, initial="s", finals={"t"})  # {b}
    r = dividend.language_right_quotient(divisor)
    assert isinstance(r, ExtendedDFA)
    assert dfa_language(r, {"a", "b"}, 3) == {"a"}
    assert r.final_states == {"t"} and r.transitions == dividend.transitions


def test_48_argument_order_is_dividend_first():
    dividend = make_dfa({"s": {"a": "t"}, "t": {"b": "u"}, "u": {}}, initial="s", finals={"u"})
    divisor = make_dfa({"s": {"b": "t"}, "t": {}}, initial="s", finals={"t"})
    assert dfa_language(divisor.language_right_quotient(dividend), {"a", "b"}, 3) == set()


def test_48_second_language_empty(even_a, empty_ab):
    r = even_a.language_right_quotient(empty_ab)
    assert dfa_language(r, {"a", "b"}, 4) == set()


def test_48_first_language_empty(empty_ab, even_a):
    assert dfa_language(empty_ab.language_right_quotient(even_a), {"a", "b"}, 4) == set()


def test_48_second_language_contains_the_empty_word(only_ab, b_star):
    # ε in L2 (b* contains ε): L1 is included in L1 / L2.
    r = only_ab.language_right_quotient(b_star)
    assert "ab" in dfa_language(r, {"a", "b"}, 3)
    assert dfa_language(r, {"a", "b"}, 3) == {"a", "ab"}


def test_48_several_suffixes():
    dividend = make_dfa(
        {"s": {"a": "x", "b": "y"}, "x": {"c": "f"}, "y": {"d": "f"}, "f": {}},
        initial="s",
        finals={"f"},
    )  # {ac, bd}
    divisor = make_dfa({"s": {"c": "t", "d": "t"}, "t": {}}, initial="s", finals={"t"})  # {c, d}
    assert dfa_language(dividend.language_right_quotient(divisor), {"a", "b", "c", "d"}, 2) == {"a", "b"}


def test_48_several_final_states(nfa_diamond, even_a):
    r = nfa_diamond.language_right_quotient(
        make_dfa({"s": {"b": "t"}, "t": {}}, initial="s", finals={"t"})
    )
    assert isinstance(r, ExtendedNFA)
    assert nfa_language(r, 3) == {"a"}
    assert r.final_states == {"q1", "q2"}


def test_48_the_result_keeps_the_structure_of_the_dividend(even_a, only_c):
    r = even_a.language_right_quotient(only_c)  # no common symbol: only ε-like divisors matter
    assert r.states == even_a.states and r.initial_state == even_a.initial_state
    assert r.input_symbols == even_a.input_symbols


def test_48_different_alphabets(only_c, even_a):
    for first, second in ((only_c, even_a), (even_a, only_c), (only_c, only_c)):
        r = first.language_right_quotient(second)
        alphabet = union_alphabet(first, second)
        assert r.input_symbols == first.input_symbols
        for word in words_up_to(sorted(alphabet), 4):
            assert accepts_any(r, word) == right_oracle(first, second, word, alphabet)


def test_48_matches_the_definition_on_many_pairs(request):
    for first, second in quotient_pairs(request):
        alphabet = union_alphabet(first, second)
        r = first.language_right_quotient(second)
        for word in words_up_to(sorted(alphabet), 3):
            assert accepts_any(r, word) == right_oracle(first, second, word, alphabet), word


def test_48_is_dual_to_the_left_quotient_through_reversal(request):
    # L1 / L2 = ( (L2^R) ^-1 L1^R )^R : checked with functions 36 and 47.
    for first, second in quotient_pairs(request)[::3]:
        alphabet = sorted(union_alphabet(first, second))
        direct = first.language_right_quotient(second)
        dual = second.reversed_automaton().language_left_quotient(first.reversed_automaton())
        for word in words_up_to(alphabet, 3):
            assert accepts_any(direct, word) == nfa_accepts(dual, word[::-1]), word


def test_48_does_not_modify_the_operands(even_a, nondeterministic, simple):
    for first, second in ((even_a, simple), (nondeterministic, even_a), (simple, nondeterministic)):
        before = (operand_snapshot(first), operand_snapshot(second))
        first.language_right_quotient(second)
        assert (operand_snapshot(first), operand_snapshot(second)) == before


def test_new_names_46_47_48_are_not_native():
    for name in ("language_star", "language_left_quotient", "language_right_quotient"):
        assert not hasattr(DFA, name)


# =====================================================================================
# Function 52: NFA -> DFA (subset construction, name of the course statement)
# =====================================================================================


def test_52_nfa_without_epsilon(nondeterministic):
    dfa = nondeterministic.to_dfa()
    assert isinstance(dfa, ExtendedDFA) and dfa.is_deterministic()
    assert dfa.initial_state == frozenset({"q0"})
    assert dfa_language(dfa, {"a", "b"}, 6) == nfa_language(nondeterministic, 6)


def test_52_nfa_with_epsilon_transitions(eps_chain, eps_cycle, nfa_eps_mix):
    assert eps_chain.to_dfa().initial_state == frozenset({"q0", "q1", "q2"})  # closure of q0
    for nfa in (eps_chain, eps_cycle, nfa_eps_mix):
        dfa = nfa.to_dfa()
        assert dfa_language(dfa, nfa.input_symbols - {EPS}, 5) == nfa_language(nfa, 5)
        assert EPS not in dfa.input_symbols


def test_52_several_final_states_and_non_determinism(nfa_diamond):
    dfa = nfa_diamond.to_dfa()
    assert dfa.final_states == {frozenset({"q3", "q4"})}  # a subset is final if it meets F
    assert dfa.transitions[frozenset({"q0"})]["a"] == frozenset({"q1", "q2"})


def test_52_transition_to_the_empty_set(nfa_diamond):
    partial = nfa_diamond.to_dfa()
    full = nfa_diamond.to_dfa(complete=True)
    assert frozenset() not in partial.states and not partial.is_complete()
    assert frozenset() in full.states and full.is_complete()
    assert full.transitions[frozenset()] == {"a": frozenset(), "b": frozenset()}
    assert dfa_language(full, {"a", "b"}, 5) == dfa_language(partial, {"a", "b"}, 5)


def test_52_is_the_same_algorithm_as_function_35(request):
    for name, nfa in all_nfas(request).items():
        for complete in (False, True):
            assert snapshot(nfa.to_dfa(complete)) == snapshot(nfa.determinize(complete)), name


def test_52_language_and_alphabet_are_preserved(request):
    for name, nfa in all_nfas(request).items():
        dfa = nfa.to_dfa()
        assert dfa.input_symbols == nfa.input_symbols - {EPS}, name
        assert dfa_language(dfa, dfa.input_symbols, 5) == nfa_language(nfa, 5), name
        assert dfa.accessible_states() == dfa.states, name


def test_52_subset_names_are_unique_and_the_nfa_is_unchanged(request):
    for name, nfa in all_nfas(request).items():
        before = nfa_snapshot(nfa)
        dfa = nfa.to_dfa()
        assert len(dfa.states) == len(set(dfa.states)), name
        assert nfa_snapshot(nfa) == before, name
