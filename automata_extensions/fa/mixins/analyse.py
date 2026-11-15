"""Graph-analysis algorithms for finite automata (traversals, reachability)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import (
    AbstractSet,
    Any,
    Callable,
    FrozenSet,
    Hashable,
    Iterable,
    Iterator,
    Optional,
    Tuple,
)

State = Hashable

#: What an automaton "holds" while reading a word: one state for a DFA, a
#: frozenset of states for an NFA.
Configuration = Any


@dataclass(frozen=True)
class ExecutionTrace:
    """Result of :meth:`AnalyseMixin.execution_trace`.

    The object is immutable.  It keeps the trace (what happened while reading
    the word) apart from the recognition answer (``accepted``).

    Attributes
    ----------
    word : str
        The word that was read.
    configurations : tuple
        ``configurations[i]`` is the configuration after reading the first
        ``i`` symbols, so ``configurations[0]`` is the initial configuration
        (the empty word is therefore a trace of length one).  A
        configuration is a single state for a DFA and a ``frozenset`` of
        states for an NFA.
    accepted : bool
        ``True`` if the whole word was read and the last configuration is
        accepting.
    blocked_at : int or None
        ``None`` if the whole word was read.  Otherwise the index of the
        first symbol that could not be read (no transition, or a symbol that
        is not in the alphabet).  The trace then stops just before it:
        ``len(configurations) == blocked_at + 1``.
    """

    word: str
    configurations: Tuple[Configuration, ...]
    accepted: bool
    blocked_at: Optional[int] = None


class AnalyseMixin:
    """Analysis algorithms based on the transition graph of an automaton.

    The mixin defines no ``__init__`` and stores no state of its own.  Every
    algorithm relies on a single hook, :meth:`_successors`, which the concrete
    automaton class must provide.  This keeps the algorithms independent of the
    transition format: a DFA, an NFA or a GNFA only needs to say "which states
    can I reach in one step from ``state``?".

    The class must be combined with an automaton class that has the attributes
    ``states`` and ``initial_state``.
    """

    def _successors(self, state: State) -> list[State]:
        """Return the states reachable from ``state`` in exactly one step.

        Must be overridden by the concrete automaton class.  The result must be
        in a deterministic order so that traversals are reproducible.

        Parameters
        ----------
        state : State
            A state of the automaton.

        Returns
        -------
        list
            Successors of ``state`` (possibly with repetitions).
        """
        raise NotImplementedError(
            "The concrete automaton class must implement _successors()."
        )

    def _check_state(self, state: State) -> State:
        """Return ``state`` if it belongs to the automaton.

        Parameters
        ----------
        state : State
            The state to check.

        Returns
        -------
        State
            The same state, unchanged.

        Raises
        ------
        ValueError
            If ``state`` is not a state of the automaton.
        """
        if state not in self.states:
            raise ValueError(f"{state!r} is not a state of the automaton.")
        return state

    def _check_start(self, start_state: State | None) -> State:
        """Return the state a traversal must start from.

        Parameters
        ----------
        start_state : State or None
            Requested start state.  ``None`` means the initial state.

        Returns
        -------
        State
            The validated start state.

        Raises
        ------
        ValueError
            If ``start_state`` is not a state of the automaton.
        """
        if start_state is None:
            return self.initial_state
        return self._check_state(start_state)

    def _bfs_from(
        self,
        starts: Iterable[State],
        neighbours: Callable[[State], Iterable[State]],
    ) -> list[State]:
        """Breadth-first traversal from several start states, on any graph.

        The graph is given by the function ``neighbours``.  Using
        ``self._successors`` explores the automaton; using a function that
        returns predecessors explores the reversed automaton.

        Parameters
        ----------
        starts : iterable of State
            States where the traversal starts.
        neighbours : callable
            ``neighbours(state)`` returns the states one step away.

        Returns
        -------
        list
            Every state reached (start states included), once each, level by
            level.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of edges.
        """
        visited: set[State] = set()
        queue: deque[State] = deque()
        for start in starts:
            if start not in visited:
                visited.add(start)
                queue.append(start)
        order: list[State] = []
        while queue:
            state = queue.popleft()
            order.append(state)
            for neighbour in neighbours(state):
                if neighbour not in visited:
                    # Marked when enqueued, so a state is never queued twice.
                    visited.add(neighbour)
                    queue.append(neighbour)
        return order

    def _reverse_graph(self) -> dict[State, list[State]]:
        """Build the predecessors of every state (the reversed transitions).

        Returns
        -------
        dict
            ``reverse[q]`` is the list of states ``p`` that have at least one
            transition ``p -> q``.  Every state is a key, possibly with an
            empty list.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.
        """
        reverse: dict[State, list[State]] = {state: [] for state in self.states}
        for state in self.states:
            for successor in self._successors(state):
                reverse[successor].append(state)
        return reverse

    def dfs_iterative(self, start_state: State | None = None) -> list[State]:
        """Depth-first traversal using an explicit stack.

        Parameters
        ----------
        start_state : State, optional
            State where the traversal starts.  Defaults to the initial state.

        Returns
        -------
        list
            The states reachable from ``start_state`` (itself included), each
            appearing exactly once, in the order they are first visited
            (pre-order).  The order is the same as :meth:`dfs_recursive`.

        Raises
        ------
        ValueError
            If ``start_state`` is not a state of the automaton.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.

        References
        ----------
        Cormen, Leiserson, Rivest, Stein. Introduction to Algorithms, DFS.
        """
        start = self._check_start(start_state)
        visited: set[State] = set()
        order: list[State] = []
        stack: list[State] = [start]
        while stack:
            state = stack.pop()
            if state in visited:
                continue  # already visited through another path
            visited.add(state)
            order.append(state)
            # Pushed in reverse order so that the first successor is popped
            # first, exactly like the recursive version.
            for successor in reversed(self._successors(state)):
                if successor not in visited:
                    stack.append(successor)
        return order

    def dfs_recursive(self, start_state: State | None = None) -> list[State]:
        """Depth-first traversal using recursion.

        Parameters
        ----------
        start_state : State, optional
            State where the traversal starts.  Defaults to the initial state.

        Returns
        -------
        list
            The states reachable from ``start_state`` (itself included), each
            appearing exactly once, in the order they are first visited
            (pre-order).

        Raises
        ------
        ValueError
            If ``start_state`` is not a state of the automaton.
        RecursionError
            On very long paths (Python's recursion limit, about 1000 by
            default).  Use :meth:`dfs_iterative` for large automata.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.

        References
        ----------
        Cormen, Leiserson, Rivest, Stein. Introduction to Algorithms, DFS.
        """
        start = self._check_start(start_state)
        visited: set[State] = set()
        order: list[State] = []

        def visit(state: State) -> None:
            visited.add(state)
            order.append(state)
            for successor in self._successors(state):
                if successor not in visited:
                    visit(successor)

        visit(start)
        return order

    def bfs(self, start_state: State | None = None) -> list[State]:
        """Breadth-first traversal using a FIFO queue.

        Parameters
        ----------
        start_state : State, optional
            State where the traversal starts.  Defaults to the initial state.

        Returns
        -------
        list
            The states reachable from ``start_state`` (itself included), each
            appearing exactly once, level by level: first the start state,
            then its successors, then their successors, and so on.

        Raises
        ------
        ValueError
            If ``start_state`` is not a state of the automaton.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.

        References
        ----------
        Cormen, Leiserson, Rivest, Stein. Introduction to Algorithms, BFS.
        """
        start = self._check_start(start_state)
        return self._bfs_from([start], self._successors)

    def accessible_states(self) -> FrozenSet[State]:
        """Compute the set of accessible states.

        A state is accessible if some path leads to it from the initial state.

        Returns
        -------
        frozenset
            Accessible states, always including the initial state.  A
            ``frozenset`` is immutable and compares equal to a ``set`` with
            the same elements.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.

        References
        ----------
        Hopcroft, Motwani & Ullman. Introduction to Automata Theory,
        Languages, and Computation.
        """
        return frozenset(self.bfs())

    def is_accessible(self, state: State) -> bool:
        """Test whether a state is accessible from the initial state.

        Parameters
        ----------
        state : State
            The state to test.

        Returns
        -------
        bool
            ``True`` if ``state`` belongs to :meth:`accessible_states`.

        Raises
        ------
        ValueError
            If ``state`` is not a state of the automaton.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.
        """
        return self._check_state(state) in self.accessible_states()

    def coaccessible_states(self) -> FrozenSet[State]:
        """Compute the set of coaccessible states.

        A state is coaccessible if some path leads from it to a final state.
        The computation is a breadth-first traversal of the reversed
        automaton, starting from all the final states at once.

        Returns
        -------
        frozenset
            Coaccessible states.  Every final state is coaccessible (empty
            path); the set is empty if the automaton has no final state.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.

        References
        ----------
        Hopcroft, Motwani & Ullman. Introduction to Automata Theory,
        Languages, and Computation.
        """
        reverse = self._reverse_graph()

        def predecessors(state: State) -> list[State]:
            return reverse[state]

        return frozenset(self._bfs_from(self.final_states, predecessors))

    def is_coaccessible(self, state: State) -> bool:
        """Test whether a state is coaccessible (can reach a final state).

        Parameters
        ----------
        state : State
            The state to test.

        Returns
        -------
        bool
            ``True`` if ``state`` belongs to :meth:`coaccessible_states`.

        Raises
        ------
        ValueError
            If ``state`` is not a state of the automaton.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.
        """
        return self._check_state(state) in self.coaccessible_states()

    def _finish_order(self) -> list[State]:
        """Depth-first traversal of the whole automaton, in finish order.

        Every state is visited, including the states that are not accessible
        from the initial state.  A state is appended to the result when all
        its successors have been fully explored (post-order).  The traversal
        is iterative, so a long path cannot trigger ``RecursionError``.

        Returns
        -------
        list
            All the states, each exactly once, in order of finishing time.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.
        """
        visited: set[State] = set()
        order: list[State] = []
        for start in self.states:
            if start in visited:
                continue
            visited.add(start)
            # Each stack item keeps the state and the iterator of its
            # successors not yet explored: this is the "call stack" of the
            # recursive version, written by hand.
            stack = [(start, iter(self._successors(start)))]
            while stack:
                state, successors = stack[-1]
                for successor in successors:
                    if successor not in visited:
                        visited.add(successor)
                        stack.append((successor, iter(self._successors(successor))))
                        break
                else:
                    # No unexplored successor left: the state is finished.
                    stack.pop()
                    order.append(state)
        return order

    def _contains_cycle(
        self,
        starts: Iterable[State],
        within: AbstractSet[State] | None = None,
    ) -> bool:
        """Tell whether a cycle exists in the part of the graph explored from
        ``starts``.

        Depth-first traversal that distinguishes two kinds of already seen
        states:

        * ``on_path``: states of the current DFS path (the "recursion stack").
          Reaching one of them again closes a cycle.
        * ``done``: states completely explored.  Reaching one of them again
          is *not* a cycle (two paths may simply join).

        Parameters
        ----------
        starts : iterable of State
            States from which the exploration starts.
        within : set of State, optional
            If given, the graph is restricted to these states: transitions
            leading outside the set are ignored.  By default the whole
            graph is used.

        Returns
        -------
        bool
            ``True`` if a cycle (a self-loop included) is found.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.
        """

        def successors_of(state: State) -> Iterator[State]:
            found = self._successors(state)
            if within is None:
                return iter(found)
            return iter([s for s in found if s in within])

        done: set[State] = set()
        for start in starts:
            if start in done:
                continue
            on_path: set[State] = {start}
            stack = [(start, successors_of(start))]
            while stack:
                state, successors = stack[-1]
                for successor in successors:
                    if successor in on_path:
                        return True  # back to a state of the current path
                    if successor not in done:
                        on_path.add(successor)
                        stack.append((successor, successors_of(successor)))
                        break
                else:
                    stack.pop()
                    on_path.discard(state)
                    done.add(state)
        return False

    def strongly_connected_components(self) -> FrozenSet[FrozenSet[State]]:
        """Compute the strongly connected components (Kosaraju's algorithm).

        A strongly connected component (SCC) is a maximal set of states in
        which every state can reach every other one by following the
        transitions in their direction.  An isolated state, or a state
        without cycle, forms an SCC on its own.

        Algorithm
        ---------
        1. Depth-first traversal of the automaton, recording the order in
           which states are finished (:meth:`_finish_order`).
        2. Take the states by decreasing finishing time.  For each state not
           yet in a component, a traversal of the *reversed* automaton
           (:meth:`_reverse_graph`) collects its component.

        Kosaraju was chosen over Tarjan because it is built from the two
        tools the mixin already has (a DFS and the reversed graph), and
        its correctness proof is easy to follow.

        Returns
        -------
        frozenset of frozenset
            The components.  They form a partition of the states: every state
            belongs to exactly one component.  No order is guaranteed.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.

        References
        ----------
        Cormen, Leiserson, Rivest, Stein. Introduction to Algorithms,
        strongly connected components.
        """
        reverse = self._reverse_graph()
        assigned: set[State] = set()
        components: list[FrozenSet[State]] = []
        for state in reversed(self._finish_order()):
            if state in assigned:
                continue

            def unassigned_predecessors(current: State) -> list[State]:
                return [p for p in reverse[current] if p not in assigned]

            component = frozenset(self._bfs_from([state], unassigned_predecessors))
            assigned |= component
            components.append(component)
        return frozenset(components)

    def has_cycle(self) -> bool:
        """Test whether the automaton contains at least one cycle.

        The whole transition graph is explored, including the states that
        are not accessible from the initial state.  A self-loop is a cycle.
        To restrict the search to what a given state can reach, use
        :meth:`has_cycle_from`.

        Returns
        -------
        bool
            ``True`` if some path leads from a state back to itself.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.

        References
        ----------
        Cormen, Leiserson, Rivest, Stein. Introduction to Algorithms,
        depth-first search (back edges).
        """
        return self._contains_cycle(self.states)

    def has_cycle_from(self, state: State) -> bool:
        """Test whether a cycle can be reached from a given state.

        Only the subgraph accessible from ``state`` (itself included) is
        explored.  The answer is ``True`` if ``state`` belongs to a cycle or
        leads to one.  A cycle elsewhere in the automaton is ignored.

        Parameters
        ----------
        state : State
            The state from which the exploration starts.

        Returns
        -------
        bool
            ``True`` if a cycle exists in the subgraph accessible from
            ``state``.

        Raises
        ------
        ValueError
            If ``state`` is not a state of the automaton.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.
        """
        return self._contains_cycle([self._check_state(state)])

    def is_empty(self) -> bool:
        """Test whether the recognised language is empty.

        The language is empty if and only if no accessible state is final:
        a word is accepted exactly when it leads from the initial state to a
        final state, so such a state must be accessible.

        Returns
        -------
        bool
            ``True`` if no accessible state is final, ``False`` otherwise.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions (one traversal of
        the accessible part, see :meth:`accessible_states`).

        References
        ----------
        Hopcroft, Motwani & Ullman. Introduction to Automata Theory,
        Languages, and Computation.
        """
        return self.accessible_states().isdisjoint(self.final_states)

    def is_infinite(self) -> bool:
        """Test whether the recognised language is infinite.

        The language is infinite if and only if there is a cycle made of
        states that are both accessible and coaccessible ("useful" states).

        * A cycle that is not accessible can never be entered, so it
          produces no word.
        * A cycle that is not coaccessible can be entered but no final state
          can be reached from it, so it produces no accepted word either.
        * A cycle that is accessible *and* coaccessible can be repeated as
          many times as we want before reaching a final state: this gives
          arbitrarily long accepted words.

        The cycle search (:meth:`_contains_cycle`) is restricted to the
        intersection of :meth:`accessible_states` and
        :meth:`coaccessible_states`.

        Returns
        -------
        bool
            ``True`` if the language is infinite, ``False`` if it is finite
            (the empty language is finite).

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions (two traversals and
        one cycle search).

        References
        ----------
        Hopcroft, Motwani & Ullman. Introduction to Automata Theory,
        Languages, and Computation (pumping lemma).
        """
        useful = self.accessible_states() & self.coaccessible_states()
        return self._contains_cycle(useful, within=useful)

    def accepts(self, word: str) -> bool:
        """Test whether the automaton recognises a word.

        This is a thin wrapper around the native ``accepts_input`` of
        automata-lib, which already simulates the automaton (DFA and NFA)
        from the initial state and checks the states reached after the
        whole word is read.  The wrapper only makes one case explicit: a word
        containing a symbol that is not in the alphabet is not a word of
        the language, so the answer is ``False`` (no exception).

        Parameters
        ----------
        word : str
            The word to read.  The empty string is the empty word: it is
            accepted exactly when the initial state is final (or, for an NFA,
            when a final state can be reached from it by epsilon moves).

        Returns
        -------
        bool
            ``True`` if the word is accepted, ``False`` otherwise.  A
            missing transition on the way also gives ``False``.

        Complexity
        ----------
        O(|w|) for a DFA (one transition per symbol).  For an NFA, the
        native simulation follows all the possible paths at once.
        """
        if any(symbol not in self.input_symbols for symbol in word):
            return False
        return bool(self.accepts_input(word))

    # --- reading a word: hooks and trace ----------------------------------

    def _initial_configuration(self) -> Configuration:
        """Return the configuration before any symbol is read.

        Must be overridden by the concrete automaton class: the initial state
        for a DFA, the set of initial states (epsilon closure included) for
        an NFA.
        """
        raise NotImplementedError(
            "The concrete automaton class must implement "
            "_initial_configuration()."
        )

    def _next_configuration(
        self, configuration: Configuration, symbol: str
    ) -> Optional[Configuration]:
        """Return the configuration reached by reading one symbol.

        Must be overridden by the concrete automaton class.

        Parameters
        ----------
        configuration : Configuration
            The current configuration.
        symbol : str
            The symbol to read.

        Returns
        -------
        Configuration or None
            The new configuration, or ``None`` if the symbol cannot be read
            (no transition, symbol outside the alphabet, empty set of states
            for an NFA).  As a consequence ``None`` cannot itself be used as
            a state name.
        """
        raise NotImplementedError(
            "The concrete automaton class must implement "
            "_next_configuration()."
        )

    def _is_accepting_configuration(self, configuration: Configuration) -> bool:
        """Tell whether a configuration accepts the word read so far.

        Must be overridden by the concrete automaton class.
        """
        raise NotImplementedError(
            "The concrete automaton class must implement "
            "_is_accepting_configuration()."
        )

    def execution_trace(self, word: str) -> ExecutionTrace:
        """Compute the trace of the automaton while it reads a word.

        The trace is the list of configurations met: the initial one, then
        one after each symbol.  For a DFA a configuration is the single
        current state; for an NFA it is the set of current states.

        This is different from :meth:`accepts`, which only answers yes or no.
        Here the whole run is kept, and the method never raises for a
        rejected word: reading simply stops where the automaton is blocked
        (missing transition or symbol outside the alphabet), which is
        what automata-lib calls a rejection.

        Parameters
        ----------
        word : str
            The word to read (one character per symbol, like the native
            ``accepts_input``).  The empty string is the empty word.

        Returns
        -------
        ExecutionTrace
            Immutable object with the word, the configurations, the
            acceptance answer and, if the automaton was blocked, the index of
            the symbol that could not be read.

        Complexity
        ----------
        O(|w|) for a DFA (one transition per symbol).  For an NFA, each
        step costs at most the size of the current set times its
        transitions.

        Examples
        --------
        For a DFA ``q0 --a--> q1 --b--> q2`` where ``q2`` is final::

            >>> trace = dfa.execution_trace("ab")      # doctest: +SKIP
            >>> trace.configurations                   # doctest: +SKIP
            ('q0', 'q1', 'q2')
            >>> trace.accepted                         # doctest: +SKIP
            True
        """
        configuration = self._initial_configuration()
        configurations = [configuration]
        for index, symbol in enumerate(word):
            following = self._next_configuration(configuration, symbol)
            if following is None:
                return ExecutionTrace(
                    word, tuple(configurations), accepted=False, blocked_at=index
                )
            configuration = following
            configurations.append(configuration)
        return ExecutionTrace(
            word,
            tuple(configurations),
            accepted=self._is_accepting_configuration(configuration),
        )

    # --- reachability from / to a given state -----------------------------

    def reachable_from(self, state: State) -> FrozenSet[State]:
        """Compute the states reachable from a given state.

        Unlike :meth:`accessible_states`, the traversal starts at ``state``
        and not at the initial state.  The convention is that a state is
        reachable from itself by a path of length zero, so ``state`` is
        always in the result.

        Parameters
        ----------
        state : State
            The state where the exploration starts.

        Returns
        -------
        frozenset
            ``state`` and every state that can be reached from it by
            following transitions in their direction.

        Raises
        ------
        ValueError
            If ``state`` is not a state of the automaton.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions (one :meth:`bfs`).
        """
        return frozenset(self.bfs(self._check_state(state)))

    def _direct_predecessors(self, state: State) -> list[State]:
        """Return the states having a transition to ``state`` (one step).

        Complexity
        ----------
        O(|Q| + |E|): the reversed graph is built on each call.
        """
        return self._reverse_graph()[self._check_state(state)]

    def predecessor_states(self, state: State) -> FrozenSet[State]:
        """Compute all the states from which ``state`` can be reached.

        The reversed transitions are explored completely: first the direct
        predecessors (one transition to ``state``), then their own
        predecessors (indirect ones, reached through two or more
        transitions), and so on until nothing new is found.

        The path must have **at least one transition**.  Hence ``state``
        itself is in the result only if it lies on a cycle (a self-loop
        included); the path of length zero does not count.

        The method is named ``predecessor_states`` because the native
        ``DFA.predecessors`` of automata-lib already exists and works on
        words, not on states.

        Parameters
        ----------
        state : State
            The target state.

        Returns
        -------
        frozenset
            Direct and indirect predecessors of ``state``.

        Raises
        ------
        ValueError
            If ``state`` is not a state of the automaton.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.
        """
        reverse = self._reverse_graph()
        direct = reverse[self._check_state(state)]
        return frozenset(self._bfs_from(direct, reverse.__getitem__))

    # --- determinism ------------------------------------------------------------

    #: Key used by automata-lib for epsilon (lambda) transitions of an NFA.
    _EPSILON = ""

    def _targets(self, value: Any) -> list[State]:
        """Return the list of destinations stored in a transition table cell.

        In automata-lib a DFA cell is one state and an NFA cell is a *set* of
        states.  A class that stores single states which are themselves
        collections (for instance ``frozenset`` names) must override this
        hook, as :class:`ExtendedDFA` does.
        """
        if isinstance(value, (set, frozenset, list, tuple)):
            return list(value)
        return [value]

    def is_deterministic(self) -> bool:
        """Test whether the automaton is deterministic.

        An automaton is deterministic when

        * it has a single initial state (automata-lib stores one
          ``initial_state``; an ``initial_states`` attribute, if a model has
          one, must hold at most one state),
        * it has no epsilon transition (key ``""``),
        * for every state and symbol there is at most one destination.

        Determinism is **not** completeness: a deterministic automaton may
        lack transitions.  The test only reads ``transitions``, so it works
        for any class that follows the layout of automata-lib
        (``{state: {symbol: destination(s)}}``).

        Returns
        -------
        bool
            ``True`` if the automaton is deterministic.

        Complexity
        ----------
        O(|E|), where E is the set of table cells.
        """
        initials = getattr(self, "initial_states", None)
        if initials is not None and len(initials) > 1:
            return False
        for outgoing in self.transitions.values():
            for symbol, value in outgoing.items():
                destinations = self._targets(value)
                if symbol == self._EPSILON and destinations:
                    return False
                if len(destinations) > 1:
                    return False
        return True

    # --- epsilon closure --------------------------------------------------------

    def _epsilon_successors(self, state: State) -> list[State]:
        """Return the states reached from ``state`` by one epsilon transition."""
        value = self.transitions.get(state, {}).get(self._EPSILON)
        if value is None:
            return []
        return sorted(self._targets(value), key=repr)

    def epsilon_closure(self, state: State) -> FrozenSet[State]:
        """Return the epsilon closure of one state.

        The closure of ``q`` is the set of states reachable from ``q`` using
        epsilon transitions only (key ``""`` in automata-lib), ``q`` itself
        included, even without any epsilon transition.  The traversal is a
        breadth-first search with a set of visited states
        (:meth:`_bfs_from`), so epsilon cycles end.

        A DFA has no epsilon transition, so its closures are singletons.

        Parameters
        ----------
        state : State
            A state of the automaton.

        Returns
        -------
        frozenset
            The closure.

        Raises
        ------
        ValueError
            If ``state`` is not a state of the automaton.

        Complexity
        ----------
        O(|Q| + |E_ε|), where E_ε is the set of epsilon transitions.
        """
        return self.epsilon_closure_of([self._check_state(state)])

    def epsilon_closure_of(self, states: Iterable[State]) -> FrozenSet[State]:
        """Return the epsilon closure of a set of states.

        This is the union of the closures of its members.  Two methods
        (single state / set of states) are used instead of one that guesses
        the kind of its argument, because a state may itself be a tuple or a
        ``frozenset`` (as in the automata built by ``minimize`` or
        ``determinize``).

        Parameters
        ----------
        states : iterable of State
            States of the automaton; may be empty.

        Returns
        -------
        frozenset
            The union of the closures; the empty set for an empty input.

        Raises
        ------
        ValueError
            If a state does not belong to the automaton.
        """
        starts = [self._check_state(state) for state in states]
        return frozenset(self._bfs_from(starts, self._epsilon_successors))

    # --- useful states, trim ----------------------------------------------------

    def useful_states(self) -> FrozenSet[State]:
        """Return the useful states: accessible **and** coaccessible.

        A useful state lies on some path from the initial state to a final
        state, so it matters for the language.  Computed as the intersection
        of :meth:`accessible_states` and :meth:`coaccessible_states` (nothing
        is recomputed).  The set is empty exactly when the language is empty.

        Complexity
        ----------
        O(|Q| + |E|).
        """
        return self.accessible_states() & self.coaccessible_states()

    def is_trim(self) -> bool:
        """Test whether the automaton is trim (has no useless state).

        Every state must be useful (:meth:`useful_states`).  Convention for
        the **empty language**: no state is useful, but an automaton needs an
        initial state, so the one-state automaton without final state (what
        ``trim()`` returns) is also trim.  Hence ``is_trim()`` is true iff
        ``trim()`` would keep all the states.

        Complexity
        ----------
        O(|Q| + |E|).
        """
        kept = self.useful_states() or frozenset([self.initial_state])
        return kept == frozenset(self.states)
