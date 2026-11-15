"""Extended DFA: the original ``DFA`` of automata-lib plus our mixins."""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable

from automata.fa.dfa import DFA

from .mixins.analyse import AnalyseMixin, State
from .mixins.conversions import ConversionMixin
from .mixins.grammaire import GrammaireMixin
from .mixins.morphisme import MorphismeMixin
from .mixins.visualisation import VisualisationMixin
from .morphism import AutomatonMorphism


class ExtendedDFA(VisualisationMixin, GrammaireMixin, MorphismeMixin, ConversionMixin, AnalyseMixin, DFA):
    """A ``DFA`` with the analysis algorithms of :class:`AnalyseMixin`.

    The mixin comes before ``DFA`` in the bases, so its methods have priority
    in the method resolution order (MRO).  The original class is not modified.
    """

    def _successors(self, state: State) -> list[State]:
        """Return the states reachable from ``state`` in one transition.

        In automata-lib 9.2.0, ``transitions`` has the form
        ``{state: {symbol: next_state}}``.  A state with no outgoing
        transition has an empty dict (only possible with
        ``allow_partial=True``); ``.get`` also protects against a missing key.

        Parameters
        ----------
        state : State
            A state of the automaton.

        Returns
        -------
        list
            Next states, ordered by symbol (``a`` before ``b``) so that
            traversals are reproducible.  A state may appear several times if
            several symbols lead to it.
        """
        outgoing = self.transitions.get(state, {})
        return [outgoing[symbol] for symbol in sorted(outgoing)]

    def _targets(self, value: State) -> list[State]:
        """A DFA table cell always holds exactly one state, whatever its type."""
        return [value]

    def _restrict(self, kept: frozenset) -> ExtendedDFA:
        """Sub-automaton on ``kept`` (function 19, ``induced_subautomaton``)."""
        return self.induced_subautomaton(kept)

    def _initial_configuration(self) -> State:
        """Return the configuration before reading: the initial state."""
        return self.initial_state

    def _next_configuration(self, configuration: State, symbol: str) -> State | None:
        """Return the state reached from ``configuration`` by ``symbol``.

        In a DFA the configuration is the single current state.  ``None`` is
        returned when no move exists: either the transition is missing
        (partial DFA) or ``symbol`` is not in the alphabet, which in both
        cases means the word is rejected.
        """
        return self.transitions.get(configuration, {}).get(symbol)

    def _is_accepting_configuration(self, configuration: State) -> bool:
        """Tell whether the current state is final."""
        return configuration in self.final_states

    # --- building new automata from this one ------------------------------
    #
    # These methods are in ExtendedDFA and not in AnalyseMixin because they
    # depend on the way a DFA is represented and built in automata-lib
    # (transitions as {state: {symbol: state}}, DFA constructor, allow_partial).
    # The definitions differ for an NFA, which will get its own versions.

    def induced_subautomaton(self, states: Iterable[State]) -> ExtendedDFA:
        """Build the sub-automaton induced by a set of states.

        Only the requested states are kept, and only the transitions whose
        source **and** target are both kept.  The alphabet is unchanged, and
        the final states are those of the original automaton that are kept.
        The original automaton is not modified.

        Because transitions may be removed, the result is built with
        ``allow_partial=True``.  Every kept state is still a key of
        ``transitions`` (an empty dict if it has no transition left), as
        automata-lib requires.

        Parameters
        ----------
        states : iterable of State
            The states to keep.

        Returns
        -------
        ExtendedDFA
            A new automaton.  Every word it accepts is also accepted by the
            original one, since an accepting run inside the kept states is
            also a run of the original automaton.

        Raises
        ------
        ValueError
            If ``states`` is empty, contains an unknown state, or does not
            contain the initial state.  A DFA must have an initial state that
            is one of its states, so no valid DFA exists in these cases and
            no other initial state is chosen arbitrarily.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.
        """
        kept = frozenset(states)
        unknown = kept - self.states
        if unknown:
            raise ValueError(
                f"Not states of the automaton: {sorted(map(repr, unknown))}."
            )
        if not kept:
            raise ValueError("The set of states is empty: no valid DFA exists.")
        if self.initial_state not in kept:
            raise ValueError(
                f"The initial state {self.initial_state!r} must be kept: a DFA "
                "needs an initial state."
            )
        transitions = {
            state: {
                symbol: target
                for symbol, target in self.transitions.get(state, {}).items()
                if target in kept
            }
            for state in kept
        }
        return type(self)(
            states=kept,
            input_symbols=self.input_symbols,
            transitions=transitions,
            initial_state=self.initial_state,
            final_states=self.final_states & kept,
            allow_partial=True,
        )

    def is_complete(self) -> bool:
        """Test whether every state has a transition for every symbol.

        Completeness is different from determinism.  A DFA is deterministic
        by construction (one target per state and symbol at most), but it may
        be incomplete when built with ``allow_partial=True``.  The flag
        itself does not matter: a table with every transition defined is
        complete whatever the flag says.  For an NFA the notion differs
        (at least one target per state and symbol, and the NFA may be
        non-deterministic); it is not handled here.

        Returns
        -------
        bool
            ``True`` if no transition is missing.

        Complexity
        ----------
        O(|Q| * |Σ|).
        """
        return all(
            self.input_symbols.issubset(self.transitions.get(state, {}))
            for state in self.states
        )

    def complete(self, sink_state: State | None = None) -> ExtendedDFA:
        """Return an equivalent complete DFA (sink state added if needed).

        This is a wrapper around the native ``DFA.to_complete`` of
        automata-lib, which adds a sink (trap) state, sends every missing
        transition to it, and gives it a loop on every symbol.  The sink is
        not final, so the recognised language is unchanged.  If the
        automaton is already complete, no state is added.  The wrapper makes
        sure the result is an ``ExtendedDFA`` (and not a plain ``DFA``) and
        is built with ``allow_partial=False``, so automata-lib itself
        checks that it is complete.  The original automaton is not modified.

        Parameters
        ----------
        sink_state : State, optional
            Name of the sink state.  By default automata-lib chooses the
            largest negative integer that is not already a state name.

        Returns
        -------
        ExtendedDFA
            A new complete automaton recognising the same language.

        Complexity
        ----------
        O(|Q| * |Σ|).
        """
        native = self.to_complete(trap_state=sink_state)
        return type(self)(
            states=native.states,
            input_symbols=native.input_symbols,
            transitions=native.transitions,
            initial_state=native.initial_state,
            final_states=native.final_states,
            allow_partial=False,
        )

    # --- operations on languages ------------------------------------------
    #
    # Naming: automata-lib already has DFA.complement(retain_names, minify) and
    # DFA.issubset(other).  They are NOT replaced (the mixin comes first in the
    # MRO, so the same names would hide them): the methods below are called
    # complemented() and is_included_in().

    def complemented(self) -> ExtendedDFA:
        """Build a DFA recognising the complement of the language.

        The complement is taken over the alphabet of the automaton:
        ``Σ* \\ L(A)``.  A word containing a symbol outside the alphabet is in
        neither language (the automaton cannot read it), so it is rejected by
        the result too.

        Algorithm: make the DFA complete if it is not (:meth:`complete`, which
        relies on the native ``to_complete``), then swap final and non-final
        states.  Completeness is essential: with a missing transition a word
        would be blocked, hence rejected by both the DFA and its "complement".
        State names, the initial state and the alphabet are kept, and the
        automaton is not minimised (unlike the native ``complement``, which
        renames the states by default).  The original is not modified.

        Only DFAs are handled: an NFA must be determinised first (a
        determinisation is not part of the project yet; automata-lib has
        ``DFA.from_nfa``).

        Returns
        -------
        ExtendedDFA
            A new complete automaton, built with ``allow_partial=False``.

        Complexity
        ----------
        O(|Q| * |Σ|).
        """
        base = self if self.is_complete() else self.complete()
        return type(self)(
            states=base.states,
            input_symbols=base.input_symbols,
            transitions=base.transitions,
            initial_state=base.initial_state,
            final_states=base.states - base.final_states,
            allow_partial=False,
        )

    # Pairs of states (used by the product and the inclusion test).  A pair is
    # (state of self, state of other); ``None`` in a pair means "this automaton
    # cannot read the word any more" (missing transition or symbol outside its
    # alphabet), i.e. the word is rejected by it.  ``None`` is therefore never
    # a state name.

    @staticmethod
    def _step(automaton: DFA, state: State | None, symbol: str) -> State | None:
        """Return the state reached by ``symbol``, or ``None`` if blocked."""
        if state is None:
            return None
        return automaton.transitions.get(state, {}).get(symbol)

    def _pair_alphabet(self, other: DFA) -> list[str]:
        """Return the union of the two alphabets, in a reproducible order."""
        return sorted(set(self.input_symbols) | set(other.input_symbols), key=repr)

    def _pair_step(self, other: DFA, pair: tuple, symbol: str) -> tuple:
        """Return the pair reached from ``pair`` by reading ``symbol``."""
        first, second = pair
        return (self._step(self, first, symbol), self._step(other, second, symbol))

    def cartesian_product(
        self, other: DFA, accept: Callable[[bool, bool], bool]
    ) -> ExtendedDFA:
        """Build the product of two DFAs, with a rule choosing the final pairs.

        The states are pairs ``(q1, q2)``, ``q1`` from ``self`` and ``q2`` from
        ``other``.  The initial state is ``(initial of self, initial of
        other)``.  On a symbol, both components move at the same time.  Only
        the pairs reachable from the initial pair are built (breadth-first),
        not all the ``|Q1| * |Q2|`` pairs.

        The product by itself is not an intersection: which pairs are final
        is decided by ``accept(in_first, in_second)``, called with two
        booleans ("the first component is final", "the second component is
        final").  The result recognises the words ``w`` for which
        ``accept(w in L(A), w in L(B))`` is true.  Examples:

        * intersection: ``lambda a, b: a and b``
        * union: ``lambda a, b: a or b``
        * difference ``L(A) \\ L(B)``: ``lambda a, b: a and not b``
        * symmetric difference: ``lambda a, b: a != b``

        **Different alphabets.**  The product is built over the union of the
        two alphabets.  An automaton that cannot read a symbol (no
        transition, or symbol outside its alphabet) rejects the word, so its
        component becomes ``None`` and stays ``None``.  The pair ``(q, None)``
        is final only if ``accept(q is final, False)`` is true.  Hence the
        result is correct for any rule, ``or`` included.  When both DFAs are
        complete and share their alphabet, ``None`` never appears.

        Parameters
        ----------
        other : DFA
            The second automaton (a native ``DFA`` is enough: only
            ``transitions``, ``initial_state``, ``final_states`` and
            ``input_symbols`` are read).
        accept : callable
            Rule giving the final pairs, see above.  It has no default on
            purpose, so the choice of the rule is always explicit.

        Returns
        -------
        ExtendedDFA
            A new complete automaton over the union of the alphabets.  Neither
            operand is modified.

        Complexity
        ----------
        O(|Q1| * |Q2| * |Σ|) in the worst case.
        """
        alphabet = self._pair_alphabet(other)
        initial = (self.initial_state, other.initial_state)
        transitions: dict[tuple, dict[str, tuple]] = {}
        seen = {initial}
        queue = deque([initial])
        while queue:
            pair = queue.popleft()
            outgoing = {}
            for symbol in alphabet:
                following = self._pair_step(other, pair, symbol)
                outgoing[symbol] = following
                if following not in seen:
                    seen.add(following)
                    queue.append(following)
            transitions[pair] = outgoing
        finals = {
            (first, second)
            for first, second in transitions
            if accept(
                first is not None and first in self.final_states,
                second is not None and second in other.final_states,
            )
        }
        return type(self)(
            states=frozenset(transitions),
            input_symbols=frozenset(alphabet),
            transitions=transitions,
            initial_state=initial,
            final_states=finals,
            allow_partial=False,
        )

    def is_included_in(self, other: DFA) -> bool:
        """Test whether ``L(self)`` is included in ``L(other)``.

        ``L(A) ⊆ L(B)`` means every word accepted by A is accepted by B.  This
        is not equality: ``L(A) ⊆ L(B)`` and ``L(B) ⊆ L(A)`` must both hold
        for equal languages.

        Instead of building ``A ∩ complement(B)`` and testing it for
        emptiness, the pairs ``(p, q)`` reachable from the initial pair are
        explored directly (breadth-first), the same pairs as in
        :meth:`cartesian_product`.  The answer is ``False`` as soon as a
        reachable pair has ``p`` final and ``q`` not final (or blocked):
        the word leading to it is accepted by A and rejected by B.  No
        automaton is built, and the search stops at the first witness.
        Pairs whose first component is blocked are not explored further, as A
        accepts nothing more from them.

        Incomplete automata need no completion, and different alphabets are
        handled as in :meth:`cartesian_product`: a word using a symbol that
        B cannot read is rejected by B.

        Parameters
        ----------
        other : DFA
            The automaton whose language should contain ``L(self)``.

        Returns
        -------
        bool
            ``True`` if every word accepted by ``self`` is accepted by
            ``other``.

        Complexity
        ----------
        O(|Q1| * |Q2| * |Σ|) in the worst case.
        """
        alphabet = self._pair_alphabet(other)
        initial = (self.initial_state, other.initial_state)
        seen = {initial}
        queue = deque([initial])
        while queue:
            pair = queue.popleft()
            first, second = pair
            if first is None:
                continue  # self rejects every continuation of this word
            if first in self.final_states and (
                second is None or second not in other.final_states
            ):
                return False  # a word accepted by self and rejected by other
            for symbol in alphabet:
                following = self._pair_step(other, pair, symbol)
                if following not in seen:
                    seen.add(following)
                    queue.append(following)
        return True

    # --- comparing automata -----------------------------------------------
    #
    # ``==`` is NOT redefined (automata-lib does not document it): the three
    # methods below have their own names.

    def has_same_language_as(self, other: DFA) -> bool:
        """Test whether two automata recognise exactly the same language.

        ``L(A) = L(B)`` holds when both inclusions hold: ``L(A) ⊆ L(B)`` and
        ``L(B) ⊆ L(A)``.  Both are decided by :meth:`is_included_in`, so
        incomplete automata and different alphabets are handled the same way:
        a word using a symbol that an automaton cannot read is rejected by it.
        Two automata with different alphabets can therefore have the same
        language (for example ``a*`` read by a DFA over ``{a}`` and by a
        partial DFA over ``{a, b}`` without any ``b`` transition).

        This compares languages, not structures: states may have different
        names, numbers or transitions.  For the structural comparison see
        :meth:`is_isomorphic_to`.

        Parameters
        ----------
        other : DFA
            The second automaton (a native ``DFA`` is enough).

        Returns
        -------
        bool
            ``True`` if both automata accept exactly the same words.

        Complexity
        ----------
        O(|Q1| * |Q2| * |Σ|), two explorations of the pairs of states.
        """
        reverse = other if isinstance(other, ExtendedDFA) else type(self)(
            states=other.states,
            input_symbols=other.input_symbols,
            transitions=other.transitions,
            initial_state=other.initial_state,
            final_states=other.final_states,
            allow_partial=True,
        )
        return self.is_included_in(other) and reverse.is_included_in(self)

    def _propagate(
        self,
        other: DFA,
        mapping: dict[State, State],
        inverse: dict[State, State],
        state: State,
        image: State,
    ) -> list[tuple[State, State]] | None:
        """Map ``state`` to ``image`` and everything this choice forces.

        Because the automata are deterministic, once ``p`` is mapped to ``q``
        the target of ``p`` on a symbol must be mapped to the target of ``q``
        on the same symbol, and so on.  Each new pair is checked: same
        finality, same symbols with a transition, and no conflict with the
        pairs already chosen (``mapping`` and ``inverse`` keep the mapping a
        bijection).

        Returns
        -------
        list or None
            The new pairs added to ``mapping``/``inverse``, or ``None`` if the
            choice is impossible (in which case nothing is added).
        """
        added: list[tuple[State, State]] = []
        stack = [(state, image)]
        while stack:
            p, q = stack.pop()
            if p in mapping or q in inverse:
                if mapping.get(p) == q and inverse.get(q) == p:
                    continue  # already mapped this way
                ok = False
            else:
                outgoing_p = self.transitions.get(p, {})
                outgoing_q = other.transitions.get(q, {})
                ok = (p in self.final_states) == (q in other.final_states) and set(
                    outgoing_p
                ) == set(outgoing_q)
            if not ok:
                for undone_p, undone_q in added:
                    del mapping[undone_p]
                    del inverse[undone_q]
                return None
            mapping[p] = q
            inverse[q] = p
            added.append((p, q))
            for symbol, target in outgoing_p.items():
                stack.append((target, outgoing_q[symbol]))
        return added

    def _extend_isomorphism(
        self, other: DFA, mapping: dict[State, State], inverse: dict[State, State]
    ) -> bool:
        """Map the states not reached yet (backtracking search).

        States reachable from the initial state are all forced by
        :meth:`_propagate`.  Only states that are **not accessible** can still
        be free: for one of them, every still unused state of ``other`` is
        tried, and the choice is undone if it leads nowhere.
        """
        free = next((s for s in self.states if s not in mapping), None)
        if free is None:
            return True
        for candidate in other.states:
            if candidate in inverse:
                continue
            added = self._propagate(other, mapping, inverse, free, candidate)
            if added is None:
                continue
            if self._extend_isomorphism(other, mapping, inverse):
                return True
            for p, q in added:
                del mapping[p]
                del inverse[q]
        return False

    def is_isomorphic_to(self, other: DFA) -> bool:
        """Test whether two DFAs are the same up to a renaming of the states.

        They are isomorphic if a **bijection** ``f`` between the states
        exists such that:

        * ``f`` maps the initial state to the initial state;
        * ``q`` is final if and only if ``f(q)`` is final;
        * for every state ``q`` and symbol ``a``, ``q`` has a transition on
          ``a`` if and only if ``f(q)`` has one, and then
          ``f(δ(q, a)) = δ'(f(q), a)`` (same symbols, same targets).

        The alphabets must be equal: the alphabet is part of the structure.

        This is stricter than :meth:`has_same_language_as`: two DFAs may
        accept the same language without being isomorphic (for example ``a*``
        with one state and ``a*`` with two final states).

        Algorithm: quick necessary conditions first (same alphabet, same
        number of states, same number of final states).  Then the initial
        states are mapped and everything reachable is **forced** by
        determinism, in a single pass.  States that are not accessible are
        matched by a backtracking search; this is the only part that may take
        exponential time, and only when many unaccessible states look alike.

        Only DFAs are handled (a DFA has exactly one initial state).  An NFA
        has a set of initial states and non-deterministic transitions, so the
        search would be different.

        Parameters
        ----------
        other : DFA
            The second automaton (a native ``DFA`` is enough).

        Returns
        -------
        bool
            ``True`` if the automata are the same up to the names of states.

        Complexity
        ----------
        O(|Q| * |Σ|) when every state is accessible.
        """
        if set(self.input_symbols) != set(other.input_symbols):
            return False
        if len(self.states) != len(other.states):
            return False
        if len(self.final_states) != len(other.final_states):
            return False
        mapping: dict[State, State] = {}
        inverse: dict[State, State] = {}
        start = self._propagate(
            other, mapping, inverse, self.initial_state, other.initial_state
        )
        if start is None:
            return False
        return self._extend_isomorphism(other, mapping, inverse)

    def is_prefix_closed(self) -> bool:
        """Test whether the recognised language is prefix-closed.

        ``L`` is prefix-closed if every prefix of every word of ``L`` is in
        ``L`` (so, as soon as ``L`` is not empty, it contains the empty word).

        Criterion for a DFA: ``L`` is prefix-closed if and only if **every
        useful state is final**, a useful state being both accessible and
        coaccessible (:meth:`accessible_states`, :meth:`coaccessible_states`).

        * If a useful state ``s`` is reached by ``u`` and ``u v`` is accepted
          (``s`` is coaccessible), ``u`` is a prefix of an accepted word, so
          ``u`` must be accepted: ``s`` is final.
        * Conversely, if all useful states are final, a prefix ``u`` of an
          accepted word leads to a state that is accessible (by ``u``) and
          coaccessible (by the rest of the word): it is useful, hence final.

        The empty language is prefix-closed (nothing to check).  Non-useful
        states are ignored: a non-final dead state, such as the sink of a
        complete DFA, does not matter.  The criterion relies on
        determinism, so it is not valid as such for an NFA.

        Returns
        -------
        bool
            ``True`` if the language is prefix-closed.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.

        References
        ----------
        Hopcroft, Motwani & Ullman. Introduction to Automata Theory,
        Languages, and Computation.
        """
        useful = self.accessible_states() & self.coaccessible_states()
        return useful <= self.final_states

    def largest_prefix_closed_sublanguage(self) -> ExtendedDFA:
        """Build a DFA for the largest prefix-closed sublanguage of ``L``.

        Let ``K`` be the set of words ``w`` of ``L`` such that **every**
        prefix of ``w`` (the empty word and ``w`` itself included) is in
        ``L``.  This is not the test of :meth:`is_prefix_closed`: here the
        language is *repaired*, not just checked.

        Construction
        ------------
        Keep the states that can be reached from the initial state by a path
        whose states are **all final** (the initial state included), and
        keep only the transitions between kept states
        (:meth:`induced_subautomaton`).  If the initial state is not final,
        ``K`` is empty and the result is the one-state automaton without
        final state.

        Mathematical justification
        --------------------------
        1. *K is included in L(A)*: a word of ``K`` is one of its own
           prefixes, so it is in ``L``.
        2. *K is prefix-closed*: if ``w`` is in ``K`` and ``u`` is a prefix
           of ``w``, every prefix of ``u`` is a prefix of ``w``, hence in
           ``L``; and ``u`` itself is in ``L``.  So ``u`` is in ``K``.
        3. *K is the largest*: let ``M`` be prefix-closed with ``M`` included
           in ``L``.  For ``w`` in ``M`` all its prefixes are in ``M``, hence
           in ``L``, so ``w`` is in ``K``.  Thus ``M`` is included in ``K``.
        4. *The automaton recognises K*: the run on ``w`` visits exactly the
           states reached by the prefixes of ``w``.  The new automaton
           accepts ``w`` iff the whole run stays in the kept states, i.e. iff
           every prefix leads to a final state, i.e. iff every prefix is in
           ``L`` (determinism: one run per word).  All kept states are
           final, so the result also passes :meth:`is_prefix_closed`.

        Conventions
        -----------
        * The empty word is in ``K`` iff the initial state is final.
        * Inaccessible states are never reached, so they are dropped.
        * The original automaton is not modified; the result is a new
          (generally partial) automaton of the same class.

        Returns
        -------
        ExtendedDFA
            A DFA whose language is ``K``.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.
        """
        if self.initial_state not in self.final_states:
            return self.induced_subautomaton([self.initial_state])
        kept = self._bfs_from(
            [self.initial_state],
            lambda state: [
                target
                for target in self._successors(state)
                if target in self.final_states
            ],
        )
        return self.induced_subautomaton(kept)

    def distinguishing_word(self, first: State, second: State) -> str | None:
        """Return a shortest word that distinguishes two states, if any.

        A word ``w`` distinguishes ``p`` and ``q`` if exactly one of
        ``δ*(p, w)`` and ``δ*(q, w)`` is final.

        Method
        ------
        Breadth-first search in the product of the automaton with itself,
        starting from ``(p, q)``.  The first pair containing exactly one
        final state gives, through the parent links, a shortest
        distinguishing word.  A missing transition is treated as a move to a
        non-final dead state, represented by ``None`` (never a state name),
        as in :meth:`is_included_in`.

        Parameters
        ----------
        first, second : State
            States of the automaton.

        Returns
        -------
        str or None
            A shortest distinguishing word (``""`` if exactly one state is
            final), or ``None`` if the states are equivalent.  Identical
            states are always equivalent.

        Raises
        ------
        ValueError
            If one of the states does not belong to the automaton.

        Notes
        -----
        Valid for DFA only: the determinism makes one run per word.

        Complexity
        ----------
        O(|Q|² * |Σ|) in the worst case.
        """
        self._check_state(first)
        self._check_state(second)
        if first == second:
            return None
        alphabet = self._pair_alphabet(self)
        start = (first, second)
        parent: dict[tuple, tuple[tuple, str] | None] = {start: None}
        queue: deque[tuple] = deque([start])
        while queue:
            pair = queue.popleft()
            if (pair[0] in self.final_states) != (pair[1] in self.final_states):
                word: list[str] = []
                link = parent[pair]
                while link is not None:
                    pair, symbol = link
                    word.append(symbol)
                    link = parent[pair]
                return "".join(reversed(word))
            for symbol in alphabet:
                nxt = self._pair_step(self, pair, symbol)
                if nxt == (None, None) or nxt in parent:
                    continue
                parent[nxt] = (pair, symbol)
                queue.append(nxt)
        return None

    def are_distinguishable(self, first: State, second: State) -> bool:
        """Tell whether two states are distinguishable.

        ``True`` iff :meth:`distinguishing_word` finds a word.  Same
        validation, same DFA-only limit.
        """
        return self.distinguishing_word(first, second) is not None

    def myhill_nerode_classes(
        self, accessible_only: bool = False
    ) -> frozenset[frozenset[State]]:
        """Partition the states into Myhill-Nerode equivalence classes.

        ``p ~ q`` iff no word distinguishes them (:meth:`distinguishing_word`
        returns ``None``).  The relation is an equivalence, and it is
        computed here by Moore's partition refinement, which agrees with
        :meth:`are_distinguishable` on every pair.

        Method
        ------
        Start with the partition {final states, non-final states}; then
        split every class according to the classes reached by each symbol,
        until nothing changes.  A missing transition leads to the dead block
        ``None`` (not a state name), so partial DFAs are handled as their
        completion would be: all states that cannot reach a final state end
        up together, with no invented state in the output.

        Parameters
        ----------
        accessible_only : bool, default False
            Convention for inaccessible states.  ``False``: every state is
            partitioned (the relation is defined on all states).  ``True``:
            only accessible states are partitioned (the classes of the
            minimal DFA); inaccessible states are left out.

        Returns
        -------
        frozenset of frozenset
            The classes.  They are non-empty, disjoint, and cover the
            considered states.  The result is a partition, not the split
            into final and non-final states, which is only its starting
            point.

        Notes
        -----
        For DFA only (an NFA would first have to be determinized).

        Complexity
        ----------
        O(|Q|² * |Σ|) in the worst case.
        """
        considered = (
            self.accessible_states() if accessible_only else self.states
        )
        alphabet = self._pair_alphabet(self)
        # ``None`` is the dead block; it takes part in the refinement only.
        nodes: list[State | None] = list(considered) + [None]
        block: dict[State | None, int] = {node: int(node in self.final_states) for node in nodes}
        count = len(set(block.values()))
        while True:
            signature = {
                node: (
                    block[node],
                    tuple(
                        block[self._step(self, node, symbol)] for symbol in alphabet
                    ),
                )
                for node in nodes
            }
            ids: dict[tuple, int] = {}
            block = {node: ids.setdefault(signature[node], len(ids)) for node in nodes}
            if len(ids) == count:
                break
            count = len(ids)
        groups: dict[int, set[State]] = {}
        for node in considered:
            groups.setdefault(block[node], set()).add(node)
        return frozenset(frozenset(group) for group in groups.values())

    def quotient_by(self, classes: Iterable[Iterable[State]]) -> AutomatonMorphism:
        """Merge the states of each class and return the projection (function 61, fusion).

        The result is the **strict** morphism ``h : self → quotient`` that
        sends every state to its class; ``h.target`` is the quotient DFA,
        whose states are the classes (``frozenset``), whose initial state is
        the class of the initial state and whose final states are the classes
        of final states.  Because the morphism is strict,
        ``L(quotient) = L(self)``.

        A partition is accepted only if merging is **compatible** with the
        automaton: a class must not mix final and non-final states, the members
        of a class must agree on which symbols have a transition, and their
        successors on a symbol must lie in one single class.  Otherwise the
        merge would change the language or create non-determinism, and the
        error says which class and symbol are responsible.

        Parameters
        ----------
        classes : iterable of iterables of states
            A partition of ``self.states`` (non-empty, disjoint, covering).

        Raises
        ------
        ValueError
            If ``classes`` is not a partition, or is not a congruence.
        """
        blocks = [frozenset(c) for c in classes]
        owner: dict[State, frozenset] = {}
        for block in blocks:
            if not block:
                raise ValueError("a class of the partition is empty.")
            for q in block:
                if q in owner:
                    raise ValueError(f"state {q!r} belongs to two classes.")
                owner[q] = block
        outside = set(owner) - set(self.states)
        if outside:
            raise ValueError(f"unknown states in the partition: {sorted(outside, key=repr)}.")
        uncovered = set(self.states) - set(owner)
        if uncovered:
            raise ValueError(f"the partition does not cover the states {sorted(uncovered, key=repr)}.")
        transitions: dict[frozenset, dict[str, frozenset]] = {}
        for block in blocks:
            ordered = sorted(block, key=repr)
            if len({q in self.final_states for q in block}) > 1:
                raise ValueError(f"class {sorted(block, key=repr)} mixes final and non-final states.")
            out: dict[str, frozenset] = {}
            for symbol in sorted(self.input_symbols):
                cells = [self.transitions.get(q, {}).get(symbol) for q in ordered]
                if len({c is None for c in cells}) > 1:
                    raise ValueError(
                        f"class {ordered}: its states do not all have a transition on {symbol!r}."
                    )
                if cells[0] is None:
                    continue
                reached = {owner[self._step(self, q, symbol)] for q in ordered}
                if len(reached) > 1:
                    raise ValueError(
                        f"class {ordered}: on {symbol!r} its states lead to different classes."
                    )
                out[symbol] = next(iter(reached))
            transitions[block] = out
        quotient = type(self)(
            states=frozenset(blocks),
            input_symbols=self.input_symbols,
            transitions=transitions,
            initial_state=owner[self.initial_state],
            final_states=frozenset(b for b in blocks if b <= self.final_states),
            allow_partial=True,
        )
        return AutomatonMorphism(self, quotient, owner, strict=True)

    def minimization_morphism(self) -> AutomatonMorphism:
        """Return the projection of the completed accessible automaton onto its minimal DFA.

        The source is ``self`` completed (function 21, implicit sink) and
        restricted to its accessible states; the classes merged are the
        Myhill-Nerode classes of function 30, so the target is the minimal
        complete DFA (same as ``minimize`` of the source, function 32).  The
        morphism is strict and surjective, and not injective unless the source
        is already minimal.
        """
        full = self.complete()
        accessible = full.induced_subautomaton(full.accessible_states())
        return accessible.quotient_by(accessible.myhill_nerode_classes(accessible_only=True))

    def transition_monoid(self, max_elements: int = 100_000):
        """Return the transition monoid of this DFA (function 62).

        Transformations ``f_w(q) = delta*(q, w)`` of the accessible states of the
        completed DFA, closed under composition.  It depends on the automaton;
        see :meth:`syntactic_monoid` for the monoid of the language.

        Raises
        ------
        RuntimeError
            If there are more than ``max_elements`` transformations.
        """
        from .monoid import TransitionMonoid

        return TransitionMonoid.from_dfa(self, max_elements)

    def syntactic_monoid(self, max_elements: int = 100_000):
        """Return the syntactic monoid of the language (function 62).

        Quotient of the transition monoid by the syntactic congruence
        ``u ~ v  iff  xuy in L <=> xvy in L``, computed on the Myhill-Nerode
        classes (function 59) without enumerating ``Sigma*``.
        """
        from .monoid import SyntacticMonoid

        return SyntacticMonoid.from_dfa(self, max_elements)

    def nerode_partition(self):
        """Return the Myhill-Nerode partition of the language (function 59).

        The partition is made of classes of **words** (not of states): two
        words are in the same class iff no suffix separates them.  The classes
        of states come from :meth:`myhill_nerode_classes` (function 30) on the
        completed automaton (an implicit sink is added when transitions are
        missing, see :meth:`complete`).

        Returns
        -------
        MyhillNerodePartition
            Immutable object with the classes, their shortest representative
            words, ``class_of_word``, ``separating_suffix`` and the quotient
            automaton.  This DFA is not modified.
        """
        from .nerode import MyhillNerodePartition

        return MyhillNerodePartition.from_dfa(self)

    def residual_automaton(self, complete: bool = True) -> ExtendedDFA:
        """Build the residual automaton of the language (function 60, DFA version).

        The residuals ``u⁻¹L`` are the Nerode classes: reading ``u`` from the
        initial state gives a state whose language is ``u⁻¹L``, and two words
        have the same residual iff their states are equivalent.  The
        automaton of the distinct residuals is therefore the quotient by the
        partition of :meth:`nerode_partition` (the minimal DFA, function 32).

        Parameters
        ----------
        complete : bool
            ``True`` (default): complete DFA, with the residual ``∅`` when it
            occurs.  ``False``: partial DFA without the ``∅`` residual.

        Returns
        -------
        ExtendedDFA
            State names are ``frozenset`` of the original states of the class
            (the sink added by completion may appear in one of them).
        """
        return self.nerode_partition().quotient_automaton(complete)

    def residual_by_word(self, word: str) -> ExtendedDFA:
        """Return a DFA for the residual ``u⁻¹L`` (``u`` is ``word``).

        The state reached by ``u`` becomes the initial state; only the states
        reachable from it are kept.  If ``u`` leaves a partial automaton, the
        residual is empty: a single non-final initial state is returned.

        Raises
        ------
        ValueError
            If a symbol of ``word`` is not in the alphabet.
        """
        state: State | None = self.initial_state
        for symbol in word:
            if symbol not in self.input_symbols:
                raise ValueError(f"{symbol!r} is not a symbol of the alphabet.")
            state = self._step(self, state, symbol)
            if state is None:
                break
        if state is None:
            kept = frozenset({self.initial_state})
            return type(self)(
                states=kept,
                input_symbols=self.input_symbols,
                transitions={self.initial_state: {}},
                initial_state=self.initial_state,
                final_states=set(),
                allow_partial=True,
            )
        kept = self.reachable_from(state)
        return type(self)(
            states=kept,
            input_symbols=self.input_symbols,
            transitions={q: dict(self.transitions.get(q, {})) for q in kept},
            initial_state=state,
            final_states=self.final_states & kept,
            allow_partial=True,
        )

    def is_minimal(self) -> bool:
        """Test whether the DFA is minimal.

        Convention of the project (consistent with :meth:`minimize`): the
        automaton must be minimal **among the automata of its own kind**.

        * every state is accessible;
        * no two distinct states are equivalent, i.e. all the
          Myhill-Nerode classes (:meth:`myhill_nerode_classes`) are
          singletons (see :meth:`distinguishing_word`);
        * if the DFA is **incomplete**, no state other than the initial one
          may be dead (unable to reach a final state): a missing transition
          already plays the role of the sink, so such a state is useless.
          A *complete* DFA needs its sink, which is then not counted as
          superfluous.

        Minimality is more than "no equivalent states": an inaccessible
        state can be distinguishable from all others and still be removable.

        Returns
        -------
        bool
            ``True`` if no smaller equivalent DFA of the same kind exists.

        Notes
        -----
        DFA only; an NFA is never tested here (it has no unique minimal
        automaton).

        Complexity
        ----------
        O(|Q|² * |Σ|).
        """
        if self.accessible_states() != self.states:
            return False
        if len(self.myhill_nerode_classes()) != len(self.states):
            return False
        if self.is_complete():
            return True
        useful = self.coaccessible_states()
        return all(
            state == self.initial_state or state in useful for state in self.states
        )

    def minimize(self) -> ExtendedDFA:
        """Build the minimal DFA recognising the same language.

        Algorithm: Moore, through :meth:`myhill_nerode_classes` (so it is
        the same relation as functions 29 and 30; no code is duplicated).
        Moore's refinement is O(|Q|²·|Σ|) but short and close to the
        definition, which fits this project better than Hopcroft's
        O(|Σ|·|Q|·log|Q|) with its splitter bookkeeping.

        Steps
        -----
        1. Keep only accessible states (``accessible_only=True``).
        2. Each equivalence class becomes one state.  Its name is the
           ``frozenset`` of the original states it groups; classes are
           disjoint, so **names never collide**.
        3. Initial state: the class of the initial state.  Final states: the
           classes made of final states (a class is never mixed).
        4. Transition ``class --a--> class'`` taken from any representative
           (equivalent states give the same classes).

        Incomplete DFA (convention of automata-lib's ``allow_partial``): the
        result has the same kind as the input.  A complete DFA gives a
        complete minimal DFA (its sink class stays).  An incomplete DFA gives
        a partial minimal DFA: the class of the dead states is removed (the
        missing transitions replace it), except when the language is empty,
        where the single initial class stays.

        The original automaton is not modified.

        Returns
        -------
        ExtendedDFA
            A minimal DFA (:meth:`is_minimal` is ``True``) with the same
            language.

        Notes
        -----
        DFA only.  An NFA must be determinized first; this project has no
        determinization yet, so nothing is minimized "directly" on an NFA.

        Complexity
        ----------
        O(|Q|² * |Σ|).
        """
        complete = self.is_complete()
        classes = self.myhill_nerode_classes(accessible_only=True)
        if not complete:
            useful = self.coaccessible_states()
            classes = frozenset(
                cls
                for cls in classes
                if self.initial_state in cls or not cls.isdisjoint(useful)
            )
        owner = {state: cls for cls in classes for state in cls}
        transitions = {}
        for cls in classes:
            representative = next(iter(cls))
            transitions[cls] = {
                symbol: owner[target]
                for symbol, target in self.transitions.get(representative, {}).items()
                if target in owner
            }
        return type(self)(
            states=classes,
            input_symbols=self.input_symbols,
            transitions=transitions,
            initial_state=owner[self.initial_state],
            final_states={cls for cls in classes if cls <= self.final_states},
            allow_partial=not complete,
        )
