"""Conversions between kinds of finite automata (subsets, reversal)."""

from __future__ import annotations

from collections import deque
from typing import Any

from ...regex.regex import EMPTY, EPSILON, Regex, Symbol, concat, star, union
from .analyse import State


class ConversionMixin:
    """Constructions that build a new automaton from an existing one.

    Like :class:`AnalyseMixin`, it only relies on the layout of automata-lib
    (``states``, ``input_symbols``, ``transitions``, ``initial_state``,
    ``final_states``) and on the hooks ``_targets``, ``_epsilon_successors``
    and ``epsilon_closure_of`` of the analysis mixin.  The concrete classes
    are imported inside the methods: ``dfa.py`` and ``nfa.py`` import this
    module, so a top-level import would be circular.
    """

    def _fresh_state(self) -> int:
        """Return a state name that is not used: the largest unused negative int.

        This is the convention of automata-lib's ``to_complete`` for its
        trap state, which keeps the project consistent.
        """
        candidate = -1
        while candidate in self.states:
            candidate -= 1
        return candidate

    def _targets_of(self, state: State, symbol: str) -> list[State]:
        """Return the destinations of ``state`` on ``symbol`` (possibly none)."""
        value = self.transitions.get(state, {}).get(symbol)
        return [] if value is None else self._targets(value)

    def determinize(self, complete: bool = False):
        """Build an equivalent DFA by the subset construction.

        Native method: ``DFA.from_nfa(nfa, retain_names, minify)`` exists in
        automata-lib 9.2.0, but it minimises by default and chooses its own
        names, so it is not the plain construction asked here.  It is only
        used as an independent check in the tests.

        Algorithm
        ---------
        * Initial state: ``ε-closure({q0})``.
        * From a subset ``S`` and a symbol ``a``: ``ε-closure`` of the union
          of the destinations of ``a`` from every state of ``S``.
        * New subsets are explored breadth-first until none appears; a
          ``dict`` of discovered subsets avoids creating one twice.
        * A subset is final if it contains a final state of the NFA.

        For an NFA without epsilon transitions the closures are singletons,
        so the same code is the classic construction.

        Conventions
        -----------
        * Names of the DFA states are ``frozenset`` of NFA states (unique by
          construction).
        * The empty subset (dead end) is **not** created by default: the DFA
          is partial, as ``allow_partial`` of automata-lib permits.  With
          ``complete=True`` the empty subset is kept as an explicit sink and
          the DFA is complete.
        * Automata-lib allows one initial state only, so only one start
          subset exists.  Only symbols of ``input_symbols`` are used (not the
          epsilon key).
        * Applied to a DFA, it returns an equivalent DFA whose states are
          singleton subsets.
        * The original automaton is not modified.

        Parameters
        ----------
        complete : bool, default False
            Add the empty subset so that every transition exists.

        Returns
        -------
        ExtendedDFA
            A deterministic automaton recognising the same language.

        Complexity
        ----------
        O(2^|Q| * |Σ|) in the worst case (the subsets may all appear).
        """
        from ..dfa import ExtendedDFA

        alphabet = sorted(
            (symbol for symbol in self.input_symbols if symbol != self._EPSILON),
            key=repr,
        )
        start = self.epsilon_closure_of([self.initial_state])
        transitions: dict[frozenset, dict[str, frozenset]] = {}
        queue: deque[frozenset] = deque([start])
        transitions[start] = {}
        while queue:
            subset = queue.popleft()
            for symbol in alphabet:
                moved = {
                    target
                    for state in subset
                    for target in self._targets_of(state, symbol)
                }
                target_subset = self.epsilon_closure_of(moved)
                if not target_subset and not complete:
                    continue
                transitions[subset][symbol] = target_subset
                if target_subset not in transitions:
                    transitions[target_subset] = {}
                    queue.append(target_subset)
        return ExtendedDFA(
            states=frozenset(transitions),
            input_symbols=frozenset(alphabet),
            transitions=transitions,
            initial_state=start,
            final_states={s for s in transitions if s & self.final_states},
            allow_partial=not complete,
        )

    def reversed_automaton(self):
        """Build an NFA recognising the reversed language ``L^R``.

        Native method: ``NFA.reverse()`` exists in automata-lib 9.2.0 (and is
        not hidden by this one, which has another name).

        Construction
        ------------
        1. Every transition ``p --a--> q`` becomes ``q --a--> p``,
           **epsilon transitions included** (``a`` may be ``""``).
        2. The old initial state becomes the only final state.
        3. The old final states must become initial, but automata-lib
           accepts **one** initial state.  If there is exactly one old final
           state, it is the new initial state.  Otherwise a fresh state
           (:meth:`_fresh_state`, a negative integer never used) is the
           initial state, with an epsilon transition to each old final state
           (none if there is no final state: the language stays empty).

        A path ``q0 -> f`` reading ``w`` gives a path ``f -> q0`` reading
        ``w^R``, hence ``L(result) = L^R``.  The empty word is kept: it is
        accepted iff the initial state was final.

        The result is an NFA even for a DFA input: it is **not** forced into
        a DFA (use :meth:`determinize` for that).  The original automaton is
        not modified.

        Returns
        -------
        ExtendedNFA
            The reversed automaton.

        Complexity
        ----------
        O(|Q| + |E|), where E is the set of transitions.
        """
        from ..nfa import ExtendedNFA

        reverse: dict[State, dict[str, set[State]]] = {s: {} for s in self.states}
        for source, outgoing in self.transitions.items():
            for symbol, value in outgoing.items():
                for target in self._targets(value):
                    reverse[target].setdefault(symbol, set()).add(source)
        finals = self.final_states
        states = set(self.states)
        if len(finals) == 1:
            (initial,) = finals
        else:
            initial = self._fresh_state()
            states.add(initial)
            reverse[initial] = {self._EPSILON: set(finals)} if finals else {}
        return ExtendedNFA(
            states=states,
            input_symbols=self.input_symbols,
            transitions={
                s: {a: frozenset(t) for a, t in out.items()}
                for s, out in reverse.items()
            },
            initial_state=initial,
            final_states={self.initial_state},
        )

    def remove_epsilon_transitions(self):
        """Build an NFA without epsilon transitions with the same language.

        Native method: ``NFA.eliminate_lambda()`` exists in automata-lib 9.2.0.
        It is not hidden (this method has another name) and is only used as
        an independent check in the tests.

        Construction (propagation through the closures of
        :meth:`epsilon_closure`), with ``E(q)`` the epsilon closure of ``q``:

        * states, alphabet and initial state are unchanged;
        * ``q --a--> r`` in the result iff ``r`` is a destination on ``a``
          of **some state of** ``E(q)``, i.e. ``δ'(q, a) = ∪ δ(p, a)`` for
          ``p`` in ``E(q)``;
        * ``q`` is final iff ``E(q)`` contains a final state.

        Why the language is kept: a run ``q0 ε* p1 a1 ε* p2 a2 ... ε* pn``
        of the original, with ``pn`` followed by ``ε*`` to a final state,
        becomes ``q0 --a1--> r1 --a2--> ...``: each ``ε*`` before a letter is
        absorbed by ``E`` of the current state, and the last ``ε*`` is
        absorbed by the new final states.  The reverse direction is
        symmetric.  The empty word is accepted iff ``E(q0)`` meets ``F``,
        which is exactly "``q0`` is final" in the result.

        No epsilon key remains in the result.  Applied to a DFA it gives an
        equivalent NFA whose cells are singleton sets.  The original
        automaton is not modified.

        Returns
        -------
        ExtendedNFA
            An epsilon-free NFA.

        Complexity
        ----------
        O(|Q| * (|Q| + |E|) + |Q|² * |Σ|) with E the transitions.
        """
        from ..nfa import ExtendedNFA

        alphabet = sorted(
            (symbol for symbol in self.input_symbols if symbol != self._EPSILON),
            key=repr,
        )
        transitions: dict[State, dict[str, frozenset]] = {}
        finals = set()
        for state in self.states:
            closure = self.epsilon_closure(state)
            if closure & self.final_states:
                finals.add(state)
            cells: dict[str, frozenset] = {}
            for symbol in alphabet:
                reached = {
                    target
                    for member in closure
                    for target in self._targets_of(member, symbol)
                }
                if reached:
                    cells[symbol] = frozenset(reached)
            transitions[state] = cells
        return ExtendedNFA(
            states=self.states,
            input_symbols=frozenset(alphabet),
            transitions=transitions,
            initial_state=self.initial_state,
            final_states=finals,
        )

    def is_universal(self) -> bool:
        """Decide whether ``L(A) = Σ*`` over the alphabet of the automaton.

        Algorithm (as specified: complement, then emptiness test)

        1. Get a **complete DFA**: the automaton itself if it already is a
           DFA, otherwise :meth:`determinize` with ``complete=True`` (the
           empty subset is kept as an explicit sink, so no word is lost).
        2. Take its complement (``complemented``: completion, then swap the
           final states).
        3. ``L = Σ*`` iff the complement is empty (``is_empty``).

        Not to be confused with completeness: a complete automaton may
        reject words (its sink, for instance), and an incomplete one is never
        universal as soon as a reachable state lacks a transition, because
        the missing word is rejected.  The empty word is covered: it is
        in ``Σ*``, so the initial state must be final.  With an empty
        alphabet, ``Σ* = {ε}``.

        The original automaton is not modified.

        Returns
        -------
        bool
            ``True`` iff every word over the alphabet is accepted.

        Complexity
        ----------
        Linear for a DFA; exponential in the worst case for an NFA (subset
        construction).
        """
        from ..dfa import ExtendedDFA

        if isinstance(self, ExtendedDFA):
            dfa = self
        else:
            dfa = self.determinize(complete=True)
        return dfa.complemented().is_empty()

    @staticmethod
    def _as_table(automaton: Any) -> dict:
        """Transitions of any automaton as ``{state: {symbol: frozenset}}``.

        A native ``DFA`` cell is one state; an NFA cell is a set.  The type
        of the automaton decides (not the type of the cell), because a DFA
        state may itself be a ``frozenset``.
        """
        from automata.fa.dfa import DFA

        single = isinstance(automaton, DFA)
        return {
            state: {
                symbol: frozenset([cell] if single else cell)
                for symbol, cell in outgoing.items()
            }
            for state, outgoing in automaton.transitions.items()
        }

    def language_union(self, other: Any):
        """Build an automaton recognising ``L(self) ∪ L(other)``.

        Native methods ``DFA.union`` and ``NFA.union`` exist in automata-lib;
        they are not hidden (other name).  Reasons for not simply calling
        them: ``DFA.union`` is meant for DFAs sharing their alphabet and
        minimises its result by default, while this project needs different
        alphabets, incomplete automata and unchanged structure.

        Two constructions, chosen by the operands:

        * **both are DFAs** (this automaton an ``ExtendedDFA``): the product
          of function 23, ``cartesian_product(other, lambda a, b: a or b)``.
          The result is a complete DFA over the union of the alphabets; the
          pair ``(q, None)`` stands for "the other automaton is blocked".
        * **otherwise (an NFA is involved)**: an epsilon-NFA with a new
          initial state ``("start",)`` and epsilon transitions to the two
          initial states.  The states of the operands are renamed
          ``(0, s)`` and ``(1, s)``, so names can never collide, and the
          fresh initial state is a 1-tuple, different from every renamed
          state.  The alphabet is the union of the alphabets.  Determinize
          afterwards with :meth:`determinize` if a DFA is wanted.

        A word is accepted iff one of the operands accepts it: in the
        product through the ``or`` rule, in the NFA because a run starts by
        choosing one side and then stays in it (the sides share no state).
        A symbol unknown to one operand simply blocks that side.

        Neither operand is modified.

        Parameters
        ----------
        other : DFA or NFA
            The second automaton (native or extended class).

        Returns
        -------
        ExtendedDFA or ExtendedNFA
            See above.

        Complexity
        ----------
        Product: O(|Q1| * |Q2| * |Σ|); epsilon-NFA: O(|Q1| + |Q2| + |E|).
        """
        from automata.fa.dfa import DFA

        from ..dfa import ExtendedDFA
        from ..nfa import ExtendedNFA

        if isinstance(self, ExtendedDFA) and isinstance(other, DFA):
            return self.cartesian_product(other, lambda first, second: first or second)
        start = ("start",)
        tables = (self._as_table(self), self._as_table(other))
        operands = (self, other)
        transitions: dict = {start: {self._EPSILON: frozenset()}}
        finals = set()
        for tag, (table, operand) in enumerate(zip(tables, operands)):
            for state, outgoing in table.items():
                transitions[(tag, state)] = {
                    symbol: frozenset((tag, t) for t in targets)
                    for symbol, targets in outgoing.items()
                }
            for state in operand.states:
                transitions.setdefault((tag, state), {})
            finals |= {(tag, f) for f in operand.final_states}
        transitions[start] = {
            self._EPSILON: frozenset(
                (tag, operand.initial_state) for tag, operand in enumerate(operands)
            )
        }
        alphabet = (set(self.input_symbols) | set(other.input_symbols)) - {self._EPSILON}
        return ExtendedNFA(
            states=frozenset(transitions),
            input_symbols=alphabet,
            transitions=transitions,
            initial_state=start,
            final_states=finals,
        )

    # --- trim --------------------------------------------------------------------

    def trim(self):
        """Return the trim automaton: only the useful states are kept.

        The useful states (:meth:`useful_states`) and the transitions between
        them are kept; accessible-but-dead states (sinks) and inaccessible
        states disappear, so the result is generally **partial**, as
        ``allow_partial`` of automata-lib permits.  The language is unchanged:
        an accepting run only visits useful states.

        Convention for the empty language: no state is useful, but a valid
        automaton needs an initial state, so the result is the initial state
        alone, without transition or final state.

        The result has the same class as the input; the original is not
        modified.

        Complexity
        ----------
        O(|Q| + |E|).
        """
        kept = self.useful_states() or frozenset([self.initial_state])
        return self._restrict(kept)

    # --- function 39: state elimination on a normalised epsilon-NFA ----------------

    def is_normalized(self) -> bool:
        """Test the **normalised** form required by :meth:`eliminate_states`.

        Definition used by the project (the course statement only says
        "ε-AFEND normalisé" without defining it; this is the usual form for
        the state-elimination method of Brzozowski and McCluskey):

        1. the initial state has **no incoming transition**;
        2. there is **exactly one final state**, different from the initial
           state;
        3. the final state has **no outgoing transition**.

        Epsilon transitions are allowed anywhere else.
        """
        if len(self.final_states) != 1:
            return False
        (final,) = self.final_states
        if final == self.initial_state:
            return False
        if self.transitions.get(final) and any(
            self._targets(value) for value in self.transitions[final].values()
        ):
            return False
        return not any(
            self.initial_state in self._targets(value)
            for outgoing in self.transitions.values()
            for value in outgoing.values()
        )

    def normalized(self):
        """Return an equivalent epsilon-NFA in normalised form.

        A new initial state ``("start",)`` gets an epsilon transition to the
        old initial state, and every old final state gets an epsilon
        transition to a new final state ``("accept",)``.  The old states are
        renamed ``(0, s)``, so no name collides with the two new ones.  The
        language is unchanged; the original is not modified.

        Returns
        -------
        ExtendedNFA
            A normalised automaton (:meth:`is_normalized` is true).
        """
        from ..nfa import ExtendedNFA

        start, accept = ("start",), ("accept",)
        table = {
            (0, state): {
                symbol: frozenset((0, t) for t in targets)
                for symbol, targets in outgoing.items()
            }
            for state, outgoing in self._as_table(self).items()
        }
        for state in self.states:
            table.setdefault((0, state), {})
        for final in self.final_states:
            cell = table[(0, final)]
            cell[self._EPSILON] = cell.get(self._EPSILON, frozenset()) | {accept}
        table[start] = {self._EPSILON: frozenset([(0, self.initial_state)])}
        table[accept] = {}
        return ExtendedNFA(
            states=frozenset(table),
            input_symbols={a for a in self.input_symbols if a != self._EPSILON},
            transitions=table,
            initial_state=start,
            final_states={accept},
        )

    def _eliminate(self, order=None) -> Regex:
        """State elimination on a normalised automaton (no precondition check).

        The edges carry :class:`Regex` objects of the project (parallel edges
        are united, an epsilon edge is ``ε``, a missing edge is ``∅``).  To
        eliminate a state ``s``, for every predecessor ``p`` and successor
        ``q`` (both different from ``s``)::

            R(p, q)  :=  R(p, q) | R(p, s) R(s, s)* R(s, q)

        If ``order`` is ``None``, the next state is chosen greedily: the one
        with the smallest product (incoming edges) x (outgoing edges), which
        keeps the expressions small (ties are broken by ``repr``).
        """
        (final,) = self.final_states
        initial = self.initial_state
        remaining = [s for s in self.states if s not in (initial, final)]
        remaining.sort(key=repr)
        if order is not None:
            order = list(order)
            if sorted(order, key=repr) != remaining:
                raise ValueError(
                    "order must contain each intermediate state exactly once."
                )
        edge: dict[tuple, Regex] = {}
        for source, outgoing in self.transitions.items():
            for symbol, value in outgoing.items():
                label = EPSILON if symbol == self._EPSILON else Symbol(symbol)
                for target in self._targets(value):
                    key = (source, target)
                    edge[key] = union(edge.get(key, EMPTY), label)
        todo = list(order) if order is not None else remaining

        def cost(state):
            ins = sum(1 for (p, q) in edge if q == state and p != state)
            outs = sum(1 for (p, q) in edge if p == state and q != state)
            return (ins * outs, repr(state))

        while todo:
            state = todo.pop(0) if order is not None else min(todo, key=cost)
            if order is None:
                todo.remove(state)
            loop = star(edge.pop((state, state), EMPTY))
            incoming = {p: r for (p, q), r in edge.items() if q == state}
            outgoing = {q: r for (p, q), r in edge.items() if p == state}
            for p in incoming:
                del edge[(p, state)]
            for q in outgoing:
                del edge[(state, q)]
            for p, r_in in incoming.items():
                for q, r_out in outgoing.items():
                    edge[(p, q)] = union(
                        edge.get((p, q), EMPTY), concat(r_in, loop, r_out)
                    )
        return edge.get((initial, final), EMPTY)

    def eliminate_states(self, order=None):
        """Eliminate the intermediate states and return the language's regex as text.

        **Precondition:** the automaton is normalised (:meth:`is_normalized`;
        :meth:`normalized` builds one), otherwise ``ValueError``.  The
        initial and the final state are never eliminated; with no incoming
        edge on the first and no outgoing edge on the last, the edge between
        them at the end is a regular expression of the whole language (see
        :meth:`_eliminate` for the formula and the default order).

        The expression is built with the project's regex algebra
        (:mod:`automata_extensions.regex`, normal form included), and
        returned as text in its syntax (``ε``, ``∅``, ``|``, ``*``).

        Parameters
        ----------
        order : iterable of State, optional
            Order of elimination; it must list each intermediate state
            exactly once.  The expression depends on the order, the language
            does not.

        Returns
        -------
        str or None
            The expression, or ``None`` if the language is empty.

        Raises
        ------
        ValueError
            If the automaton is not normalised, if ``order`` is not a
            permutation of the intermediate states, or if a symbol has more
            than one character (not representable in the regex syntax).
        """
        if not self.is_normalized():
            raise ValueError(
                "The automaton is not normalised: it needs one initial state "
                "without incoming transition and one distinct final state "
                "without outgoing transition (see normalized())."
            )
        result = self._eliminate(order)
        return None if result == EMPTY else str(result)

    # --- function 52: NFA -> DFA (subset construction) -------------------------------

    def to_dfa(self, complete: bool = False):
        """NFA to DFA by the subset construction (function 52).

        The construction is the one of :meth:`determinize` (function 35),
        which implements exactly the algorithm asked here (initial subset =
        epsilon closure of the initial state, subsets built from the
        destinations then closed by epsilon, a subset is final if it contains
        a final state, exploration until nothing new appears).  It is **not**
        written a second time: this method is the name given by the course
        statement and calls it.  Automata-lib has a single initial state, so
        the initial subset is the closure of that state.

        Parameters
        ----------
        complete : bool, default False
            Keep the empty subset as an explicit sink (complete DFA); by
            default a transition to the empty subset is simply absent
            (partial DFA).

        Returns
        -------
        ExtendedDFA
            A DFA with the same language; the original is not modified.
        """
        return self.determinize(complete=complete)

    # --- functions 53 and 54: DFA -> regular expression -------------------------------

    @staticmethod
    def _symbol_regex(symbol: str) -> Regex:
        if len(symbol) != 1:
            raise ValueError(
                f"The symbol {symbol!r} has several characters: the regex syntax "
                "only supports one-character symbols."
            )
        return Symbol(symbol)

    def to_regex_elimination(self, order=None) -> Regex:
        """Regular expression of the language by state elimination (function 53).

        Steps (those of the course statement):

        1. :meth:`trim` removes the useless states (language unchanged);
        2. :meth:`normalized` adds a new initial state with an epsilon edge to
           the old one and a new final state with an epsilon edge from every
           old final state;
        3. the old states are eliminated one by one with
           ``R'(i, j) = R(i, j) | R(i, k) R(k, k)* R(k, j)`` (a missing edge
           is ``∅``; loops are the stars ``R(k, k)*``);
        4. the edge from the new initial to the new final state is the result.

        Parameters
        ----------
        order : iterable of State, optional
            Order of elimination of the states of the **normalised** trimmed
            automaton (default: greedy, smallest in x out degree first).

        Returns
        -------
        Regex
            An expression in the normal form of the project; ``∅`` for the
            empty language.

        Raises
        ------
        ValueError
            For a symbol longer than one character.

        Notes
        -----
        Works for a DFA and also for an NFA (epsilon edges are handled); the
        automaton is never modified.  Size may grow exponentially in the
        worst case, as for every state-elimination method.
        """
        for symbol in self.input_symbols:
            if symbol != self._EPSILON:
                self._symbol_regex(symbol)
        return self.trim().normalized()._eliminate(order)

    def to_equation_system(self, form: str = "AX"):
        """Return the system of rational equations associated with the automaton (function 58).

        ``form="AX"``: ``X_q = ⋃ a·X_r ∪ (ε if q final)`` (``X0`` = initial
        state); ``form="XA"``: ``X_q = ⋃ X_p·a ∪ (ε if q initial)``.  The result
        is an ``AutomatonSystem``; see :class:`...regex.equations.EquationSystem`.
        """
        from ...regex.equations import EquationSystem

        return EquationSystem.from_automaton(self, form)

    def to_regex_arden(self) -> Regex:
        """Regular expression of the language by Arden's lemma (function 54).

        System of equations (``X_q`` = words that lead from ``q`` to a final
        state, so each letter is written **before** the variable)::

            X_q = ∪_a  a · X_δ(q,a)  ∪  (ε if q is final)

        i.e. ``X_i = ∪_j C_ij X_j ∪ B_i`` with ``C_ij`` the union of the
        symbols going from ``i`` to ``j`` and ``B_i`` equal to ``ε`` or
        ``∅``.  The answer is ``X_initial``.

        The system is built and solved by :class:`~...regex.equations.EquationSystem`
        (function 58); no graph elimination, only equations are rewritten.
        For each variable ``X_k`` other than the initial one:

        1. isolate it with **Arden's lemma**: ``X = A X ∪ B`` with
           ``ε ∉ L(A)`` has the solution ``X = A* B``; here
           ``X_k = C_kk X_k ∪ B'_k`` gives ``X_k = C_kk* B'_k``, where ``B'_k``
           gathers the other terms ``C_kj X_j`` and ``B_k``;
        2. substitute this value in every other equation;

        and finally apply the lemma to the equation of the initial state.

        Condition of the lemma: ``ε ∉ L(C_kk)`` is **checked** at every
        application (``ValueError`` otherwise).  It always holds for a DFA:
        the coefficients start as unions of symbols and substitutions only
        produce products containing such a coefficient.  An automaton with
        epsilon edges (NFA) is refused for this reason.

        The system is built on the accessible states only (the others do not
        affect ``X_initial``).  Final states: ``B_q = ε``.  States without
        transition: all ``C_qj`` are ``∅`` (``X_q = B_q``).

        Returns
        -------
        Regex
            The expression (project normal form); ``∅`` for the empty
            language.

        Raises
        ------
        ValueError
            For an automaton with epsilon edges, a multi-character symbol, or
            if a lemma condition fails.
        """
        from automata.fa.dfa import DFA

        if not isinstance(self, DFA):
            raise ValueError(
                "Arden's method is implemented for DFA (epsilon edges break the "
                "condition of the lemma): determinize first (to_dfa())."
            )
        from ...regex.equations import EquationSystem

        accessible = self.accessible_states()
        system = EquationSystem.from_automaton(self, "AX", states=accessible)
        others = sorted((q for q in accessible if q != self.initial_state), key=repr)
        order = [system.names[q] for q in others] + [system.names[self.initial_state]]
        return system.solve(order=order)


    # --- functions 43, 44, 45: intersection, difference, concatenation ---------------
    #
    # Native methods ``DFA.intersection``, ``DFA.difference`` and
    # ``NFA.concatenate`` exist in automata-lib 9.2.0.  They are not hidden
    # (names ``language_*`` are used, like ``language_union``).  They are not
    # called because the project needs different alphabets, incomplete
    # automata, unchanged state names and NFA/DFA mixtures.

    @staticmethod
    def _lift(automaton: Any):
        """Return an ``ExtendedDFA``/``ExtendedNFA`` copy of a native automaton.

        An automaton that already belongs to the project is returned as is.
        """
        from automata.fa.dfa import DFA

        from ..dfa import ExtendedDFA
        from ..nfa import ExtendedNFA

        if isinstance(automaton, (ExtendedDFA, ExtendedNFA)):
            return automaton
        if isinstance(automaton, DFA):
            return ExtendedDFA(
                states=automaton.states,
                input_symbols=automaton.input_symbols,
                transitions={s: dict(t) for s, t in automaton.transitions.items()},
                initial_state=automaton.initial_state,
                final_states=automaton.final_states,
                allow_partial=True,
            )
        return ExtendedNFA(
            states=automaton.states,
            input_symbols=automaton.input_symbols,
            transitions={
                s: {a: frozenset(t) for a, t in out.items()}
                for s, out in automaton.transitions.items()
            },
            initial_state=automaton.initial_state,
            final_states=automaton.final_states,
        )

    def language_intersection(self, other: Any):
        """Build an automaton recognising ``L(self) ∩ L(other)``.

        Product construction (function 23): the states are pairs
        ``(p, q)``, built lazily from ``(initial, initial)`` (only the
        accessible pairs are created), and a pair is final iff **both**
        components are final.

        * **Two DFAs:** ``cartesian_product`` with the rule ``a and b``.  The
          result is a complete DFA over the **union** of the alphabets.  When
          the alphabets differ, a symbol unknown to one automaton blocks it
          (``None`` component), so any word containing it is rejected, which
          is what intersection requires.
        * **An NFA is involved:** both operands are first made epsilon-free
          (:meth:`remove_epsilon_transitions`, function 37); then
          ``(p, q) --a--> (p', q')`` iff ``p --a--> p'`` and ``q --a--> q'``,
          for the symbols of both alphabets.  The result is an epsilon-free
          NFA over the union of the alphabets (no determinization needed).

        Neither operand is modified.

        Returns
        -------
        ExtendedDFA or ExtendedNFA

        Complexity
        ----------
        O(|Q1| * |Q2| * |Σ|) in the worst case.
        """
        from ..dfa import ExtendedDFA
        from ..nfa import ExtendedNFA

        other = self._lift(other)
        if isinstance(self, ExtendedDFA) and isinstance(other, ExtendedDFA):
            return self.cartesian_product(other, lambda first, second: first and second)
        first, second = self.remove_epsilon_transitions(), other.remove_epsilon_transitions()
        common = sorted(first.input_symbols & second.input_symbols, key=repr)
        start = (first.initial_state, second.initial_state)
        transitions: dict = {start: {}}
        queue = deque([start])
        while queue:
            pair = queue.popleft()
            for symbol in common:
                targets = frozenset(
                    (x, y)
                    for x in first.transitions.get(pair[0], {}).get(symbol, ())
                    for y in second.transitions.get(pair[1], {}).get(symbol, ())
                )
                if not targets:
                    continue
                transitions[pair][symbol] = targets
                for target in targets:
                    if target not in transitions:
                        transitions[target] = {}
                        queue.append(target)
        return ExtendedNFA(
            states=frozenset(transitions),
            input_symbols=first.input_symbols | second.input_symbols,
            transitions=transitions,
            initial_state=start,
            final_states={
                (p, q)
                for (p, q) in transitions
                if p in first.final_states and q in second.final_states
            },
        )

    def language_difference(self, other: Any):
        """Build an automaton recognising ``L(self) \\ L(other)``.

        The difference is ``L(A) ∩ complement(L(B))``.  Instead of building
        the complement explicitly (function 22) and then an intersection, the
        product of function 23 is used with the final-pair rule
        ``a and not b``: it computes exactly the same language in one pass,
        and it needs no completion of ``B`` because a blocked ``B`` (missing
        transition, or symbol outside its alphabet) **rejects**, so the word
        is in the difference iff ``A`` accepts it.  This handles different
        alphabets and incomplete automata correctly (a complement taken over
        the alphabet of ``B`` alone would miss the words using symbols of
        ``A`` only).

        An NFA operand is first determinized (:meth:`determinize`, function
        35), because the complement of an NFA needs a deterministic
        automaton.  The result is always a complete DFA over the union of the
        alphabets.  Neither operand is modified.

        Returns
        -------
        ExtendedDFA

        Complexity
        ----------
        O(|Q1| * |Q2| * |Σ|) for DFAs; exponential in the worst case if a
        determinization is needed.
        """
        from ..dfa import ExtendedDFA

        other = self._lift(other)
        left = self if isinstance(self, ExtendedDFA) else self.determinize()
        right = other if isinstance(other, ExtendedDFA) else other.determinize()
        return left.cartesian_product(right, lambda first, second: first and not second)

    def language_concatenation(self, other: Any):
        """Build an epsilon-NFA recognising ``{uv | u ∈ L(self), v ∈ L(other)}``.

        Constraints of automata-lib 9.2.0: **one** initial state, and epsilon
        transitions written with the key ``""``.  A new initial state is not
        needed here: the initial state of the first automaton is the
        initial state of the result.

        Construction
        ------------
        * states of the first automaton become ``(0, s)``, those of the
          second ``(1, s)`` (names never collide, even if both automata use
          the same names);
        * all transitions are kept (epsilon transitions of the operands
          too);
        * for every final state ``f`` of the first automaton, an **epsilon
          transition** ``(0, f) --ε--> (1, initial of the second)``;
        * the final states are the final states of the **second** automaton
          only, ``(1, f)``.

        A word is accepted iff a run reads ``u`` in the first part up to a
        final state, jumps by epsilon, then reads ``v`` up to a final state
        of the second part.  The empty word is accepted iff both operands
        accept it.  If one language is empty the result is empty (no final
        state is reachable).  The alphabet is the union of the alphabets
        (each part only reads its own symbols).  Neither operand is
        modified.

        Returns
        -------
        ExtendedNFA

        Complexity
        ----------
        O(|Q1| + |Q2| + |E1| + |E2| + |F1|).
        """
        from ..nfa import ExtendedNFA

        tables = (self._as_table(self), self._as_table(other))
        operands = (self, other)
        transitions: dict = {}
        for tag, (table, operand) in enumerate(zip(tables, operands)):
            for state in operand.states:
                transitions[(tag, state)] = {
                    symbol: frozenset((tag, t) for t in targets)
                    for symbol, targets in table.get(state, {}).items()
                }
        jump = (1, other.initial_state)
        for final in self.final_states:
            cell = transitions[(0, final)]
            cell[self._EPSILON] = cell.get(self._EPSILON, frozenset()) | {jump}
        alphabet = (set(self.input_symbols) | set(other.input_symbols)) - {self._EPSILON}
        return ExtendedNFA(
            states=frozenset(transitions),
            input_symbols=alphabet,
            transitions=transitions,
            initial_state=(0, self.initial_state),
            final_states={(1, f) for f in other.final_states},
        )

    # --- functions 46, 47, 48: Kleene star, left and right quotients --------------------
    #
    # Native: ``NFA.kleene_star()`` and ``DFA.right_quotient(other)`` exist in
    # automata-lib 9.2.0 (to my knowledge; the left quotient has no native
    # method).  They are not hidden: the names below start with ``language_``.

    def language_star(self):
        """Build an epsilon-NFA recognising the Kleene star ``L(self)*``.

        ``L* = ∪_{n≥0} L^n``: all concatenations of zero or more words of L,
        so the empty word is always accepted.

        Construction
        ------------
        * the states of the automaton become ``(0, s)``; a **new initial
          state** ``("start",)`` is added (a 1-tuple, never equal to a renamed
          state, so no collision is possible);
        * ``("start",)`` is **final** and has an epsilon transition to the old
          initial state;
        * every old final state keeps its status and gets an epsilon
          transition back to the old initial state;
        * the alphabet and the old transitions (epsilon ones included) are
          unchanged.

        Correctness.  (⊆) A word ``w1 ... wn`` with every ``wi`` in L is read
        by: start --ε--> initial, accepting run of ``w1``, ε back to the
        initial state, run of ``w2``, ..., run of ``wn``, ending in a final
        state; for ``n = 0`` the new initial state is final.  (⊇) The only
        way back to the initial state is an epsilon transition leaving a
        final state, so a run cuts into pieces that each go from the initial
        state to a final state, i.e. into words of L.  The new initial state
        (instead of marking the old one final) is needed because the old
        initial state may have incoming transitions: making it final would
        accept words of the form ``u v`` with ``u`` read from a loop that is
        not in L.

        The star of the empty language is ``{ε}``.  The original automaton is
        not modified.

        Returns
        -------
        ExtendedNFA

        Complexity
        ----------
        O(|Q| + |E| + |F|).
        """
        from ..nfa import ExtendedNFA

        start = ("start",)
        table = self._as_table(self)
        transitions: dict = {
            (0, state): {
                symbol: frozenset((0, t) for t in targets)
                for symbol, targets in table.get(state, {}).items()
            }
            for state in self.states
        }
        back = (0, self.initial_state)
        for final in self.final_states:
            cell = transitions[(0, final)]
            cell[self._EPSILON] = cell.get(self._EPSILON, frozenset()) | {back}
        transitions[start] = {self._EPSILON: frozenset([back])}
        return ExtendedNFA(
            states=frozenset(transitions),
            input_symbols={a for a in self.input_symbols if a != self._EPSILON},
            transitions=transitions,
            initial_state=start,
            final_states={(0, f) for f in self.final_states} | {start},
        )

    def language_left_quotient(self, other: Any):
        """Left quotient ``{ w | ∃ u ∈ L(self), u w ∈ L(other) }``.

        **Order of arguments** (the course statement gives no signature):
        ``first.language_left_quotient(second)`` computes the words ``w``
        such that some ``u`` of the language of ``first`` (``self``) satisfies
        ``u w`` in the language of ``second`` (``other``).  In the usual
        notation this is ``L1⁻¹ L2`` with ``L1 = L(self)``, ``L2 = L(other)``.
        Not to be confused with :meth:`language_right_quotient`, where
        ``self`` is the dividend.

        Construction
        ------------
        1. Make both automata epsilon-free (function 37; the language read
           from each state is preserved).
        2. Explore, breadth-first and lazily, the pairs ``(p, q)`` reachable
           from ``(initial of self, initial of other)`` by symbols known to
           both automata (``p`` is a state of ``self``, ``q`` of ``other``).
        3. Let ``S`` be the set of states ``q`` of ``other`` met in a pair
           ``(p, q)`` with ``p`` final: ``S`` is the set of states of
           ``other`` reachable after reading a word ``u`` of ``L(self)``.
        4. The result is ``other`` where every state of ``S`` may be the
           start.  automata-lib allows **one** initial state, so a new
           initial state ``("start",)`` gets an epsilon transition to each
           state of ``S`` (the states of ``other`` are renamed ``(0, q)``).

        Why it is right: ``w`` is in the quotient iff there are ``u`` in
        ``L(self)`` and an accepting run of ``other`` on ``u w``; the run
        reaches some ``q`` in ``S`` after ``u`` and then accepts ``w``; the
        converse is the same argument backwards.

        Special cases: if ``L(self)`` is empty, ``S`` is empty and the result
        is empty; the empty word is in the result iff ``S`` contains a final
        state of ``other``.  The alphabet of the result is that of
        ``other`` (a word ``w`` with a foreign symbol cannot be in ``L2``).
        Neither operand is modified.

        Returns
        -------
        ExtendedNFA

        Complexity
        ----------
        O(|Q1| * |Q2| * |Σ|) for the exploration.
        """
        from ..nfa import ExtendedNFA

        first = self._lift(self).remove_epsilon_transitions()
        second = self._lift(other).remove_epsilon_transitions()
        common = sorted(first.input_symbols & second.input_symbols, key=repr)
        start_pair = (first.initial_state, second.initial_state)
        seen = {start_pair}
        queue = deque([start_pair])
        reached = set()
        while queue:
            p, q = queue.popleft()
            if p in first.final_states:
                reached.add(q)
            for symbol in common:
                for p2 in first.transitions.get(p, {}).get(symbol, ()):
                    for q2 in second.transitions.get(q, {}).get(symbol, ()):
                        if (p2, q2) not in seen:
                            seen.add((p2, q2))
                            queue.append((p2, q2))
        start = ("start",)
        transitions: dict = {
            (0, q): {s: frozenset((0, t) for t in ts) for s, ts in out.items()}
            for q, out in second.transitions.items()
        }
        transitions[start] = {self._EPSILON: frozenset((0, q) for q in reached)}
        return ExtendedNFA(
            states=frozenset(transitions),
            input_symbols=second.input_symbols,
            transitions=transitions,
            initial_state=start,
            final_states={(0, f) for f in second.final_states},
        )

    def language_right_quotient(self, other: Any):
        """Right quotient ``{ w | ∃ u ∈ L(other), w u ∈ L(self) }``.

        **Order of arguments:** ``first.language_right_quotient(second)``
        takes the words ``w`` such that ``w u`` is in the language of
        ``first`` (``self``, the dividend) for some ``u`` in the language of
        ``second`` (``other``): ``L1 / L2`` with ``L1 = L(self)``,
        ``L2 = L(other)``.  Not to be confused with
        :meth:`language_left_quotient`.

        Construction (only the final states change)
        ------------------------------------------
        Keep the states, the initial state and the transitions of ``self``
        (made epsilon-free when it is an NFA).  A state ``p`` becomes final
        iff **some word of L2 leads from p to a final state of self**, i.e.
        iff, in the product of ``self`` with ``other`` (pairs explored
        lazily, common symbols only), a pair ``(f, g)`` with ``f`` final in
        ``self`` and ``g`` final in ``other`` is reachable from
        ``(p, initial of other)``.  This is computed for all ``p`` at once by
        a backward search from the final pairs.

        Why it is right: ``w`` leads from the initial state to a state ``p``;
        ``w`` is in the quotient iff some ``u`` of ``L2`` leads from ``p`` to
        a final state, which is exactly the new finality of ``p``.  The
        result is built on the dividend, so it is a **DFA** if ``self`` is a
        DFA (the transitions are unchanged, hence deterministic) and an
        epsilon-free NFA otherwise.  The alphabet is that of ``self``.

        Special cases: an empty ``L2`` or ``L1`` gives the empty language;
        if ``ε`` is in ``L2`` then ``L1`` is included in the quotient.
        Neither operand is modified.  The theoretical dual (reverse both,
        take the left quotient, reverse) gives the same language; the tests
        check it with functions 36 and 47.

        Returns
        -------
        ExtendedDFA or ExtendedNFA

        Complexity
        ----------
        O(|Q1| * |Q2| * |Σ|).
        """
        from ..dfa import ExtendedDFA

        dfa_case = isinstance(self, ExtendedDFA)
        base = self if dfa_case else self.remove_epsilon_transitions()
        table = self._as_table(base)
        second = self._lift(other).remove_epsilon_transitions()
        common = sorted(set(base.input_symbols) & second.input_symbols, key=repr)
        initial_other = second.initial_state
        seen = set()
        queue: deque = deque()
        for state in base.states:
            pair = (state, initial_other)
            seen.add(pair)
            queue.append(pair)
        reverse_edges: dict = {}
        while queue:
            p, q = queue.popleft()
            for symbol in common:
                for p2 in table.get(p, {}).get(symbol, ()):
                    for q2 in second.transitions.get(q, {}).get(symbol, ()):
                        reverse_edges.setdefault((p2, q2), set()).add((p, q))
                        if (p2, q2) not in seen:
                            seen.add((p2, q2))
                            queue.append((p2, q2))
        good = {
            pair
            for pair in seen
            if pair[0] in base.final_states and pair[1] in second.final_states
        }
        stack = list(good)
        while stack:
            for before in reverse_edges.get(stack.pop(), ()):
                if before not in good:
                    good.add(before)
                    stack.append(before)
        finals = {state for state in base.states if (state, initial_other) in good}
        if dfa_case:
            return type(self)(
                states=self.states,
                input_symbols=self.input_symbols,
                transitions={s: dict(t) for s, t in self.transitions.items()},
                initial_state=self.initial_state,
                final_states=finals,
                allow_partial=True,
            )
        return type(base)(
            states=base.states,
            input_symbols=base.input_symbols,
            transitions=base.transitions,
            initial_state=base.initial_state,
            final_states=finals,
        )
