"""Grammar-related conversions of finite automata (function 55, automaton → grammar)."""

from __future__ import annotations

from collections import deque
from typing import Any

from ...grammar.grammar import LeftRegularGrammar, RegularGrammar
from ...grammar.tree import Production


class GrammaireMixin:
    """Bridge between finite automata and linear grammars.

    The opposite direction, grammar → automaton, is ``to_nfa`` of
    :class:`automata_extensions.grammar.RegularGrammar` and of
    :class:`automata_extensions.grammar.LeftRegularGrammar`.
    """

    def _epsilon_free(self) -> Any:
        """Return an automaton without epsilon transition (``self`` if it has none)."""
        automaton: Any = self
        has_epsilon = any(
            automaton._targets(value)
            for outgoing in automaton.transitions.values()
            for symbol, value in outgoing.items()
            if symbol == automaton._EPSILON
        )
        return automaton.remove_epsilon_transitions() if has_epsilon else automaton

    @staticmethod
    def _grammar_names(
        automaton: Any, terminals: set[str], reserved: set[str], initial_name: str | None = None
    ) -> dict:
        """Name every state ``A0``, ``A1``, ... in breadth-first order from the initial one.

        Unreachable states come last, ordered by ``repr``.  A name that is a
        terminal symbol or in ``reserved`` is skipped, so nonterminals never
        collide with anything.  If ``initial_name`` is given, the initial state
        takes it and does not use up a number.
        """
        order: list[Any] = []
        seen = {automaton.initial_state}
        queue = deque([automaton.initial_state])
        while queue:
            state = queue.popleft()
            order.append(state)
            outgoing = automaton.transitions.get(state, {})
            for symbol in sorted(outgoing):
                for target in sorted(automaton._targets(outgoing[symbol]), key=repr):
                    if target not in seen:
                        seen.add(target)
                        queue.append(target)
        order += sorted((s for s in automaton.states if s not in seen), key=repr)
        taken = set(terminals) | set(reserved)
        names: dict[Any, str] = {}
        if initial_name is not None:
            names[automaton.initial_state] = initial_name
        counter = 0
        for state in order:
            if state in names:
                continue
            while f"A{counter}" in taken:
                counter += 1
            names[state] = f"A{counter}"
            taken.add(names[state])
        return names

    def to_right_grammar(self) -> RegularGrammar:
        """Return a right-linear grammar generating the language of the automaton.

        Construction: one nonterminal ``A_q`` per state ``q``;

        * the start symbol is the nonterminal of the initial state;
        * every transition ``q --a--> r`` gives ``A_q → a A_r``;
        * every final state ``q`` gives ``A_q → ε``.

        Naming: the initial state is ``S``; the other states get ``A0``,
        ``A1``, ... in breadth-first order from the initial state (unreachable
        states afterwards, ordered by ``repr``).  A name equal to a terminal
        symbol is skipped, so nonterminals never collide with terminals.

        Epsilon transitions would give unit productions ``A → B`` (not
        regular): they are removed first with ``remove_epsilon_transitions``.
        The automaton is not modified.

        Returns
        -------
        RegularGrammar
            An immutable grammar; ``L(G) = L(automaton)``.
        """
        automaton = self._epsilon_free()
        terminals = {s for s in automaton.input_symbols if s != automaton._EPSILON}
        start = "S"
        while start in terminals:
            start += "'"
        names = self._grammar_names(automaton, terminals, {start}, initial_name=start)
        productions: set[Production] = set()
        for state in automaton.states:
            for symbol, value in automaton.transitions.get(state, {}).items():
                if symbol == automaton._EPSILON:
                    continue
                for target in automaton._targets(value):
                    productions.add(Production(names[state], (symbol, names[target])))
            if state in automaton.final_states:
                productions.add(Production(names[state], ()))
        return RegularGrammar(
            frozenset(names.values()), frozenset(terminals), frozenset(productions), start
        )

    def to_left_grammar(self) -> LeftRegularGrammar:
        """Return a left-linear grammar generating the language of the automaton.

        Construction: one nonterminal ``A_q`` per state ``q`` plus a fresh start
        symbol ``S``.  ``A_q`` generates the words that lead from the initial
        state to ``q``:

        * the initial state gives ``A_q0 → ε``;
        * every transition ``q --a--> r`` gives ``A_r → A_q a``.

        The language is the union over the final states ``f`` of ``L(A_f)``.
        To avoid unit productions ``S → A_f``, ``S`` receives directly the
        productions of every final ``A_f``: ``S → A_q a`` for each transition
        ``q --a--> f`` and ``S → ε`` if the initial state is final.

        Epsilon transitions are removed first.  The automaton is not modified.
        """
        automaton = self._epsilon_free()
        terminals = {s for s in automaton.input_symbols if s != automaton._EPSILON}
        start = "S"
        while start in terminals:
            start += "'"
        names = self._grammar_names(automaton, terminals, {start})
        productions: set[Production] = {Production(names[automaton.initial_state], ())}
        if automaton.initial_state in automaton.final_states:
            productions.add(Production(start, ()))
        for state in automaton.states:
            for symbol, value in automaton.transitions.get(state, {}).items():
                if symbol == automaton._EPSILON:
                    continue
                for target in automaton._targets(value):
                    productions.add(Production(names[target], (names[state], symbol)))
                    if target in automaton.final_states:
                        productions.add(Production(start, (names[state], symbol)))
        return LeftRegularGrammar(
            frozenset({*names.values(), start}),
            frozenset(terminals),
            frozenset(productions),
            start,
        )
