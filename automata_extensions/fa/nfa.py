"""Extended NFA (with epsilon transitions) built on automata-lib's ``NFA``."""

from __future__ import annotations

from automata.fa.nfa import NFA

from .mixins.analyse import AnalyseMixin, State
from .mixins.conversions import ConversionMixin
from .mixins.grammaire import GrammaireMixin
from .mixins.morphisme import MorphismeMixin
from .mixins.visualisation import VisualisationMixin


class ExtendedNFA(VisualisationMixin, GrammaireMixin, MorphismeMixin, ConversionMixin, AnalyseMixin, NFA):
    """``NFA`` of automata-lib plus the project's mixins.

    automata-lib 9.2.0 represents an NFA as
    ``transitions = {state: {symbol: set_of_states}}`` with **one**
    ``initial_state``; epsilon (lambda) transitions use the key ``""``.
    Names of methods that already exist natively (``reverse``,
    ``eliminate_lambda`` ...) are not reused.

    The step-by-step hooks of :class:`AnalyseMixin` are provided too, so
    ``execution_trace`` returns, for an NFA, the successive **sets** of states
    (each one closed under epsilon moves).
    """

    def _initial_configuration(self) -> frozenset:
        """Return the set of states before reading: the epsilon closure of the initial state."""
        return self.epsilon_closure_of([self.initial_state])

    def _next_configuration(self, configuration: frozenset, symbol: str) -> frozenset | None:
        """Return the epsilon-closed set reached from ``configuration`` by ``symbol``.

        ``None`` means the automaton is blocked: ``symbol`` is not in the
        alphabet, or no state of the set has a ``symbol`` transition (the set
        of current states would be empty, so the word is rejected).
        """
        if symbol not in self.input_symbols:
            return None
        reached = {
            target
            for state in configuration
            for target in self.transitions.get(state, {}).get(symbol, ())
        }
        return self.epsilon_closure_of(reached) if reached else None

    def _is_accepting_configuration(self, configuration: frozenset) -> bool:
        """Tell whether the set contains a final state."""
        return not configuration.isdisjoint(self.final_states)

    def _successors(self, state: State) -> list[State]:
        """Return every state reached in one transition (epsilon included)."""
        outgoing = self.transitions.get(state, {})
        found: list[State] = []
        for symbol in sorted(outgoing):
            found.extend(sorted(self._targets(outgoing[symbol]), key=repr))
        return found

    def _restrict(self, kept: frozenset) -> ExtendedNFA:
        """Sub-automaton on ``kept`` (which contains the initial state)."""
        return type(self)(
            states=kept,
            input_symbols=self.input_symbols,
            transitions={
                state: {
                    symbol: frozenset(t for t in targets if t in kept)
                    for symbol, targets in self.transitions.get(state, {}).items()
                    if any(t in kept for t in targets)
                }
                for state in kept
            },
            initial_state=self.initial_state,
            final_states=self.final_states & kept,
        )

    def residual_automaton(self, complete: bool = True):
        """Residual automaton of the language of this NFA (function 60).

        The NFA is first determinized (``to_dfa``, function 52); the residual
        automaton of the resulting DFA is :meth:`ExtendedDFA.residual_automaton`.
        """
        return self.to_dfa(complete=True).residual_automaton(complete)

    def transition_monoid(self, max_elements: int = 100_000):
        """Transition monoid of the determinized automaton (function 62)."""
        return self.to_dfa(complete=True).transition_monoid(max_elements)

    def syntactic_monoid(self, max_elements: int = 100_000):
        """Syntactic monoid of the language of this NFA (function 62), via ``to_dfa``."""
        return self.to_dfa(complete=True).syntactic_monoid(max_elements)
