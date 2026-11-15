"""Residual automaton of a regular expression (function 60).

The residual of a language ``L`` by a word ``u`` is ``u⁻¹L = { w | uw ∈ L }``.
For an expression, the residual by a letter is the Brzozowski derivative, and
``u⁻¹L`` is obtained by deriving letter after letter.  The states of the
residual automaton are the **distinct residuals** (distinct as languages).
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

from .regex import Regex


def residual_automaton(
    regex: Regex,
    alphabet: Iterable[str] | None = None,
    complete: bool = True,
    max_states: int = 10_000,
):
    """Build the residual automaton of ``L(regex)``.

    Algorithm (breadth-first, so representatives are the residuals of the
    shortest words, in alphabetical order for equal lengths):

    1. the first residual is the language itself (the ``ε``-residual);
    2. for a residual ``R`` and a symbol ``a``, ``a⁻¹R`` is the derivative
       ``D_a(R)``;
    3. a new derivative is **merged** with an existing residual when they have
       the same *language*.  Syntactic equality is only a shortcut: two
       different expressions such as ``a*`` and ``ε|aa*`` are the same residual,
       decided with :meth:`Regex.same_language_as` (Brzozowski automata
       compared by :meth:`ExtendedDFA.has_same_language_as`);
    4. a residual is final iff it contains the empty word (``nullable``).

    The set of syntactic derivatives is finite (normal form of the regex
    module), so the construction stops; merging only makes it smaller.

    Parameters
    ----------
    regex : Regex
        The language.
    alphabet : iterable of str, optional
        Symbols of the automaton; it must contain those of the expression.
        Default: the symbols of the expression.
    complete : bool
        ``True`` (default): the automaton is complete and keeps the residual
        ``∅`` when it occurs.  ``False``: the ``∅`` residual and the
        transitions to it are removed (partial DFA); an empty language gives
        the single initial state.
    max_states : int
        Safety limit on the number of syntactic derivatives explored.

    Returns
    -------
    ExtendedDFA
        States are :class:`Regex` objects: the representative of each
        residual (the first one found).  It is the minimal DFA of the
        language, whose states are the residuals.

    Raises
    ------
    ValueError
        If the alphabet is empty (the library refuses automata without
        symbols) or misses a symbol of the expression.
    RuntimeError
        If more than ``max_states`` derivatives would be explored.
    """
    from ..fa.dfa import ExtendedDFA

    symbols = regex._alphabet_argument(alphabet)
    if not symbols:
        raise ValueError("Pass a non-empty alphabet: the library refuses automata without symbols.")
    automata: dict[Regex, object] = {}

    def dfa_of(r: Regex):
        if r not in automata:
            automata[r] = r.brzozowski_dfa(symbols, max_states=max_states)
        return automata[r]

    representatives: list[Regex] = [regex]
    owner: dict[Regex, Regex] = {regex: regex}
    transitions: dict[Regex, dict[str, Regex]] = {regex: {}}
    queue: deque[Regex] = deque([regex])
    seen = 1
    while queue:
        current = queue.popleft()
        for symbol in symbols:
            derivative = current.derivative(symbol)
            if derivative not in owner:
                seen += 1
                if seen > max_states:
                    raise RuntimeError(f"More than {max_states} derivatives: construction stopped.")
                twin = next(
                    (
                        r
                        for r in representatives
                        if r.nullable() == derivative.nullable()
                        and dfa_of(r).has_same_language_as(dfa_of(derivative))
                    ),
                    None,
                )
                if twin is None:
                    representatives.append(derivative)
                    transitions[derivative] = {}
                    queue.append(derivative)
                    owner[derivative] = derivative
                else:
                    owner[derivative] = twin
            transitions[current][symbol] = owner[derivative]
    finals = {r for r in representatives if r.nullable()}
    if complete:
        return ExtendedDFA(
            states=frozenset(representatives),
            input_symbols=frozenset(symbols),
            transitions=transitions,
            initial_state=regex,
            final_states=finals,
        )
    dead = {r for r in representatives if dfa_of(r).is_empty()}
    kept = [r for r in representatives if r not in dead or r == regex]
    if regex in dead:
        kept = [regex]
    return ExtendedDFA(
        states=frozenset(kept),
        input_symbols=frozenset(symbols),
        transitions={
            r: {a: t for a, t in transitions[r].items() if t in kept and not (regex in dead)}
            for r in kept
        },
        initial_state=regex,
        final_states={r for r in finals if r in kept},
        allow_partial=True,
    )
