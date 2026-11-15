"""Myhill-Nerode partition of a regular language given by a DFA (function 59).

Words and states are different objects.  The Nerode congruence is a relation
on **words**::

    u ~_L v   ⟺   ∀ w ∈ Σ*,  uw ∈ L ⟺ vw ∈ L

and its classes are in one-to-one correspondence with the states of the minimal
complete DFA.  In a DFA, two words read to the same state are equivalent, and
two words read to *different but equivalent* states (same future behaviour)
are equivalent too; the equivalence of states is computed by
``ExtendedDFA.myhill_nerode_classes`` (function 30, Moore's refinement), which
this module only reuses.  An incomplete DFA is completed first with an
implicit sink: the words that leave the automaton form (or join) the "dead"
class of the words that have no continuation in ``L``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence


@dataclass(frozen=True)
class NerodeClass:
    """One class of words of the Nerode congruence (immutable).

    Attributes
    ----------
    index : int
        Position in the partition (classes are ordered by their representative;
        class 0 contains the empty word).
    states : frozenset
        The states of the **completed** DFA that the words of the class lead
        to (accessible states only).  They are pairwise equivalent.  When the
        DFA was incomplete, the sink added by the completion may belong here.
    representative : str
        The shortest word of the class (the alphabetically first among the
        shortest).
    is_final : bool
        ``True`` iff the words of the class belong to ``L`` (``ε`` is in the
        residual).
    is_dead : bool
        ``True`` iff no word of the class can be extended into ``L`` (its
        residual is ``∅``).
    """

    index: int
    states: frozenset
    representative: str
    is_final: bool
    is_dead: bool


@dataclass(frozen=True)
class MyhillNerodePartition:
    """Partition of ``Σ*`` into Myhill-Nerode classes (immutable).

    Build it with ``ExtendedDFA.nerode_partition()``.

    Attributes
    ----------
    classes : tuple of NerodeClass
        The classes, ordered by (length, text) of their representative.
    """

    classes: tuple[NerodeClass, ...]
    _automaton: Any = field(repr=False, compare=False)
    _source: Any = field(repr=False, compare=False)

    # ---- construction -------------------------------------------------------------

    @classmethod
    def from_dfa(cls, dfa: Any) -> MyhillNerodePartition:
        """Compute the partition of the language of ``dfa`` (the DFA is not modified)."""
        full = dfa.complete()
        groups = full.myhill_nerode_classes(accessible_only=True)
        alphabet = sorted(full.input_symbols)
        shortest = {full.initial_state: ""}
        queue: deque = deque([full.initial_state])
        while queue:
            state = queue.popleft()
            for symbol in alphabet:
                target = full.transitions[state][symbol]
                if target not in shortest:
                    shortest[target] = shortest[state] + symbol
                    queue.append(target)
        useful = full.coaccessible_states()

        def key(word: str) -> tuple[int, str]:
            return (len(word), word)

        classes = tuple(
            NerodeClass(
                index=i,
                states=frozenset(group),
                representative=min((shortest[s] for s in group), key=key),
                is_final=bool(group & full.final_states),
                is_dead=group.isdisjoint(useful),
            )
            for i, group in enumerate(
                sorted(groups, key=lambda g: key(min((shortest[s] for s in g), key=key)))
            )
        )
        return cls(classes, full, dfa)

    # ---- basic access -------------------------------------------------------------

    def __len__(self) -> int:
        """Number of classes: the index of the Nerode congruence."""
        return len(self.classes)

    def __iter__(self) -> Iterator[NerodeClass]:
        return iter(self.classes)

    @property
    def index(self) -> int:
        """Number of classes (``|Q|`` of the minimal complete DFA)."""
        return len(self.classes)

    def representatives(self) -> tuple[str, ...]:
        """Shortest word of every class, in class order."""
        return tuple(c.representative for c in self.classes)

    def class_of_state(self, state: Any) -> NerodeClass:
        """Return the class of an accessible state of the completed DFA.

        Raises
        ------
        ValueError
            If the state is unknown or inaccessible (no word leads to it).
        """
        for cls in self.classes:
            if state in cls.states:
                return cls
        raise ValueError(f"{state!r} is not an accessible state of the automaton.")

    def _state_of_word(self, word: Sequence[str]) -> Any:
        state = self._automaton.initial_state
        for symbol in word:
            if symbol not in self._automaton.input_symbols:
                raise ValueError(f"{symbol!r} is not a symbol of the alphabet.")
            state = self._automaton.transitions[state][symbol]
        return state

    def class_of_word(self, word: Sequence[str]) -> NerodeClass:
        """Return the class of a **word** (read it from the initial state).

        Raises
        ------
        ValueError
            If a symbol is not in the alphabet.
        """
        return self.class_of_state(self._state_of_word(word))

    def are_equivalent(self, first: Sequence[str], second: Sequence[str]) -> bool:
        """Tell whether two words satisfy ``u ~_L v``."""
        return self.class_of_word(first) == self.class_of_word(second)

    def separating_suffix(self, first: Sequence[str], second: Sequence[str]) -> str | None:
        """Return a shortest suffix ``w`` with exactly one of ``uw``, ``vw`` in ``L``.

        ``None`` when the words are equivalent.  Uses
        ``ExtendedDFA.distinguishing_word`` (function 29), not a new search.
        """
        return self._automaton.distinguishing_word(
            self._state_of_word(first), self._state_of_word(second)
        )

    def transition(self, cls: NerodeClass, symbol: str) -> NerodeClass:
        """Return the class of ``u·a`` for the words ``u`` of ``cls`` (well defined).

        This is the transition of the minimal automaton; it does not depend on
        the word chosen in the class.
        """
        if symbol not in self._automaton.input_symbols:
            raise ValueError(f"{symbol!r} is not a symbol of the alphabet.")
        state = next(iter(cls.states))
        return self.class_of_state(self._automaton.transitions[state][symbol])

    # ---- residuals ----------------------------------------------------------------

    def residual_regex(self, cls: NerodeClass):
        """Return the residual ``u⁻¹L`` of the class (a regex), ``∅`` for a dead class.

        The DFA is re-rooted at a state of the class and converted with
        ``to_regex_elimination`` (function 53).
        """
        full = self._automaton
        rooted = type(full)(
            states=full.states,
            input_symbols=full.input_symbols,
            transitions=full.transitions,
            initial_state=next(iter(cls.states)),
            final_states=full.final_states,
            allow_partial=False,
        )
        return rooted.to_regex_elimination()

    def quotient_automaton(self, complete: bool = True):
        """Return the minimal DFA whose states are the classes of the partition.

        State names are the ``frozenset`` of states of the class, exactly as
        ``ExtendedDFA.minimize`` (function 32), which does the work.  With
        ``complete=False`` the dead class is left out (a partial DFA).
        """
        return self._automaton.minimize() if complete else self._source.minimize()

    def __str__(self) -> str:
        lines = []
        for c in self.classes:
            tags = (["final"] if c.is_final else []) + (["dead"] if c.is_dead else [])
            word = c.representative or "ε"
            states = sorted(map(str, c.states))
            suffix = f" ({', '.join(tags)})" if tags else ""
            lines.append(f"[{c.index}] {word!r} ~ states {states}{suffix}")
        return "\n".join(lines)
