"""Morphisms of finite automata (function 61).

Definitions used by the project
-------------------------------
Let ``A`` and ``B`` be two finite automata over the **same alphabet** and
``h : Q_A → Q_B`` a map defined on **every** state of ``A``.

``h`` is a **morphism** (also called homomorphism here) when:

1. ``h(i_A) = i_B``  (initial states);
2. ``h(F_A) ⊆ F_B``  (final states are preserved);
3. for every transition ``q --a--> r`` of ``A`` there is a transition
   ``h(q) --a--> h(r)`` in ``B``: for deterministic automata
   ``h(δ_A(q, a)) = δ_B(h(q), a)`` **whenever ``δ_A(q, a)`` is defined**.

Nothing is required where ``δ_A(q, a)`` is undefined, and ``B`` may have more
transitions (a *partial* DFA is allowed on both sides).  Consequently a morphism
only guarantees ``L(A) ⊆ L(B)``.

``h`` is a **strict morphism** if, in addition:

4. ``h⁻¹(F_B) = F_A``  (final states are reflected);
5. every transition ``h(q) --a--> p`` of ``B`` comes from a transition
   ``q --a--> r`` of ``A`` with ``h(r) = p`` (for a DFA: ``δ_A(q, a)`` is
   defined iff ``δ_B(h(q), a)`` is).

A strict morphism satisfies ``L(A) = L(B)``: a run of ``B`` on a word is
lifted step by step to a run of ``A`` and ends in a final state of ``A``.

``h`` is an **isomorphism** when it is a strict morphism and a bijection (its
inverse is then a strict morphism too); this is exactly what
``ExtendedDFA.is_isomorphic_to`` (function 26) searches for.  A morphism is
**not** required to be injective or surjective: merging states (quotient,
minimization) gives non-injective strict morphisms.

Epsilon transitions of NFAs are treated like any other label.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

EPSILON_KEY = ""


@dataclass(frozen=True)
class Violation:
    """One reason why a map is not a morphism.

    Attributes
    ----------
    kind : str
        ``"alphabet"``, ``"domain"``, ``"codomain"``, ``"initial"``,
        ``"final"``, ``"transition"``, ``"missing-transition"``,
        ``"strict-final"`` or ``"strict-transition"``.
    message : str
        Explicit explanation.
    """

    kind: str
    message: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.message}"


class MorphismError(ValueError):
    """The map is not a morphism; ``violations`` lists every reason found."""

    def __init__(self, violations: Iterable[Violation]) -> None:
        self.violations: tuple[Violation, ...] = tuple(violations)
        super().__init__("not a morphism:\n  " + "\n  ".join(str(v) for v in self.violations))


def _targets(automaton: Any, value: Any) -> list[Any]:
    """Destinations stored in one cell: a set for an NFA, a single state for a DFA."""
    from automata.fa.nfa import NFA

    return list(value) if isinstance(automaton, NFA) else [value]


def _out(automaton: Any, state: Any, symbol: str) -> list[Any]:
    """Destinations of ``state`` on ``symbol`` (empty list when there is no transition)."""
    cell = automaton.transitions.get(state, {}).get(symbol)
    return [] if cell is None else _targets(automaton, cell)


def _alphabet(automaton: Any) -> frozenset[str]:
    return frozenset(s for s in automaton.input_symbols if s != EPSILON_KEY)


def _by_repr(items: Iterable[Any]) -> list[Any]:
    return sorted(items, key=repr)


def morphism_violations(
    source: Any, target: Any, mapping: Mapping[Any, Any], strict: bool = False
) -> tuple[Violation, ...]:
    """Return every reason why ``mapping`` is not a morphism ``source → target``.

    An empty tuple means that it is one.  Checks are done in this order, and a
    structural problem (alphabet, domain, codomain) stops the later checks,
    which would be meaningless:

    1. alphabets must be equal;
    2. the domain must be exactly ``Q_source`` and the images in ``Q_target``;
    3. initial state; 4. final states; 5. transitions (existing transitions
       of ``source`` must exist in ``target``); 6. with ``strict=True``,
       reflection of final states and of transitions.

    Parameters
    ----------
    source, target : DFA or NFA
        Automata of automata-lib (or extensions); never modified.
    mapping : mapping
        ``state of source → state of target``.
    strict : bool
        Also require the reflection conditions (4. and 5. of the module
        docstring).

    Raises
    ------
    TypeError
        If ``mapping`` is not a mapping.
    """
    if not isinstance(mapping, Mapping):
        raise TypeError(f"mapping must be a mapping, got {type(mapping).__name__}.")
    found: list[Violation] = []
    sa, sb = _alphabet(source), _alphabet(target)
    if sa != sb:
        found.append(
            Violation(
                "alphabet",
                f"the alphabets differ: only in source {sorted(sa - sb)}, only in target {sorted(sb - sa)}.",
            )
        )
    missing = _by_repr(q for q in source.states if q not in mapping)
    unknown = _by_repr(q for q in mapping if q not in source.states)
    if missing:
        found.append(Violation("domain", f"the map is not defined on the states {missing}."))
    if unknown:
        found.append(Violation("domain", f"the map is defined on unknown source states {unknown}."))
    outside = _by_repr(q for q in source.states if q in mapping and mapping[q] not in target.states)
    for q in outside:
        found.append(Violation("codomain", f"h({q!r}) = {mapping[q]!r} is not a state of the target."))
    if found:
        return tuple(found)

    if mapping[source.initial_state] != target.initial_state:
        found.append(
            Violation(
                "initial",
                f"h(initial) = h({source.initial_state!r}) = {mapping[source.initial_state]!r}, "
                f"but the initial state of the target is {target.initial_state!r}.",
            )
        )
    for q in _by_repr(source.final_states):
        if mapping[q] not in target.final_states:
            found.append(
                Violation("final", f"{q!r} is final but h({q!r}) = {mapping[q]!r} is not final in the target.")
            )
    for q in _by_repr(source.states):
        for symbol in sorted(source.transitions.get(q, {})):
            image = _out(target, mapping[q], symbol)
            for r in _by_repr(_out(source, q, symbol)):
                if mapping[r] in image:
                    continue
                label = "ε" if symbol == EPSILON_KEY else repr(symbol)
                if not image:
                    found.append(
                        Violation(
                            "missing-transition",
                            f"{q!r} --{label}--> {r!r} exists but the target has no transition "
                            f"from h({q!r}) = {mapping[q]!r} on {label}.",
                        )
                    )
                else:
                    found.append(
                        Violation(
                            "transition",
                            f"{q!r} --{label}--> {r!r} gives h({r!r}) = {mapping[r]!r}, but the target "
                            f"goes from {mapping[q]!r} on {label} to {_by_repr(image)}.",
                        )
                    )
    if strict:
        for q in _by_repr(source.states):
            if mapping[q] in target.final_states and q not in source.final_states:
                found.append(
                    Violation(
                        "strict-final",
                        f"h({q!r}) = {mapping[q]!r} is final but {q!r} is not (final states are not reflected).",
                    )
                )
            for symbol in sorted(target.transitions.get(mapping[q], {})):
                lifted = {mapping[r] for r in _out(source, q, symbol)}
                for p in _by_repr(_out(target, mapping[q], symbol)):
                    if p not in lifted:
                        label = "ε" if symbol == EPSILON_KEY else repr(symbol)
                        found.append(
                            Violation(
                                "strict-transition",
                                f"the target has {mapping[q]!r} --{label}--> {p!r} but {q!r} has no "
                                f"{label}-transition leading to a state mapped to {p!r}.",
                            )
                        )
    return tuple(found)


@dataclass(frozen=True, eq=False)
class AutomatonMorphism:
    """A checked morphism ``h : source → target`` (immutable).

    Build it with ``ExtendedDFA.morphism_to`` / ``ExtendedNFA.morphism_to``;
    the constructor does **not** check anything, use :func:`morphism_violations`.

    Attributes
    ----------
    source, target : automata
        The two automata.
    mapping : read-only mapping
        ``h`` on every state of ``source``.
    strict : bool
        Whether the strict conditions were checked.
    """

    source: Any
    target: Any
    mapping: Mapping[Any, Any]
    strict: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mapping", MappingProxyType(dict(self.mapping)))

    def __call__(self, state: Any) -> Any:
        """Return ``h(state)``."""
        try:
            return self.mapping[state]
        except KeyError:
            raise ValueError(f"{state!r} is not a state of the source automaton.") from None

    def image(self) -> frozenset:
        """Return ``h(Q_source)``."""
        return frozenset(self.mapping.values())

    def kernel(self) -> frozenset[frozenset]:
        """Return the partition of ``Q_source`` into the fibres ``h⁻¹(p)`` (merged states)."""
        fibres: dict[Any, set] = {}
        for q, p in self.mapping.items():
            fibres.setdefault(p, set()).add(q)
        return frozenset(frozenset(f) for f in fibres.values())

    @property
    def is_injective(self) -> bool:
        """``True`` if no two states are merged."""
        return len(self.image()) == len(self.mapping)

    @property
    def is_surjective(self) -> bool:
        """``True`` if every state of the target is an image."""
        return self.image() == frozenset(self.target.states)

    @property
    def is_bijective(self) -> bool:
        """``True`` for an injective and surjective map."""
        return self.is_injective and self.is_surjective

    @property
    def is_isomorphism(self) -> bool:
        """``True`` iff it is a strict morphism and a bijection."""
        return self.is_bijective and not morphism_violations(
            self.source, self.target, self.mapping, strict=True
        )

    def compose(self, other: AutomatonMorphism) -> AutomatonMorphism:
        """Return ``other ∘ self``: first this morphism, then ``other``.

        Raises
        ------
        ValueError
            If the target of this morphism is not the source of ``other``
            (same states and alphabet) or an image is outside ``other``'s domain.
        """
        if set(self.target.states) != set(other.source.states) or _alphabet(self.target) != _alphabet(other.source):
            raise ValueError("the target of the first morphism must be the source of the second.")
        mapping = {q: other(self(q)) for q in self.mapping}
        return AutomatonMorphism(self.source, other.target, mapping, self.strict and other.strict)
