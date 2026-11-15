"""Morphisms between finite automata (function 61): methods shared by DFA and NFA."""

from __future__ import annotations

from typing import Any, Mapping

from ..morphism import AutomatonMorphism, MorphismError, Violation, morphism_violations


class MorphismeMixin:
    """Check and build morphisms ``h : Q_self → Q_target`` (see :mod:`..morphism`)."""

    def morphism_violations(
        self, target: Any, mapping: Mapping[Any, Any], strict: bool = False
    ) -> tuple[Violation, ...]:
        """Return every reason why ``mapping`` is not a morphism ``self → target``.

        An empty tuple means that ``mapping`` is a morphism.  The conditions
        are those of the module :mod:`automata_extensions.fa.morphism`:
        same alphabet, map defined on every state with images in the target,
        initial state mapped to the initial state, final states preserved
        (``h(F) ⊆ F'``) and every existing transition ``q --a--> r`` mapped to
        a transition ``h(q) --a--> h(r)``.  Missing transitions of ``self`` are
        not constrained (convention for partial DFAs).  With ``strict=True``
        final states and transitions must also be *reflected*, which gives
        ``L(self) = L(target)``.

        Neither automaton is modified.
        """
        return morphism_violations(self, target, mapping, strict)

    def is_morphism_to(self, target: Any, mapping: Mapping[Any, Any], strict: bool = False) -> bool:
        """Tell whether ``mapping`` is a (strict) morphism ``self → target``."""
        return not self.morphism_violations(target, mapping, strict)

    def morphism_to(
        self, target: Any, mapping: Mapping[Any, Any], strict: bool = False
    ) -> AutomatonMorphism:
        """Return the checked morphism ``mapping : self → target``.

        Raises
        ------
        MorphismError
            With one explicit message per violation (``error.violations``).
        TypeError
            If ``mapping`` is not a mapping.
        """
        found = self.morphism_violations(target, mapping, strict)
        if found:
            raise MorphismError(found)
        return AutomatonMorphism(self, target, mapping, strict)
