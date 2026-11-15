"""Arden's lemma: solving the language equations ``X = AX ∪ B`` and ``X = XA ∪ B``.

Statement (``A`` and ``B`` are regular languages, ``X`` is the unknown):

* ``X = A·X ∪ B``  (unknown on the **right** of the coefficient)  ⟹  ``X = A*·B``;
* ``X = X·A ∪ B``  (unknown on the **left** of the coefficient)   ⟹  ``X = B·A*``.

The solution is unique when ``ε ∉ L(A)``.  When ``ε ∈ L(A)`` the equation has
several solutions (``A*·B`` is only the smallest one), so this module refuses
to answer instead of silently choosing one.

The order of the concatenation matters: ``A*·B`` and ``B·A*`` are different
languages in general (``A = a``, ``B = b`` gives ``a*b`` and ``ba*``).
"""

from __future__ import annotations

from typing import Literal

from .regex import Regex, concat, star, union

Form = Literal["AX", "XA"]


class ArdenError(ValueError):
    """The hypotheses of Arden's lemma are not satisfied."""


def _check_regex(name: str, value: object) -> None:
    if not isinstance(value, Regex):
        raise TypeError(f"{name} must be a Regex, got {type(value).__name__}.")


def arden_star(coefficient: Regex, least: bool = False) -> Regex:
    """Return ``A*`` after checking the hypothesis ``ε ∉ L(A)``.

    Parameters
    ----------
    coefficient : Regex
        The coefficient ``A`` of the equation.
    least : bool
        If true, the check is skipped: ``A*`` is then the *smallest* solution
        coefficient (see :func:`solve_arden`), not a unique one.

    Returns
    -------
    Regex
        ``A*`` in normal form (``∅* = ε``).

    Raises
    ------
    ArdenError
        If the empty word belongs to ``L(A)``.
    """
    _check_regex("coefficient", coefficient)
    if coefficient.nullable() and not least:
        raise ArdenError(
            f"Arden's lemma does not apply: the empty word belongs to L({coefficient}), "
            "so the equation has several solutions (A*B is only the smallest one)."
        )
    return star(coefficient)


def solve_arden(a: Regex, b: Regex, form: Form = "AX", *, least: bool = False) -> Regex:
    """Solve ``X = A·X ∪ B`` (``form="AX"``) or ``X = X·A ∪ B`` (``form="XA"``).

    Parameters
    ----------
    a : Regex
        Coefficient ``A``; ``ε ∉ L(A)`` is required.  ``∅`` is allowed.
    b : Regex
        Constant term ``B``; ``∅`` and ``ε`` are allowed.
    form : {"AX", "XA"}
        ``"AX"``: the coefficient is written before the unknown, the solution
        is ``A*·B``.  ``"XA"``: the coefficient is written after the unknown,
        the solution is ``B·A*``.
    least : bool
        Default ``False``: the lemma is applied only when ``ε ∉ L(A)``.  With
        ``True`` and ``ε ∈ L(A)``, the same formula returns the **least**
        solution for inclusion (every solution contains it); it is not unique
        in that case (``X = A*B ∪ Y`` works for any ``Y`` with ``A·Y ⊆ Y``),
        so use it only when the smallest solution is what you want.

    Returns
    -------
    Regex
        The unique solution, simplified by the smart constructors of the regex
        module (``∅`` absorbs, ``ε`` disappears from a concatenation,
        ``∅* = ε``).  The arguments are never modified.

    Raises
    ------
    ArdenError
        If ``ε ∈ L(A)``.
    TypeError
        If ``a`` or ``b`` is not a :class:`Regex`.
    ValueError
        If ``form`` is neither ``"AX"`` nor ``"XA"``.
    """
    _check_regex("a", a)
    _check_regex("b", b)
    if form not in ("AX", "XA"):
        raise ValueError(f"form must be 'AX' or 'XA', got {form!r}.")
    loop = arden_star(a, least=least)
    return concat(loop, b) if form == "AX" else concat(b, loop)


def equation_side(a: Regex, b: Regex, x: Regex, form: Form = "AX") -> Regex:
    """Return the right-hand side of the equation for a candidate ``x``.

    ``A·x ∪ B`` for ``form="AX"`` and ``x·A ∪ B`` for ``form="XA"``.  Used to
    check that a solution really satisfies its equation.
    """
    if form not in ("AX", "XA"):
        raise ValueError(f"form must be 'AX' or 'XA', got {form!r}.")
    return union(concat(a, x) if form == "AX" else concat(x, a), b)
