"""Systems of rational (regular) language equations (function 58).

A system has one unknown language ``X_i`` per equation and regular
coefficients.  Two forms exist, named like in :mod:`.arden`:

* ``form="AX"`` (coefficient *before* the unknown, "right-linear")::

      X_i = A_i1·X_1 ∪ A_i2·X_2 ∪ … ∪ A_in·X_n ∪ B_i

* ``form="XA"`` (coefficient *after* the unknown, "left-linear")::

      X_i = X_1·A_i1 ∪ X_2·A_i2 ∪ … ∪ X_n·A_in ∪ B_i

Solving = Gaussian elimination where the "division" is Arden's lemma
(:func:`.arden.arden_star`): the unknown ``X_k`` is isolated with
``X_k = A_kk*·(rest)`` (or ``(rest)·A_kk*``), substituted in the following
equations, then the closed forms are found back to front.  The order of every
concatenation is kept: coefficients are only ever multiplied on the side where
they were written.

When every self-coefficient met during the elimination satisfies ``ε ∉ L(A)``
the solution is **unique**.  Otherwise the system is refused
(:class:`ArdenError`) unless ``least=True`` is given, which returns the
*least* solution for inclusion (the one an automaton defines).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .arden import ArdenError, Form, arden_star
from .regex import EMPTY, EPSILON, Regex, Symbol, concat, union


class EquationSystemError(ValueError):
    """The system is ill-formed (unknown names, wrong types, bad order...)."""


def _join(form: Form, via: Regex, loop: Regex, to: Regex) -> Regex:
    """Chain ``via``, the loop and ``to`` on the side imposed by ``form``.

    ``AX``: ``via·loop·to`` (``via`` stands before the unknown being replaced);
    ``XA``: ``to·loop·via`` (the replaced unknown stands before its coefficient).
    """
    return concat(via, loop, to) if form == "AX" else concat(to, loop, via)


@dataclass(frozen=True)
class EquationSystem:
    """Immutable system of ``n`` rational equations in ``n`` unknowns.

    Parameters
    ----------
    unknowns : tuple of str
        Distinct non-empty names, in the order of the equations.
    coefficients : tuple of tuples of Regex
        ``coefficients[i][j]`` is ``A_ij``, the coefficient of ``X_j`` in the
        equation of ``X_i`` (``∅`` when ``X_j`` does not occur).
    constants : tuple of Regex
        ``constants[i]`` is ``B_i``.
    form : {"AX", "XA"}
        Side of the coefficients (see the module docstring).

    Raises
    ------
    EquationSystemError
        If the system is ill-formed.  Prefer :meth:`build`.
    """

    unknowns: tuple[str, ...]
    coefficients: tuple[tuple[Regex, ...], ...]
    constants: tuple[Regex, ...]
    form: Form = "AX"

    def __post_init__(self) -> None:
        n = len(self.unknowns)
        object.__setattr__(self, "unknowns", tuple(self.unknowns))
        object.__setattr__(self, "coefficients", tuple(tuple(r) for r in self.coefficients))
        object.__setattr__(self, "constants", tuple(self.constants))
        if n == 0:
            raise EquationSystemError("a system needs at least one unknown.")
        if any(not isinstance(u, str) or not u for u in self.unknowns):
            raise EquationSystemError("unknown names must be non-empty strings.")
        if len(set(self.unknowns)) != n:
            raise EquationSystemError("unknown names must be distinct.")
        if self.form not in ("AX", "XA"):
            raise EquationSystemError(f"form must be 'AX' or 'XA', got {self.form!r}.")
        if len(self.coefficients) != n or any(len(row) != n for row in self.coefficients):
            raise EquationSystemError("the coefficients must form an n×n table.")
        if len(self.constants) != n:
            raise EquationSystemError("there must be exactly one constant per equation.")
        for value in (*self.constants, *(r for row in self.coefficients for r in row)):
            if not isinstance(value, Regex):
                raise EquationSystemError(f"coefficients must be Regex, got {type(value).__name__}.")

    # ---- construction -------------------------------------------------------------

    @classmethod
    def build(
        cls,
        unknowns: Sequence[str],
        coefficients: Mapping[tuple[str, str], Regex] | None = None,
        constants: Mapping[str, Regex] | None = None,
        form: Form = "AX",
    ) -> EquationSystem:
        """Build a system from sparse tables; a missing coefficient or constant is ``∅``.

        Parameters
        ----------
        unknowns : sequence of str
            Names of the unknowns.
        coefficients : mapping ``(i, j) -> Regex``
            ``A_ij`` for the pair of names ``(X_i, X_j)``.
        constants : mapping ``name -> Regex``
            ``B_i``.
        form : {"AX", "XA"}
            Side of the coefficients.

        Raises
        ------
        EquationSystemError
            For a name that is not an unknown, a duplicate, or a non-Regex value.
        """
        names = tuple(unknowns)
        position = {name: i for i, name in enumerate(names)}
        table: list[list[Regex]] = [[EMPTY] * len(names) for _ in names]
        constant: list[Regex] = [EMPTY] * len(names)
        for key, value in (coefficients or {}).items():
            if not (isinstance(key, tuple) and len(key) == 2):
                raise EquationSystemError(f"a coefficient key must be a pair of names, got {key!r}.")
            for name in key:
                if name not in position:
                    raise EquationSystemError(f"{name!r} is not an unknown of the system.")
            table[position[key[0]]][position[key[1]]] = value
        for name, value in (constants or {}).items():
            if name not in position:
                raise EquationSystemError(f"{name!r} is not an unknown of the system.")
            constant[position[name]] = value
        return cls(names, tuple(tuple(r) for r in table), tuple(constant), form)

    @classmethod
    def from_automaton(
        cls, automaton: Any, form: Form = "AX", states: Iterable[Any] | None = None
    ) -> AutomatonSystem:
        """Return the system associated with a finite automaton.

        One unknown ``X0``, ``X1``, … per state (``X0`` is the initial state,
        the others follow in breadth-first order, unreachable states last).

        * ``form="AX"``: ``X_q`` is the language of the words that lead from
          ``q`` to a final state, so ``X_q = ⋃ a·X_r ∪ (ε if q is final)``.
        * ``form="XA"``: ``X_q`` is the language of the words that lead from
          the initial state to ``q``, so ``X_q = ⋃ X_p·a ∪ (ε if q is initial)``.

        Epsilon transitions (NFA) give the coefficient ``ε``; a cycle of them
        makes the solution non-unique, and only ``solve(least=True)`` works.

        Parameters
        ----------
        automaton : DFA or NFA of automata-lib (or an extension)
            Only ``states``, ``transitions``, ``initial_state`` and
            ``final_states`` are read; symbols must be single characters.
        form : {"AX", "XA"}
            Which system to build.
        states : iterable, optional
            Restrict the system to these states (the initial one is always kept).

        Returns
        -------
        AutomatonSystem
            The system with the unknown of every state and the way to read the
            language of the automaton from a solution.
        """
        if form not in ("AX", "XA"):
            raise EquationSystemError(f"form must be 'AX' or 'XA', got {form!r}.")
        keep = set(states) if states is not None else set(automaton.states)
        keep.add(automaton.initial_state)

        def targets(value: Any) -> list[Any]:
            hook = getattr(automaton, "_targets", None)
            if hook is not None:
                return list(hook(value))
            return list(value) if isinstance(value, (set, frozenset, list, tuple)) else [value]

        edges: list[tuple[Any, str, Any]] = []
        for state in keep:
            for symbol, value in automaton.transitions.get(state, {}).items():
                for target in targets(value):
                    if target in keep:
                        edges.append((state, symbol, target))
        order: list[Any] = []
        seen = {automaton.initial_state}
        queue = [automaton.initial_state]
        while queue:
            state = queue.pop(0)
            order.append(state)
            out = sorted(
                {(s, repr(t)): t for (p, s, t) in edges if p == state}.items(), key=lambda kv: kv[0]
            )
            for _, target in out:
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        order += sorted((s for s in keep if s not in seen), key=repr)
        names = {state: f"X{i}" for i, state in enumerate(order)}
        eps = getattr(automaton, "_EPSILON", "")
        coefficients: dict[tuple[str, str], Regex] = {}
        for source, symbol, target in edges:
            if symbol != eps and len(symbol) != 1:
                raise EquationSystemError(
                    f"symbol {symbol!r} has several characters; the regex module needs one-character symbols."
                )
            label = EPSILON if symbol == eps else Symbol(symbol)
            key = (names[source], names[target]) if form == "AX" else (names[target], names[source])
            coefficients[key] = union(coefficients.get(key, EMPTY), label)
        if form == "AX":
            constants = {names[q]: EPSILON for q in order if q in automaton.final_states}
            reading: tuple[str, ...] = (names[automaton.initial_state],)
        else:
            constants = {names[automaton.initial_state]: EPSILON}
            reading = tuple(names[q] for q in order if q in automaton.final_states)
        system = cls.build([names[q] for q in order], coefficients, constants, form)
        return AutomatonSystem(system, MappingProxyType(dict(names)), reading)

    # ---- queries ------------------------------------------------------------------

    def right_hand_side(self, name: str, solution: Mapping[str, Regex]) -> Regex:
        """Return the right-hand side of the equation of ``name`` for the given values."""
        i = self._index(name)
        parts = []
        for j, other in enumerate(self.unknowns):
            coefficient = self.coefficients[i][j]
            if coefficient != EMPTY:
                value = solution[other]
                parts.append(concat(coefficient, value) if self.form == "AX" else concat(value, coefficient))
        return union(*parts, self.constants[i])

    def is_solution(self, solution: Mapping[str, Regex]) -> bool:
        """Check that ``solution`` satisfies **every** equation (equality of languages).

        The comparison is semantic (``Regex.same_language_as``), not
        syntactic: two different expressions of the same language are fine.

        Raises
        ------
        EquationSystemError
            If a value is missing for an unknown.
        """
        missing = [u for u in self.unknowns if u not in solution]
        if missing:
            raise EquationSystemError(f"missing value for {missing}.")
        return all(
            solution[name].same_language_as(self.right_hand_side(name, solution))
            for name in self.unknowns
        )

    def __str__(self) -> str:
        lines = []
        for i, name in enumerate(self.unknowns):
            terms = []
            for j, other in enumerate(self.unknowns):
                coefficient = self.coefficients[i][j]
                if coefficient != EMPTY:
                    c = f"({coefficient})"
                    terms.append(f"{c}{other}" if self.form == "AX" else f"{other}{c}")
            if self.constants[i] != EMPTY or not terms:
                terms.append(str(self.constants[i]))
            lines.append(f"{name} = " + " ∪ ".join(terms))
        return "\n".join(lines)

    def _index(self, name: str) -> int:
        try:
            return self.unknowns.index(name)
        except ValueError:
            raise EquationSystemError(f"{name!r} is not an unknown of the system.") from None

    # ---- solving ------------------------------------------------------------------

    def solve(
        self, order: Sequence[str] | None = None, least: bool = False
    ) -> Mapping[str, Regex]:
        """Solve the system and return one regular expression per unknown.

        Algorithm (``form="AX"``; ``"XA"`` is the mirror image):

        1. pick the next unknown ``X_k`` to eliminate;
        2. Arden: ``X_k = A_kk*·(Σ_{j later} A_kj X_j ∪ B_k)``;
        3. substitute in every later equation ``i``:
           ``A_ij ← A_ij ∪ A_ik·A_kk*·A_kj`` and ``B_i ← B_i ∪ A_ik·A_kk*·B_k``;
        4. when all unknowns are eliminated, go back from the last one to the
           first to replace the later unknowns by their expressions.

        Without ``order``, the next unknown is the one that creates the fewest
        new terms (``#incoming × #outgoing`` non-empty coefficients), which
        keeps the expressions small; all results are in the normal form of
        the regex module (``∅`` absorbs, ``ε`` disappears, ``∅* = ε``...).

        Parameters
        ----------
        order : sequence of str, optional
            Elimination order: every unknown exactly once.
        least : bool
            If false (default) a self-coefficient containing ``ε`` raises
            :class:`ArdenError` (the solution would not be unique).  If true,
            the least solution is returned instead.

        Returns
        -------
        mapping
            Read-only ``{name: Regex}``; the system itself is never modified.

        Raises
        ------
        EquationSystemError
            If ``order`` is not a permutation of the unknowns.
        ArdenError
            If Arden's condition fails and ``least`` is false; the message
            names the unknown.
        """
        n = len(self.unknowns)
        if order is None:
            chosen: list[int] | None = None
        else:
            chosen = [self._index(name) for name in order]
            if sorted(chosen) != list(range(n)):
                raise EquationSystemError("order must contain every unknown exactly once.")
        a = [list(row) for row in self.coefficients]
        b = list(self.constants)
        remaining = list(range(n))
        eliminated: list[tuple[int, Regex, dict[int, Regex], Regex]] = []
        while remaining:
            if chosen is not None:
                k = chosen[len(eliminated)]
            else:
                def cost(x: int) -> tuple[int, int]:
                    ins = sum(1 for i in remaining if i != x and a[i][x] != EMPTY)
                    outs = sum(1 for j in remaining if j != x and a[x][j] != EMPTY)
                    return (ins * outs, x)

                k = min(remaining, key=cost)
            remaining.remove(k)
            try:
                loop = arden_star(a[k][k], least=least)
            except ArdenError as error:
                raise ArdenError(f"cannot eliminate {self.unknowns[k]}: {error}") from None
            row = {j: a[k][j] for j in remaining if a[k][j] != EMPTY}
            eliminated.append((k, loop, row, b[k]))
            for i in remaining:
                via = a[i][k]
                if via == EMPTY:
                    continue
                for j, coefficient in row.items():
                    a[i][j] = union(a[i][j], _join(self.form, via, loop, coefficient))
                b[i] = union(b[i], _join(self.form, via, loop, b[k]))
        value: dict[int, Regex] = {}
        for k, loop, row, constant in reversed(eliminated):
            if self.form == "AX":
                value[k] = concat(
                    loop, union(*[concat(c, value[j]) for j, c in row.items()], constant)
                )
            else:
                value[k] = concat(
                    union(*[concat(value[j], c) for j, c in row.items()], constant), loop
                )
        return MappingProxyType({name: value[i] for i, name in enumerate(self.unknowns)})


@dataclass(frozen=True)
class AutomatonSystem:
    """The system of an automaton with the names of its unknowns.

    Attributes
    ----------
    system : EquationSystem
        The equations.
    names : mapping
        ``state → unknown name``.
    reading : tuple of str
        Unknowns whose union is the language of the automaton (the initial
        state for ``"AX"``, the final states for ``"XA"``).
    """

    system: EquationSystem
    names: Mapping[Any, str]
    reading: tuple[str, ...]

    def language(self, solution: Mapping[str, Regex]) -> Regex:
        """Return the regular expression of the automaton from a solution."""
        return union(*[solution[name] for name in self.reading])

    def solve(self, order: Sequence[str] | None = None, least: bool = False) -> Regex:
        """Solve the system and return the expression of the automaton's language.

        Automata give systems with ``ε`` coefficients in some cases (epsilon
        transitions); use ``least=True`` for them.
        """
        return self.language(self.system.solve(order=order, least=least))
