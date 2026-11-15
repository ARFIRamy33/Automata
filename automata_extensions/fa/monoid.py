"""Transition monoid and syntactic monoid of a regular language (function 62).

Two different objects
---------------------
* The **transition monoid** ``T(A)`` of a complete DFA ``A`` is the set of
  transformations ``f_w : Q → Q``, ``f_w(q) = δ*(q, w)``, with the law
  ``f_u · f_v = f_{uv}`` (``f_u`` first, then ``f_v``) and the identity
  ``f_ε``.  It depends on the automaton, not only on the language.
* The **syntactic monoid** ``M(L)`` is the quotient of ``Σ*`` by the syntactic
  congruence ::

      u ≈_L v   ⟺   ∀ x, y ∈ Σ*,   x·u·y ∈ L  ⟺  x·v·y ∈ L

  It depends only on ``L``.  ``M(L)`` is a quotient of ``T(A)`` for every DFA
  ``A`` of ``L`` (the map ``f_w ↦ [w]≈`` is a surjective monoid morphism),
  but two transformations of ``T(A)`` may be syntactically equal.

Finite computation (no word of ``Σ*`` is ever enumerated)
---------------------------------------------------------
Let ``A`` be a DFA of ``L``, completed (implicit sink) and restricted to its
accessible states.  For two words, ``x·u·y ∈ L`` means: read ``x`` from the
initial state to some accessible state ``p``, then ``u`` then ``y``.  Hence ::

    u ≈_L v   ⟺   for every accessible state p:  f_u(p) ~ f_v(p)

where ``~`` is the Myhill-Nerode equivalence of states (same future, function 30,
reused through ``ExtendedDFA.nerode_partition``, function 59).  Because ``~``
is a congruence for the transitions, a word acts on the **classes** of states,
and two words are syntactically equivalent iff they act identically on the
classes.  So ``M(L)`` is the transition monoid of the minimal DFA, computed by
the closure of the generators ``f_a`` (``a ∈ Σ``) under composition: a
breadth-first search on transformations, at most ``|classes|^|classes|`` of
them.  The elements are found in order of their shortest word, so element 0 is
the class of the empty word, the identity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

Action = tuple[int, ...]


@dataclass(frozen=True)
class MonoidElement:
    """One element of a monoid of transformations (immutable).

    Attributes
    ----------
    index : int
        Position in the monoid (0 is the class of the empty word).
    word : str
        The shortest word (alphabetically first among the shortest) whose
        transformation is this element.
    action : tuple of int
        ``action[i]`` is the image of the ``i``-th point of the set acted on.
    """

    index: int
    word: str
    action: Action


def _compose(first: Action, second: Action) -> Action:
    """Transformation "``first`` then ``second``"."""
    return tuple(second[i] for i in first)


def _close(
    generators: Mapping[str, Action], size: int, max_elements: int
) -> tuple[MonoidElement, ...]:
    """Close the generators under composition (breadth-first, shortlex order)."""
    identity: Action = tuple(range(size))
    found: dict[Action, MonoidElement] = {identity: MonoidElement(0, "", identity)}
    queue: deque[MonoidElement] = deque([found[identity]])
    symbols = sorted(generators)
    while queue:
        element = queue.popleft()
        for symbol in symbols:
            action = _compose(element.action, generators[symbol])
            if action not in found:
                if len(found) >= max_elements:
                    raise RuntimeError(
                        f"more than {max_elements} transformations: construction stopped "
                        "(raise max_elements if this is expected)."
                    )
                found[action] = MonoidElement(len(found), element.word + symbol, action)
                queue.append(found[action])
    return tuple(found.values())


@dataclass(frozen=True)
class TransformationMonoid:
    """A finite monoid of transformations, with its law (immutable).

    Base of :class:`TransitionMonoid` and :class:`SyntacticMonoid`.

    Attributes
    ----------
    alphabet : tuple of str
        Sorted letters.
    acted : tuple
        Labels of the points on which the transformations act.
    elements : tuple of MonoidElement
        Distinct transformations, in order of their shortest word.
    """

    alphabet: tuple[str, ...]
    acted: tuple
    elements: tuple[MonoidElement, ...]
    _generators: Mapping[str, Action] = field(repr=False, compare=False)
    _index: Mapping[Action, int] = field(repr=False, compare=False)

    @classmethod
    def _make(cls, alphabet: Sequence[str], acted: Sequence[Any], generators: Mapping[str, Action],
              max_elements: int, **extra: Any):
        elements = _close(generators, len(acted), max_elements)
        return cls(
            tuple(alphabet), tuple(acted), elements, dict(generators),
            {e.action: e.index for e in elements}, **extra,
        )

    # ---- access -------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of elements (the order of the monoid)."""
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def element(self, index: int) -> MonoidElement:
        """Return the element at ``index``."""
        return self.elements[index]

    def _coerce(self, value: MonoidElement | int) -> MonoidElement:
        return value if isinstance(value, MonoidElement) else self.elements[value]

    def element_of_word(self, word: str) -> MonoidElement:
        """Return the element of the word ``word`` (its transformation).

        The word is read letter by letter and composed with the generators;
        nothing else is enumerated.

        Raises
        ------
        ValueError
            If a letter is not in the alphabet.
        """
        action: Action = tuple(range(len(self.acted)))
        for symbol in word:
            if symbol not in self._generators:
                raise ValueError(f"{symbol!r} is not a letter of the alphabet.")
            action = _compose(action, self._generators[symbol])
        return self.elements[self._index[action]]

    # ---- the law -------------------------------------------------------------------

    def product(self, first: MonoidElement | int, second: MonoidElement | int) -> MonoidElement:
        """Return ``first · second``: the element of the concatenation (``first`` read first).

        Raises
        ------
        ValueError
            If the result is not an element (the set would not be closed).
        """
        a, b = self._coerce(first), self._coerce(second)
        action = _compose(a.action, b.action)
        if action not in self._index:
            raise ValueError("the product is not an element of the monoid: the set is not closed.")
        return self.elements[self._index[action]]

    def _check_size(self, max_size: int, what: str) -> None:
        if len(self) > max_size:
            raise ValueError(
                f"{what} needs {len(self)}² (or ³) operations: too large "
                f"(more than max_size={max_size}); raise max_size to force it."
            )

    def multiplication_table(self, max_size: int = 200) -> tuple[tuple[int, ...], ...]:
        """Return the table ``t[i][j] = index of (element i · element j)``.

        Raises
        ------
        ValueError
            If the monoid has more than ``max_size`` elements (the table would
            have more than ``max_size²`` cells).
        """
        self._check_size(max_size, "the multiplication table")
        return tuple(
            tuple(self.product(a, b).index for b in self.elements) for a in self.elements
        )

    def neutral_element(self) -> MonoidElement | None:
        """Find the neutral element by testing ``e·x = x·e = x`` for every ``x``.

        Not assumed to be element 0: it is looked for.  Returns ``None`` if
        there is none (never the case here).
        """
        for e in self.elements:
            if all(self.product(e, x) == x and self.product(x, e) == x for x in self.elements):
                return e
        return None

    def is_closed(self) -> bool:
        """Check the closure: every product of two elements is an element."""
        try:
            return all(
                _compose(a.action, b.action) in self._index
                for a in self.elements for b in self.elements
            )
        except IndexError:  # pragma: no cover - defensive
            return False

    def is_associative(self, max_size: int = 120) -> bool:
        """Check ``(a·b)·c = a·(b·c)`` for **every** triple of elements.

        Raises
        ------
        ValueError
            If there are more than ``max_size`` elements (the check is cubic).
        """
        self._check_size(max_size, "the associativity check")
        table = self.multiplication_table(max_size)
        n = len(self)
        return all(
            table[table[i][j]][k] == table[i][table[j][k]]
            for i in range(n) for j in range(n) for k in range(n)
        )

    def is_monoid(self, max_size: int = 120) -> bool:
        """Check closure, associativity and the existence of a neutral element."""
        return self.is_closed() and self.neutral_element() is not None and self.is_associative(max_size)


@dataclass(frozen=True)
class TransitionMonoid(TransformationMonoid):
    """Transition monoid ``T(A)`` of a DFA: transformations of its accessible states.

    Attributes
    ----------
    acted : tuple
        The accessible states of the completed DFA, ordered by their shortest
        access word (the sink added by the completion may appear).
    """

    @classmethod
    def from_dfa(cls, dfa: Any, max_elements: int = 100_000) -> TransitionMonoid:
        """Compute ``T(A)`` from a DFA (the DFA is not modified).

        The DFA is completed (implicit sink, function 21) and restricted to its
        accessible states, which are the only ones a word read from the
        initial state can visit.
        """
        full = dfa.complete()
        alphabet = sorted(full.input_symbols)
        order = [full.initial_state]
        seen = {full.initial_state}
        queue: deque = deque(order)
        while queue:
            state = queue.popleft()
            for symbol in alphabet:
                target = full.transitions[state][symbol]
                if target not in seen:
                    seen.add(target)
                    order.append(target)
                    queue.append(target)
        position = {q: i for i, q in enumerate(order)}
        generators = {
            a: tuple(position[full.transitions[q][a]] for q in order) for a in alphabet
        }
        return cls._make(alphabet, order, generators, max_elements)

    def explain(self) -> str:
        """Describe the method in plain words, with the numbers of this monoid."""
        return (
            f"Transition monoid: the {len(self.acted)} accessible states of the completed DFA are "
            f"transformed by each letter; closing these {len(self.alphabet)} generators under "
            f"composition gives {len(self)} distinct transformations f_w (f_u·f_v = f_uv). "
            "This monoid depends on the automaton: two transformations may be syntactically equal."
        )


@dataclass(frozen=True)
class SyntacticMonoid(TransformationMonoid):
    """Syntactic monoid ``M(L)``: ``Σ*`` modulo the syntactic congruence (immutable).

    Elements act on the classes of Myhill-Nerode (``acted`` holds the shortest
    word of each class).  Extra attributes:

    Attributes
    ----------
    partition : MyhillNerodePartition
        The partition (function 59) used to quotient the states.
    transition_monoid : TransitionMonoid
        The monoid of transformations of the automaton, of which this one is
        a quotient.
    projection : tuple of int
        ``projection[i]`` is the index in ``M(L)`` of the class of the
        ``i``-th element of ``transition_monoid`` (the quotient map).
    """

    partition: Any = field(default=None, repr=False, compare=False)
    transition_monoid: TransitionMonoid | None = field(default=None, repr=False, compare=False)
    projection: tuple[int, ...] = field(default=(), repr=False, compare=False)

    @classmethod
    def from_dfa(cls, dfa: Any, max_elements: int = 100_000) -> SyntacticMonoid:
        """Compute ``M(L)`` for the language of ``dfa`` (the DFA is not modified).

        Steps (see the module docstring for the justification):

        1. Myhill-Nerode partition of the states (reused: function 59);
        2. generators: the action of each letter on the classes;
        3. closure under composition: the distinct actions are the elements of
           ``M(L)`` (equal action ⟺ syntactically equivalent);
        4. the transition monoid is computed too, and the projection
           ``T(A) → M(L)`` is recorded to show the quotient.
        """
        partition = dfa.nerode_partition()
        alphabet = sorted(dfa.input_symbols)
        generators = {
            a: tuple(partition.transition(c, a).index for c in partition.classes) for a in alphabet
        }
        transition = TransitionMonoid.from_dfa(dfa, max_elements)
        syntactic = cls._make(
            alphabet, [c.representative for c in partition.classes], generators, max_elements,
            partition=partition, transition_monoid=transition,
        )
        projection = tuple(syntactic.element_of_word(e.word).index for e in transition.elements)
        return cls(
            syntactic.alphabet, syntactic.acted, syntactic.elements, syntactic._generators,
            syntactic._index, partition, transition, projection,
        )

    # ---- relation with the transition monoid -------------------------------------------

    def verify_projection(self, max_size: int = 200) -> bool:
        """Check that ``transition_monoid → M(L)`` is a surjective monoid morphism.

        It must send the identity to the identity, every product to the product
        of the images, and reach every element of ``M(L)``.
        """
        t = self.transition_monoid
        assert t is not None
        t._check_size(max_size, "the projection check")
        if set(self.projection) != set(range(len(self))):
            return False
        if self.projection[0] != 0:
            return False
        return all(
            self.projection[t.product(a, b).index]
            == self.product(self.projection[a.index], self.projection[b.index]).index
            for a in t.elements for b in t.elements
        )

    # ---- the language -------------------------------------------------------------------

    def accepting_elements(self) -> frozenset[int]:
        """Indices of the elements ``m`` such that ``[w] = m`` implies ``w ∈ L``.

        Element ``m`` is accepting iff it sends the class of the empty word
        (the initial class, point 0) to a final class.  ``M(L)`` recognizes
        ``L`` with this set.
        """
        finals = {c.index for c in self.partition.classes if c.is_final}
        return frozenset(e.index for e in self.elements if e.action[0] in finals)

    def recognizes(self, word: str) -> bool:
        """Tell whether ``word ∈ L``, using only the monoid (``[word]`` accepting)."""
        return self.element_of_word(word).index in self.accepting_elements()

    def explain(self) -> str:
        """Describe how the syntactic quotient was obtained, with this monoid's numbers."""
        t = self.transition_monoid
        assert t is not None
        return (
            "Syntactic monoid: u ≈ v iff x·u·y ∈ L ⟺ x·v·y ∈ L for all x, y. Instead of testing "
            "infinitely many contexts, the states of the DFA are grouped by Myhill-Nerode "
            f"equivalence ({len(self.partition)} classes: same future), and two words are "
            "equivalent iff they act identically on these classes. The distinct actions, found "
            f"by closing the {len(self.alphabet)} letters under composition, are the "
            f"{len(self)} elements. The transition monoid of the automaton has {len(t)} "
            f"transformations; the syntactic monoid is its quotient by this relation."
        )
