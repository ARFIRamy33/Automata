r"""Formal regular expressions: AST, parser, Brzozowski derivatives, automata.

Syntax (the one of the theory of automata, **not** Python's ``re``)
-------------------------------------------------------------------
The course statement asks for an AST and for derivatives but does not fix a
concrete syntax, and automata-lib's own syntax could not be checked here, so
this module defines its own, documented below.

========== =====================================================
``a``      a symbol: any single character not listed below
``ε``      the empty word (``()`` is accepted as well)
``∅``      the empty language
``r|s``    union
``rs``     concatenation (juxtaposition)
``r*``     Kleene star
``(r)``    grouping
``\x``     the character ``x`` as a literal symbol (``\*``, ``\(``...)
========== =====================================================

Priorities: ``*`` binds tighter than concatenation, which binds tighter
than ``|``.  Reserved characters: ``( ) | * + ? \ ε ∅``.  ``+`` and ``?`` are
**not** operators here (they are rejected with an explicit message, write
``rr*`` or ``ε|r``); use ``\+`` and ``\?`` for literal symbols.  Spaces are
ordinary symbols (they are not skipped).  The empty text is an error: write
``ε`` or ``∅``.

Normal form
-----------
Expressions are immutable and always built through :func:`union`,
:func:`concat` and :func:`star`, which apply only identities that preserve
the language:

* union: flattened, ``∅`` removed, duplicates removed, operands sorted
  (associativity, commutativity, idempotence, ``∅|r = r``);
* concatenation: flattened, ``ε`` removed, ``∅`` absorbs the whole product;
* star: ``∅* = ε* = ε`` and ``(r*)* = r*``.

Two expressions that differ only by these identities are *equal*, which
makes them usable as DFA states and guarantees that Brzozowski's
construction ends (finitely many derivatives modulo associativity,
commutativity and idempotence of the union).  Nothing else is simplified
(for instance ``r|r*`` is left as is): a wrong simplification would change
the language.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

RESERVED = frozenset("()|*+?\\ε∅")


class RegexSyntaxError(ValueError):
    """The text is not a valid regular expression (the message gives the position)."""


class Regex:
    """Base class of the regular-expression AST (see the module docstring)."""

    __slots__ = ()

    # --- to be defined by the node classes ------------------------------------

    def nullable(self) -> bool:
        """Return whether the empty word belongs to the language (``ε ∈ L``)."""
        raise NotImplementedError

    def derivative(self, symbol: str) -> Regex:
        """Return the Brzozowski derivative ``D_a(R) = { w | a w ∈ L(R) }``."""
        raise NotImplementedError

    def alphabet(self) -> frozenset[str]:
        """Return the symbols occurring in the expression."""
        raise NotImplementedError

    def _render(self, context: int) -> str:
        raise NotImplementedError

    # --- shared ----------------------------------------------------------------

    def __str__(self) -> str:
        return self._render(0)

    @staticmethod
    def parse(text: str) -> Regex:
        """Parse a text with the syntax of the module (``RegexSyntaxError`` if invalid)."""
        return _Parser(text).parse()

    def brzozowski_dfa(self, alphabet: Iterable[str] | None = None, max_states: int = 10_000):
        """Build the Brzozowski automaton (function 50).

        * the initial state is the expression itself;
        * ``δ(R, a) = D_a(R)`` (expressions in normal form are the states, so
          two derivatives that are equal modulo the normal form give the
          **same** state);
        * a state ``R`` is final iff ``ε ∈ L(R)`` (:meth:`nullable`).

        Only the derivatives reachable from the initial state are built
        (breadth-first).  The automaton is **complete** over ``alphabet``:
        the sink state ``∅`` appears as soon as some derivative is empty.

        Parameters
        ----------
        alphabet : iterable of str, optional
            Symbols of the DFA; it must contain every symbol of the
            expression.  Default: the symbols of the expression.  Extra
            symbols lead to ``∅``.
        max_states : int
            Safety limit against runaway constructions.

        Returns
        -------
        ExtendedDFA
            States are :class:`Regex` objects (displayed as text by the
            exports).

        Raises
        ------
        ValueError
            If ``alphabet`` misses a symbol of the expression, or if the
            library refuses an empty alphabet (expression without symbol:
            pass an alphabet).
        RuntimeError
            If more than ``max_states`` states would be created.
        """
        from ..fa.dfa import ExtendedDFA

        symbols = self._alphabet_argument(alphabet)
        transitions: dict[Regex, dict[str, Regex]] = {self: {}}
        queue: deque[Regex] = deque([self])
        while queue:
            current = queue.popleft()
            for symbol in symbols:
                target = current.derivative(symbol)
                transitions[current][symbol] = target
                if target not in transitions:
                    if len(transitions) >= max_states:
                        raise RuntimeError(
                            f"More than {max_states} derivatives: construction stopped."
                        )
                    transitions[target] = {}
                    queue.append(target)
        try:
            return ExtendedDFA(
                states=frozenset(transitions),
                input_symbols=frozenset(symbols),
                transitions=transitions,
                initial_state=self,
                final_states={r for r in transitions if r.nullable()},
            )
        except Exception as error:  # e.g. the library refuses an empty alphabet
            if not symbols:
                raise ValueError(
                    "The library refused an automaton without symbols; pass a "
                    f"non-empty alphabet ({error})."
                ) from error
            raise

    def to_nfa(self, alphabet: Iterable[str] | None = None):
        """Build an epsilon-NFA by Thompson's construction (function 51).

        Each sub-expression gives a fragment with **one** start state and
        **one** accepting state, and no transition enters the start or
        leaves the accepting state.  With ``s, f`` the new states of a
        fragment:

        * ``∅``: ``s`` and ``f`` without any transition;
        * ``ε``: ``s --ε--> f``;
        * symbol ``a``: ``s --a--> f``;
        * ``R|S``: ``s --ε--> start(R), start(S)`` and
          ``accept(R), accept(S) --ε--> f``;
        * ``RS``: ``accept(R) --ε--> start(S)``, start ``start(R)``, accept
          ``accept(S)``;
        * ``R*``: ``s --ε--> f``, ``s --ε--> start(R)``,
          ``accept(R) --ε--> start(R)``, ``accept(R) --ε--> f``.

        States are consecutive integers from a single counter, so two
        generated states never collide.  The empty word is the key ``""``
        of automata-lib.  The expression is not modified.

        Returns
        -------
        ExtendedNFA
            Initial state: the start of the whole fragment; one final state.
        """
        from ..fa.nfa import ExtendedNFA

        symbols = self._alphabet_argument(alphabet)
        transitions: dict[int, dict[str, set[int]]] = {}
        counter = [0]

        def new_state() -> int:
            state = counter[0]
            counter[0] += 1
            transitions[state] = {}
            return state

        def link(source: int, symbol: str, target: int) -> None:
            transitions[source].setdefault(symbol, set()).add(target)

        def build(node: Regex) -> tuple[int, int]:
            if isinstance(node, Concat):
                fragments = [build(part) for part in node.parts]
                for (_, accept), (start, _) in zip(fragments, fragments[1:]):
                    link(accept, "", start)
                return fragments[0][0], fragments[-1][1]
            start, accept = new_state(), new_state()
            if isinstance(node, Epsilon):
                link(start, "", accept)
            elif isinstance(node, Symbol):
                link(start, node.char, accept)
            elif isinstance(node, Union):
                for part in node.parts:
                    inner_start, inner_accept = build(part)
                    link(start, "", inner_start)
                    link(inner_accept, "", accept)
            elif isinstance(node, Star):
                inner_start, inner_accept = build(node.inner)
                link(start, "", accept)
                link(start, "", inner_start)
                link(inner_accept, "", inner_start)
                link(inner_accept, "", accept)
            # Empty: the two states stay disconnected.
            return start, accept

        start, accept = build(self)
        try:
            return ExtendedNFA(
                states=frozenset(transitions),
                input_symbols=frozenset(symbols),
                transitions={
                    s: {a: frozenset(t) for a, t in out.items()}
                    for s, out in transitions.items()
                },
                initial_state=start,
                final_states={accept},
            )
        except Exception as error:
            if not symbols:
                raise ValueError(
                    "The library refused an automaton without symbols; pass a "
                    f"non-empty alphabet ({error})."
                ) from error
            raise

    def same_language_as(self, other: Regex, alphabet: Iterable[str] | None = None) -> bool:
        """Return whether two expressions denote the **same language**.

        This is semantic, unlike ``==`` (syntactic, up to the normal form):
        ``a*`` and ``ε|aa*`` are different objects with the same language.
        Both Brzozowski automata (function 50) are built over the union of the
        alphabets and compared with ``ExtendedDFA.has_same_language_as``
        (function 25); nothing is re-implemented here.

        Parameters
        ----------
        other : Regex
            The second expression.
        alphabet : iterable of str, optional
            Extra symbols (never changes the answer).
        """
        symbols = sorted(set(self.alphabet()) | set(other.alphabet()) | set(alphabet or ()))
        if not symbols:  # no symbol at all: only ∅ and ε are possible
            return self.nullable() == other.nullable()
        return self.brzozowski_dfa(symbols).has_same_language_as(other.brzozowski_dfa(symbols))

    def residual_automaton(
        self,
        alphabet: Iterable[str] | None = None,
        complete: bool = True,
        max_states: int = 10_000,
    ):
        """Build the residual automaton of the language (function 60).

        See :func:`automata_extensions.regex.residual.residual_automaton`.
        """
        from .residual import residual_automaton

        return residual_automaton(self, alphabet, complete, max_states)

    def _alphabet_argument(self, alphabet: Iterable[str] | None) -> list[str]:
        own = self.alphabet()
        if alphabet is None:
            return sorted(own)
        given = set(alphabet)
        if not own <= given:
            raise ValueError(
                f"The alphabet must contain the symbols of the expression: missing {sorted(own - given)}."
            )
        return sorted(given)


@dataclass(frozen=True)
class Empty(Regex):
    """``∅``: the empty language."""

    def nullable(self) -> bool:
        return False

    def derivative(self, symbol: str) -> Regex:
        return EMPTY  # D_a(∅) = ∅

    def alphabet(self) -> frozenset[str]:
        return frozenset()

    def _render(self, context: int) -> str:
        return "∅"


@dataclass(frozen=True)
class Epsilon(Regex):
    """``ε``: the language containing only the empty word."""

    def nullable(self) -> bool:
        return True

    def derivative(self, symbol: str) -> Regex:
        return EMPTY  # D_a(ε) = ∅

    def alphabet(self) -> frozenset[str]:
        return frozenset()

    def _render(self, context: int) -> str:
        return "ε"


@dataclass(frozen=True)
class Symbol(Regex):
    """A single symbol ``a``."""

    char: str

    def __post_init__(self) -> None:
        if not isinstance(self.char, str) or len(self.char) != 1:
            raise ValueError(f"A symbol is one character, got {self.char!r}.")

    def nullable(self) -> bool:
        return False

    def derivative(self, symbol: str) -> Regex:
        return EPSILON if symbol == self.char else EMPTY  # D_a(b) = ε if a = b else ∅

    def alphabet(self) -> frozenset[str]:
        return frozenset([self.char])

    def _render(self, context: int) -> str:
        return ("\\" + self.char) if self.char in RESERVED else self.char


@dataclass(frozen=True)
class Union(Regex):
    """``R1|R2|...`` (build it with :func:`union`)."""

    parts: tuple[Regex, ...]

    def nullable(self) -> bool:
        return any(part.nullable() for part in self.parts)

    def derivative(self, symbol: str) -> Regex:
        # D_a(R|S) = D_a(R) | D_a(S)
        return union(*(part.derivative(symbol) for part in self.parts))

    def alphabet(self) -> frozenset[str]:
        return frozenset().union(*(part.alphabet() for part in self.parts))

    def _render(self, context: int) -> str:
        text = "|".join(part._render(0) for part in self.parts)
        return f"({text})" if context >= 1 else text


@dataclass(frozen=True)
class Concat(Regex):
    """``R1 R2 ...`` (build it with :func:`concat`)."""

    parts: tuple[Regex, ...]

    def nullable(self) -> bool:
        return all(part.nullable() for part in self.parts)

    def derivative(self, symbol: str) -> Regex:
        # D_a(R S) = D_a(R) S  |  (D_a(S) if ε ∈ L(R) else ∅)
        head, tail = self.parts[0], concat(*self.parts[1:])
        first = concat(head.derivative(symbol), tail)
        if head.nullable():
            return union(first, tail.derivative(symbol))
        return first

    def alphabet(self) -> frozenset[str]:
        return frozenset().union(*(part.alphabet() for part in self.parts))

    def _render(self, context: int) -> str:
        text = "".join(part._render(1) for part in self.parts)
        return f"({text})" if context >= 2 else text


@dataclass(frozen=True)
class Star(Regex):
    """``R*`` (build it with :func:`star`)."""

    inner: Regex

    def nullable(self) -> bool:
        return True

    def derivative(self, symbol: str) -> Regex:
        # D_a(R*) = D_a(R) R*
        return concat(self.inner.derivative(symbol), self)

    def alphabet(self) -> frozenset[str]:
        return self.inner.alphabet()

    def _render(self, context: int) -> str:
        return self.inner._render(2) + "*"


EMPTY = Empty()
EPSILON = Epsilon()


def _sort_key(node: Regex) -> tuple[str, str]:
    return (type(node).__name__, str(node))


def union(*parts: Regex) -> Regex:
    """Union in normal form (flattened, without ``∅``, deduplicated, sorted)."""
    flat: dict[Regex, None] = {}
    for part in parts:
        for item in part.parts if isinstance(part, Union) else (part,):
            if not isinstance(item, Empty):
                flat[item] = None
    ordered = sorted(flat, key=_sort_key)
    if not ordered:
        return EMPTY
    return ordered[0] if len(ordered) == 1 else Union(tuple(ordered))


def concat(*parts: Regex) -> Regex:
    """Concatenation in normal form (flattened, without ``ε``; ``∅`` absorbs)."""
    flat: list[Regex] = []
    for part in parts:
        if isinstance(part, Empty):
            return EMPTY
        flat.extend(part.parts if isinstance(part, Concat) else (part,))
    flat = [item for item in flat if not isinstance(item, Epsilon)]
    if not flat:
        return EPSILON
    return flat[0] if len(flat) == 1 else Concat(tuple(flat))


def star(inner: Regex) -> Regex:
    """Star in normal form (``∅* = ε* = ε``, ``(R*)* = R*``)."""
    if isinstance(inner, (Empty, Epsilon)):
        return EPSILON
    return inner if isinstance(inner, Star) else Star(inner)


class _Parser:
    """Recursive-descent parser: union > concatenation > star > atom."""

    def __init__(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("A regular expression is given as a string.")
        self.text = text
        self.pos = 0

    def _peek(self) -> str | None:
        return self.text[self.pos] if self.pos < len(self.text) else None

    def _error(self, message: str) -> RegexSyntaxError:
        return RegexSyntaxError(f"{message} (position {self.pos} in {self.text!r}).")

    def parse(self) -> Regex:
        if self.text == "":
            raise RegexSyntaxError(
                "Empty text: write ε (or ()) for the empty word and ∅ for the empty language."
            )
        result = self._union()
        if self._peek() is not None:  # only an unmatched ')' can stop the union
            raise self._error("Unmatched ')'")
        return result

    def _union(self) -> Regex:
        alternatives = [self._concat()]
        while self._peek() == "|":
            self.pos += 1
            alternatives.append(self._concat())
        return union(*alternatives)

    def _concat(self) -> Regex:
        factors: list[Regex] = []
        while self._peek() not in (None, "|", ")"):
            factors.append(self._star())
        if not factors:
            raise self._error("Empty alternative: use ε for the empty word")
        return concat(*factors)

    def _star(self) -> Regex:
        node = self._atom()
        while self._peek() == "*":
            self.pos += 1
            node = star(node)
        return node

    def _atom(self) -> Regex:
        char = self._peek()
        if char == "(":
            self.pos += 1
            if self._peek() == ")":
                self.pos += 1
                return EPSILON
            if self._peek() is None:
                raise self._error("Missing ')'")
            inner = self._union()
            if self._peek() != ")":
                raise self._error("Missing ')'")
            self.pos += 1
            return inner
        if char == "*":
            raise self._error("Nothing to repeat before '*'")
        if char in ("+", "?"):
            raise self._error(
                f"The operator {char!r} is not supported: write rr* or ε|r, or \\{char} for the symbol"
            )
        if char == "\\":
            self.pos += 1
            escaped = self._peek()
            if escaped is None:
                raise self._error("Dangling '\\' at the end")
            self.pos += 1
            return Symbol(escaped)
        self.pos += 1
        if char == "ε":
            return EPSILON
        if char == "∅":
            return EMPTY
        return Symbol(char)
