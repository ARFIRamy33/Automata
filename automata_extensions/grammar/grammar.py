"""Right-linear (regular) grammars, conversions with automata and parsing.

Conventions of the project
--------------------------
* nonterminals and terminals are non-empty strings and the two sets are
  disjoint; ``ε`` is never a symbol, it is the empty body ``()``;
* a production is a :class:`Production` ``head → body`` with a body of one of
  the three accepted shapes ``()``, ``(a,)`` or ``(a, B)`` (``a`` terminal,
  ``B`` nonterminal);
* grammars are immutable; every operation returns a new object.

The classes :class:`RegularGrammar` (right-linear: ``A → aB``) and
:class:`LeftRegularGrammar` (left-linear: ``A → Ba``) refuse every other shape
with an explicit :class:`GrammarError`.  Grammars written in a more liberal way
(``A → aabB``, unit productions ``A → B``) can be rewritten into the strict form
with :meth:`RegularGrammar.from_linear`, which keeps the language.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence

from .tree import EPSILON_LABEL, ParseTree, Production


class GrammarError(ValueError):
    """The grammar is invalid or not regular (right-linear)."""


class NotDerivableError(ValueError):
    """The word cannot be derived from the grammar."""


def _parse_body(body: str | Sequence[str], heads: set[str]) -> tuple[str, ...]:
    """Turn a rule body into a tuple of symbols (``""`` and ``"ε"`` are empty).

    A string is read character by character; any other sequence is used as is.
    """
    if isinstance(body, str):
        return () if body in ("", EPSILON_LABEL) else tuple(body)
    return tuple(body)


def _split(word: Sequence[str]) -> tuple[str, ...]:
    """A ``str`` is read character by character, any other sequence as is."""
    return tuple(word)


@dataclass(frozen=True)
class _LinearGrammar:
    """Common part of right-linear and left-linear grammars ``G = (N, T, P, S)``.

    Parameters
    ----------
    nonterminals, terminals : frozenset[str]
        Disjoint sets of non-empty strings.
    productions : frozenset[Production]
        Productions of the accepted shapes (see the subclasses).
    start : str
        Start symbol, a nonterminal.

    Raises
    ------
    GrammarError
        At construction, if the grammar is not valid.
    """

    nonterminals: frozenset[str]
    terminals: frozenset[str]
    productions: frozenset[Production]
    start: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "nonterminals", frozenset(self.nonterminals))
        object.__setattr__(self, "terminals", frozenset(self.terminals))
        object.__setattr__(self, "productions", frozenset(self.productions))
        for name in (*self.nonterminals, *self.terminals):
            if not isinstance(name, str) or not name:
                raise GrammarError(f"symbols must be non-empty strings, got {name!r}.")
        clash = self.nonterminals & self.terminals
        if clash:
            raise GrammarError(f"symbols both terminal and nonterminal: {sorted(clash)}.")
        if self.start not in self.nonterminals:
            raise GrammarError(f"start symbol {self.start!r} is not a nonterminal.")
        for production in self.productions:
            if production.head not in self.nonterminals:
                raise GrammarError(f"{production}: the head is not a nonterminal.")
            for symbol in production.body:
                if symbol not in self.nonterminals and symbol not in self.terminals:
                    raise GrammarError(f"{production}: unknown symbol {symbol!r}.")
            self._check(production)

    def _check(self, production: Production) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- construction -------------------------------------------------------------

    @classmethod
    def from_rules(cls, start: str, rules: Mapping[str, Iterable[str | Sequence[str]]]):
        """Build a grammar from ``{head: [body, ...]}``.

        A body is a sequence of symbols, or a string read character by
        character (so nonterminals of a string body have one character).
        ``""`` and ``"ε"`` are the empty body.  Every key of ``rules`` is a
        nonterminal; every other symbol is a terminal.
        """
        heads = set(rules)
        productions: set[Production] = set()
        terminals: set[str] = set()
        for head, bodies in rules.items():
            for body in bodies:
                symbols = _parse_body(body, heads)
                terminals.update(s for s in symbols if s not in heads)
                productions.add(Production(head, symbols))
        return cls(frozenset(heads), frozenset(terminals), frozenset(productions), start)

    @classmethod
    def from_text(cls, text: str, start: str | None = None, *, normalize: bool = False):
        """Read a grammar written as text, one line per head.

        Format: ``S -> a A | b | ε`` (``→`` is accepted instead of ``->``).
        **Symbols of a body are separated by spaces**, so names of several
        characters are possible (``Expr -> if Expr``); ``ε`` or an empty
        alternative is the empty body; blank lines and lines starting with
        ``#`` are ignored; the same head may appear on several lines.  Every
        head is a nonterminal and every other symbol a terminal.  ``|`` cannot
        be a terminal in this format.

        Parameters
        ----------
        text : str
            The grammar.
        start : str, optional
            Start symbol; by default the head of the first line.
        normalize : bool
            If true, rewrite liberal rules into the strict form (see
            :meth:`RegularGrammar.from_linear`) instead of refusing them.

        Raises
        ------
        GrammarError
            On a malformed line or a non-regular rule.
        """
        rules: dict[str, list[tuple[str, ...]]] = {}
        for number, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            arrow = "->" if "->" in line else "→" if "→" in line else None
            if arrow is None:
                raise GrammarError(f"line {number}: missing '->' in {line!r}.")
            head, _, right = line.partition(arrow)
            head = head.strip()
            if not head or len(head.split()) != 1:
                raise GrammarError(f"line {number}: the head must be one symbol, got {head!r}.")
            for alternative in right.split("|"):
                tokens = alternative.split()
                body = () if tokens in ([], [EPSILON_LABEL]) else tuple(tokens)
                rules.setdefault(head, []).append(body)
        if not rules:
            raise GrammarError("the grammar text contains no production.")
        first = next(iter(rules))
        maker = cls.from_linear if normalize else cls.from_rules
        return maker(start if start is not None else first, rules)

    def productions_of(self, head: str) -> tuple[Production, ...]:
        """Productions of ``head`` in a fixed order (by body, ``ε`` first)."""
        return tuple(
            sorted((p for p in self.productions if p.head == head), key=lambda p: p.body)
        )

    def rules(self) -> dict[str, tuple[tuple[str, ...], ...]]:
        """Return ``{head: bodies}``: start symbol first, then sorted heads; bodies sorted."""
        heads = [self.start, *sorted(self.nonterminals - {self.start})]
        return {h: tuple(p.body for p in self.productions_of(h)) for h in heads}

    def __str__(self) -> str:
        lines = []
        for head, bodies in self.rules().items():
            if bodies:
                shown = " | ".join("".join(b) if b else EPSILON_LABEL for b in bodies)
                lines.append(f"{head} → {shown}")
        return "\n".join(lines)


def _flip(production: Production) -> Production:
    """Mirror a body: ``A → aB`` becomes ``A → Ba`` and conversely."""
    return Production(production.head, production.body[::-1])


@dataclass(frozen=True)
class RegularGrammar(_LinearGrammar):
    """Right-linear grammar: productions ``A → ε``, ``A → a`` and ``A → aB``."""

    def _check(self, production: Production) -> None:
        body = production.body
        if len(body) == 0:
            return
        if body[0] not in self.terminals:
            if len(body) == 1:
                raise GrammarError(
                    f"{production}: unit productions (A → B) are not supported "
                    "in a regular grammar (from_linear can remove them)."
                )
            raise GrammarError(
                f"{production}: not right-linear (a body must start with a "
                "terminal; use LeftRegularGrammar for left-linear rules)."
            )
        if len(body) == 1:
            return
        if len(body) == 2 and body[1] in self.nonterminals:
            return
        raise GrammarError(
            f"{production}: unsupported shape; only A → ε, A → a and A → aB are accepted "
            "(from_linear can split longer bodies)."
        )

    @classmethod
    def from_linear(cls, start: str, rules: Mapping[str, Iterable[str | Sequence[str]]]):
        """Build a strict grammar from liberal right-linear rules, keeping the language.

        Accepted bodies: any number of terminals, optionally followed by one
        nonterminal (``ε``, ``a``, ``abc``, ``aB``, ``abB``, ``B``).  The result
        only has ``A → ε``, ``A → a`` and ``A → aB``:

        1. a body ``t1 … tk [N]`` with ``k ≥ 2`` is cut with fresh nonterminals
           ``A → t1 X1``, ``X1 → t2 X2``, …, ``Xk-1 → tk [N]``;
        2. a unit production ``A → B`` is removed: ``A`` receives every other
           production of the nonterminals reachable from ``A`` by unit
           productions (so ``A → ε`` when one of them is nullable).

        Fresh names are ``<head>_1``, ``<head>_2``, … (never an existing symbol).

        Raises
        ------
        GrammarError
            If a nonterminal appears before the end of a body (not right-linear).
        """
        heads = set(rules)
        used = set(heads)
        strict: dict[str, set[tuple[str, ...]]] = {h: set() for h in heads}
        units: dict[str, set[str]] = {h: set() for h in heads}
        order = list(rules)

        def fresh(base: str) -> str:
            n = 1
            while f"{base}_{n}" in used:
                n += 1
            used.add(f"{base}_{n}")
            strict[f"{base}_{n}"] = set()
            units[f"{base}_{n}"] = set()
            return f"{base}_{n}"

        for head in order:
            for raw in rules[head]:
                body = _parse_body(raw, heads)
                for i, symbol in enumerate(body[:-1]):
                    if symbol in heads:
                        raise GrammarError(
                            f"{head} → {' '.join(body)}: nonterminal {symbol!r} before the "
                            "end of the body; the rule is not right-linear."
                        )
                tail = body[-1] if body and body[-1] in heads else None
                letters = body[:-1] if tail is not None else body
                if not letters:
                    if tail is None:
                        strict[head].add(())
                    elif tail != head:
                        units[head].add(tail)
                    continue
                current = head
                for position, letter in enumerate(letters):
                    last = position == len(letters) - 1
                    if last:
                        strict[current].add((letter,) if tail is None else (letter, tail))
                    else:
                        following = fresh(head)
                        strict[current].add((letter, following))
                        current = following
        # unit closure
        for head in order:
            reach, todo = {head}, [head]
            while todo:
                for nxt in units.get(todo.pop(), ()):
                    if nxt not in reach:
                        reach.add(nxt)
                        todo.append(nxt)
            gathered = set()
            for other in reach:
                gathered |= strict[other]
            strict[head] = gathered
        productions = {Production(h, b) for h, bodies in strict.items() for b in bodies}
        terminals = {s for p in productions for s in p.body if s not in strict}
        return cls(frozenset(strict), frozenset(terminals), frozenset(productions), start)

    def to_left_grammar(self) -> LeftRegularGrammar:
        """Return a left-linear grammar with the same language (through an NFA)."""
        return self.to_nfa().to_left_grammar()

    # ---- function 55: grammar -> automaton ---------------------------------------

    def _final_name(self) -> str:
        """Name of the extra final state, different from every nonterminal."""
        name = "F"
        while name in self.nonterminals:
            name += "'"
        return name

    def to_nfa(self):
        """Build an :class:`ExtendedNFA` recognizing the language of the grammar.

        States are the nonterminals plus one fresh final state (named ``F``,
        ``F'``, ... to avoid any collision).  Production by production:

        * ``A → aB``  gives the transition ``A --a--> B``;
        * ``A → a``   gives ``A --a--> F`` (``F`` is final);
        * ``A → ε``   makes ``A`` final.

        There is no epsilon transition in the result and the grammar is not
        modified.
        """
        from ..fa.nfa import ExtendedNFA

        final = self._final_name()
        table: dict[str, dict[str, set[str]]] = {n: {} for n in self.nonterminals}
        table[final] = {}
        finals = {final} if any(len(p.body) == 1 for p in self.productions) else set()
        for production in self.productions:
            body = production.body
            if not body:
                finals.add(production.head)
            else:
                target = body[1] if len(body) == 2 else final
                table[production.head].setdefault(body[0], set()).add(target)
        used_final = final in finals
        states = set(self.nonterminals) | ({final} if used_final else set())
        if not used_final:
            table.pop(final)
        return ExtendedNFA(
            states=states,
            input_symbols=set(self.terminals),
            transitions={s: {a: frozenset(t) for a, t in table[s].items()} for s in states},
            initial_state=self.start,
            final_states=finals,
        )

    def to_dfa(self, complete: bool = False):
        """Return a DFA for the language (subset construction of :meth:`to_nfa`)."""
        return self.to_nfa().to_dfa(complete=complete)

    # ---- function 56: derivation trees --------------------------------------------

    def _reach_table(self, word: tuple[str, ...]) -> list[dict[str, bool]]:
        """``table[i][A]`` is true iff ``A`` derives ``word[i:]`` (right to left DP)."""
        n = len(word)
        table: list[dict[str, bool]] = [dict() for _ in range(n + 1)]
        for i in range(n, -1, -1):
            for head in self.nonterminals:
                ok = False
                for p in self.productions_of(head):
                    b = p.body
                    if not b:
                        ok = i == n
                    elif len(b) == 1:
                        ok = i == n - 1 and word[i] == b[0]
                    else:
                        ok = i < n and word[i] == b[0] and table[i + 1][b[1]]
                    if ok:
                        break
                table[i][head] = ok
        return table

    def accepts(self, word: Sequence[str]) -> bool:
        """Return whether ``word`` belongs to the language of the grammar.

        ``word`` is a string (one terminal per character) or a sequence of
        terminals.  A symbol that is not a terminal makes the answer ``False``.
        Time ``O(|word| · |P|)``: no search, so recursive productions are safe.
        """
        w = _split(word)
        if any(s not in self.terminals for s in w):
            return False
        return self._reach_table(w)[0][self.start]

    def count_trees(self, word: Sequence[str]) -> int:
        """Number of distinct derivation trees of ``word`` (0 if not derivable)."""
        w = _split(word)
        if any(s not in self.terminals for s in w):
            return 0
        n = len(w)
        count: list[dict[str, int]] = [dict() for _ in range(n + 1)]
        for i in range(n, -1, -1):
            for head in self.nonterminals:
                total = 0
                for p in self.productions_of(head):
                    b = p.body
                    if not b:
                        total += int(i == n)
                    elif len(b) == 1:
                        total += int(i == n - 1 and w[i] == b[0])
                    elif i < n and w[i] == b[0]:
                        total += count[i + 1][b[1]]
                count[i][head] = total
        return count[0][self.start]

    def is_ambiguous_for(self, word: Sequence[str]) -> bool:
        """Return whether ``word`` has at least two derivation trees."""
        return self.count_trees(word) >= 2

    def parse_all(self, word: Sequence[str], limit: int | None = None) -> tuple[ParseTree, ...]:
        """Return the derivation trees of ``word`` (all, or at most ``limit``).

        Trees come in a fixed order (productions sorted by body).  The list is
        empty when the word is not derivable.  The search never explores a
        dead end (it is guided by the table of :meth:`_reach_table`) and every
        production with a nonterminal consumes one letter, so it always stops.
        """
        w = _split(word)
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative.")
        if limit == 0:
            return ()
        if any(s not in self.terminals for s in w):
            return ()
        table = self._reach_table(w)
        if not table[0][self.start]:
            return ()
        found: list[ParseTree] = []
        for tree in self._trees(self.start, 0, w, table):
            found.append(tree)
            if limit is not None and len(found) >= limit:
                break
        return tuple(found)

    def _trees(
        self, head: str, i: int, w: tuple[str, ...], table: list[dict[str, bool]]
    ) -> Iterator[ParseTree]:
        n = len(w)
        for p in self.productions_of(head):
            b = p.body
            if not b:
                if i == n:
                    yield ParseTree.node(p, (ParseTree.epsilon(),))
            elif len(b) == 1:
                if i == n - 1 and w[i] == b[0]:
                    yield ParseTree.node(p, (ParseTree.terminal(b[0]),))
            elif i < n and w[i] == b[0] and table[i + 1][b[1]]:
                for sub in self._trees(b[1], i + 1, w, table):
                    yield ParseTree.node(p, (ParseTree.terminal(b[0]), sub))

    def parse(self, word: Sequence[str]) -> ParseTree:
        """Return **one** derivation tree of ``word`` (the first in the fixed order).

        Policy for ambiguity: the method is deterministic and returns the first
        tree; use :meth:`parse_all` for every tree, or
        :meth:`count_trees` / :meth:`is_ambiguous_for` to detect ambiguity.

        Raises
        ------
        NotDerivableError
            If the word is not in the language (explicit message).
        """
        w = _split(word)
        unknown = [s for s in w if s not in self.terminals]
        if unknown:
            raise NotDerivableError(f"{unknown[0]!r} is not a terminal of the grammar.")
        trees = self.parse_all(w, limit=1)
        if not trees:
            shown = "".join(w) if all(len(s) == 1 for s in w) else " ".join(w)
            raise NotDerivableError(
                f"the word {shown or EPSILON_LABEL!r} cannot be derived from {self.start!r}."
            )
        return trees[0]

    def generate(self, max_length: int) -> frozenset[tuple[str, ...]]:
        """Return every word of the language of length at most ``max_length``."""
        words: set[tuple[str, ...]] = set()
        frontier: list[tuple[tuple[str, ...], str]] = [((), self.start)]
        for _ in range(max_length + 2):
            nxt: list[tuple[tuple[str, ...], str]] = []
            for prefix, head in frontier:
                for p in self.productions_of(head):
                    b = p.body
                    if not b:
                        words.add(prefix)
                    elif len(b) == 1:
                        if len(prefix) + 1 <= max_length:
                            words.add(prefix + b)
                    elif len(prefix) + 1 <= max_length:
                        nxt.append((prefix + (b[0],), b[1]))
            frontier = nxt
        return frozenset(words)




@dataclass(frozen=True)
class LeftRegularGrammar(_LinearGrammar):
    """Left-linear grammar: productions ``A → ε``, ``A → a`` and ``A → Ba``.

    Its language is the mirror image of the language of the right-linear
    grammar obtained by writing every body backwards (:meth:`reversed`); every
    operation below uses that fact, so the algorithms are not duplicated.
    """

    def _check(self, production: Production) -> None:
        body = production.body
        if len(body) == 0:
            return
        if len(body) == 1:
            if body[0] in self.terminals:
                return
            raise GrammarError(
                f"{production}: unit productions (A → B) are not supported "
                "in a regular grammar (from_linear can remove them)."
            )
        if len(body) == 2 and body[0] in self.nonterminals and body[1] in self.terminals:
            return
        if len(body) == 2 and body[0] in self.terminals:
            raise GrammarError(
                f"{production}: not left-linear (a body must end with a terminal; "
                "use RegularGrammar for right-linear rules)."
            )
        raise GrammarError(
            f"{production}: unsupported shape; only A → ε, A → a and A → Ba are accepted "
            "(from_linear can split longer bodies)."
        )

    @classmethod
    def from_linear(cls, start: str, rules: Mapping[str, Iterable[str | Sequence[str]]]):
        """Build a strict left-linear grammar from liberal rules, keeping the language.

        Same idea as :meth:`RegularGrammar.from_linear` with the mirrored shape:
        any nonterminal first, then any number of terminals (``B``, ``aB`` is
        ``Ba``, ``Bab``, ``ab``, ``ε``).  Longer bodies are cut with fresh
        nonterminals and unit productions are removed.
        """
        heads = set(rules)
        mirrored = {
            head: [_parse_body(body, heads)[::-1] for body in bodies]
            for head, bodies in rules.items()
        }
        try:
            right = RegularGrammar.from_linear(start, mirrored)
        except GrammarError as error:
            raise GrammarError(str(error).replace("right-linear", "left-linear")) from None
        return cls(
            right.nonterminals,
            right.terminals,
            frozenset(_flip(p) for p in right.productions),
            right.start,
        )

    def reversed(self) -> RegularGrammar:
        """Right-linear grammar with every body written backwards (mirror language)."""
        return RegularGrammar(
            self.nonterminals,
            self.terminals,
            frozenset(_flip(p) for p in self.productions),
            self.start,
        )

    def to_nfa(self):
        """Build an :class:`ExtendedNFA` recognizing the language of the grammar.

        States: the nonterminals plus a fresh initial state ``I`` (``I'``, … if
        needed).  ``A → Ba`` gives ``B --a--> A``; ``A → a`` gives
        ``I --a--> A``; ``A → ε`` gives ``I --ε--> A`` (removed afterwards by
        ``remove_epsilon_transitions``).  The final state is the start symbol.
        """
        from ..fa.nfa import ExtendedNFA

        initial = "I"
        while initial in self.nonterminals:
            initial += "'"
        table: dict[str, dict[str, set[str]]] = {n: {} for n in self.nonterminals}
        table[initial] = {}
        has_epsilon = False
        for p in self.productions:
            body = p.body
            if not body:
                table[initial].setdefault("", set()).add(p.head)
                has_epsilon = True
            elif len(body) == 1:
                table[initial].setdefault(body[0], set()).add(p.head)
            else:
                table[body[0]].setdefault(body[1], set()).add(p.head)
        nfa = ExtendedNFA(
            states=set(table),
            input_symbols=set(self.terminals),
            transitions={s: {a: frozenset(t) for a, t in out.items()} for s, out in table.items()},
            initial_state=initial,
            final_states={self.start},
        )
        return nfa.remove_epsilon_transitions() if has_epsilon else nfa

    def to_dfa(self, complete: bool = False):
        """Return a DFA for the language (subset construction of :meth:`to_nfa`)."""
        return self.to_nfa().to_dfa(complete=complete)

    def to_right_grammar(self) -> RegularGrammar:
        """Return a right-linear grammar with the same language (through an NFA)."""
        return self.to_nfa().to_right_grammar()

    def accepts(self, word: Sequence[str]) -> bool:
        """Return whether ``word`` is in the language (a left derivation of ``w`` is a
        right derivation of the mirror word)."""
        return self.reversed().accepts(_split(word)[::-1])

    def count_trees(self, word: Sequence[str]) -> int:
        """Number of distinct derivation trees of ``word`` (0 if not derivable)."""
        return self.reversed().count_trees(_split(word)[::-1])

    def is_ambiguous_for(self, word: Sequence[str]) -> bool:
        """Return whether ``word`` has at least two derivation trees."""
        return self.count_trees(word) >= 2

    @staticmethod
    def _mirror(tree: ParseTree) -> ParseTree:
        """Turn a tree of the reversed grammar into a tree of this grammar."""
        if tree.is_leaf:
            return tree
        return ParseTree.node(
            _flip(tree.production),  # type: ignore[arg-type]
            tuple(LeftRegularGrammar._mirror(c) for c in reversed(tree.children)),
        )

    def parse_all(self, word: Sequence[str], limit: int | None = None) -> tuple[ParseTree, ...]:
        """Return the derivation trees of ``word`` (all, or at most ``limit``).

        Children of a node follow the body of its production (``B`` before
        ``a`` for ``A → Ba``), so the frontier of a tree reads the word from
        left to right.
        """
        trees = self.reversed().parse_all(_split(word)[::-1], limit)
        return tuple(self._mirror(t) for t in trees)

    def parse(self, word: Sequence[str]) -> ParseTree:
        """Return one derivation tree of ``word`` (same policy as the right-linear class).

        Raises
        ------
        NotDerivableError
            If the word is not in the language.
        """
        w = _split(word)
        unknown = [s for s in w if s not in self.terminals]
        if unknown:
            raise NotDerivableError(f"{unknown[0]!r} is not a terminal of the grammar.")
        trees = self.parse_all(w, limit=1)
        if not trees:
            shown = "".join(w) if all(len(s) == 1 for s in w) else " ".join(w)
            raise NotDerivableError(
                f"the word {shown or EPSILON_LABEL!r} cannot be derived from {self.start!r}."
            )
        return trees[0]

    def generate(self, max_length: int) -> frozenset[tuple[str, ...]]:
        """Return every word of the language of length at most ``max_length``."""
        return frozenset(w[::-1] for w in self.reversed().generate(max_length))
