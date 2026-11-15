"""Derivation (parse) trees of regular grammars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["nonterminal", "terminal", "epsilon"]
EPSILON_LABEL = "ε"


@dataclass(frozen=True)
class Production:
    """A production ``head → body`` (immutable, hashable).

    ``body`` is a tuple of symbols; the empty tuple is the production
    ``head → ε``.
    """

    head: str
    body: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.head} → {' '.join(self.body) if self.body else EPSILON_LABEL}"


@dataclass(frozen=True)
class ParseTree:
    """Node of a derivation tree (immutable).

    Three kinds of nodes:

    * ``"nonterminal"``: an inner node; ``production`` is the production used
      to rewrite ``symbol`` and ``children`` are the symbols of its body, in
      order (an ``ε`` production has one ``"epsilon"`` child);
    * ``"terminal"``: a leaf holding a terminal symbol;
    * ``"epsilon"``: a leaf for the empty word (``symbol`` is ``"ε"``).
    """

    symbol: str
    kind: Kind
    production: Production | None = None
    children: tuple[ParseTree, ...] = ()

    # ---- constructors -------------------------------------------------------------

    @staticmethod
    def terminal(symbol: str) -> ParseTree:
        """Return a terminal leaf."""
        return ParseTree(symbol, "terminal")

    @staticmethod
    def epsilon() -> ParseTree:
        """Return the leaf of the empty word."""
        return ParseTree(EPSILON_LABEL, "epsilon")

    @staticmethod
    def node(production: Production, children: tuple[ParseTree, ...]) -> ParseTree:
        """Return the inner node of ``production`` with the given children."""
        return ParseTree(production.head, "nonterminal", production, tuple(children))

    # ---- queries ------------------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        """``True`` for terminal and epsilon leaves."""
        return self.kind != "nonterminal"

    def word(self) -> tuple[str, ...]:
        """Return the frontier: the derived word as a tuple of terminals."""
        if self.kind == "terminal":
            return (self.symbol,)
        if self.kind == "epsilon":
            return ()
        return tuple(t for child in self.children for t in child.word())

    def productions(self) -> tuple[Production, ...]:
        """Productions used, in derivation order (root first, then leftmost)."""
        if self.is_leaf:
            return ()
        found: tuple[Production, ...] = (self.production,)  # type: ignore[assignment]
        for child in self.children:
            found += child.productions()
        return found

    def height(self) -> int:
        """Number of edges of the longest path from this node to a leaf."""
        return 0 if not self.children else 1 + max(c.height() for c in self.children)

    def size(self) -> int:
        """Number of nodes of the tree."""
        return 1 + sum(c.size() for c in self.children)

    def derivation_steps(self) -> tuple[tuple[str, ...], ...]:
        """Sentential forms of the leftmost derivation, from the root symbol.

        Each step rewrites the leftmost nonterminal with the body of its
        production.  For the tree of ``S → aA, A → bB, B → ε`` this gives
        ``(S) (a A) (a b B) (a b)``; the last step erases ``B`` with
        ``B → ε``.  Works for right-linear and left-linear trees alike (in a
        regular grammar there is only one nonterminal at a time).
        """
        nodes: list[ParseTree] = [self]
        forms: list[tuple[str, ...]] = [(self.symbol,)]
        while True:
            index = next((i for i, n in enumerate(nodes) if n.kind == "nonterminal"), None)
            if index is None:
                break
            expanded = [c for c in nodes[index].children if c.kind != "epsilon"]
            nodes[index : index + 1] = expanded
            forms.append(tuple(n.symbol for n in nodes))
        return tuple(forms)

    # ---- rendering ----------------------------------------------------------------

    def __str__(self) -> str:
        if self.is_leaf:
            return self.symbol
        return f"{self.symbol}({' '.join(str(c) for c in self.children)})"

    def pretty(self) -> str:
        """Indented multi-line rendering (one node per line)."""
        lines: list[str] = []

        def walk(tree: ParseTree, depth: int) -> None:
            label = tree.symbol
            if tree.production is not None:
                label += f"    [{tree.production}]"
            lines.append("  " * depth + label)
            for child in tree.children:
                walk(child, depth + 1)

        walk(self, 0)
        return "\n".join(lines)

    def to_dot(self) -> str:
        """Return the tree in Graphviz DOT syntax (nonterminals boxed)."""
        out = ["digraph tree {", "  node [fontname=Helvetica];"]
        counter = [0]

        def quote(text: str) -> str:
            return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'

        def walk(tree: ParseTree) -> int:
            counter[0] += 1
            ident = counter[0]
            shape = "box" if tree.kind == "nonterminal" else "plaintext"
            out.append(f"  n{ident} [label={quote(tree.symbol)}, shape={shape}];")
            for child in tree.children:
                out.append(f"  n{ident} -> n{walk(child)};")
            return ident

        walk(self)
        out.append("}")
        return "\n".join(out)

    def to_svg(self) -> str:
        """Return the tree as a self-contained SVG document (no Graphviz needed).

        Layout: leaves are placed left to right in one row per depth level, an
        inner node is centred above its children.  Nonterminals are rounded
        boxes with the production in a tooltip, terminals are plain bold text
        and ``ε`` is grey.  Colours come from CSS variables with a dark-mode
        fallback, on an explicit background.
        """
        from xml.sax.saxutils import escape

        column = [0]
        placed: list[tuple[ParseTree, float, int, list[int]]] = []
        char = 9.0
        width = max(40.0, max(char * len(t.symbol) + 18 for t in self._walk()) + 20)

        def place(node: ParseTree, depth: int) -> float:
            if not node.children:
                x = column[0] * width + width / 2
                column[0] += 1
            else:
                xs = [place(child, depth + 1) for child in node.children]
                x = (xs[0] + xs[-1]) / 2
            placed.append((node, x, depth, []))
            return x

        place(self, 0)
        row = 64
        top = 30
        total_w = max(column[0], 1) * width
        total_h = (self.height() + 1) * row + 20
        position = {id(n): (x, top + d * row) for n, x, d, _ in placed}
        out = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w:g} {total_h:g}" '
            f'width="{total_w:g}" height="{total_h:g}" font-family="sans-serif" font-size="15">',
            "<style>:root{--bg:#ffffff;--fg:#1f2933;--nt:#e8eefc;--mute:#7b8794}"
            "@media (prefers-color-scheme: dark){:root{--bg:#16191d;--fg:#e4e7eb;"
            "--nt:#27324a;--mute:#9aa5b1}}"
            ".bg{fill:var(--bg)}.e{stroke:var(--fg);stroke-width:1.2}"
            ".nt{fill:var(--nt);stroke:var(--fg);stroke-width:1.2}"
            ".t{fill:var(--fg)}.m{fill:var(--mute);font-style:italic}</style>",
            f'<rect class="bg" width="{total_w:g}" height="{total_h:g}"/>',
        ]
        for node, x, d, _ in placed:
            for child in node.children:
                cx, cy = position[id(child)]
                out.append(f'<line class="e" x1="{x:g}" y1="{top + d * row + 14:g}" '
                           f'x2="{cx:g}" y2="{cy - 14:g}"/>')
        for node, x, d, _ in placed:
            y = top + d * row
            label = escape(node.symbol)
            if node.kind == "nonterminal":
                w = char * len(node.symbol) + 18
                tip = f"<title>{escape(str(node.production))}</title>"
                out.append(f'<g>{tip}<rect class="nt" x="{x - w / 2:g}" y="{y - 14:g}" '
                           f'width="{w:g}" height="28" rx="7"/>'
                           f'<text class="t" x="{x:g}" y="{y + 5:g}" text-anchor="middle">{label}</text></g>')
            elif node.kind == "epsilon":
                out.append(f'<text class="m" x="{x:g}" y="{y + 5:g}" text-anchor="middle">{label}</text>')
            else:
                out.append(f'<text class="t" x="{x:g}" y="{y + 5:g}" text-anchor="middle" '
                           f'font-weight="bold">{label}</text>')
        out.append("</svg>")
        return "\n".join(out)

    def _walk(self):
        yield self
        for child in self.children:
            yield from child._walk()
