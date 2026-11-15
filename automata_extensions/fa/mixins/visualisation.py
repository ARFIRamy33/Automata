"""Exports of an automaton as DOT, SVG, PDF and TikZ.

Which tool does what
--------------------
* ``dot()`` and ``latex_tikz()`` are **pure Python**: they need no Graphviz and
  no LaTeX installed (functions 40 and 41, TikZ part).
* ``svg()`` and ``pdf()`` call the ``dot`` program of Graphviz on the text of
  ``dot()``; if the program is missing a :class:`GraphvizNotFoundError` is
  raised and **no file is written** (functions 41, SVG and PDF).

automata-lib 9.2.0 has its own ``show_diagram`` (it needs the optional
``pygraphviz`` package and a Graphviz installation); it is not replaced, and
none of the names used here hides it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .analyse import State


class GraphvizNotFoundError(RuntimeError):
    """The Graphviz ``dot`` program is not installed (or not on the PATH)."""


class VisualisationMixin:
    """Export methods; they only read the automaton, never modify it.

    The layout of ``transitions`` is that of automata-lib
    (``{state: {symbol: target(s)}}``); an epsilon transition has the key
    ``""`` and is drawn as ``ε``.  The hooks ``_targets`` and ``_EPSILON`` of
    :class:`AnalyseMixin` are used, so a DFA and an NFA are drawn by the same
    code.
    """

    # --- shared helpers ---------------------------------------------------------

    def _state_text(self, state: State) -> str:
        """Readable text of a state name (sets and tuples are expanded).

        ``frozenset`` names (made by ``determinize`` and ``minimize``) are
        shown as ``{q0, q1}`` with sorted members; the empty set is ``∅``.
        """
        if isinstance(state, (set, frozenset)):
            if not state:
                return "∅"
            inner = sorted((self._state_text(item) for item in state))
            return "{" + ", ".join(inner) + "}"
        if isinstance(state, tuple):
            return "(" + ", ".join(self._state_text(item) for item in state) + ")"
        return str(state)

    def _ordered_states(self) -> list[State]:
        """Return the states in a reproducible order (initial state first)."""
        others = sorted(
            (s for s in self.states if s != self.initial_state), key=repr
        )
        return [self.initial_state] + others

    def _edges(self) -> dict[tuple[State, State], list[str]]:
        """Group the transitions: ``(source, target) -> sorted symbol texts``.

        Several symbols between the same two states give one edge with a
        label such as ``a, b``; epsilon is written ``ε``.
        """
        grouped: dict[tuple[State, State], set[str]] = {}
        for source in self._ordered_states():
            for symbol, value in self.transitions.get(source, {}).items():
                text = "ε" if symbol == self._EPSILON else str(symbol)
                for target in self._targets(value):
                    grouped.setdefault((source, target), set()).add(text)
        return {
            edge: sorted(texts, key=lambda t: (t == "ε", t))
            for edge, texts in grouped.items()
        }

    @staticmethod
    def _check_output_path(path: str | Path) -> Path:
        """Return ``path`` as a ``Path``; its directory must exist.

        Nothing is created implicitly: a typo in the directory is reported
        instead of silently creating folders.

        Raises
        ------
        FileNotFoundError
            If the parent directory does not exist.
        IsADirectoryError
            If ``path`` is an existing directory.
        """
        target = Path(path)
        if target.is_dir():
            raise IsADirectoryError(f"{target} is a directory, not a file.")
        if not target.parent.is_dir():
            raise FileNotFoundError(
                f"The directory {target.parent} does not exist: cannot write {target.name}."
            )
        return target

    # --- function 40: DOT ---------------------------------------------------------

    @staticmethod
    def _dot_escape(text: str) -> str:
        """Escape a text for a double-quoted DOT string."""
        return (
            text.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "")
        )

    def dot(self, path: str | Path | None = None, name: str = "automaton") -> str:
        """Return the automaton in the DOT language (and optionally save it).

        * Every state is a node ``n0, n1, ...`` (identifiers are generated, so
          any state name is safe) whose ``label`` is the escaped state name;
          backslashes, double quotes and line breaks are escaped.
        * A final state is a ``doublecircle``, the others are ``circle``.
        * The initial state receives an arrow coming from an invisible point.
        * Transitions between the same two states are grouped in one edge
          (``a, b``); epsilon is shown as ``ε``.
        * ``rankdir=LR`` draws the automaton from left to right.

        Parameters
        ----------
        path : str or Path, optional
            If given, the text is also written there (UTF-8).  The directory
            must exist.
        name : str
            Name of the graph.

        Returns
        -------
        str
            The DOT source.

        Raises
        ------
        FileNotFoundError
            If the directory of ``path`` does not exist.

        Complexity
        ----------
        O(|Q| + |E|) (plus sorting).
        """
        target = self._check_output_path(path) if path is not None else None
        ids = {state: f"n{index}" for index, state in enumerate(self._ordered_states())}
        esc = self._dot_escape
        lines = [
            f'digraph "{esc(name)}" {{',
            "  rankdir=LR;",
            '  node [shape=circle, fontname="Helvetica"];',
            '  edge [fontname="Helvetica"];',
            '  __start__ [shape=point, label="", width=0.1];',
        ]
        for state, node in ids.items():
            shape = "doublecircle" if state in self.final_states else "circle"
            label = esc(self._state_text(state))
            lines.append(f'  {node} [shape={shape}, label="{label}"];')
        lines.append(f"  __start__ -> {ids[self.initial_state]};")
        for (source, dest), symbols in self._edges().items():
            label = esc(", ".join(symbols))
            lines.append(f'  {ids[source]} -> {ids[dest]} [label="{label}"];')
        lines.append("}")
        text = "\n".join(lines) + "\n"
        if target is not None:
            target.write_text(text, encoding="utf-8")
        return text

    # --- function 41: SVG, PDF (Graphviz) and TikZ (pure Python) --------------------

    def _graphviz(self, output_format: str) -> bytes:
        """Run ``dot -T<format>`` on :meth:`dot` and return its output.

        Raises
        ------
        GraphvizNotFoundError
            If the ``dot`` program cannot be found.
        RuntimeError
            If Graphviz fails.
        """
        executable = shutil.which("dot")
        if executable is None:
            raise GraphvizNotFoundError(
                "The Graphviz program 'dot' was not found. Install Graphviz "
                "(Windows: winget install Graphviz.Graphviz; Linux: apt install "
                "graphviz) and make sure 'dot' is on the PATH.  dot() and "
                "latex_tikz() work without it."
            )
        result = subprocess.run(
            [executable, f"-T{output_format}"],
            input=self.dot().encode("utf-8"),
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Graphviz failed: " + result.stderr.decode("utf-8", "replace")
            )
        return result.stdout

    def svg(self, path: str | Path | None = None) -> str:
        """Return a real SVG drawing (made by Graphviz) and optionally save it.

        The file is written only after Graphviz succeeded, so a failure never
        leaves a fake or empty file.

        Raises
        ------
        GraphvizNotFoundError
            If Graphviz is not installed.
        FileNotFoundError
            If the directory of ``path`` does not exist.
        """
        target = self._check_output_path(path) if path is not None else None
        text = self._graphviz("svg").decode("utf-8")
        if target is not None:
            target.write_text(text, encoding="utf-8")
        return text

    def pdf(self, path: str | Path) -> Path:
        """Write a real PDF drawing (made by Graphviz) and return its path.

        Raises
        ------
        GraphvizNotFoundError
            If Graphviz is not installed.
        FileNotFoundError
            If the directory of ``path`` does not exist.
        """
        target = self._check_output_path(path)
        target.write_bytes(self._graphviz("pdf"))
        return target

    @staticmethod
    def _latex_escape(text: str) -> str:
        """Escape the special characters of LaTeX in a text."""
        table = {
            "\\": r"\textbackslash{}",
            "{": r"\{",
            "}": r"\}",
            "#": r"\#",
            "$": r"\$",
            "%": r"\%",
            "&": r"\&",
            "_": r"\_",
            "^": r"\^{}",
            "~": r"\~{}",
            '"': '\\texttt{"}',
        }
        return "".join(table.get(char, char) for char in text)

    def _tikz_state_text(self, state: State) -> str:
        """LaTeX text of a state (``∅`` becomes ``$\\emptyset$``)."""
        if isinstance(state, (set, frozenset)):
            if not state:
                return r"$\emptyset$"
            inner = sorted(self._state_text(item) for item in state)
            return r"\{" + ", ".join(self._latex_escape(t) for t in inner) + r"\}"
        return self._latex_escape(self._state_text(state))

    def _tikz_positions(self) -> dict[State, tuple[int, int]]:
        """Layout: breadth-first layers from the initial state, left to right."""
        layer = {self.initial_state: 0}
        order = [self.initial_state]
        successors: dict[State, list[State]] = {}
        for (source, dest) in self._edges():
            successors.setdefault(source, []).append(dest)
        index = 0
        while index < len(order):
            current = order[index]
            index += 1
            for nxt in successors.get(current, []):
                if nxt not in layer:
                    layer[nxt] = layer[current] + 1
                    order.append(nxt)
        extra = max(layer.values()) + 1
        for state in self._ordered_states():
            if state not in layer:  # inaccessible states: one more column
                layer[state] = extra
                order.append(state)
        rows: dict[int, int] = {}
        positions = {}
        for state in order:
            row = rows.get(layer[state], 0)
            rows[layer[state]] = row + 1
            positions[state] = (layer[state], row)
        return positions

    def latex_tikz(self, path: str | Path | None = None, standalone: bool = False) -> str:
        """Return TikZ code (``tikzpicture``) for LaTeX, and optionally save it.

        Written directly in Python: no Graphviz is needed.  The picture uses
        the TikZ libraries ``automata`` and ``positioning``: add
        ``\\usetikzlibrary{automata,positioning}`` to the preamble, or use
        ``standalone=True`` to get a complete document that compiles as is.

        Layout: columns follow the breadth-first distance from the initial
        state; a pair of opposite edges is drawn with ``bend left``, a loop
        with ``loop above``.  Special characters of LaTeX are escaped and
        epsilon is ``$\\varepsilon$``.

        Raises
        ------
        FileNotFoundError
            If the directory of ``path`` does not exist.
        """
        target = self._check_output_path(path) if path is not None else None
        positions = self._tikz_positions()
        ids = {state: f"s{i}" for i, state in enumerate(positions)}
        edges = self._edges()
        lines = [
            r"\begin{tikzpicture}[->, >=latex, shorten >=1pt, auto, semithick, "
            r"initial text=]"
        ]
        for state, (column, row) in positions.items():
            options = ["state"]
            if state == self.initial_state:
                options.append("initial")
            if state in self.final_states:
                options.append("accepting")
            lines.append(
                f"  \\node[{', '.join(options)}] ({ids[state]}) "
                f"at ({3 * column},{-2 * row}) {{{self._tikz_state_text(state)}}};"
            )
        for (source, dest), symbols in edges.items():
            label = ", ".join(
                r"$\varepsilon$" if s == "ε" else self._latex_escape(s) for s in symbols
            )
            if source == dest:
                style = "[loop above]"
            elif (dest, source) in edges:
                style = "[bend left]"
            else:
                style = ""
            lines.append(
                f"  \\path ({ids[source]}) edge{style} node {{{label}}} ({ids[dest]});"
            )
        lines.append(r"\end{tikzpicture}")
        text = "\n".join(lines) + "\n"
        if standalone:
            text = (
                "\\documentclass[tikz,border=5pt]{standalone}\n"
                "\\usetikzlibrary{automata,positioning}\n"
                "\\begin{document}\n" + text + "\\end{document}\n"
            )
        if target is not None:
            target.write_text(text, encoding="utf-8")
        return text
