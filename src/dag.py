"""Build the project DAG programmatically.

Defining the DAG in code (rather than as a static drawing) means:
    * the adjustment set can be derived automatically with NetworkX
    * the figure regenerates if the DAG changes
    * the conditional independencies are testable
    * the assumptions are version-controlled and code-reviewable

Usage
-----
    python -m src.dag                  # prints adjustment set + saves figure
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from .paths import FIGURES_DIR

# ----------------------------------------------------------------------------
# DAG definition
# ----------------------------------------------------------------------------
# Encoded as a list of directed edges. Symbols match docs/01_treatment_and_dag.md.

EDGES: list[tuple[str, str]] = [
    # Confounders -> treatment
    ("C", "T"),     # category drives basket subtotal -> eligibility
    ("S", "T"),     # large sellers push larger baskets
    ("G", "T"),     # geography affects baseline freight, hence eligibility
    ("M", "T"),     # season affects basket sizes
    # Confounders -> outcome
    ("C", "Y"),
    ("S", "Y"),
    ("G", "Y"),
    ("M", "Y"),
    # Treatment -> outcome (the path of interest)
    ("T", "Y"),
    # Treatment -> mediators -> outcome (post-treatment, do NOT condition)
    ("T", "F"),     # treatment lowers freight
    ("F", "Y"),
    ("T", "B"),     # treatment incentivises larger baskets
    ("B", "D"),     # bigger baskets take longer to ship
    ("D", "Y"),
    # Collider - caused by both T (review prompts vary by treatment) and Y
    ("T", "R"),
    ("Y", "R"),
]

CONFOUNDERS: set[str] = {"C", "S", "G", "M"}
MEDIATORS:   set[str] = {"F", "B", "D"}
COLLIDERS:   set[str] = {"R"}
EXPOSURE:   str = "T"
OUTCOME:    str = "Y"


# ----------------------------------------------------------------------------
# DAG analysis
# ----------------------------------------------------------------------------
def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(EDGES)
    return g


def adjustment_set(g: nx.DiGraph) -> set[str]:
    """The minimal adjustment set under our assumptions.

    Standard backdoor recipe: condition on every fork-style confounder of (T, Y). Do NOT
    condition on mediators or colliders. We compute it explicitly rather
    than calling networkx's algorithm so the logic is auditable.
    """
    confounders_of_pair: set[str] = set()
    for v in g.nodes:
        if v in (EXPOSURE, OUTCOME):
            continue
        # A fork on (T, Y) is a node with directed paths to BOTH T and Y.
        if (
            nx.has_path(g, v, EXPOSURE)
            and nx.has_path(g, v, OUTCOME)
            # exclude colliders (no incoming edges from T or Y)
            and EXPOSURE not in nx.ancestors(g, v)
            and OUTCOME not in nx.ancestors(g, v)
        ):
            confounders_of_pair.add(v)
    return confounders_of_pair


def implied_independencies(g: nx.DiGraph) -> list[tuple[str, str, frozenset[str]]]:
    """Return (a, b, conditioning_set) triples that should be independent."""
    out: list[tuple[str, str, frozenset[str]]] = []
    nodes = sorted(g.nodes)
    for a, b in product(nodes, nodes):
        if a >= b or b in nx.descendants(g, a) or a in nx.descendants(g, b):
            continue
        # Try conditioning on the parents of both - a sufficient set.
        cond = frozenset(set(g.predecessors(a)) | set(g.predecessors(b))) - {a, b}
        # NetworkX renamed d_separated -> is_d_separator in v3.3+. Try both
        # so the script keeps working under either version.
        d_sep_fn = getattr(nx, "is_d_separator", None) or nx.d_separated
        if not d_sep_fn(g, {a}, {b}, cond):
            continue
        out.append((a, b, cond))
    return out


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
NODE_LABELS = {
    "T": "T\ntreatment",
    "Y": "Y\noutcome",
    "C": "C\ncategory",
    "S": "S\nseller tier",
    "G": "G\ngeography",
    "M": "M\nmonth",
    "F": "F\nfreight",
    "B": "B\nbasket size",
    "D": "D\ndelivery days",
    "R": "R\nreview\nleft",
}

NODE_POSITIONS = {
    "M": (-2.5,  2.0),
    "G": (-1.0,  2.0),
    "C": (-2.5,  0.0),
    "S": (-1.0,  0.0),
    "T": ( 0.5,  1.0),
    "F": ( 1.5,  2.0),
    "B": ( 1.5,  0.0),
    "D": ( 2.5,  0.0),
    "Y": ( 3.0,  1.0),
    "R": ( 3.5, -1.0),
}

NODE_COLOURS = {
    "T": "#5a9bd5",
    "Y": "#ed7d31",
    **{n: "#a9d18e" for n in CONFOUNDERS},
    **{n: "#f4b183" for n in MEDIATORS},
    **{n: "#bfbfbf" for n in COLLIDERS},
}


def render(g: nx.DiGraph, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw_networkx_nodes(
        g, NODE_POSITIONS,
        node_size=2200,
        node_color=[NODE_COLOURS[n] for n in g.nodes],
        edgecolors="black",
        linewidths=1.2,
        ax=ax,
    )
    nx.draw_networkx_edges(
        g, NODE_POSITIONS,
        arrowsize=18,
        edge_color="#444444",
        width=1.4,
        connectionstyle="arc3,rad=0.05",
        ax=ax,
    )
    nx.draw_networkx_labels(
        g, NODE_POSITIONS, labels=NODE_LABELS,
        font_size=8.5, font_family="sans-serif", ax=ax,
    )

    # Legend
    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=10,
                   markerfacecolor="#5a9bd5", markeredgecolor="black",
                   label="exposure (T)"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=10,
                   markerfacecolor="#ed7d31", markeredgecolor="black",
                   label="outcome (Y)"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=10,
                   markerfacecolor="#a9d18e", markeredgecolor="black",
                   label="confounder - adjust"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=10,
                   markerfacecolor="#f4b183", markeredgecolor="black",
                   label="mediator - do NOT adjust"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=10,
                   markerfacecolor="#bfbfbf", markeredgecolor="black",
                   label="collider - do NOT adjust"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, frameon=False)

    ax.set_title("Causal DAG - Olist free-shipping treatment", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    g = build_graph()
    adj = adjustment_set(g)
    print(f"Adjustment set for ({EXPOSURE} -> {OUTCOME}): {sorted(adj)}")
    print("Implied conditional independencies (a _||_ b | conditioning set):")
    for a, b, cond in implied_independencies(g):
        cond_str = ", ".join(sorted(cond)) if cond else "{}"
        print(f"  {a} _||_ {b} | {{{cond_str}}}")
    out_path = FIGURES_DIR / "dag.png"
    render(g, out_path)
    print(f"\nFigure written to: {out_path}")


if __name__ == "__main__":
    main()
