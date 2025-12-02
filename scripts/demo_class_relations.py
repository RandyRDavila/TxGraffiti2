#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo: Class relations & implication forms (NetworkX corpus)
- Builds a small corpus of graphs (classics + random)
- Uses DataModel/MaskCache from txgraffiti2025
- ClassLogic: base + boolean conjunctions, redundancy, equivalence, generality order
- ClassRelationsMiner: C1 ⊆ C2 and C1 ⇔ C2
- Implication forms C ⇒ R with tight exemplars
- IncomparabilityAnalyzer: registers |x-y|, min(x,y), max(x,y)

Run:
    PYTHONPATH=src python scripts/demo_class_relations.py
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Optional rich UI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.theme import Theme
    from rich import box
except Exception:
    Console = Table = Panel = Theme = box = None

# NetworkX (robust to version differences)
try:
    import networkx as nx
except Exception as e:
    raise RuntimeError("This demo requires networkx. Please install it: pip install networkx") from e

# TxGraffiti forms
from txgraffiti2025.relations.core import DataModel, MaskCache
from txgraffiti2025.relations.class_logic import ClassLogic
from txgraffiti2025.relations.class_relations_miner import ClassRelationsMiner
from txgraffiti2025.relations.incomparability import IncomparabilityAnalyzer


# ───────────────────────────── UI helpers ───────────────────────────── #

def _console(enable=True):
    if not enable or Console is None:
        return None
    theme = Theme({
        "title": "bold cyan",
        "ok": "bold green",
        "warn": "bold yellow",
        "err": "bold red",
        "dim": "dim",
    })
    return Console(theme=theme)

def _print(console, text="", style=None):
    if console:
        console.print(text, style=style)
    else:
        print(text)

def _h(console, title):
    if console and Panel:
        console.print(Panel.fit(title, style="title", box=box.ROUNDED))
    else:
        print(f"\n== {title} ==")

def _table(console, title, rows, headers):
    if not rows:
        _print(console, "[dim]∅[/dim]" if console else "∅")
        return
    if console and Table:
        t = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False)
        for h in headers:
            t.add_column(h)
        for r in rows:
            t.add_row(*[str(x) for x in r])
        console.print(t)
    else:
        print("\n" + title)
        print("-" * len(title))
        print("\t".join(headers))
        for r in rows:
            print("\t".join(map(str, r)))


# ─────────────────────── NetworkX compatibility ─────────────────────── #

def _nx_random_tree(n: int, seed: int | None = None):
    """Random tree across NX versions."""
    # Newer location
    if hasattr(nx, "generators") and hasattr(nx.generators, "trees") and hasattr(nx.generators.trees, "random_tree"):
        return nx.generators.trees.random_tree(n, seed=seed)
    # Older top-level (rare)
    if hasattr(nx, "random_tree"):
        return nx.random_tree(n, seed=seed)  # type: ignore
    # Fallback: make a random Prüfer sequence
    rng = random.Random(seed)
    prufer = [rng.randrange(0, n) for _ in range(n - 2)]
    # Simple Prüfer to tree converter
    G = nx.Graph()
    G.add_nodes_from(range(n))
    degree = [1] * n
    for v in prufer:
        degree[v] += 1
    from heapq import heapify, heappush, heappop
    leaves = [i for i in range(n) if degree[i] == 1]
    heapify(leaves)
    for v in prufer:
        u = heappop(leaves)
        G.add_edge(u, v)
        degree[u] -= 1
        degree[v] -= 1
        if degree[v] == 1:
            heappush(leaves, v)
    u, v = leaves[0], leaves[1]
    G.add_edge(u, v)
    return G

def _is_chordal(G: nx.Graph) -> bool:
    """Chordal predicate across NX versions."""
    if hasattr(nx, "is_chordal"):
        try:
            return bool(nx.is_chordal(G))  # type: ignore
        except Exception:
            pass
    try:
        from networkx.algorithms.chordal import is_chordal as _is_ch
        return bool(_is_ch(G))
    except Exception:
        return False

def _dodecahedral_graph():
    if hasattr(nx, "dodecahedral_graph"):
        return nx.dodecahedral_graph()
    # Older location
    try:
        return nx.generators.classic.dodecahedral_graph()
    except Exception:
        # Fallback: 20-vertex 3-regular graph from networkx generators if available
        return nx.cycle_graph(20)

def _heawood_graph():
    if hasattr(nx, "heawood_graph"):
        return nx.heawood_graph()
    try:
        return nx.generators.classic.heawood_graph()
    except Exception:
        # Fallback: (3,6)-regular-ish alternative if missing
        return nx.cycle_graph(14)

def _is_planar(G: nx.Graph) -> bool:
    try:
        ok, _ = nx.check_planarity(G)
        return bool(ok)
    except Exception:
        return False

def _triangle_free(G: nx.Graph) -> bool:
    try:
        tri = sum(nx.triangles(G).values())
        return tri == 0
    except Exception:
        return False

def _safe_radius(G: nx.Graph):
    if not nx.is_connected(G):
        return np.nan
    try:
        return nx.radius(G)
    except Exception:
        return np.nan

def _safe_diameter(G: nx.Graph):
    if not nx.is_connected(G):
        return np.nan
    try:
        return nx.diameter(G)
    except Exception:
        return np.nan


# ───────────────────────────── Data builder ─────────────────────────── #

def _random_corpus(n=50, seed=0) -> List[nx.Graph]:
    rng = random.Random(seed)
    graphs: List[nx.Graph] = []

    classics = [
        ("Petersen", nx.petersen_graph()),
        ("Complete_6", nx.complete_graph(6)),
        ("Cycle_8", nx.cycle_graph(8)),
        ("Path_10", nx.path_graph(10)),
        ("Grid_4x4", nx.grid_2d_graph(4, 4)),
        ("Balanced_Bipartite_6_6", nx.complete_bipartite_graph(6, 6)),
        ("Tree_12", _nx_random_tree(12, seed=42)),
        ("Star_12", nx.star_graph(12)),
        ("Dodecahedral", _dodecahedral_graph()),
        ("Heawood", _heawood_graph()),
    ]
    for name, G in classics:
        H = nx.convert_node_labels_to_integers(G)
        H.graph["name"] = name
        graphs.append(H)

    while len(graphs) < n:
        n_nodes = rng.randint(8, 24)
        p = rng.uniform(0.12, 0.25)
        G = nx.fast_gnp_random_graph(n_nodes, p, seed=rng.randint(0, 10**6))
        if G.number_of_nodes() == 0:
            continue
        # take giant component
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            G = G.subgraph(max(comps, key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
        G.graph["name"] = f"Gnp_{n_nodes}_{p:.2f}_{len(graphs)}"
        graphs.append(G)

    return graphs[:n]

def build_dataframe(graphs: List[nx.Graph]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for G in graphs:
        name = G.graph.get("name", f"G_{G.number_of_nodes()}_{G.number_of_edges()}")
        order = G.number_of_nodes()
        size = G.number_of_edges()
        connected = nx.is_connected(G)
        tree = connected and (size == order - 1)
        bipartite = nx.is_bipartite(G)
        planar = _is_planar(G)
        chordal = _is_chordal(G) if order > 0 else False
        triangle_free = _triangle_free(G)
        degs = dict(G.degree())
        minimum_degree = int(min(degs.values())) if order else 0
        maximum_degree = int(max(degs.values())) if order else 0
        radius = _safe_radius(G)
        diameter = _safe_diameter(G)
        try:
            avg_clustering = nx.average_clustering(G) if order > 0 else np.nan
        except Exception:
            avg_clustering = np.nan

        rows.append(dict(
            name=name,
            order=order,
            size=size,
            minimum_degree=minimum_degree,
            maximum_degree=maximum_degree,
            diameter=diameter,
            radius=radius,
            avg_clustering=avg_clustering,

            connected=connected,
            tree=tree,
            bipartite=bipartite,
            planar=planar,
            chordal=chordal,
            triangle_free=triangle_free,

            dataset_nonempty=True,
        ))
    return pd.DataFrame(rows)


# ───────────────────────────── Forms blocks ─────────────────────────── #

def run_class_logic(model: DataModel, cache: MaskCache, *, console, max_arity=2, min_support=0):
    _h(console, "Build hypothesis classes (ClassLogic)")
    logic = ClassLogic(model, cache)

    # inventory
    bool_cols = model.boolean_cols
    num_cols = model.numeric_cols
    _table(console, "Columns by type",
           rows=[
               ("Boolean-like", ", ".join(bool_cols)),
               ("Numeric", ", ".join(num_cols)),
           ],
           headers=["Type", "Columns"])

    # base hypothesis
    base = logic.base_name()
    base_supp = int(cache.mask(logic.base_predicate()).sum())
    _table(console, "Base hypothesis", [(base, base_supp)], headers=["Base", "Support"])

    # enumerate and normalize
    enumerated = logic.enumerate(max_arity=max_arity)
    nonred, red, equiv = logic.normalize()
    logic.sort_by_generality()

    _print(console, f"Enumerated {len(enumerated)} conjunction(s) (max_arity={max_arity}, min_support={min_support})")
    _print(console, f"Nonredundant: {len(nonred)}   Redundant: {len(red)}   Equivalence groups: {len(equiv)}")

    top = []
    for name, pred in (logic.sorted_by_generality() or [])[:16]:
        top.append((name, int(cache.mask(pred).sum())))
    _table(console, "Top classes by generality (support)", top, headers=["Hypothesis (conjunction)", "Support"])

    if red:
        rrows = []
        for name, pred in red[:6]:
            rrows.append((name, int(cache.mask(pred).sum())))
        _table(console, "Sample redundant classes", rrows, headers=["Hypothesis (redundant)", "Support"])

    if equiv:
        gr = []
        for k, group in enumerate(equiv[:5], start=1):
            gr.append((f"group {k}", ", ".join([n for n, _ in group])))
        _table(console, "Equivalence groups (first 5)", gr, headers=["Group", "Members (same mask, same arity)"])

    return logic


def run_class_relations(model: DataModel, cache: MaskCache, logic: ClassLogic, *, console):
    _h(console, "Class relations C1 ⊆ C2 and C1 ⇔ C2 (ClassRelationsMiner)")
    miner = ClassRelationsMiner(model, cache)

    hyps = logic.sort_by_generality()
    stats, conjs = miner.make_conjectures(hyps, min_support=0.05, max_violations=0, min_implication_rate=1.0)

    if stats.empty:
        _print(console, "[dim]No relations under current thresholds.[/dim]")
        return stats, []

    # filter out trivial base ⇒ dataset_nonempty
    view = stats[stats["name_j"] != logic.base_name()].copy().head(20)
    rows = [
        (
            row["name_i"], row["name_j"],
            row["nA"], row["nB"], row["nI"],
            row["viol_i_to_j"], row["viol_j_to_i"],
            f'{row["rec_i_to_j"]:.3f}', f'{row["rec_j_to_i"]:.3f}',
            f'{row["jaccard"]:.3f}',
            "True" if row["imp_i_to_j"] else "False",
            "True" if row["imp_j_to_i"] else "False",
            "True" if row["equiv"] else "False",
        )
        for _, row in view.iterrows()
    ]
    _table(console, "ClassRelationsMiner: top relations",
           rows, headers=[
               "name_i", "name_j", "nA", "nB", "nI",
               "viol_i_to_j", "viol_j_to_i",
               "rec_i_to_j", "rec_j_to_i", "jaccard",
               "imp_i_to_j", "imp_j_to_i", "equiv",
           ])

    # preview conjectures, suppress base-tautologies
    preview = []
    for c in conjs:
        s = str(c)
        if "dataset_nonempty" in s:
            continue
        preview.append((type(c).__name__, s))
    if preview:
        _table(console, "Conjectures (preview)", preview[:20], headers=["Type", "Repr"])
    else:
        _print(console, "[dim]No nontrivial class relations after filtering base-tautologies.[/dim]")

    return stats, conjs


# ───────────────────────── Implication rules: C ⇒ R ─────────────────── #

def _check_connected_radius_diameter(df: pd.DataFrame):
    ok = (df["radius"] <= df["diameter"]) & (df["diameter"] <= 2 * df["radius"])
    holds = int(ok.sum())
    tight = int(((df["diameter"] == df["radius"]) | (df["diameter"] == 2 * df["radius"])).sum())
    violations = int((~ok).sum())
    return len(df), holds, tight, violations, ok

def _check_planar_edge_bound(df: pd.DataFrame):
    ok = pd.Series(True, index=df.index)
    mask = df["order"] >= 3
    ok.loc[mask] = (df.loc[mask, "size"] <= (3 * df.loc[mask, "order"] - 6))
    holds = int(ok.sum())
    tight = int((mask & (df["size"] == (3 * df["order"] - 6))).sum())
    violations = int((~ok).sum())
    return len(df), holds, tight, violations, ok

def _check_triangle_free_clustering_zero(df: pd.DataFrame, tol=1e-12):
    ok = df["avg_clustering"].abs() <= tol
    holds = int(ok.sum())
    tight = int(ok.sum())  # equality throughout
    violations = int((~ok).sum())
    return len(df), holds, tight, violations, ok

def _check_tree_edge_identity(df: pd.DataFrame):
    ok = (df["size"] == df["order"] - 1)
    holds = int(ok.sum())
    tight = int(ok.sum())
    violations = int((~ok).sum())
    return len(df), holds, tight, violations, ok

FORMS_RULES = [
    ("Connected: r ≤ D ≤ 2r", ["dataset_nonempty", "connected"], _check_connected_radius_diameter, "Inequality"),
    ("Planar: m ≤ 3n - 6 (n ≥ 3)", ["dataset_nonempty", "planar"], _check_planar_edge_bound, "Inequality"),
    ("Triangle-free: avg_clustering = 0", ["dataset_nonempty", "triangle_free"], _check_triangle_free_clustering_zero, "Equality"),
    ("Tree: m = n-1", ["dataset_nonempty", "tree"], _check_tree_edge_identity, "Equality"),
]

def run_implication_forms(model: DataModel, cache: MaskCache, *, console):
    _h(console, "Implication forms C ⇒ R (equalities / inequalities)")
    rows = []
    tight_exemplars = []

    for rule_name, pred_names, checker, kind in FORMS_RULES:
        # skip if predicate columns not present (robustness)
        if any(p not in model.preds for p in pred_names):
            continue

        preds = [model.preds[p] for p in pred_names]
        m = cache.mask(preds[0])
        for p in preds[1:]:
            m = m & cache.mask(p)
        df_sub = model.df[m.to_numpy(dtype=bool, copy=False)]
        total, holds, tight, violations, ok_mask = checker(df_sub)
        rows.append((
            " ∧ ".join(pred_names) + " ⇒ " + rule_name,
            kind,
            total, holds, tight, violations
        ))

        # tight exemplars by name (first 5)
        if "name" in df_sub.columns and total > 0:
            if "Tree: m = n-1" in rule_name:
                tight_mask = (df_sub["size"] == (df_sub["order"] - 1))
            elif "Planar: m ≤ 3n - 6" in rule_name:
                tight_mask = (df_sub["order"] >= 3) & (df_sub["size"] == (3 * df_sub["order"] - 6))
            elif "Connected: r ≤ D ≤ 2r" in rule_name:
                tight_mask = (df_sub["diameter"] == df_sub["radius"]) | (df_sub["diameter"] == 2 * df_sub["radius"])
            elif "Triangle-free: avg_clustering = 0" in rule_name:
                tight_mask = (df_sub["avg_clustering"].abs() <= 1e-12)
            else:
                tight_mask = pd.Series(False, index=df_sub.index)

            names = df_sub.loc[tight_mask, "name"].head(5).tolist()
            if names:
                tight_exemplars.append((" ∧ ".join(pred_names) + " ⇒ " + rule_name, ", ".join(names)))

    _table(console, "Implication forms (C ⇒ R) on this corpus",
           rows, headers=["Antecedent ⇒ Relation", "Form", "n (C)", "# holds", "# tight", "# violations"])

    if tight_exemplars:
        _table(console, "Tight exemplars (first 5 per form)", tight_exemplars, headers=["Form", "Graph names"])
    else:
        _print(console, "[dim]No tight exemplars to display.[/dim]")


# ───────────────────── Incomparability registration ─────────────────── #

def run_incomparability(model: DataModel, cache: MaskCache, logic: ClassLogic, *, console):
    _h(console, "Incomparability → |x−y|, min(x,y), max(x,y) (IncomparabilityAnalyzer)")
    inc = IncomparabilityAnalyzer(model, cache)
    C = logic.base_predicate()
    C_name = logic.base_name()

    diag = inc.analyze(condition=C, use_base_if_none=True, include_all_pairs=False)
    if diag.empty:
        _print(console, f"[dim]No incomparable numeric pairs under C = {C_name}.[/dim]")
        return

    cols = ["inv1","inv2","n_rows","n_lt","n_gt","n_eq","balance","support"]
    rows = diag.loc[:, cols].head(15).itertuples(index=False, name=None)
    _table(console, f"Incomparability under C = {C_name} (top 15)", rows, headers=cols)

    absdf = inc.register_absdiff_exprs_for_meaningful_pairs(
        condition=C,
        min_support=0.10, min_side_rate=0.10, min_side_count=5,
        min_median_gap=0.5, min_mean_gap=0.5,
        name_style="pretty",
        overwrite_existing=False,
        top_n_store=20,
    )
    if not absdf.empty:
        cols = ["inv1","inv2","n_rows","median_gap","mean_gap","support","selected","expr_name"]
        rows = absdf.loc[:, cols].head(12).itertuples(index=False, name=None)
        _table(console, "Registered |x-y| (ranked by mean_gap)", rows, headers=cols)
    else:
        _print(console, "[dim]No |x-y| expressions passed thresholds.[/dim]")

    mm_bal = inc.register_minmax_exprs_for_meaningful_pairs(
        condition=C,
        min_support=0.10, min_side_rate=0.10, min_side_count=5, max_eq_rate=0.70,
        key_style="pretty", overwrite_existing=False,
    )
    if not mm_bal.empty:
        cols = ["inv1","inv2","n_rows","rate_lt","rate_gt","rate_eq","support","selected","key_min","key_max"]
        rows = mm_bal.loc[:, cols].query("selected == True").head(12).itertuples(index=False, name=None)
        _table(console, "Registered min/max for balanced incomparable pairs", rows, headers=cols)

    # Summary of new expr names
    added = []
    if inc.abs_exprs_top:
        for k, _e in inc.abs_exprs_top:
            added.append(("absdiff", k))
    if inc.minmax_exprs_top:
        for kmin, kmax in inc.minmax_exprs_top:
            added.append(("minmax", f"{kmin} / {kmax}"))
    if added:
        _table(console, "New expressions registered (preview)", added[:15], headers=["Kind", "Expr name(s)"])
    else:
        _print(console, "[dim]No new expressions registered.[/dim]")


# ───────────────────────────── Main ─────────────────────────────────── #

def main():
    console = _console(enable=True)
    # _print(console, "Building a NetworkX graph corpus…", style="dim")
    # graphs = _random_corpus(n=50, seed=7)
    # df = build_dataframe(graphs)
    from txgraffiti.example_data import graph_data as df
    df['nontrivial'] = df['connected']

    _h(console, "DataFrame snapshot")
    _print(console, f"DataFrame: {len(df)} graphs, {df.shape[1]} feature columns", style="ok")

    # Show column types
    bool_cols = [c for c in df.columns if df[c].dropna().isin([0, 1]).all() or str(df[c].dtype).lower().startswith("bool")]
    num_cols = [c for c in df.columns if c not in bool_cols]
    _table(console, "Heuristic column partition (pre-DataModel)",
           rows=[
               ("Boolean-like", ", ".join(sorted([c for c in df.columns if c in ["connected","tree","bipartite","planar","chordal","triangle_free","dataset_nonempty"]]))),
               ("Numeric", ", ".join(sorted([c for c in df.columns if c not in ["connected","tree","bipartite","planar","chordal","triangle_free","dataset_nonempty"]]))),
           ],
           headers=["Type", "Columns"])

    # Build model
    model = DataModel(df)
    cache = MaskCache(model)

    # 1) Hypothesis classes
    logic = run_class_logic(model, cache, console=console, max_arity=2, min_support=0)

    # 2) Class relations
    run_class_relations(model, cache, logic, console=console)

    # 3) Implication forms with tight exemplars
    run_implication_forms(model, cache, console=console)

    # 4) Incomparability constructors
    run_incomparability(model, cache, logic, console=console)

    # 5) Inspect the model
    print(model.summary())

if __name__ == "__main__":
    main()
