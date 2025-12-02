#!/usr/bin/env python3
# Demo: full pipeline — Class logic → Class relations → Incomparability → Equality → Ratios
# Run:  PYTHONPATH=src python scripts/demo_pipeline.py --n 50 --seed 7 --max-arity 2

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional Rich pretty printing
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.theme import Theme
    from rich import box
except Exception:
    Console = None
    Table = None
    Panel = None
    Theme = None
    box = None

# NetworkX is required for corpus creation
import networkx as nx


# ──────────────────────────────────────────────────────────────────────────────
# Optional console helpers
# ──────────────────────────────────────────────────────────────────────────────
def _console(enabled: bool) -> Optional[Console]:
    if not enabled or Console is None:
        return None
    theme = Theme({
        "ok": "bold green",
        "warn": "bold yellow",
        "err": "bold red",
        "title": "bold cyan",
        "dim": "dim",
    })
    return Console(theme=theme)


def _hr(console: Optional[Console]):
    if console:
        console.print("─" * 80, style="dim")
    else:
        print("─" * 80)


def _panel(console: Optional[Console], title: str):
    if console and Panel is not None:
        console.print(Panel.fit(title, style="title"))
    else:
        print(title)


def _table(console: Optional[Console], title: str, rows: List[Tuple[str, str]]):
    if console and Table is not None:
        t = Table(title=title, box=box.SIMPLE_HEAVY if box else None)
        t.add_column("Type", style="bold")
        t.add_column("Columns")
        for k, v in rows:
            t.add_row(k, v)
        console.print(t)
    else:
        print(title)
        for k, v in rows:
            print(f"  {k:14s} {v}")


# ──────────────────────────────────────────────────────────────────────────────
# TxGraffiti 2025 imports (new architecture)
# ──────────────────────────────────────────────────────────────────────────────
from txgraffiti2025.relations.core import DataModel, MaskCache
from txgraffiti2025.relations.equality import EqualityMiner
from txgraffiti2025.relations.ratios import RatiosMiner, RatiosConfig

# Class logic + relations + incomparability miners
# (These names match your earlier demos/prints; adjust import paths if yours differ.)
from txgraffiti2025.relations.class_logic import ClassLogic          # provides enumerate_conjunctions, etc.
from txgraffiti2025.relations.class_relations_miner import ClassRelationsMiner
from txgraffiti2025.relations.incomparability import IncomparabilityAnalyzer


# ──────────────────────────────────────────────────────────────────────────────
# Graph corpus
# ──────────────────────────────────────────────────────────────────────────────
def _rand_tree(n: int, rng: random.Random) -> nx.Graph:
    # Portable “random tree”: attach each new node to a random previous node
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for v in range(1, n):
        u = rng.randrange(0, v)
        G.add_edge(u, v)
    return G


def _grid_graph(m: int, n: int) -> nx.Graph:
    H = nx.grid_2d_graph(m, n)
    # relabel (r,c) -> r*n + c
    mapping = {rc: rc[0] * n + rc[1] for rc in H.nodes()}
    return nx.relabel_nodes(H, mapping=mapping, copy=True)


def _tri_free(G: nx.Graph) -> bool:
    # Triangle-free iff no cycles of length 3 in undirected simple graph
    # Quick: check any edge participates in a triangle via common neighbors
    for u, v in G.edges():
        if len(set(G[u]).intersection(G[v])) > 0:
            return False
    return True


def _safe_diameter_radius(G: nx.Graph) -> Tuple[float, float]:
    if not nx.is_connected(G):
        return (np.nan, np.nan)
    try:
        d = nx.diameter(G)
    except Exception:
        d = np.nan
    try:
        r = nx.radius(G)
    except Exception:
        r = np.nan
    return float(d), float(r)


def _safe_chordal(G: nx.Graph) -> bool:
    try:
        return bool(nx.is_chordal(G))
    except Exception:
        # Some NX versions lack this; default False (safe)
        return False


def _safe_planar(G: nx.Graph) -> bool:
    try:
        planar, _ = nx.check_planarity(G)
        return bool(planar)
    except Exception:
        return False


def _name(G: nx.Graph, label: str) -> str:
    return label


def _random_corpus(n: int, seed: int) -> List[Tuple[str, nx.Graph]]:
    rng = random.Random(seed)
    graphs: List[Tuple[str, nx.Graph]] = []

    # Some classics
    graphs.append(("Petersen", nx.petersen_graph()))
    graphs.append(("Complete_6", nx.complete_graph(6)))
    graphs.append(("Cycle_8", nx.cycle_graph(8)))
    graphs.append(("Path_10", nx.path_graph(10)))
    graphs.append(("Star_12", nx.star_graph(12)))  # has 13 nodes, we’ll record order from G
    graphs.append(("Balanced_Bipartite_6_6", nx.complete_bipartite_graph(6, 6)))
    graphs.append(("Grid_4x4", _grid_graph(4, 4)))

    # Random trees, ER graphs, and a few small chordal-ish constructions
    for k in [8, 10, 12]:
        graphs.append((f"Tree_{k}", _rand_tree(k, rng)))

    for i in range(max(0, n - len(graphs))):
        # alternate: ER and “slightly chordal-ish” (attach cliques) mini graphs
        if i % 3 == 0:
            m = rng.randrange(8, 14)
            p = rng.uniform(0.1, 0.22)
            G = nx.erdos_renyi_graph(m, p, seed=rng.randrange(10_000))
            graphs.append((f"Gnp_{m}_{p:.2f}_{i}", G))
        elif i % 3 == 1:
            k = rng.randrange(8, 14)
            G = _rand_tree(k, rng)
            # add a random extra edge to create a single cycle
            u, v = rng.sample(list(G.nodes()), 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
            graphs.append((f"Unicyclic_{k}_{i}", G))
        else:
            # small chordal: start with a clique, then attach leaves
            k = rng.randrange(4, 7)
            core = nx.complete_graph(k)
            add = rng.randrange(4, 8)
            for j in range(add):
                v = max(core.nodes()) + 1 if core.nodes() else 0
                u = rng.choice(list(core.nodes()))
                core.add_node(v)
                core.add_edge(u, v)
            graphs.append((f"Chordalish_{k}c+{add}l_{i}", core))

    return graphs[:n]


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction → DataFrame
# ──────────────────────────────────────────────────────────────────────────────
def _feature_row(name: str, G: nx.Graph) -> Dict[str, float]:
    # numeric basics
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degs = [d for _, d in G.degree()]
    min_deg = float(min(degs)) if degs else np.nan
    max_deg = float(max(degs)) if degs else np.nan
    avg_cl = float(nx.average_clustering(G)) if n > 0 else np.nan
    diam, rad = _safe_diameter_radius(G)

    # booleans
    connected = bool(nx.is_connected(G)) if n > 0 else False
    is_tree = bool(nx.is_tree(G)) if n > 0 else False
    bipartite = bool(nx.is_bipartite(G)) if n > 0 else False
    planar = _safe_planar(G)
    chordal = _safe_chordal(G)
    tri_free = _tri_free(G)

    return dict(
        name=name,
        order=float(n),
        size=float(m),
        minimum_degree=min_deg,
        maximum_degree=max_deg,
        diameter=float(diam),
        radius=float(rad),
        avg_clustering=float(avg_cl),

        connected=connected,
        tree=is_tree,
        bipartite=bipartite,
        planar=planar,
        chordal=chordal,
        triangle_free=tri_free,

        dataset_nonempty=True,
    )


def build_dataframe(graphs: List[Tuple[str, nx.Graph]]) -> pd.DataFrame:
    rows = []
    for nm, G in graphs:
        rows.append(_feature_row(nm, G))
    df = pd.DataFrame(rows)
    # Ensure stable dtypes for booleans
    for col in ["connected", "tree", "bipartite", "planar", "chordal", "triangle_free", "dataset_nonempty"]:
        if col in df.columns:
            df[col] = df[col].astype(bool, copy=False)
    return df


def _heuristic_partition(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    bool_like = []
    numeric = []
    for c in df.columns:
        if df[c].dtype == bool:
            bool_like.append(c)
        elif pd.api.types.is_numeric_dtype(df[c].dtype):
            numeric.append(c)
        else:
            # floatify "name" column to keep numeric ops simple (we will still keep the original string in df)
            if c == "name":
                numeric.append(c)
            else:
                # ignore others for mining
                pass
    # keep name numeric only if truly numeric; otherwise drop from numeric list
    if "name" in numeric:
        try:
            pd.to_numeric(df["name"])
        except Exception:
            numeric.remove("name")
    return bool_like, numeric


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PipelineArgs:
    n: int = 50
    seed: int = 7
    max_arity: int = 2
    min_support: float = 0.0
    conjunction_limit: Optional[int] = None
    ratios_target: str = "size"     # choose among numeric columns
    ratios_weight: float = 0.5
    use_rich: bool = True


def run_pipeline(args: PipelineArgs):
    console = _console(args.use_rich)

    _panel(console, "Building a NetworkX graph corpus…")
    graphs = _random_corpus(n=args.n, seed=args.seed)
    df = build_dataframe(graphs)

    bool_like, numeric = _heuristic_partition(df)
    _table(console, "Heuristic column partition (pre-DataModel)", [
        ("Boolean-like", ", ".join(sorted(bool_like))),
        ("Numeric",      ", ".join(sorted(numeric))),
    ])

    # Build model + cache; DataModel infers Exprs and Predicates internally
    model = DataModel(df=df)
    cache = MaskCache(model)

    # ── Class logic: enumerate conjunctions, redundancy/equivalence ──
    _panel(console, "Build hypothesis classes (ClassLogic)")
    logic = ClassLogic(model, cache)
    nonred, red, equiv = logic.normalize()
    logic.sort_by_generality()
    # Summary tables
    # Top (by support)
    # top = sorted([(n, int(cache.mask(p).sum())) for (n, p) in logic._sorted],
    #              key=lambda t: (-t[1], t[0]))
    if console and Table is not None:
        t = Table(title="Top classes by generality (support)", box=box.SIMPLE_HEAVY if box else None)
        t.add_column("Hypothesis (conjunction)")
        t.add_column("Support", justify="right")
        for name, sup in logic._sorted[:12]:
            t.add_row(name, str(sup))
        console.print(t)
    else:
        print("Top classes by generality (support):")
        for name, sup in logic._sorted[:12]:
            print(f"  {name:55s} {sup}")
    # Redundant sample
    if red:
        sample = red[:5]
        if console and Table is not None:
            t = Table(title="Sample redundant classes", box=box.SIMPLE_HEAVY if box else None)
            t.add_column("Hypothesis (redundant)")
            t.add_column("Support", justify="right")
            for n, p in sample:
                t.add_row(n, str(int(cache.mask(p).sum())))
            console.print(t)
        else:
            print("\nSample redundant classes:")
            for n, p in sample:
                print(f"  {n:55s} {int(cache.mask(p).sum())}")

    # Equivalence groups
    if equiv:
        if console and Table is not None:
            t = Table(title="Equivalence groups (first 5)", box=box.SIMPLE_HEAVY if box else None)
            t.add_column("Group")
            t.add_column("Members (same mask, same arity)")
            for i, grp in enumerate(equiv[:5], start=1):
                names = [n for (n, _) in grp]
                t.add_row(f"group {i}", ", ".join(names))
            console.print(t)
        else:
            print("\nEquivalence groups (first 5):")
            for i, grp in enumerate(equiv[:5], start=1):
                names = [n for (n, _) in grp]
                print(f"  group {i}: {', '.join(names)}")

    # ── Class relations ⊆ and ⇔ ──
    _panel(console, "Class relations C1 ⊆ C2 and C1 ⇔ C2 (ClassRelationsMiner)")
    rel_miner = ClassRelationsMiner(model, cache)
    rel_df, _ = rel_miner.make_conjectures(hyps=logic._sorted)
    if rel_df is not None and len(rel_df):
        # show a few
        if console and Table is not None:
            t = Table(title="ClassRelationsMiner: top relations", box=box.SIMPLE_HEAVY if box else None, show_lines=False)
            cols = ["name_i","name_j","nA","nB","nI","viol_i_to_j","viol_j_to_i","rec_i_to_j","rec_j_to_i","jaccard","imp_i_to_j","imp_j_to_i","equiv"]
            for c in cols:
                t.add_column(c)
            for _, r in rel_df.head(20).iterrows():
                t.add_row(*(str(r.get(c)) for c in cols))
            console.print(t) if console else print(rel_df.head(20))
        else:
            print(rel_df.head(20))
    else:
        print("No nontrivial class relations after filtering base-tautologies.")

    # ── Incomparability → |x−y|, min(x,y), max(x,y) ──
    _panel(console, "Incomparability → |x−y|, min(x,y), max(x,y) (IncomparabilityAnalyzer)")
    inc = IncomparabilityAnalyzer(model, cache)
    # analyze under base domain (or TRUE)
    base = getattr(model, "base_pred", None)
    inc_summary = inc.summarize(condition=base, top_k=10, register_new=True, require_balanced=True)
    # Pretty print from analyzer (it returns preformatted blocks)
    print(inc_summary)

    # ── Equality mining: constants + pairwise equalities ──
    _panel(console, "Equality mining: constants and pairwise equalities")
    em = EqualityMiner(model, cache)
    const_df = em.analyze_constants(condition=base, use_base_if_none=True, tol=1e-9, rationalize=True, max_denom=64)
    pair_df  = em.analyze_pair_equalities(condition=base, use_base_if_none=True, tol=1e-9, min_support=0.10, min_eq_rate=0.95)
    # Print small tables
    if console and Table is not None:
        t1 = Table(title="Detected constant-valued invariants on domain", box=box.SIMPLE_HEAVY if box else None)
        for c in ["inv","n_domain","n_finite","support","value","value_frac","max_abs_dev","selected"]:
            t1.add_column(c)
        for _, r in const_df.head(12).iterrows():
            t1.add_row(*(str(r.get(c)) for c in ["inv","n_domain","n_finite","support","value","value_frac","max_abs_dev","selected"]))
        console.print(t1)

        t2 = Table(title="Detected near-equalities x ≈ y on domain", box=box.SIMPLE_HEAVY if box else None)
        for c in ["inv1","inv2","n_domain","n_rows","n_eq","n_neq","rate_eq","support","max_abs_gap","selected"]:
            t2.add_column(c)
        for _, r in pair_df.head(12).iterrows():
            t2.add_row(*(str(r.get(c)) for c in ["inv1","inv2","n_domain","n_rows","n_eq","n_neq","rate_eq","support","max_abs_gap","selected"]))
        console.print(t2)
    else:
        print("Constants (head):\n", const_df.head(12))
        print("\nPairwise equalities (head):\n", pair_df.head(12))

    eq_conjs = em.make_eq_conjectures(constants=const_df, pairs=pair_df, condition=base)
    print(f"\nConstructed {len(eq_conjs)} Eq-conjecture(s).")

    # ── Ratios miner: intricate mixed bounds ──
    _panel(console, "Ratios miner: intricate mixed bounds (sqrt/square mixes)")
    rm = RatiosMiner(model, cache, class_logic=logic, config=RatiosConfig(
        min_touch_keep=3,
        max_denom=30,
        include_base_hypothesis=True,
        use_sorted_conjunctions=True,
        conjunction_limit=args.conjunction_limit,
        exclude_nonpositive_x=True,
        exclude_nonpositive_y=True,
    ))

    # pick a safe target if user’s choice isn’t present
    tgt = args.ratios_target
    if tgt not in model.numeric_cols:
        # prefer size/order/maximum_degree in that order
        for fallback in ["size", "order", "maximum_degree", "diameter", "radius"]:
            if fallback in model.numeric_cols:
                tgt = fallback
                break
    results = rm.run_pipeline(
        target_col=tgt,
        primary=None,
        secondary=None,
        weight=args.ratios_weight,
        min_touch=3,
        hyps=None,
    )
    rm.summarize(results, title=f"Ratios • Intricate Mixed Bounds (target={tgt})")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> PipelineArgs:
    p = argparse.ArgumentParser(description="TxGraffiti2025 full pipeline demo")
    p.add_argument("--n", type=int, default=50, help="number of graphs in corpus")
    p.add_argument("--seed", type=int, default=7, help="random seed")
    p.add_argument("--max-arity", type=int, default=2, help="max conjunction arity")
    p.add_argument("--min-support", type=float, default=0.0, help="min support for conjunctions")
    p.add_argument("--conjunction-limit", type=int, default=None, help="limit printed/used conjunctions")
    p.add_argument("--ratios-target", type=str, default="size", help="target column for ratios miner")
    p.add_argument("--ratios-weight", type=float, default=0.5, help="mix weight w in [0,1]")
    p.add_argument("--no-rich", action="store_true", help="disable rich output")
    a = p.parse_args()
    return PipelineArgs(
        n=a.n,
        seed=a.seed,
        max_arity=a.max_arity,
        min_support=a.min_support,
        conjunction_limit=a.conjunction_limit,
        ratios_target=a.ratios_target,
        ratios_weight=a.ratios_weight,
        use_rich=(not a.no_rich),
    )


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
